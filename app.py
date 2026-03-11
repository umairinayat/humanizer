import threading
import os
import time
import json
import logging
from flask import Flask, render_template, request, Response, jsonify

from config import ADAPTER_DIR, CHECKPOINT_DIR
import config
from inference import load_model, humanize, humanize_batch

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_STATUS = {"ready": False, "checkpoint": "Not Loaded"}
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint-10000")

def initialize_model_thread():
    global MODEL_STATUS
    log.info("Initializing model in background...")
    try:
        ckpt_path = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else str(ADAPTER_DIR)
        
        # Load the model directly
        load_model(adapter_path=ckpt_path)
        
        MODEL_STATUS["ready"] = True
        MODEL_STATUS["checkpoint"] = os.path.basename(ckpt_path)
        log.info("Model initialization complete.")
    except Exception as e:
        log.error(f"Error loading model: {e}")
        MODEL_STATUS["checkpoint"] = "Load Failed"

# Start the loading thread before running the app
thread = threading.Thread(target=initialize_model_thread)
thread.daemon = True
thread.start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    return jsonify(MODEL_STATUS)

def split_into_chunks(text, words_per_chunk):
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunks.append(" ".join(words[i:i+words_per_chunk]))
    return chunks

@app.route("/humanize_stream", methods=["POST"])
def humanize_stream():
    data = request.json
    text = data.get("text", "")
    words_per_chunk = data.get("words_per_chunk", 200)
    max_new_tokens = data.get("max_new_tokens", 512)
    temperature = data.get("temperature", 0.85)
    top_p = data.get("top_p", 0.92)
    repetition_penalty = data.get("repetition_penalty", 1.20)
    apply_post = data.get("apply_post", True)
    show_raw = data.get("show_raw", False)

    if not MODEL_STATUS["ready"]:
        return jsonify({"error": "Model not loaded yet. Please wait..."}), 400

    chunks = split_into_chunks(text, words_per_chunk)
    
    def generate():
        start_time_total = time.time()
        
        # Send init
        init_data = {"type": "init", "total": len(chunks)}
        yield f"data: {json.dumps(init_data)}\n\n"
        
        for idx, chunk in enumerate(chunks):
            start_time_chunk = time.time()
            try:
                # Patching GENERATION settings temporarily for this request context
                # Note: In a heavily multi-threaded environment this could cause race conditions, 
                # but for a demo app running synchronously per generation request, it's acceptable.
                orig_top_p = config.GENERATION["top_p"]
                orig_rep = config.GENERATION["repetition_penalty"]
                config.GENERATION["top_p"] = top_p
                config.GENERATION["repetition_penalty"] = repetition_penalty
                
                # Inference
                result = humanize(chunk, temperature=temperature, max_new_tokens=max_new_tokens)
                
                # Restore
                config.GENERATION["top_p"] = orig_top_p
                config.GENERATION["repetition_penalty"] = orig_rep

                raw_text = result if show_raw else None
                if apply_post:
                    # Simple post-processing
                    result = result.replace("  ", " ").strip()
                
                elapsed = round(time.time() - start_time_chunk, 2)
                chunk_data = {
                    "type": "chunk",
                    "idx": idx,
                    "result": result,
                    "raw": raw_text,
                    "elapsed": elapsed
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                
            except Exception as e:
                log.error(f"Error processing chunk {idx}: {e}")
                err_data = {"type": "error", "idx": idx, "error": str(e)}
                yield f"data: {json.dumps(err_data)}\n\n"
                
        total_elapsed = round(time.time() - start_time_total, 2)
        done_data = {"type": "done", "elapsed": total_elapsed}
        yield f"data: {json.dumps(done_data)}\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route("/humanize_batch", methods=["POST"])
def humanize_batch_endpoint():
    """Process all chunks in one batched GPU call, return combined result."""
    data = request.json
    text = data.get("text", "")
    words_per_chunk = data.get("words_per_chunk", 200)
    max_new_tokens = data.get("max_new_tokens", 512)
    temperature = data.get("temperature", 0.85)
    top_p = data.get("top_p", 0.92)
    repetition_penalty = data.get("repetition_penalty", 1.20)
    apply_post = data.get("apply_post", True)

    if not MODEL_STATUS["ready"]:
        return jsonify({"error": "Model not loaded yet. Please wait..."}), 400

    chunks = split_into_chunks(text, words_per_chunk)
    start = time.time()

    orig_top_p = config.GENERATION["top_p"]
    orig_rep = config.GENERATION["repetition_penalty"]
    config.GENERATION["top_p"] = top_p
    config.GENERATION["repetition_penalty"] = repetition_penalty

    try:
        results = humanize_batch(chunks, temperature=temperature, max_new_tokens=max_new_tokens)
    except Exception as e:
        config.GENERATION["top_p"] = orig_top_p
        config.GENERATION["repetition_penalty"] = orig_rep
        log.error(f"Batch inference error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        config.GENERATION["top_p"] = orig_top_p
        config.GENERATION["repetition_penalty"] = orig_rep

    if apply_post:
        results = [r.replace("  ", " ").strip() for r in results]

    elapsed = round(time.time() - start, 2)
    return jsonify({
        "results": results,
        "combined": "\n\n".join(results),
        "total_chunks": len(chunks),
        "elapsed": elapsed,
    })


@app.route("/api/humanize", methods=["POST"])
def api_humanize():
    """
    Simple POST API for external access.

    Request JSON:
        {
            "text": "Your AI-generated text here...",
            "words_per_chunk": 200,        (optional, default 200)
            "temperature": 0.85,           (optional)
            "top_p": 0.92,                 (optional)
            "repetition_penalty": 1.20     (optional)
        }

    Response JSON:
        {
            "output": "Humanized text...",
            "input_words": 123,
            "output_words": 118,
            "elapsed": 4.2
        }
    """
    # Accept JSON regardless of Content-Type header (handles Postman quirks)
    data = request.get_json(force=True, silent=True) or {}
    log.info(f"API request — Content-Type: {request.content_type} | body: {request.data[:200]}")
    text = data.get("text", "").strip()
    if not text:
        return jsonify({
            "error": "Field 'text' is required and cannot be empty",
            "received_body": request.data.decode("utf-8", errors="replace")[:300],
            "content_type": request.content_type,
        }), 400

    if not MODEL_STATUS["ready"]:
        return jsonify({"error": "Model not loaded yet. Please wait..."}), 503

    words_per_chunk     = int(data.get("words_per_chunk", 200))
    temperature         = float(data.get("temperature", 0.85))
    top_p               = float(data.get("top_p", 0.92))
    repetition_penalty  = float(data.get("repetition_penalty", 1.20))

    chunks = split_into_chunks(text, words_per_chunk)
    start  = time.time()

    orig_top_p = config.GENERATION["top_p"]
    orig_rep   = config.GENERATION["repetition_penalty"]
    config.GENERATION["top_p"]              = top_p
    config.GENERATION["repetition_penalty"] = repetition_penalty

    try:
        results = humanize_batch(chunks, temperature=temperature)
    except Exception as e:
        log.error(f"API inference error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        config.GENERATION["top_p"]              = orig_top_p
        config.GENERATION["repetition_penalty"] = orig_rep

    results = [r.replace("  ", " ").strip() for r in results]
    output  = "\n\n".join(results)

    # Trim output to match input word count exactly
    input_word_count  = len(text.split())
    output_words      = output.split()
    if len(output_words) > input_word_count:
        output = " ".join(output_words[:input_word_count])

    elapsed = round(time.time() - start, 2)

    return jsonify({
        "output":  output,
        "elapsed": elapsed,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6005)
