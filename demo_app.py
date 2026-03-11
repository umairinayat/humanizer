"""
Humanizer — Chunk-parallel streaming demo (LoRA checkpoint-4000).

Strategy:
  1. Try the LoRA adapter first.
  2. If the adapter echoes the input (Jaccard > 0.75), **retry with the adapter
     disabled** so the bare Llama-3.2-3B-Instruct model handles the rewrite.
  3. Apply aggressive post-processing: vocabulary de-AI, contractions, sentence
     restructuring, conversational inserts, and human-mistake injection.
"""
from __future__ import annotations

import argparse, json, queue, random, re, threading, time
from pathlib import Path

import torch
from flask import Flask, Response, jsonify, render_template, request
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import BASE_MODEL, CHECKPOINT_DIR, SYSTEM_PROMPT, QLORA

app = Flask(__name__)

_model            = None
_tokenizer        = None
_checkpoint_label = "not loaded"
_model_lock       = threading.Lock()
CHUNK_WORDS       = 200

# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING TABLES (expanded)
# ══════════════════════════════════════════════════════════════════════════════

_WORD_SWAPS: dict[str, str] = {
    # --- verbs ---
    "utilize": "use", "utilizes": "uses", "utilized": "used", "utilizing": "using",
    "leverage": "use", "leverages": "uses", "leveraged": "used", "leveraging": "using",
    "facilitate": "help", "facilitates": "helps", "facilitated": "helped",
    "implement": "set up", "implements": "sets up", "implemented": "set up",
    "optimize": "improve", "optimizes": "improves", "optimized": "improved",
    "streamline": "simplify", "streamlines": "simplifies", "streamlined": "simplified",
    "delve": "dig", "delves": "digs", "delved": "dug", "delving": "digging",
    "foster": "encourage", "fosters": "encourages", "fostered": "encouraged",
    "navigate": "handle", "navigates": "handles", "navigated": "handled",
    "embark": "start", "embarks": "starts", "embarked": "started",
    "underscore": "highlight", "underscores": "highlights",
    "encompass": "cover", "encompasses": "covers", "encompassed": "covered",
    "necessitate": "need", "necessitates": "needs",
    "elucidate": "explain", "elucidates": "explains",
    "illuminate": "show", "illuminates": "shows",
    "demonstrate": "show", "demonstrates": "shows",
    "constitute": "make up", "constitutes": "makes up",
    "commence": "begin", "commences": "begins", "commenced": "began",
    "ascertain": "find out", "ascertains": "finds out",
    "endeavor": "try", "endeavors": "tries",
    "mitigate": "reduce", "mitigates": "reduces", "mitigated": "reduced",
    "exacerbate": "worsen", "exacerbates": "worsens",
    "proliferate": "spread", "proliferates": "spreads",
    "juxtapose": "compare", "juxtaposes": "compares",
    "substantiate": "back up", "substantiates": "backs up",
    "augment": "boost", "augments": "boosts",
    "perpetuate": "keep going", "perpetuates": "keeps going",
    "exemplify": "show", "exemplifies": "shows",
    "underscore": "stress",
    # --- adjectives / adverbs ---
    "comprehensive": "thorough", "robust": "solid", "pivotal": "key",
    "crucial": "important", "nuanced": "subtle", "holistic": "overall",
    "multifaceted": "complex", "paramount": "top",
    "substantial": "big", "significant": "major", "meticulous": "careful",
    "indispensable": "essential", "intricate": "detailed",
    "profound": "deep", "groundbreaking": "new",
    "unprecedented": "never seen before", "cutting-edge": "latest",
    "state-of-the-art": "top-notch", "innovative": "fresh",
    "transformative": "game-changing", "compelling": "strong",
    "seamless": "smooth", "seamlessly": "smoothly",
    "notably": "especially", "fundamentally": "basically",
    "inherently": "naturally", "ultimately": "in the end",
    "essentially": "basically", "undeniably": "clearly",
    "arguably": "some would say",
    "consequently": "so", "subsequently": "then", "furthermore": "also",
    "moreover": "plus", "additionally": "also", "nevertheless": "still",
    "nonetheless": "even so", "henceforth": "from now on",
    "notwithstanding": "despite", "conversely": "on the flip side",
    "concurrently": "at the same time",
    # --- nouns ---
    "paradigm": "approach", "synergy": "teamwork", "tapestry": "mix",
    "realm": "area", "plethora": "bunch", "myriad": "many",
    "landscape": "scene", "trajectory": "path", "framework": "setup",
    "methodology": "method", "infrastructure": "setup",
    "facet": "side", "spectrum": "range", "catalyst": "trigger",
    "cornerstone": "foundation", "endeavor": "effort",
    "ramification": "consequence", "implication": "effect",
    "discourse": "discussion", "narrative": "story",
    "stakeholder": "person involved", "proponent": "supporter",
    "detriment": "harm", "magnitude": "size",
    # --- phrases ---
    "aforementioned": "mentioned", "In conclusion": "All in all",
    "To summarize": "To sum up", "Furthermore": "Also",
    "Moreover": "Plus", "Additionally": "On top of that",
    "In addition": "Also", "It is worth noting": "Worth mentioning",
    "It should be noted": "Note that", "In today's world": "These days",
    "plays a crucial role": "matters a lot",
    "a wide range of": "all kinds of", "In order to": "To",
    "Due to the fact that": "Because",
    "At the end of the day": "When it comes down to it",
    "It goes without saying": "Obviously",
    "needless to say": "obviously",
    "It is important to note": "Keep in mind",
    "In light of": "Given", "With regard to": "About",
    "In the context of": "When it comes to",
    "On the other hand": "Then again",
    "As a result": "Because of this", "For instance": "Like",
    "In particular": "Especially", "To a great extent": "Mostly",
    "In essence": "Basically", "By and large": "Mostly",
    "In the realm of": "In", "With respect to": "About",
    "As mentioned earlier": "Like I said",
    "Serves as": "Works as", "serves as": "works as",
    "Plays a role": "Matters", "plays a role": "matters",
    "It is evident": "It's clear", "it is evident": "it's clear",
    "A plethora of": "A bunch of", "a plethora of": "a bunch of",
    "In a nutshell": "Basically",
    "To put it simply": "Simply put",
    "Take into account": "Consider",
    "take into account": "consider",
}

_PHRASE_SWAPS = [
    (r'\bIt is important to\b', 'You should'),
    (r'\bIt is essential to\b', 'You need to'),
    (r'\bIt is evident that\b', "It's clear that"),
    (r'\bThere are many\b', "You'll find plenty of"),
    (r'\bThere is a\b', "There's a"),
    (r'\bdo not\b', "don't"), (r'\bcan not\b', "can't"),
    (r'\bcannot\b', "can't"), (r'\bwill not\b', "won't"),
    (r'\bdoes not\b', "doesn't"), (r'\bis not\b', "isn't"),
    (r'\bare not\b', "aren't"), (r'\bwas not\b', "wasn't"),
    (r'\bwere not\b', "weren't"), (r'\bhave not\b', "haven't"),
    (r'\bhas not\b', "hasn't"), (r'\bhad not\b', "hadn't"),
    (r'\bwould not\b', "wouldn't"), (r'\bcould not\b', "couldn't"),
    (r'\bshould not\b', "shouldn't"),
    (r'\bI am\b', "I'm"), (r'\bI have\b', "I've"), (r'\bI will\b', "I'll"),
    (r'\bwe are\b', "we're"), (r'\bthey are\b', "they're"),
    (r'\bwe have\b', "we've"), (r'\byou are\b', "you're"),
    (r'\bit is\b', "it's"), (r'\bthat is\b', "that's"),
    (r'\bwho is\b', "who's"), (r'\bwhat is\b', "what's"),
    (r'\blet us\b', "let's"), (r'\bthey have\b', "they've"),
    (r'\bwho are\b', "who're"), (r'\bthere will\b', "there'll"),
    (r'\bhe is\b', "he's"), (r'\bshe is\b', "she's"),
    (r'\bwe will\b', "we'll"), (r'\bthey will\b', "they'll"),
]


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _apply_word_swaps(text: str) -> str:
    for ai, human in _WORD_SWAPS.items():
        text = text.replace(ai, human)
        text = text.replace(ai.capitalize(), human.capitalize())
    return text

def _apply_phrase_swaps(text: str) -> str:
    for pat, repl in _PHRASE_SWAPS:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text

def _clean_markdown(text: str) -> str:
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'—', ' - ', text)
    text = re.sub(r',\s*,', ',', text)
    return text.strip()


# ── Sentence restructuring ───────────────────────────────────────────────

_INTERJECTIONS = [
    "You see, ", "Here's the thing - ", "Honestly, ", "Look, ",
    "The thing is, ", "Truth be told, ", "In a way, ", "Funny enough, ",
    "To be fair, ", "That said, ", "I mean, ", "Right, so ",
    "Basically, ", "Okay so ", "Now, ", "Well, ", "Thing is, ",
    "Point being, ", "For what it's worth, ", "Real talk, ",
]

_SENTENCE_STARTERS = [
    "And ", "But ", "So ", "Plus ", "Also ", "Still, ",
    "Anyway, ", "Now ", "Sure, ", "Granted, ",
]

def _restructure_sentences(text: str, intensity: str = "standard") -> str:
    paragraphs = text.split('\n')
    result = []

    for para in paragraphs:
        if len(para.strip()) < 30:
            result.append(para); continue

        sentences = re.split(r'(?<=[.!?])\s+', para)
        new_sents = []
        i = 0
        while i < len(sentences):
            s = sentences[i]
            words = s.split()

            # Merge short consecutive sentences (~30% chance)
            if (len(words) < 8 and i + 1 < len(sentences)
                    and len(sentences[i+1].split()) < 12
                    and random.random() < 0.35):
                conj = random.choice([', and ', ' - ', ', so ', ', plus '])
                nxt = sentences[i+1]
                merged = s.rstrip('.!?') + conj + nxt[0].lower() + nxt[1:]
                new_sents.append(merged)
                i += 2; continue

            # Split very long sentences (~40% chance)
            if len(words) > 25 and random.random() < 0.45:
                for sp in [' but ', ' and ', ' which ', ' although ', ' because ', ' however ']:
                    pos = s.lower().find(sp)
                    if len(s) * 0.25 < pos < len(s) * 0.75:
                        first = s[:pos].rstrip(',') + '.'
                        rest = s[pos + len(sp):].strip()
                        if rest:
                            starter = random.choice(_SENTENCE_STARTERS) if random.random() < 0.4 else ''
                            rest = starter + rest[0].upper() + rest[1:]
                        new_sents.append(first)
                        if rest: new_sents.append(rest)
                        break
                else:
                    new_sents.append(s)
            else:
                new_sents.append(s)
            i += 1

        # Add interjections
        freq = 4 if intensity == "comprehensive" else 6
        for j in range(len(new_sents)):
            if j > 0 and j % freq == 0 and random.random() < 0.5:
                intj = random.choice(_INTERJECTIONS)
                s = new_sents[j]
                if s and s[0].isupper():
                    new_sents[j] = intj + s[0].lower() + s[1:]

        # Randomly start some sentences with casual connector
        for j in range(len(new_sents)):
            if j > 1 and random.random() < 0.15:
                s = new_sents[j]
                if s and s[0].isupper() and not any(s.startswith(x) for x in _INTERJECTIONS + _SENTENCE_STARTERS):
                    new_sents[j] = random.choice(_SENTENCE_STARTERS) + s[0].lower() + s[1:]

        result.append(' '.join(new_sents))

    # Occasionally merge very short paragraphs
    merged = []
    for p in result:
        if merged and len(merged[-1].split()) < 15 and len(p.split()) < 15 and random.random() < 0.3:
            merged[-1] = merged[-1] + ' ' + p
        else:
            merged.append(p)
    return '\n'.join(merged)


# ── Human mistakes injection ─────────────────────────────────────────────

def _inject_human_mistakes(text: str) -> str:
    paragraphs = text.split('\n')
    processed = []
    MISTAKES = ['swap_letters','extra_space','missing_comma','repeated_word',
                'wrong_homophone','missing_apostrophe','lowercase_after_period',
                'wrong_preposition','missing_article']

    for para in paragraphs:
        if len(para.strip()) < 20:
            processed.append(para); continue
        sents = para.split('. ')
        total = len(sents)
        if total < 2:
            processed.append(para); continue

        n_mod = max(1, min(5, int(total * 0.18)))
        indices = random.sample(range(total), min(n_mod, total))

        for idx in indices:
            s = sents[idx]
            if len(s) < 15: continue
            m = random.choice(MISTAKES)

            if m == 'swap_letters':
                ws = s.split()
                c = [i for i,w in enumerate(ws) if len(w) > 4 and w.isalpha()]
                if c:
                    wi = random.choice(c); w = ws[wi]
                    p = random.randint(1, len(w)-2)
                    ws[wi] = w[:p]+w[p+1]+w[p]+w[p+2:]
                    sents[idx] = ' '.join(ws)
            elif m == 'extra_space':
                ws = s.split()
                if len(ws) > 3:
                    p = random.randint(1, len(ws)-2); ws[p] += ' '
                    sents[idx] = ' '.join(ws)
            elif m == 'missing_comma':
                if ',' in s:
                    pos = random.choice([i for i,c in enumerate(s) if c==','])
                    sents[idx] = s[:pos]+s[pos+1:]
            elif m == 'repeated_word':
                ws = s.split()
                c = [i for i,w in enumerate(ws) if w.lower() in ('the','to','a','is','in','of','and','that')]
                if c:
                    wi = random.choice(c); ws.insert(wi+1, ws[wi])
                    sents[idx] = ' '.join(ws)
            elif m == 'wrong_homophone':
                HP = {'their':'there','there':'their','its':"it's","it's":'its',
                      'your':"you're","you're":'your','than':'then','then':'than',
                      'affect':'effect','effect':'affect','lose':'loose'}
                ws = s.split()
                c = [i for i,w in enumerate(ws) if w.lower().rstrip('.,!?') in HP]
                if c:
                    wi = random.choice(c); raw = ws[wi].lower().rstrip('.,!?')
                    sfx = ws[wi][len(raw):]; ws[wi] = HP[raw]+sfx
                    sents[idx] = ' '.join(ws)
            elif m == 'missing_apostrophe':
                for correct,wrong in {"don't":'dont',"it's":'its',"can't":'cant',
                      "won't":'wont',"doesn't":'doesnt',"they're":'theyre',
                      "we're":'were',"didn't":'didnt'}.items():
                    if correct in s:
                        sents[idx] = s.replace(correct, wrong, 1); break
            elif m == 'lowercase_after_period':
                ws = s.split()
                if ws and ws[0][0:1].isupper() and random.random() > 0.5:
                    ws[0] = ws[0][0].lower()+ws[0][1:]
                    sents[idx] = ' '.join(ws)
            elif m == 'wrong_preposition':
                for o,r in {' in the ':' on the ',' on the ':' in the ',
                            ' at the ':' in the ',' for the ':' to the '}.items():
                    if o in s: sents[idx] = s.replace(o, r, 1); break
            elif m == 'missing_article':
                for phrase in [' the ',' a ',' an ']:
                    pos = s.find(phrase)
                    if pos > 0 and random.random() > 0.6:
                        sents[idx] = s[:pos]+' '+s[pos+len(phrase):]; break

        sents = [s for s in sents if s.strip()]
        processed.append('. '.join(sents))
    return '\n'.join(processed)


# ── Echo detection ───────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def _is_echo(raw: str, original: str) -> bool:
    return _jaccard(raw, original) > 0.75


# ── Full pipeline ────────────────────────────────────────────────────────

def _post_process(raw: str, original: str, strategy: str = "standard",
                  inject_mistakes: bool = True) -> str:
    text = raw.strip()

    # If still an echo after base-model retry, start from original
    if _is_echo(text, original):
        text = original

    text = _clean_markdown(text)
    text = _apply_word_swaps(text)
    text = _apply_phrase_swaps(text)

    intensity = "comprehensive" if strategy in ("comprehensive","gptzero") else "standard"
    text = _restructure_sentences(text, intensity)

    if inject_mistakes:
        text = _inject_human_mistakes(text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint() -> str:
    ckpts = sorted(
        [p for p in CHECKPOINT_DIR.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {CHECKPOINT_DIR}")
    return str(ckpts[-1])

def load_model(lora_path: str) -> None:
    global _model, _tokenizer, _checkpoint_label
    print(f"\n[demo] Base    : {BASE_MODEL}")
    print(f"[demo] Adapter : {lora_path}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=QLORA["load_in_4bit"],
        bnb_4bit_quant_type=QLORA["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=QLORA["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    base.config.use_cache = True
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    _model, _tokenizer, _checkpoint_label = model, tok, Path(lora_path).name
    print(f"[demo] Ready   : {_checkpoint_label}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def split_into_chunks(text: str, wpc: int = CHUNK_WORDS) -> list[str]:
    sents = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    chunks, cur, cur_wc = [], [], 0
    for s in sents:
        wc = len(s.split())
        if cur_wc + wc > wpc and cur:
            chunks.append(" ".join(cur)); cur, cur_wc = [], 0
        cur.append(s); cur_wc += wc
    if cur: chunks.append(" ".join(cur))
    return chunks or [text.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE  —  adapter-first → base-model fallback
# ══════════════════════════════════════════════════════════════════════════════

_REWRITE_SYSTEM = (
    "You are a skilled human writer. Rewrite the user's text so it sounds "
    "completely natural and human-written. Change sentence structures freely, "
    "use contractions where natural, vary sentence lengths between short and long, "
    "add small filler words or asides if fitting, and make it feel like a real "
    "person wrote it from scratch. Avoid bullet points, numbered lists, or "
    "markdown formatting. Keep all the original facts and meaning intact. "
    "Output ONLY the rewritten text — no preamble, no options, no commentary."
)

def _build_prompt(chunk: str, system: str, use_training_format: bool = False) -> str:
    """Build chat prompt. use_training_format=True matches the exact SFT data."""
    if use_training_format:
        user_msg = f"Rewrite the following text in a natural, human-written style:\n\n{chunk}"
    else:
        user_msg = (
            f"Rewrite the text below so it reads like a real person wrote it. "
            f"Keep all facts. Output only the rewritten text, nothing else.\n\n{chunk}"
        )
    messages = [
        {"role": "system",  "content": system},
        {"role": "user",    "content": user_msg},
    ]
    try:
        return _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return (f"<|system|>\n{system}\n"
                f"<|user|>\nRewrite this in your own words, naturally:\n\n{chunk}\n"
                "<|assistant|>\n")

def _generate(prompt: str, max_new: int, temp: float, top_p: float,
              rep_pen: float) -> str:
    """Run generation, return decoded new tokens only."""
    inputs_cpu = _tokenizer(prompt, return_tensors="pt")
    inputs     = {k: v.to(_model.device) for k, v in inputs_cpu.items()}
    input_len  = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new, temperature=temp, top_p=top_p,
            do_sample=True, repetition_penalty=rep_pen,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id,
        )
    return _tokenizer.decode(out[0][input_len:].cpu(), skip_special_tokens=True).strip()


def infer_chunk(
    chunk: str,
    strategy: str        = "comprehensive",
    max_new_tokens: int  = 400,
    temperature: float   = 0.75,
    top_p: float         = 0.92,
    rep_pen: float       = 1.15,
    inject_mistakes: bool = True,
    apply_post: bool     = True,
) -> tuple[str, str]:
    """
    1. Try with LoRA adapter (training prompt format).
    2. If echo → retry with adapter DISABLED (base instruct model + rewrite prompt).
    3. Post-process (if apply_post is True).
    Returns (final_text, raw_text).
    """
    # ── Attempt 1: LoRA adapter (EXACT training prompt format) ──
    prompt_adapter = _build_prompt(chunk, SYSTEM_PROMPT, use_training_format=True)

    with _model_lock:
        raw = _generate(prompt_adapter, max_new_tokens, temperature, top_p, rep_pen)

        # ── Attempt 2: base model if adapter echoed ──
        if _is_echo(raw, chunk):
            prompt_base = _build_prompt(chunk, _REWRITE_SYSTEM, use_training_format=False)
            with _model.disable_adapter():
                raw = _generate(prompt_base, max_new_tokens,
                                min(temperature + 0.15, 1.1),   # slightly higher temp
                                top_p, rep_pen)

    if apply_post:
        final = _post_process(raw, chunk, strategy, inject_mistakes)
    else:
        final = raw.strip()
    return final, raw.strip()


# ══════════════════════════════════════════════════════════════════════════════
# SSE ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

def _sse(d: dict) -> str:
    return f"data: {json.dumps(d)}\n\n"


@app.route("/humanize_stream", methods=["POST"])
def humanize_stream():
    if _model is None:
        return jsonify({"error": "Model not loaded"}), 503
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400

    strategy       = data.get("strategy",               "comprehensive")
    max_new_tokens = int(data.get("max_new_tokens",      400))
    temperature    = float(data.get("temperature",       0.75))
    top_p          = float(data.get("top_p",             0.92))
    rep_pen        = float(data.get("repetition_penalty",1.15))
    wpc            = int(data.get("words_per_chunk",     CHUNK_WORDS))
    apply_post     = bool(data.get("apply_post",         True))
    show_raw       = bool(data.get("show_raw",           False))
    inject_mistakes= bool(data.get("inject_mistakes",    True))

    chunks = split_into_chunks(text, wpc)

    def generate():
        yield _sse({"type": "init", "total": len(chunks), "chunks": chunks})
        t0 = time.time()
        q: queue.Queue = queue.Queue()

        def worker(idx):
            try:
                t1  = time.time()
                out = infer_chunk(chunks[idx], strategy, max_new_tokens,
                                  temperature, top_p, rep_pen,
                                  inject_mistakes if apply_post else False,
                                  apply_post)
                q.put(("ok", (idx, out, round(time.time()-t1, 2))))
            except Exception as e:
                import traceback; traceback.print_exc()
                q.put(("err", (idx, str(e))))

        threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(len(chunks))]
        for t in threads: t.start()

        done = 0
        while done < len(chunks):
            status, payload = q.get()
            if status == "err":
                idx, msg = payload
                yield _sse({"type": "error", "idx": idx, "message": msg})
            else:
                idx, result_tuple, elapsed = payload
                final, raw_text = result_tuple
                evt = {"type": "chunk", "idx": idx, "result": final,
                       "elapsed": elapsed}
                if show_raw:
                    evt["raw"] = raw_text
                yield _sse(evt)
            done += 1
        for t in threads: t.join()
        yield _sse({"type": "done", "elapsed": round(time.time()-t0, 2)})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering":"no","Cache-Control":"no-cache"})


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    return jsonify({"checkpoint": _checkpoint_label, "ready": _model is not None,
                    "chunk_words": CHUNK_WORDS})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    lora_path = args.checkpoint or find_latest_checkpoint()
    load_model(lora_path)
    print(f"[demo] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
