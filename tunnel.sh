#!/bin/bash
# Auto-restart cloudflared tunnel forever
URL_FILE="/home/kashif/magface-active-learning/u/uu/code/humanizer/tunnel_url.txt"

while true; do
    echo "[$(date)] Starting cloudflared tunnel..."

    # Start tunnel, capture URL from output
    cloudflared tunnel --url http://localhost:6005 --protocol http2 2>&1 | while IFS= read -r line; do
        echo "$line"
        # Extract and save the public URL
        if echo "$line" | grep -q "trycloudflare.com"; then
            URL=$(echo "$line" | grep -o 'https://[^ ]*trycloudflare.com')
            if [ -n "$URL" ]; then
                echo "$URL" > "$URL_FILE"
                echo "=========================================="
                echo "  PUBLIC URL: $URL"
                echo "  API:        $URL/api/humanize"
                echo "  Web UI:     $URL"
                echo "=========================================="
            fi
        fi
    done

    echo "[$(date)] Tunnel died. Restarting in 5s..."
    sleep 5
done
