# Birthday Website

This project is a static birthday website served by a small Python HTTP server with support for audio/video range requests.

## Requirements

- Python 3

## Run locally

From the project folder:

```bash
python3 server.py
```

Then open:

```text
http://localhost:6010
```

## Why use `server.py`

Do not open `index.html` directly in the browser. Run the local server instead so the media files in `Am/` load and stream correctly.

## Stop the server

Press `Ctrl+C` in the terminal where the server is running.

## Project files

- `index.html` - main birthday webpage
- `server.py` - local Python server
- `Am/` - images, videos, and birthday audio files

## Change the port

If port `6010` is already in use, update the `PORT` value in `server.py` and restart the server.

## Troubleshooting

If you get an "address already in use" error, either stop the process already using port `6010` or change the `PORT` value in `server.py`.
