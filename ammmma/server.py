#!/usr/bin/env python3
"""
Birthday Website Server - Supports range requests for video/audio playback
"""
import os
import re
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 6010

class RangeRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler that supports Range requests for video/audio streaming."""

    def do_GET(self):
        # Add proper MIME types
        mimetypes.add_type('video/mp4', '.mp4')
        mimetypes.add_type('audio/mpeg', '.mp3')
        mimetypes.add_type('image/jpeg', '.jpg')
        mimetypes.add_type('image/png', '.png')

        # Check for Range header
        range_header = self.headers.get('Range')
        if range_header:
            self.handle_range_request(range_header)
        else:
            super().do_GET()

    def handle_range_request(self, range_header):
        path = self.translate_path(self.path)
        
        if not os.path.isfile(path):
            self.send_error(404, 'File not found')
            return

        file_size = os.path.getsize(path)
        
        # Parse range header
        match = re.match(r'bytes=(\d+)-(\d*)', range_header)
        if not match:
            self.send_error(416, 'Invalid range')
            return

        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else file_size - 1
        
        if start >= file_size:
            self.send_error(416, 'Range not satisfiable')
            return

        end = min(end, file_size - 1)
        content_length = end - start + 1

        content_type, _ = mimetypes.guess_type(path)
        if content_type is None:
            content_type = 'application/octet-stream'

        self.send_response(206)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
        self.send_header('Content-Length', str(content_length))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'public, max-age=3600')
        self.end_headers()

        with open(path, 'rb') as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format, *args):
        # Quieter logging - only errors
        if '404' in str(args) or '500' in str(args):
            super().log_message(format, *args)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = HTTPServer(('0.0.0.0', PORT), RangeRequestHandler)
    print(f'Birthday server running on http://0.0.0.0:{PORT}')
    print(f'Serving files from: {os.getcwd()}')
    server.serve_forever()
