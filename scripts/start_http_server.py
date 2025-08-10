#!/usr/bin/env python3
"""
Simple HTTP server script for serving static files.
"""

import argparse
import http.server
import os
import socketserver
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a simple HTTP server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--directory", type=str, default=".", help="Directory to serve (default: current directory)")
    args = parser.parse_args()

    # Wait for the directory to be created if it doesn't exist
    while not os.path.exists(args.directory):
        print(f"Waiting for directory '{args.directory}' to be created...")
        time.sleep(10)

    print(f"Directory '{args.directory}' found, starting server on port {args.port}...")
    os.chdir(args.directory)

    # Check and kill process on port if occupied
    try:
        # sudo apt update && sudo apt install lsof -y
        result = subprocess.run(["lsof", "-ti", f":{args.port}"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            pid = result.stdout.strip()

            # Get process info
            try:
                ps_result = subprocess.run(["ps", "-p", pid, "-o", "pid,ppid,command"], capture_output=True, text=True)
                if ps_result.returncode == 0:
                    process_info = ps_result.stdout.strip().split("\n")[1]  # Skip header
                    print(f"Port {args.port} is occupied by process {pid}: {process_info}")
                else:
                    print(f"Port {args.port} is occupied by process {pid}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"Port {args.port} is occupied by process {pid}")

            print(f"Killing process {pid}...")
            subprocess.run(["kill", "-9", pid], check=True)
            time.sleep(10)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # Ignore if lsof or kill not available

    # Create the server
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"Server started at http://localhost:{args.port}")
        print(f"Serving directory: {os.path.abspath(args.directory)}")
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")


if __name__ == "__main__":
    main()
