#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CERT_DIR="$SCRIPT_DIR/certs"
STATIC_DIR="$SCRIPT_DIR/static"

if [ ! -f "$CERT_DIR/mkcert.pem" ] || [ ! -f "$CERT_DIR/mkcert-key.pem" ]; then
  echo "Missing TLS certs. Generate with: mkcert -cert-file certs/mkcert.pem -key-file certs/mkcert-key.pem localhost 127.0.0.1"
  exit 1
fi

cd "$STATIC_DIR"

python3 -c "
import http.server, ssl
server = http.server.HTTPServer(('0.0.0.0', 8080), http.server.SimpleHTTPRequestHandler)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain('$CERT_DIR/mkcert.pem', '$CERT_DIR/mkcert-key.pem')
server.socket = ctx.wrap_socket(server.socket, server_side=True)
print('Serving HTTPS on https://localhost:8080')
server.serve_forever()
"
