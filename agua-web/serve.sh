#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/static"

python3 -c "
import http.server, ssl
server = http.server.HTTPServer(('0.0.0.0', 8080), http.server.SimpleHTTPRequestHandler)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain('mkcert.pem', 'mkcert-key.pem')
server.socket = ctx.wrap_socket(server.socket, server_side=True)
print('Serving HTTPS on https://localhost:8080')
server.serve_forever()
"
