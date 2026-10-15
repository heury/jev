"""Validate every notebook, execute CPU lessons, exercise HTTP contract with a local stub."""
import ast
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'verification'
OUT.mkdir(parents=True, exist_ok=True)
paths = sorted((ROOT / 'output/jupyter-notebook').glob('*.ipynb'))
for path in paths:
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            ast.parse(cell.source, filename=f'{path.name}:cell-{i}')
    print('SCHEMA + SYNTAX PASS:', path.name)
    if path.name.startswith(('01_', '03_')):
        NotebookClient(nb, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
        nbformat.write(nb, OUT / path.name)
        print('CPU EXECUTION PASS:', path.name)

nb = nbformat.read(paths[2], as_version=4)
namespace = {}
for cell in nb.cells:
    if cell.cell_type == 'code':
        exec(compile(cell.source, '<API notebook>', 'exec'), namespace)
validate = namespace['validate_answers']
fixture, questions = namespace['fixture'], namespace['questions']
for change in ['missing', 'nan', 'sum', 'choice', 'score', 'noul']:
    bad = copy.deepcopy(fixture)
    if change == 'missing': del bad['answers']['refund']
    if change == 'nan': bad['answers']['team']['probabilities']['billing'] = float('nan')
    if change == 'sum': bad['answers']['team']['probabilities']['billing'] = .3
    if change == 'choice': bad['answers']['team']['choice'] = 'sales'
    if change == 'score': bad['answers']['urgency']['score'] = 2
    if change == 'noul': bad['answers']['refund']['noul'] = 2
    try:
        validate(bad, questions)
    except (ValueError, KeyError):
        pass
    else:
        raise AssertionError(f'invalid response accepted: {change}')
print('MALFORMED RESPONSE CHECKS PASS: 6')

class Stub(BaseHTTPRequestHandler):
    """Transport fixture only. Does not run a model."""
    seen = []
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.seen.append((self.path, self.headers.get('Authorization'), body))
        if body['state'] == 'trigger-error':
            self.send_response(401)
            self.end_headers()
            return
        data = json.dumps(fixture).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *args):
        pass

server = ThreadingHTTPServer(('127.0.0.1', 0), Stub)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    url = f'http://127.0.0.1:{server.server_port}'
    result, elapsed = namespace['request_decision'](url, 'test-model', '한국어 요청', questions, 'test-token')
    assert result == fixture and elapsed >= 0
    assert Stub.seen[-1] == ('/v1/systemone', 'Bearer test-token',
                            {'model':'test-model', 'state':'한국어 요청', 'questions':questions})
    try:
        namespace['request_decision'](url, 'test-model', 'trigger-error', questions)
    except RuntimeError as exc:
        assert 'HTTP 401' in str(exc)
    else:
        raise AssertionError('HTTP 401 was swallowed')
finally:
    server.shutdown()
    server.server_close()
    thread.join()
print('HTTP TRANSPORT PASS: local fixture server, NOT real model inference')
print('NOT RUN: NVIDIA GPU training and real Decision API inference')
