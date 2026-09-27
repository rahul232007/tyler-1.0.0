import json
import time
import urllib.request
import urllib.error

BASE = 'http://127.0.0.1:8004'

def req(path, method='GET', payload=None, token=None):
    headers = {}
    data = None
    if payload is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(payload).encode('utf-8')
    if token is not None:
        headers['Authorization'] = f'Bearer {token}'
    request = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=90) as resp:
            body = resp.read().decode('utf-8', errors='replace')
            return resp.status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8', errors='replace')
    except Exception as e:
        return 'EXCEPTION', str(e)

print('STATUS_CHECK', req('/api/v1/chat/status', 'GET'))
email = 'ollama_force_' + str(int(time.time())) + '@example.com'
register = req('/api/v1/auth/register', 'POST', {'email': email, 'password': 'TestPass123!', 'display_name': 'Ollama Force'})
print('REGISTER', register)
login = req('/api/v1/auth/login', 'POST', {'email': email, 'password': 'TestPass123!'})
print('LOGIN', login)
if login[0] != 200:
    raise SystemExit(0)
token = json.loads(login[1]).get('access_token')
conv = req('/api/v1/conversations/', 'POST', {'title': 'Ollama validation', 'model_provider': 'ollama'}, token)
print('CONVERSATION', conv)
conv_id = json.loads(conv[1]).get('id') if conv[0] == 201 else None
chat = req('/api/v1/chat/', 'POST', {'message': 'Reply with exactly: ollama-ok', 'stream': False, 'conversation_id': conv_id}, token)
print('CHAT', chat)
