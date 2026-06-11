from pathlib import Path
import os, json, requests
# Load .env manually
root = Path(__file__).resolve().parents[1]
for candidate in [root / '.env', Path.cwd() / '.env']:
    if candidate.exists():
        for line in candidate.read_text(encoding='utf-8').splitlines():
            s=line.strip()
            if not s or s.startswith('#') or '=' not in s:
                continue
            k,v = s.split('=',1)
            k=k.strip(); v=v.strip().strip('"').strip("'")
            if k and k not in os.environ and v:
                os.environ[k]=v

key = os.environ.get('GOOGLE_API_KEY')
print('Loaded key present:', bool(key))
if not key:
    print('No key available to query; aborting')
    raise SystemExit(1)

# Some environments provide an OAuth access token (starts with 'AQ.' or 'ya29.')
# which must be sent as a Bearer token in the Authorization header. If the
# value appears to be an API key, send it as the `key` query parameter.
use_auth_header = isinstance(key, str) and (key.startswith('AQ.') or key.startswith('ya29.'))
if use_auth_header:
    url = 'https://generativelanguage.googleapis.com/v1beta2/models'
    headers = {'Authorization': f'Bearer {key}'}
    print('Requesting', url, '(Authorization header)')
    resp = requests.get(url, headers=headers, timeout=15)
else:
    url = f'https://generativelanguage.googleapis.com/v1beta2/models?key={key}'
    print('Requesting', url)
    resp = requests.get(url, timeout=15)
print('Status:', resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2))
except Exception:
    print('Response text:', resp.text)
