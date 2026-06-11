from pathlib import Path
import os

root = Path(__file__).resolve().parents[1]
print('project root:', root)
for candidate in [root / '.env', Path.cwd() / '.env']:
    print('checking', candidate)
    print('exists?', candidate.exists())
    if candidate.exists():
        txt = candidate.read_text(encoding='utf-8')
        print('raw .env:')
        print(repr(txt))
        for line in txt.splitlines():
            s=line.strip()
            if not s or s.startswith('#') or '=' not in s:
                continue
            k,v = s.split('=',1)
            k=k.strip(); v=v.strip().strip('"').strip("'")
            print('parsed',k,repr(v))

print('os.environ samples:')
print('GROQ_API_KEY ->', os.environ.get('GROQ_API_KEY'))
print('GOOGLE_API_KEY ->', os.environ.get('GOOGLE_API_KEY'))
