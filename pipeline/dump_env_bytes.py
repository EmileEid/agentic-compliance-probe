from pathlib import Path
p=Path(__file__).resolve().parents[1] / '.env'
b=p.read_bytes()
print('PATH',p)
print('LENGTH',len(b))
print('REPR',repr(b))
print('LINES', [repr(x) for x in b.splitlines()])
print('DECODE', b.decode('utf-8', errors='replace'))
