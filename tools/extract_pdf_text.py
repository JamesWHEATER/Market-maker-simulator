import sys
from pathlib import Path
p = Path(sys.argv[1])
out = Path(sys.argv[2])
data = p.read_bytes()
# extract printable ASCII runs
import string
printable = set(bytes(string.printable, 'ascii'))
runs = []
run = bytearray()
for b in data:
    if b in printable:
        run.append(b)
    else:
        if len(run) >= 6:
            runs.append(run.decode('ascii', errors='ignore'))
        run = bytearray()
if len(run) >= 6:
    runs.append(run.decode('ascii', errors='ignore'))
# write joined text
out.write_text('\n'.join(runs))
print(f'Wrote {len(runs)} text runs to {out}')
