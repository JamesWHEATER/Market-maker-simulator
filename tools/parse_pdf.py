from pdfminer.high_level import extract_text
from pathlib import Path
import sys
p = Path(sys.argv[1])
out = Path(sys.argv[2])
text = extract_text(str(p))
out.write_text(text)
print('wrote', out)
