import sys
path = r'C:\Users\BIN_Diesel\Documents\tAgent\train_ppo.py'
with open(path, 'rb') as f:
    data = f.read()
clean = data.replace(b'\x00', b'')
with open(path, 'wb') as f:
    f.write(clean)
print(f'Cleaned: {len(data)} -> {len(clean)} bytes')
# Check last 100 bytes
last = clean[-100:]
print(f'Last 100 bytes repr: {repr(last)}')
