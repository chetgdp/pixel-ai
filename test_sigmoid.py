def sig(x, eff_scale):
    if x >= 0:
        return min((x * 128) // (abs(x) + eff_scale), 127)
    else:
        return max(-(-x * 128 // (abs(x) + eff_scale)), -128)

def remap(s):
    # [-127,127] -> [0,255]
    return (s + 127) * 255 // 254

scale = 16
# try different effective scales for the output layer
eff_scales = {
    'scale':    scale,          # 16 (original)
    'scale²':   scale**2,       # 256
    'scale³':   scale**3,       # 4096
    '8*scale²': 8 * scale**2,   # 2048
}

# realistic output sums
inputs = sorted(set([*range(-50000, 50001, 10000), *range(-5000, 5001, 1000), *range(-500, 501, 100), 0]))

header = f"{'x':>7}"
for name in eff_scales:
    header += f" | {name:>8}"
print(header)
print("-" * len(header))
for x in inputs:
    row = f"{x:>7}"
    for name, es in eff_scales.items():
        s = sig(x, es)
        row += f" | {remap(s):>8}"
    print(row)
