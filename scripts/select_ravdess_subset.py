from pathlib import Path
import random
import shutil

root = Path('data/raw')
extract_root = Path('data/raw/_downloads/ravdess_full')

if not extract_root.exists():
    raise SystemExit('Extracted RAVDESS files not found: data/raw/_downloads/ravdess_full')

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised',
}

wav_files = sorted(extract_root.rglob('*.wav'))
by_emotion = {v: [] for v in emotion_map.values()}
for p in wav_files:
    parts = p.stem.split('-')
    if len(parts) >= 3 and parts[2] in emotion_map:
        by_emotion[emotion_map[parts[2]]].append(p)

# reset raw directory except .gitkeep and _downloads
for child in root.iterdir():
    if child.name in {'.gitkeep', '_downloads'}:
        continue
    if child.is_dir():
        shutil.rmtree(child)
    else:
        child.unlink()

rng = random.Random(42)
selected_total = 0
for emo, files in by_emotion.items():
    if len(files) < 25:
        raise SystemExit(f'Not enough files for {emo}: {len(files)}')
    chosen = files[:]
    rng.shuffle(chosen)
    chosen = chosen[:25]
    out_dir = root / emo
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in chosen:
        shutil.copy2(src, out_dir / src.name)
    selected_total += len(chosen)

print(f'selected_total={selected_total}')
for emo in sorted(by_emotion):
    count = len(list((root / emo).glob('*.wav')))
    print(f'{emo}={count}')
