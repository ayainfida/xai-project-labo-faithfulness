import pickle
from pathlib import Path
from PIL import Image

"""
Helper script to convert CIFAR-10 from the original binary format to individual image files.
"""

src_dir = Path("cifar-10-batches-py")
out_dir = Path("datasets/CIFAR10/images")
out_dir.mkdir(parents=True, exist_ok=True)

def save_batch(batch_path: Path):
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    data = batch[b"data"]
    filenames = [x.decode("utf-8") for x in batch[b"filenames"]]

    for arr, fname in zip(data, filenames):
        img = arr.reshape(3, 32, 32).transpose(1, 2, 0)
        Image.fromarray(img).save(out_dir / fname)

for i in range(1, 6):
    save_batch(src_dir / f"data_batch_{i}")

save_batch(src_dir / "test_batch")