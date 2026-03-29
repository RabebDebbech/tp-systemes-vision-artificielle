from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps

MANIFEST_PATH = Path("data/processed/manifest.parquet")
OUT_ROOT = Path("data/processed/images_224")
OUT_MANIFEST = Path("data/processed/manifest_224.parquet")
TARGET_SIZE = (224, 224)


def process_one(path_str: str, split: str, label: str, filename: str):
    src = Path(path_str)
    dst_dir = OUT_ROOT / split / label
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / filename

    with Image.open(src) as img:
        img = img.convert("RGB")
        img = ImageOps.fit(img, TARGET_SIZE, method=Image.Resampling.LANCZOS)
        img.save(dst, format="JPEG", quality=95)

    return str(dst).replace("\\", "/")


def main():
    df = pd.read_parquet(MANIFEST_PATH)

    out_paths = []
    for _, row in df.iterrows():
        out_path = process_one(
            row["filepath"],
            row["split"],
            row["label"],
            row["filename"],
        )
        out_paths.append(out_path)

    df = df.copy()
    df["processed_filepath"] = out_paths
    df["processed_width"] = TARGET_SIZE[0]
    df["processed_height"] = TARGET_SIZE[1]

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_MANIFEST, index=False)

    print(df[["filepath", "processed_filepath", "split", "label"]].head())
    print(f"\nImages traitées: {len(df)}")


if __name__ == "__main__":
    main()