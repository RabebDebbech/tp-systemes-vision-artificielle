from pathlib import Path
import pandas as pd
from PIL import Image

SPLITS_PATH = Path("data/processed/splits.csv")
OUT_PATH = Path("data/processed/manifest.parquet")


def get_image_info(path_str: str):
    path = Path(path_str)
    try:
        file_size = path.stat().st_size
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
        return width, height, mode, file_size, None
    except Exception as e:
        return None, None, None, None, str(e)


def main():
    df = pd.read_csv(SPLITS_PATH)

    meta = df["filepath"].apply(get_image_info)
    meta_df = pd.DataFrame(
        meta.tolist(),
        columns=["width", "height", "mode", "file_size_bytes", "read_error"]
    )

    out_df = pd.concat([df, meta_df], axis=1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False)

    print(out_df.head())
    print()
    print(out_df["read_error"].isna().value_counts())


if __name__ == "__main__":
    main()