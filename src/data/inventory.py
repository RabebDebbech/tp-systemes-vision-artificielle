from pathlib import Path
import pandas as pd

ROOT = Path("data/raw/Garbage classification/Garbage classification")


def main():
    rows = []

    for class_dir in sorted(ROOT.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for img_path in class_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                rows.append(
                    {
                        "filepath": str(img_path).replace("\\", "/"),
                        "filename": img_path.name,
                        "label": class_name,
                    }
                )

    df = pd.DataFrame(rows).sort_values(["label", "filename"]).reset_index(drop=True)

    out_dir = Path("data/interim")
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "inventory.csv", index=False)

    summary = df.groupby("label").size().reset_index(name="count").sort_values("label")
    summary.to_csv(out_dir / "class_counts.csv", index=False)

    print(f"Images trouvées: {len(df)}")
    print(summary)


if __name__ == "__main__":
    main()