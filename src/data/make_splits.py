from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
INVENTORY_PATH = Path("data/interim/inventory.csv")
OUT_PATH = Path("data/processed/splits.csv")


def main():
    df = pd.read_csv(INVENTORY_PATH)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    splits_df = pd.concat([train_df, val_df, test_df], axis=0)
    splits_df = splits_df.sort_values(["split", "label", "filename"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(OUT_PATH, index=False)

    print(splits_df["split"].value_counts())
    print()
    print(pd.crosstab(splits_df["label"], splits_df["split"]))


if __name__ == "__main__":
    main()