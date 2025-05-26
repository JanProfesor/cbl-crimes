import shutil
from pathlib import Path

base_dir = Path("old_data")
target_dir = Path("data")
target_dir.mkdir(parents=True, exist_ok=True)

min_date = "2011-01"
max_date = "2022-12"

for top_folder in base_dir.iterdir():
    if not top_folder.is_dir():
        continue

    for subfolder in top_folder.iterdir():
        if not subfolder.is_dir():
            continue

        folder_date = subfolder.name
        if min_date <= folder_date <= max_date:
            for file in subfolder.glob("*-metropolitan-street.csv"):
                shutil.copy(file, target_dir / file.name)

print("Extraction complete.")
