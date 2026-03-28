import os

base_path = "../dataset_human_readable_64"
splits = ["train", "val", "test"]

def count_avi_files(folder):
    count = 0
    for root, _, files in os.walk(folder):
        count += sum(1 for file in files if file.lower().endswith('.avi'))
    return count

if __name__ == "__main__":
    for split in splits:
        split_path = os.path.join(base_path, split)
        num_files = count_avi_files(split_path)
        print(f"{split.capitalize()} set: {num_files} .avi clips found")
