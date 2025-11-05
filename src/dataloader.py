import pandas as pd
import csv
import os


def load_paraphrase_data(filepath):
    paraphrase_data = {}
    token_to_compound = {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                token = row['special_token']
                paraphrase_data[token] = {
                    'positive': [
                        row['positive_paraphrase1'],
                        row['positive_paraphrase2'],
                        row['positive_paraphrase3']
                    ],
                    'literal': row['literal_paraphrase']
                }
                compound_name = token.replace('ID', '').replace('_', ' ')
                token_to_compound[token] = compound_name

        print(f"Loaded {len(paraphrase_data)} compounds from extendedNominal.csv")
        return paraphrase_data, token_to_compound

    except Exception as e:
        print(f"Error loading paraphrase data: {e}")
        return {}, {}


def load_admire_data(filepath):
    try:
        dataset_df = pd.read_csv(filepath, delimiter='\t')
        print(f"Loaded {len(dataset_df)} ADMIRE rows")
        return dataset_df
    except Exception as e:
        print(f"Error loading ADMIRE data: {e}")
        return pd.DataFrame()


def split_train_test(dataset_df):
    subset_values = dataset_df['subset'].unique()

    train_subset = None
    test_subset = None

    for val in subset_values:
        val_str = str(val).lower().strip()
        if 'train' in val_str:
            train_subset = val
        elif 'sample' in val_str or 'test' in val_str:
            test_subset = val

    train_df = dataset_df[dataset_df['subset'] == train_subset].copy()
    test_df = dataset_df[dataset_df['subset'] == test_subset].copy()

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, test_df


def filter_valid_samples(df):
    valid = []
    for idx, row in df.iterrows():
        try:
            if ('sentence_type' in row.index and
                    row['sentence_type'] in ['idiomatic', 'literal'] and
                    'compound' in row.index and pd.notna(row['compound']) and
                    'sentence' in row.index and pd.notna(row['sentence']) and
                    'expected_order' in row.index and pd.notna(row['expected_order'])):
                valid.append(row)
        except:
            continue
    return valid


def build_image_caption_mapping(dataset_df):
    image_to_caption = {}

    caption_columns = [col for col in dataset_df.columns if 'caption' in col.lower()]
    name_columns = [col for col in dataset_df.columns if 'name' in col.lower() and 'image' in col.lower()]

    for idx, row in dataset_df.iterrows():
        for name_col in name_columns:
            base_name = name_col.replace('_name', '').replace('image', 'image').strip('_')
            caption_col = f"{base_name}_caption"

            if name_col in row.index and pd.notna(row[name_col]) and caption_col in row.index and pd.notna(
                    row[caption_col]):
                img_name = str(row[name_col]).lower().strip()
                caption = str(row[caption_col]).strip()

                if img_name and caption:
                    image_to_caption[img_name] = caption

    print(f"Built mapping for {len(image_to_caption)} image-caption pairs")
    return image_to_caption


def build_image_path_mapping(image_dir):
    image_to_path = {}

    if os.path.exists(image_dir):
        subdirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]

        for subdir in subdirs:
            subdir_path = os.path.join(image_dir, subdir)

            for img_file in os.listdir(subdir_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(subdir_path, img_file)
                    image_to_path[img_file.lower()] = full_path

    print(f"Built mapping for {len(image_to_path)} image paths")
    return image_to_path
