import json
import gzip
import os

def extract_episodes(dataset_path, output_path):
    """
    Extracts episode and scene IDs from a Habitat dataset file and saves them to a JSON file.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    with gzip.open(dataset_path, 'rt') as f:
        data = json.load(f)

    episodes_info = []
    for episode in data['episodes']:
        episodes_info.append({
            'episode_id': episode['episode_id'],
            'scene_id': episode['scene_id']
        })

    with open(output_path, 'w') as f:
        json.dump(episodes_info, f, indent=4)

    print(f"Successfully extracted {len(episodes_info)} episodes to {output_path}")

if __name__ == '__main__':
    # Path to the validation dataset
    dataset_path = '/home/hongrui/Datasets/habitat/pointnav_gibson_v1/val/val.json.gz'
    output_path = '../val_episodes.json'
    extract_episodes(dataset_path, output_path)
