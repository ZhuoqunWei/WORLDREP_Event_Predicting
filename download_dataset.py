import requests
import os

def download_worldrep_csv(save_dir="./data"):
    url = "https://huggingface.co/datasets/Daehoon/WORLDREP/resolve/main/worldrep_dataset_v2.csv"
    filename = "worldrep_dataset_v2.csv"

    # Make sure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Path to save the file
    save_path = os.path.join(save_dir, filename)

    print(f"Downloading {url}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Throw error if download failed

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Downloaded successfully to {save_path}")

    except requests.RequestException as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    download_worldrep_csv()
