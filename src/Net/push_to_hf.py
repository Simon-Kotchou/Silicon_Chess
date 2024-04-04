import pandas as pd
import zstandard as zstd
from datasets import Dataset
from huggingface_hub import HfApi, HfFolder

# Constants
zst_file_path = 'path/to/your/file.csv.zst'  # Update this path
dataset_name = 'your_dataset_name'  # Choose a name for your dataset

# Step 1: Decompress the .zst file and load the CSV
with open(zst_file_path, 'rb') as compressed:
    decompressor = zstd.ZstdDecompressor()
    with decompressor.stream_reader(compressed) as reader:
        df = pd.read_csv(reader)

# Step 2: Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Step 3: Authenticate with Hugging Face and push the dataset
hf_api = HfApi()
token = HfFolder.get_token()  # Ensure you're logged in with `huggingface-cli login`
if token is None:
    raise ValueError("You must be logged in to Hugging Face Hub. Use `huggingface-cli login`.")

# Create a repository on the Hub (if it doesn't exist) and get its URL
repo_url = hf_api.create_repo(token=token, name=dataset_name, repo_type='dataset', exist_ok=True)

# Push the dataset to the Hub
dataset.push_to_hub(dataset_name, private=True, token=token)  # Set `private=False` to make the dataset public

print(f"Dataset successfully pushed to {repo_url}")