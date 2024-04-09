from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, HfFolder
from tqdm.auto import tqdm
from datasets import Dataset, concatenate_datasets
import requests
import zstandard as zstd
import chess.pgn
import io

# Function to read URLs from a text file
def read_urls_from_file(file_path):
    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file if line.strip()]
    return urls

# Function to stream and decompress data from a single URL
def stream_decompress(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(response.raw) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        yield from text_stream

# Function to parse PGN data from a single decompressed file
def parse_pgn(text_stream):
    games_data = []
    pgn_file = io.StringIO(text_stream)
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        game_info = {key: value for key, value in game.headers.items()}
        game_info["Moves"] = game.board().variation_san(game.mainline_moves())
        games_data.append(game_info)
    return games_data

# Function to create Hugging Face datasets from URLs
def create_datasets_from_urls(file_path):
    urls = read_urls_from_file(file_path)
    datasets = []
    for url in urls:
        game_info = parse_pgn(''.join(stream_decompress(url)))
        datasets.append(Dataset.from_dict(game_info))
    return datasets

# Combine datasets and process
def process_datasets(file_path):
    datasets = create_datasets_from_urls(file_path)
    combined_dataset = concatenate_datasets(datasets)
    train_test_split = combined_dataset.train_test_split(test_size=0.15, seed=42)
    return train_test_split

# # Constants
# zst_file_path = '../../data/lichess_db_puzzle.csv.zst'  # Update this path
# dataset_name = 'Simon-Kotchou/lichess-puzzles'  # Choose a name for your dataset

# dataset = load_dataset("csv", data_files=zst_file_path, split="train")

# # Step 2: Authenticate with Hugging Face and push the dataset
# hf_api = HfApi()
# token = HfFolder.get_token()  # Ensure you're logged in with `huggingface-cli login`
# if token is None:
#     raise ValueError("You must be logged in to Hugging Face Hub. Use `huggingface-cli login`.")

# print("Pushing dataset to Hugging Face Hub...")
# dataset.push_to_hub(dataset_name, private=True, token=token)  # Set `private=False` to make the dataset public