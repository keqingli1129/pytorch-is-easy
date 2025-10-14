from datasets import load_dataset
from datasets import load_from_disk, DatasetDict


def generate_gutenberg_dataset():
    # Load the Gutenberg English dataset from Hugging Face
    # dataset = load_dataset("sedthh/gutenberg_english")
    # print(dataset)
    # print(dataset["train"][0])
    # Save it locally
    # dataset.save_to_disk("gutenberg_english_local")
    dataset = load_from_disk("gutenberg_english_local")

    # Split the train set into train, validation, and test
    split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    # Now split the test further into validation and test
    val_test = split_dataset["test"].train_test_split(test_size=0.5, seed=42)

    # Combine splits into a DatasetDict
    final_dataset = DatasetDict({
        "train": split_dataset["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })
    return final_dataset

def main():
    final_dataset = generate_gutenberg_dataset()
    print(final_dataset)

if __name__ == "__main__":
    main()