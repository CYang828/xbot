import os

from tqdm import tqdm

from src.xbot.util.path import get_data_path
from src.xbot.util.file_util import read_zipped_json, dump_json


def get_single_domain_examples(input_file_path, output_file_path):
    file_in_zip = os.path.basename(input_file_path).rsplit(".", maxsplit=1)[0]
    dataset = read_zipped_json(input_file_path, file_in_zip)

    single_domain_examples = {
        id_: dialogue for id_, dialogue in dataset.items() if dialogue["type"] == "单领域"
    }

    print(
        f"{file_in_zip} total has {len(single_domain_examples)} single domain examples"
    )

    dump_json(single_domain_examples, output_file_path)


def main():
    output_dir = os.path.join(
        get_data_path(), "crosswoz/policy_rule_single_domain_data"
    )
    input_dir = os.path.join(get_data_path(), "crosswoz/raw")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    datasets = ["train", "val", "tests"]
    for dataset in tqdm(datasets):
        output_file_name = dataset + ".json"
        input_file_name = output_file_name + ".zip"
        input_file_path = os.path.join(input_dir, input_file_name)
        output_file_path = os.path.join(output_dir, output_file_name)
        get_single_domain_examples(input_file_path, output_file_path)


if __name__ == "__main__":
    main()
