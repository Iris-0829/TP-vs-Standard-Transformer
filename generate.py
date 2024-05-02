import os


def copy_lines(source_file_path, target_dir, target_file_name, line_num):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    target_file_path = os.path.join(target_dir, target_file_name)

    try:
        with open(source_file_path, 'r') as src:
            with open(target_file_path, 'w') as tgt:
                for i in range(line_num):
                    line = src.readline()
                    if not line:
                        break
                    tgt.write(line)
        print(f"The first {line_num} lines have been copied to {target_file_path}")
    except IOError as e:
        print(f"An error occurred: {e.strerror}")


files_to_process = [
    ('./babylm_preprocessed_10mtrain/train.txt', 'train.txt', 200000),
    ('./babylm_preprocessed_10mtrain/valid.txt', 'valid.txt', 200000),
    ('./babylm_preprocessed_10mtrain/test.txt', 'test.txt', 200000)
]

target_dir = './babylm_1m'

for source_file_path, target_file_name, num_lines in files_to_process:
    copy_lines(source_file_path, target_dir, target_file_name, num_lines)
