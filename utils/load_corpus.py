import os
import sys

DATA_PATH = os.path.join('data')


def load_corpus(folders=None):
    if folders is None:
        folders = [f for f in os.scandir(DATA_PATH) if f.is_dir()]
    else:
        folders = [f for f in os.scandir(DATA_PATH) if f.name in folders]

    corpus = ''

    print('Reading in data ...')
    for folder in folders:
        folder_size = 0
        print(' ', folder.name)
        for file in os.listdir(folder.path):
            # check file format
            file_format = file.split('.')[1]
            file_path = os.path.join(folder.path, file)
            new_text = ''

            match file_format:
                case 'txt':
                    new_text = read_txt(file_path)

            new_size = sys.getsizeof(new_text) / 1024
            folder_size += new_size
            print(f' - {file} ({new_size:.2f} KB)')
            corpus += corpus + '\n\n' + new_text
        print(f' --> {folder_size / 1024:.2f} MB')
    print(f'==> corpus size {sys.getsizeof(corpus) / 1024**3} GB\n')
    return corpus


def read_txt(path):
    with open(path, 'r', encoding='utf-8-sig') as read_file:
        new_text = read_file.read()

    return new_text
