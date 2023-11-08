import os

DATA_PATH = os.path.join('data')


def load_corpus(folders=None):
    if folders is None:
        folders = [f for f in os.scandir(DATA_PATH) if f.is_dir()]

    corpus = ''

    for folder in folders:
        for file in os.listdir(folder.path):
            # check file format
            file_format = file.split('.')[1]
            file_path = os.path.join(folder.path, file)
            new_text = ''

            match file_format:
                case 'txt':
                    new_text = read_txt(file_path)

            corpus += corpus + '\n\n' + new_text

    return corpus


def read_txt(path):
    with open(path, 'r', encoding='utf-8-sig') as read_file:
        new_text = read_file.read()

    return new_text
