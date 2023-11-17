import os


def load_corpus(data_folder, folders=None):
    # generator that reads in file after file and yields text of single files
    if folders is None:
        folders = [f for f in os.scandir(data_folder) if f.is_dir()]
    else:
        folders = [f for f in os.scandir(data_folder) if f.name in folders]

    for folder in folders:
        for file in os.listdir(folder.path):
            # check file format
            file_format = file.split('.')[1]
            file_path = os.path.join(folder.path, file)
            new_text = ''

            if file_format == 'txt':
                new_text = read_txt(file_path)

            yield new_text


def read_txt(path):
    with open(path, 'r', encoding='utf-8-sig') as read_file:
        new_text = read_file.read()

    return new_text
