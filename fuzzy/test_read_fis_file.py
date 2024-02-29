def read_fis_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            print(line)


if __name__ == '__main__':
    read_fis_file('./data/sistema_guindaste.fis')