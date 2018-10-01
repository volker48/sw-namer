from pathlib import Path


def main():
    names_path = Path('output') / 'names.csv'
    sw_path = Path('output') / 'starwars_processed.csv'

    vocab = set()
    with sw_path.open('r') as infile:
        # skip header
        infile.readline()
        for


if __name__ == '__main__':
    main()