import pandas


def main(timesteps=3):
    male = pandas.read_csv('data/male.txt', header=5, names=['name'])
    female = pandas.read_csv('data/female.txt', header=5, names=['name'])
    combined = pandas.concat([male, female])

    combined = combined.sample(frac=1, random_state=38974)

#    combined.loc[:, 'name'] = '^' * timesteps + combined.name.str.lower() + '$' * timesteps

    combined.to_csv('output/names.csv', index=False)


if __name__ == '__main__':
    main()
