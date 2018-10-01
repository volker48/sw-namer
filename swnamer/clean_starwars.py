import pandas
import numpy


def main(timesteps=3):
    legend_names = pandas.read_csv('data/legend_names.csv')
    canon_names = pandas.read_csv('data/cannon_names.csv')
    cast_names = pandas.read_csv('data/cast_names.csv')
    clone_wars_names = pandas.read_csv('data/clone_wars.csv')
    kotor_names = pandas.read_csv('data/kotor.csv')
    combined = pandas.concat((legend_names, canon_names, cast_names, clone_wars_names, kotor_names))

    combined = combined.reset_index(drop=True)

    combined = combined.drop_duplicates()

#    combined.loc[:, 'name'] = '^' * timesteps + combined.name.str.lower() + '$' * timesteps

    combined.to_csv('output/starwars_processed.csv', index=False)


if __name__ == '__main__':
    main()
