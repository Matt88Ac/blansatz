from experiments import generate_datasets

if __name__ == '__main__':
    pass
    # generate_datasets('determinant', 2, 2, 100_000, 10_000, 10_000)
    generate_datasets('determinant', 10, 10, 100_000, 10_000, 20_000,
                      lower=-1.4, upper=1.4)
    generate_datasets('determinant', 15, 15, 100_000, 10_000, 20_000,
                      lower=-1.2, upper=1.2)
