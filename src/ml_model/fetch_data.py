def load_dataset():
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    dataset = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "abdallahwagih/spam-emails",
        "spam.csv"
    )

    return dataset
