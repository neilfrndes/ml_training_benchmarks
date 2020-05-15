import pandas as pd


class MLData:
    """Class for the dataset and helper functions"""

    def __init__(self, file_name: str):
        self.data = pd.read_csv(f'./data/{file_name}')

    def get_test_data(self, size: int = 1):
        """Generates a test dataset of the specified size"""
        num_rows = len(self.data)
        df = self.data.iloc[:, :-1].copy()

        while num_rows < size:
            df = df.append(df, ignore_index=True)
            num_rows = len(df)

        return df[:size]

    def get_training_data(self, size: int = 1000):
        """Generates a training dataset of the specified size"""
        num_rows = len(self.data)
        df = self.data.copy()

        while num_rows < size:
            df = df.append(df, ignore_index=True)
            num_rows = len(df)

        df = df[:size]
        return df.iloc[:, :-1], df.iloc[:, -1:]