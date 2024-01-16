from bert_loves_chemistry.chemberta.utils.molnet_dataloader import load_molnet_dataset, write_molnet_dataset_for_chemprop
import selfies as sf
from datetime import datetime


class Scaffoldprocessor:
    def __init__(self, tasks_wanted=None, split=None):
        self.tasks_wanted = tasks_wanted
        self.split = split

    def load_molnet_dataset(self, dataset_name):
        tasks, (train_df, valid_df, test_df), transformers = load_molnet_dataset(
            dataset_name,
            tasks_wanted=self.tasks_wanted,
            split=self.split
        )
        return tasks, train_df, valid_df, test_df, transformers

    def rename_columns(self, dataframe, column_mapping):
        dataframe.rename(columns=column_mapping, inplace=True)

    def process_clintox_dataset(self, train_df, valid_df, test_df):
        # Rename columns
        column_mapping = {'text': 'smiles'}
        self.rename_columns(train_df, column_mapping)
        self.rename_columns(valid_df, column_mapping)
        self.rename_columns(test_df, column_mapping)

        # Extract labels
        train_df['FDA_APPROVED'] = train_df['labels'].apply(lambda x: x[0])
        train_df['CT_TOX'] = train_df['labels'].apply(lambda x: x[1])
        valid_df['FDA_APPROVED'] = valid_df['labels'].apply(lambda x: x[0])
        valid_df['CT_TOX'] = valid_df['labels'].apply(lambda x: x[1])
        test_df['FDA_APPROVED'] = test_df['labels'].apply(lambda x: x[0])
        test_df['CT_TOX'] = test_df['labels'].apply(lambda x: x[1])

        return train_df, valid_df, test_df

    def process_tox21_dataset(self, train_df, valid_df, test_df):
        # Rename columns
        column_mapping = {'text': 'smiles'}
        self.rename_columns(train_df, column_mapping)
        self.rename_columns(valid_df, column_mapping)
        self.rename_columns(test_df, column_mapping)


        # Extract labels
        train_df['NR-AR'] = train_df['labels'].apply(lambda x: x[0])
        train_df['NR-AR-LBD'] = train_df['labels'].apply(lambda x: x[1])
        train_df['NR-AhR'] = train_df['labels'].apply(lambda x: x[2])
        train_df['NR-Aromatase'] = train_df['labels'].apply(lambda x: x[3])
        train_df['NR-ER'] = train_df['labels'].apply(lambda x: x[4])
        train_df['NR-ER-LBD'] = train_df['labels'].apply(lambda x: x[5])
        train_df['NR-PPAR-gamma'] = train_df['labels'].apply(lambda x: x[6])
        train_df['SR-ARE'] = train_df['labels'].apply(lambda x: x[7])
        train_df['SR-ATAD5'] = train_df['labels'].apply(lambda x: x[8])
        train_df['SR-HSE'] = train_df['labels'].apply(lambda x: x[9])
        train_df['SR-MMP'] = train_df['labels'].apply(lambda x: x[10])
        train_df['SR-p53'] = train_df['labels'].apply(lambda x: x[11])

        valid_df['NR-AR'] = valid_df['labels'].apply(lambda x: x[0])
        valid_df['NR-AR-LBD'] = valid_df['labels'].apply(lambda x: x[1])
        valid_df['NR-AhR'] = valid_df['labels'].apply(lambda x: x[2])
        valid_df['NR-Aromatase'] = valid_df['labels'].apply(lambda x: x[3])
        valid_df['NR-ER'] = valid_df['labels'].apply(lambda x: x[4])
        valid_df['NR-ER-LBD'] = valid_df['labels'].apply(lambda x: x[5])
        valid_df['NR-PPAR-gamma'] = valid_df['labels'].apply(lambda x: x[6])
        valid_df['SR-ARE'] = valid_df['labels'].apply(lambda x: x[7])
        valid_df['SR-ATAD5'] = valid_df['labels'].apply(lambda x: x[8])
        valid_df['SR-HSE'] = valid_df['labels'].apply(lambda x: x[9])
        valid_df['SR-MMP'] = valid_df['labels'].apply(lambda x: x[10])
        valid_df['SR-p53'] = valid_df['labels'].apply(lambda x: x[11])

        test_df['NR-AR'] = test_df['labels'].apply(lambda x: x[0])
        test_df['NR-AR-LBD'] = test_df['labels'].apply(lambda x: x[1])
        test_df['NR-AhR'] = test_df['labels'].apply(lambda x: x[2])
        test_df['NR-Aromatase'] = test_df['labels'].apply(lambda x: x[3])
        test_df['NR-ER'] = test_df['labels'].apply(lambda x: x[4])
        test_df['NR-ER-LBD'] = test_df['labels'].apply(lambda x: x[5])
        test_df['NR-PPAR-gamma'] = test_df['labels'].apply(lambda x: x[6])
        test_df['SR-ARE'] = test_df['labels'].apply(lambda x: x[7])
        test_df['SR-ATAD5'] = test_df['labels'].apply(lambda x: x[8])
        test_df['SR-HSE'] = test_df['labels'].apply(lambda x: x[9])
        test_df['SR-MMP'] = test_df['labels'].apply(lambda x: x[10])
        test_df['SR-p53'] = test_df['labels'].apply(lambda x: x[11])

        return train_df, valid_df, test_df

    def save_dataframes(self, train_df, valid_df, test_df, dataset_name):
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_df.to_csv(f'./datasets/scaffold_splits/{dataset_name}_train_{time}.csv', index=False)
        valid_df.to_csv(f'./datasets/scaffold_splits/{dataset_name}_valid_{time}.csv', index=False)
        test_df.to_csv(f'./datasets/scaffold_splits/{dataset_name}_test_{time}.csv', index=False)

    def process_data(self, dataset_name):
        tasks = None
        train_df = None
        valid_df = None
        test_df = None
        transformers = None

        if dataset_name == "clintox":
            tasks, train_df, valid_df, test_df, transformers = self.load_molnet_dataset(dataset_name)
            train_df, valid_df, test_df = self.process_clintox_dataset(train_df, valid_df, test_df)
        elif dataset_name == "tox21":
            tasks, train_df, valid_df, test_df, transformers = self.load_molnet_dataset(dataset_name)
            train_df, valid_df, test_df = self.process_tox21_dataset(train_df, valid_df, test_df)

        # self.save_dataframes(train_df, valid_df, test_df, dataset_name)

        return tasks, train_df, valid_df, test_df, transformers

