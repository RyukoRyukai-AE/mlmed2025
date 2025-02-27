import pandas as pd

def label_map (df: pd.DataFrame) -> pd.DataFrame:
    id_mapping = {
        0: 'N',
        1: 'S',
        2: 'V',
        3: 'F',
        4: 'Q'
    }

    df['Label'] = df.iloc[:,-1].map(id_mapping)
    return df