import pandas as pd
import utils

def remove_special_word(df):
    df['sentence_1'] = df['sentence_1'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
    df['sentence_2'] = df['sentence_2'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)

    return df

def swap_sentence(df:pd.DataFrame) -> pd.DataFrame:
    """
    sentence1과 sentence2를 바꾸는 data augmentation 기법입니다.

    Args:
        df (pd.DataFrame): 원본 train data

    Returns:
        pd.DataFrame
    """
    swapped_df = df.copy(deep=True)
    swapped_df['sentence_1'], swapped_df['sentence_2'] = swapped_df['sentence_2'], swapped_df['sentence_1']
    return swapped_df

# 전처리 코드 테스트
if __name__ == "__main__":
    train_df, _, _ = utils.get_data()
    preprocessed_df = swap_sentence(train_df)

    print('-'*30)
    print("전처리 전", train_df.head(5), sep='\n')
    print('-'*30)
    print("전처리 후", preprocessed_df.head(5), sep='\n')