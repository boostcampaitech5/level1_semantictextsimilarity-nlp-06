import pandas as pd
import utils
import random

from tqdm.auto import tqdm

# spellceck
from pykospacing import Spacing
from symspellpy_ko import KoSymSpell, Verbosity

def remove_special_word(df):
    """
    특수문자를 제거하는 메소드
    """
    df['sentence_1'] = df['sentence_1'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
    df['sentence_2'] = df['sentence_2'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)

    return df


def random_deletion(df):
    """
    apply로 실행시켜야 함
    k개의 워드를 랜덤으로 삭제 기법 (일단 k=1) -> 나중에 보유 문장 길이에 따라 동적으로 변경시킬 예정
    조합 3개 (ww표시가 전처리 된 거)
    1. sen_1, sen_2ww
    2. sen_1ww, sen_2
    3. sen_1ww, sen_3ww
    """
    def func(x):
        x = x.split()
        random_item = random.choice(x)
        x.remove(random_item)

        return ' '.join(x)
    
    sen_1ww = df['sentence_1'] = df['sentence_1'].apply(lambda x: func(x)).values.tolist()
    sen_2ww = df['sentence_2'] = df['sentence_2'].apply(lambda x: func(x)).values.tolist()

    sen_1, sen_2 = [], []
    labels = [label for _ in range(3) for label in df['label']] if 'label' in df.columns else []

    # 1
    sen_1.extend(df['sentence_1'].values.tolist())
    sen_2.extend(sen_2ww)
    # 2
    sen_1.extend(sen_1ww)
    sen_2.extend(df['sentence_2'].values.tolist())
    # 3
    sen_1.extend(sen_1ww)
    sen_2.extend(sen_2ww)

    new_df = pd.DataFrame()
    new_df['sentence_1'] = sen_1
    new_df['sentence_2'] = sen_2
    if labels: new_df['label'] = labels

    return new_df

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

def spellcheck(df):
    """
    sentence_1과 sentence_2의 띄어쓰기 및 맟춤법을 교정해주는 함수입니다.

    필요 라이브러리 :
    pip install git+https://github.com/haven-jeon/PyKoSpacing.git
    pip install symspellpy-ko

    Args:
        df (pd.DataFrame): 원본 train data

    Returns:
        pd.DataFrame    
    """
    spacing = Spacing()
    symspell = KoSymSpell()
    symspell.load_korean_dictionary(decompose_korean = True, load_bigrams=True)
    checked_df = df.copy()
    sen1 = []
    sen2 = []

    # 동일한 spacing 적용을 위해 공백을 제거합니다.
    checked_df['sentence_1'] = checked_df['sentence_1'].str.replace(pat=r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9//?.,;:|*~`!^/-_+<>@/#$%&/=]+', repl=r'', regex=True)
    checked_df['sentence_2'] = checked_df['sentence_2'].str.replace(pat=r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9//?.,;:|*~`!^/-_+<>@/#$%&/=]+', repl=r'', regex=True)

    # spacing을 적용합니다.
    for i in tqdm(range(len(checked_df)), desc='spacing'):
        sen1.append(spacing(checked_df['sentence_1'][i]))
        sen2.append(spacing(checked_df['sentence_2'][i]))

    # 맞춤법을 체크합니다.
    for i in tqdm(range(len(checked_df)), desc='spell'):
        term = sen1[i]
        for suggestion in symspell.lookup_compound(term, max_edit_distance=0):
            sen1[i] = suggestion.term
        term = sen2[i]
        for suggestion in symspell.lookup_compound(term, max_edit_distance=0):
            sen2[i] = suggestion.term

    checked_df['sentence_1'] = sen1
    checked_df['sentence_2'] = sen2

    return checked_df


def create_5(df):
    """
    label이 5인 데이터 생성 (조합 2개)
    가장 마지막에 사용하는 걸 추천

    1. sentence_1, sentence_1
    2. sentence_2, sentence_2
    """
    sentence_1 = pd.DataFrame()
    sentence_1['sentence_1'] = df['sentence_1']
    sentence_1['sentence_2'] = df['sentence_1']

    sentence_2 = pd.DataFrame()
    sentence_2['sentence_1'] = df['sentence_2']
    sentence_2['sentence_2'] = df['sentence_2']

    new_df = pd.concat([sentence_1, sentence_2], axis=0)
    new_df['label'] = 5.0

    return new_df


# 전처리 코드 테스트
if __name__ == "__main__":
    train_df, _, _ = utils.get_data()
    preprocessed_df = swap_sentence(train_df)

    print('-'*30)
    print("전처리 전", train_df.head(5), sep='\n')
    print('-'*30)
    print("전처리 후", preprocessed_df.head(5), sep='\n')

    random_deletion_df = random_deletion(train_df)
    print('-'*30)
    print('random deletion 전,', train_df.shape)
    print('random deletion 후,', random_deletion_df.shape)
    new_df = pd.concat([train_df, random_deletion_df], axis=0)
    print(new_df.shape)
    print('-'*30)