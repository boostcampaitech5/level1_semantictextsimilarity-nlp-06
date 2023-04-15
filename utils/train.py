import torch
import pandas as pd
from utils import utils
import utils.data_preprocessing as DP
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from utils.process_manipulator import SequentialCleaning as SC, SequentialAugmentation as SA

class Dataset(Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.inputs)

class Dataloader(pl.LightningDataModule):
    def __init__(self, tokenizer, CFG):
        super(Dataloader, self).__init__()
        self.batch_size = CFG['train']['batch_size']
        self.shuffle = CFG['train']['shuffle']
        self.seed = CFG['seed']
        self.admin = CFG['admin']
        
        train_df, val_df, predict_df = utils.get_data()
        # 김기범이고 df_split이 True라면 데이터셋을 다시 분리
        if self.admin == 'KGB':
            self.train_df = pd.concat([train_df, val_df], axis=0)
        else:
            self.train_df = train_df
            self.val_df = val_df
        self.predict_df = predict_df # test.csv

        self.train_dataset = None
        self.val_dataset = None # val == test
        self.test_dataset = None
        self.predict_dataset = None 

        self.tokenizer = tokenizer
        self.target_columns = ['label']
        self.text_columns = ['sentence_1', 'sentence_2'] 
        self.cleaning_list = CFG['select_clean']
        self.augmentation_list = CFG['select_DA']

    def tokenizing(self, df):
        data = []

        for idx, item in tqdm(df.iterrows(), desc='tokenizing', total=len(df)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        
        return data

    def preprocessing(self, data, train=False):
        sc = SC(self.cleaning_list)
        sa = SA(self.augmentation_list)

        data = sc.process(data)
        if train: # train일 때만 augmentation을 합니다.
            data = sa.process(data)
        
        # 타겟 데이터 load
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        
        # 텍스트 데이터를 전처리
        inputs = self.tokenizing(data)

        return inputs, targets
    
    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터 준비
            train_inputs, train_targets = self.preprocessing(self.train_df, train=True)
            
            # 검증 데이터 준비
            if self.admin == 'KGB':
                train_inputs, val_inputs, train_targets, val_targets = train_test_split(train_inputs, train_targets, 
                                                                                        test_size=0.055, 
                                                                                        stratify=train_targets, 
                                                                                        random_state=self.seed)
            else:
                val_inputs, val_targets = self.preprocessing(self.val_df)

            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
            self.test_dataset = self.val_dataset
        else:
            # 평가 데이터 호출
            predict_inputs, predict_targets = self.preprocessing(self.predict_df)
            self.predict_dataset = Dataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)