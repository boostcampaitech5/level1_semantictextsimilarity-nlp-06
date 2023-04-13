import yaml
import torch
import transformers
import pandas as pd
import pytorch_lightning as pl

from models.model import Model
from utils import utils, train
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # --- Setting ---
    # config file
    with open('baselines/baseline_config.yaml') as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)
    # 실험 결과 저장할 폴더 생성
    folder_name, save_path = utils.get_folder_name()
    CFG['save_path'] = save_path
    # seed 설정
    pl.seed_everything(CFG['seed'])

    # --- Fit ---
    # load data and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(CFG['train']['model_name'], max_length=CFG['train']['max_len'])
    dataloader = train.Dataloader(tokenizer, CFG['train']['batch_size'], CFG['train']['shuffle'])
    model = Model(CFG)

    # set options
    # Earlystopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # train and test
    trainer = pl.Trainer(accelerator='gpu', 
                         max_epochs=CFG['train']['epoch'],
                         default_root_dir=save_path,
                         callbacks = [early_stopping])
    
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # inference
    predictions = trainer.predict(model=model, datamodule=dataloader)
    pred_y = list(round(float(i), 1) for i in torch.cat(predictions))

    # --- save ---
    # write yaml
    # with open(f'{save_path}/{folder_name}_config.yaml', 'w') as f:
    #     yaml.dump(CFG, f)
    # save mode
    torch.save(model, f'{save_path}/{folder_name}_model.pt')
    # save submit
    submit = pd.read_csv('./data/sample_submission.csv')
    submit['target'] = pred_y
    submit.to_csv(f'{save_path}/{folder_name}_submit.csv', index=False)