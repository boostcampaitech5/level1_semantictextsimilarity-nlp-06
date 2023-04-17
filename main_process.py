import yaml
import torch
import transformers
import pandas as pd
import pytorch_lightning as pl
import wandb

from models.model import Model
from utils import utils, train
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    for DC in [['remove_special_word'], ['remove_special_word', 'spellcheck'], ['spellcheck']]:
        for DA in [['swap_sentence'], ['swap_sentence', 'create_5'], ['create_5']]:
            # --- Setting ---
            # config file
            with open('baselines/baseline_config.yaml') as f:
                CFG = yaml.load(f, Loader=yaml.FullLoader)
            # 실험 결과 저장할 폴더 생성
            folder_name, save_path = utils.get_folder_name(CFG)
            # seed 설정
            pl.seed_everything(CFG['seed'])
            # 반복 실험
            CFG['select_clean'] = DC
            CFG['select_DA'] = DA

            # logger 생성
            wandb.init(name=folder_name, project="STS", entity="boostcamp_nlp_06")
            wandb_logger = WandbLogger(save_dir=save_path)
            wandb_logger.experiment.config.update(CFG)

            # --- Fit ---
            # load data and model
            tokenizer = transformers.AutoTokenizer.from_pretrained(CFG['train']['model_name'], model_max_length=CFG['train']['max_len'])
            dataloader = train.Dataloader(tokenizer, CFG)
            model = Model(CFG)

            # set options
            # Earlystopping
            checkpoint = ModelCheckpoint(monitor='val_loss',
                                        save_top_k=3,
                                        save_last=True,
                                        save_weights_only=True,
                                        verbose=False,
                                        filename='{epoch}-{val_loss:.2f}',
                                        mode='min')
            early_stopping = EarlyStopping(monitor='val_loss', patience=CFG['patience'], mode='min')

            # train and test
            trainer = pl.Trainer(accelerator='gpu',
                                max_epochs=CFG['train']['epoch'],
                                default_root_dir=save_path,
                                log_every_n_steps=1,
                                logger = wandb_logger,
                                callbacks = [checkpoint, early_stopping])
            
            trainer.fit(model=model, datamodule=dataloader)
            trainer.test(model=model, datamodule=dataloader)

            # inference
            best_check_point = utils.get_best_check_point(save_path)
            model = Model.load_from_checkpoint(best_check_point)
            predictions = trainer.predict(model=model, datamodule=dataloader)
            pred_y = list(float(i) for i in torch.cat(predictions))

            # --- save ---
            # write yaml
            with open(f'{save_path}/{folder_name}_config.yaml', 'w') as f:
                yaml.dump(CFG, f)
            # save mode
            torch.save(model, f'{save_path}/{folder_name}_model.pt')
            # save submit
            submit = pd.read_csv('./data/sample_submission.csv')
            submit['target'] = pred_y
            submit.to_csv(f'{save_path}/{folder_name}_submit.csv', index=False)