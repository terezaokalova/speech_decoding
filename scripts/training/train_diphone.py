import argparse
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='rnn_args.yaml', help='Path to config file')
cmd_args = parser.parse_args()

args = OmegaConf.load(cmd_args.config)
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()