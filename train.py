import argparse
import logging
import json
import os
import datetime
from model import vrdOCR
import data
import torch
from paddle.io import BatchSampler, DataLoader
from src.multi_loss import MultiLoss
import paddle.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.postprocess import CTCLabelDecode
from src.metric import RecMetric
from utils.config_loader import load_config, load_weights

parser = argparse.ArgumentParser(description="Train or evaluate the vrdOCR model.")
parser.add_argument("--model_path", type=str, help="Path to the pretrained model weights.")
parser.add_argument("--run_name", type=str, default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    help="Custom name for the training run.")
parser.add_argument("--freeze_backbone", type=bool, help="Freeze backbone or not")
args = parser.parse_args()

run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("runs", run_name)
os.makedirs(run_dir, exist_ok=True)

config = load_config('data/config.json')
with open(os.path.join(run_dir, 'config.json'), 'w') as config_file:
    json.dump(config, config_file, indent=4)

log_filename = os.path.join(run_dir, f'training_log_{run_name}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

with open('src/model_config.json', 'r') as f:
    model_config = json.load(f)
backbone_config = model_config["backbone_config"]
head_config = model_config["head_config"]

model = vrdOCR(backbone_config=backbone_config, head_config=head_config).cuda()
model = load_weights(model, args.model_path)

decoder = CTCLabelDecode(character_dict_path='utils/en_dict.txt', use_space_char=True)
dataset = data.simple_dataset.MultiScaleDataSet(config=config, mode='Train')
sampler = data.multi_scale_sampler.MultiScaleSampler(dataset, **config['Train']['sampler'])

loss_fn = MultiLoss(**{'loss_config_list': [{'CTCLoss': None}, {'NRTRLoss': None}]})
optimizer = optim.Adam(model.parameters(), lr=config['Train']['optimizer']['lr'])
scheduler = StepLR(optimizer, step_size=config['Train']['scheduler']['step_size'], gamma=config['Train']['scheduler']['gamma'])

device = "gpu:{}".format(dist.ParallelEnv().dev_id)
data_loader = DataLoader(
    dataset=dataset,
    batch_sampler=sampler,
    places=device,
    num_workers=config['Train']['loader']['num_workers'],
    return_list=True,
    use_shared_memory=True
)

eval_dataset = data.simple_dataset.SimpleDataSet(config=config, mode='Eval')
eval_sampler = BatchSampler(dataset=eval_dataset, batch_size=config['Eval']['loader']['batch_size_per_card'], shuffle=False)
eval_data_loader = DataLoader(
    dataset=eval_dataset,
    batch_sampler=eval_sampler,
    places=device,
    num_workers=config['Eval']['loader']['num_workers'],
    return_list=True,
    use_shared_memory=True
)

metric = RecMetric()

def evaluate(epc, model, eval_loader):
    model.eval()
    eval_accs = []
    for eval_idx, eval_batch in enumerate(eval_loader):
        eval_images = torch.tensor(eval_batch['image'].numpy()).cuda()
        eval_outs = model(eval_images)
        eval_predictions, eval_labels_decoded = decoder(eval_outs, eval_batch['label_ctc'])
        eval_metr = metric([eval_predictions, eval_labels_decoded])
        eval_accs.append(eval_metr['acc'])
        if eval_idx % config['Train']['print_every_n_batches'] == 0:
            logger.info(f"Eval Batch {eval_idx} | Eval Accuracy: {eval_metr['acc']}%")
    logger.info(f"Epoch {epc} | Eval Accuracy: {sum(eval_accs)/len(eval_accs)}%")

for epc in range(config['Train']['num_epochs']):
    model.train()
    epoch_accuracies = []
    for batch_idx, batch in enumerate(data_loader):
        batch = [torch.tensor(elem.numpy()).cuda() for elem in batch]
        outs = model(batch[0], batch[1:])
        losses = loss_fn(outs, batch)
        total_loss = losses['CTCLoss'] + losses['NRTRLoss']
        preds, labels = decoder(outs['ctc'], batch[1])
        accuracies = metric([preds, labels])
        epoch_accuracies.append(accuracies['acc'])
        if batch_idx % config['Train']['print_every_n_batches'] == 0:
            logger.info(f"Train Batch {batch_idx} | Train Accuracy: {accuracies['acc']}% | Loss: {total_loss.item()}")
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if (batch_idx + 1) % config['Train']['eval_every_n_batches'] == 0:
            evaluate(epc, model, eval_data_loader)
    scheduler.step()
    logger.info(f"Epoch {epc} | Train Accuracy: {sum(epoch_accuracies)/len(epoch_accuracies)}%")
