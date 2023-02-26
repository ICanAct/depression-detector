from trainer_base import TrainerBase
import copy
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from utils import source_import, print_write
from torch.utils.tensorboard import SummaryWriter
from transformers import (
AdamW, get_linear_schedule_with_warmup
)

class Bert(TrainerBase):
    def __init__(self, config, test=False, device=None):
        super().__init__(config, test, device)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.f1= None
        # Initialise the tokenizer and model
        # TODO: Implement this
        self.tokenizer = None
    
        self.init_models()
        self.initalize_dataloaders()

        # Under training mode, initialize training steps, optimizers, schedulers
        if not self.test_mode:
            self.writer = SummaryWriter(log_dir=self.config['Path']['log_dir'])
            print('Using steps for training.')
            self.training_data_num = len(self.data['train'])
            self.epoch_steps = int(self.training_data_num  \
                                   / self.config['Model']['params']['batch_size'])

            # Initialize model optimizer and scheduler
            print('Initializing model optimizer.')
            self.model_optimizer, \
            self.model_optimizer_scheduler = self.init_optimizers()
    
    def initalize_dataloaders(self):
        print("Constructing dataloaders...")
        phases = self.config['Data']['phases']
        data_module = source_import(self.config['Data']['path'])
        
        self.data = {
            X: data_module.RedditDataset() for X in phases
        }
        
        for x in phases:
            print_write(["{} data size: {}".format(x, len(self.data[x]))], self.log_file)
            
        self.dataloaders = {
            X:DataLoader(
                self.data[X],
                batch_size=self.config['Model']['params']['batch_size'],
                shuffle=True if X == 'train' else False,
                num_workers=self.config['Model']['params']['num_workers']
            ) for X in phases
        }
    
    def init_optimizers(self):
        """Initialize model optimizer and scheduler"""
        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.config['Model']['params']['lr'],
            eps=self.config['Model']['params']['eps']
        )   

        # Initialize scheduler
        updates_total = self.config['Model']['params']['epoch'] * len(self.data["train"])
        scheduler = get_linear_schedule_with_warmup(
                        optimizer, 
                        num_warmup_steps=round(self.config['Model']['params']['lr_warmup']*updates_total), 
                        num_training_steps=updates_total)

        return optimizer, scheduler

    def train(self):
        best_f1 = 0
        best_epoch = 0
        loss_list = []
        self.epoch = epoch
        
        end_epoch = self.config['Model']['params']['epoch']
        
        for epoch in range(1, end_epoch+1):
            print_write(["Epoch: {}".format(epoch)], self.log_file)

            # Training
            self.model.train()
            train_loss = 0
            for step, inputs in enumerate(self.dataloaders['train']):
                # move data to cuda.
                
                for key in inputs.keys():
                    if torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(self.device)
                labels = inputs['labels']

                with torch.set_grad_enabled(True):
                    # TODO: Edit this part
                    outputs = self.model(**inputs)
                    loss = self.loss_fn(outputs, labels)
                    
                    # Backward pass
                    self.model_optimizer.zero_grad()
                    loss.backward()
                    self.model_optimizer.step()
                    self.model_optimizer_scheduler.step()

                    if step % self.config['Model']['params']['display_step'] == 0 and step != 0:
                        minibatch_loss_perf = self.loss.item()
                        _, preds = torch.max(self.logits, 2)

                        # implment compute_metrics function
                        metrics_dict = self.compute_metrics(preds, labels)
                    
                        print_str = ['Epoch: [%d/%d]'
                                     % (epoch, end_epoch),
                                     'Step: [%d/%d]'
                                     % (step, self.epoch_steps),
                                     'precision: %.3f'%(metrics_dict['precision']),
                                     'recall: %.3f'%(metrics_dict['recall']),
                                     'f1 score: %3f'%(metrics_dict['f1']),
                                     'loss: %3f'%(minibatch_loss_perf)]

                        print_write(print_str, self.log_file)
            self.writer.add_scalar('Loss/Train', sum(loss_list)/step, epoch)
             
            # validation
            #write validation code and call accordingly.
            self.eval(phase='val')
            
            # Update best model
            if self.f1 > best_f1:
                best_epoch = copy.deepcopy(epoch)
                best_model = copy.deepcopy(self.model.state_dict())
                best_f1 = self.f1
        
        print('Training Complete.')
        print_str = ['Best validation f1 is %.3f at epoch %d' % (best_f1, best_epoch)]
        print_write(print_str, self.log_file)
        # Save the best model
        self.save_model(best_model)
        print("writing prediction txt file...")
        self.eval(phase='prediction')
        print('Done')
    

    def eval(self, phase):
        raise NotImplementedError
    
    def compute_metrics(self, preds, labels):
        raise NotImplementedError