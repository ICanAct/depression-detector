import torch
import datetime
import os
from utils import source_import
class TrainerBase():
    
    def __init__(self, config, test=False, device=None) -> None:
        self.device = device
        self.config = config
        self.model_name = config.model_name
        self.test_mode = test
        
        # setup logging
        curr_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        self.config['Path']['log_dir'] = os.path.join(self.config['Path']['log_dir'], curr_time)

        os.makedirs(self.config['Path']['log_dir'], exist_ok=True)

        self.log_file = os.path.join(self.config['Path']['log_dir'], 'log.txt')
        
    def init_models(self):
        self.model = source_import(self.config['Model']['Path']).create_model(
            config=self.config,
            device=self.device
        ).to(self.device)
        
        with open(os.path.join(self.config['Path']['log_dir'], "architecture.txt"), "w") as f:
            f.write(str(self.model))
            
    def load_model(self):
        model_dir = os.path.join(self.config['Path']['log_dir'],
                                 'final_model_checkpoint.pth')

        print('Loading model from %s' % (model_dir))

        checkpoint = torch.load(model_dir)
        self.model.load_state_dict(checkpoint)

    def save_model(self, best_model):

        model_dir = os.path.join(self.config['Path']['log_dir'],
                                 'final_model_checkpoint.pth')

        torch.save(best_model, model_dir)