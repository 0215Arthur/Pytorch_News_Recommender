# from model.nrms import NRMS
# from model.nrms_v0 import NRMS_V0
# from model.nrms_v1 import NRMS_V1
"""
The wrapper of the model object. 

"""
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.parallel as P

class Model(nn.Module):
    def __init__(self, config,args):
        super(Model, self).__init__()
        print('Building model...')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #args.device = self.device
        self.n_GPUs = args.n_GPUs

        module = import_module('model.' + args.model.lower())
        self.model = module.Model(config).to(self.device)
        # load the pre-trained parameters
        # if args.load is not None:
        #     print('Loading model from', args.load)
        #     #checkpoint = torch.load(args.load, map_location=self.device)
        #     #model.load_state_dict(torch.load('../save_model/'+ckpt_file))checkpoint['model']
        #     #self.model.load_state_dict(torch.load(config.save_path+args.load))
        # else:
        #     print('Finish making model ' + args.model)

    def forward(self, batch):
        if self.train:
            if self.n_GPUs > 1 and not self.cpu:
                return P.data_parallel(self.model, batch, range(self.n_GPUs))
            else:
                return self.model(batch)
