import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
import wandb,yaml

with open('iris.yaml') as file:
    config_dict = yaml.full_load(file)



class IrisModel(LightningModule):

    def __init__(self, input_dim, hidden_dim, output_dim, lr):
        super(IrisModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.learning_rate = lr

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("val_loss", loss, sync_dist=True,prog_bar=True , on_step=False, on_epoch=True,logger=True)
        return loss
        



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



'''
def train():
    run = wandb.init(config= config_dict, 
                    project="iris",                            
                    dir='store_iris_data') #,entity="kkb")
    train_loader = DataLoader(train_data, batch_size=run.config.batch_size)
    test_loader = DataLoader(test_data, batch_size=run.config.batch_size)

    model = IrisModel(input_dim=run.config.input_dim, hidden_dim=run.config.hidden_dim, output_dim=run.config.output_dim)

    trainer = Trainer(max_epochs=run.config.max_epochs,accelerator='gpu' , devices= 'auto')
    trainer.fit(model, train_dataloaders=train_loader , val_dataloaders=test_loader)

    wandb.finish(1, False)


sweep_config = {
    'method': 'bayes',
    'metric': {'goal': 'minimize', 'name': 'HTER (val_loss)'},
    'parameters': 
    {	
        'batch_size':  {'max': 16,
                        'min': 4,
                        'distribution': 'int_uniform'},
        'learning_rate':      {'max': 0.1,
                        'min': 0.0001,
                        'distribution': 'uniform'},
        'max_epochs':  {'max': 10,
                        'min': 2,
                        'distribution': 'int_uniform'},

     }
}

sweep_id = wandb.sweep(sweep_config, project="iris")
wandb.agent(sweep_id, function=train, count=10)
'''

################


sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [16, 32, 64]
        },
        'learning_rate': {
            'values': [0.01, 0.02, 0.03]
        },
        'max_epochs': {
            'values': [10, 20, 30]
        }
    }
}




def train(config=None):
    with wandb.init(config=config_dict): #, project="Iris"):
        config = wandb.config
        iris = load_iris()
        X = iris.data
        Y = iris.target
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train).long())
        test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test).long())

        train_loader = DataLoader(train_data, batch_size=config.batch_size)
        test_loader = DataLoader(test_data, batch_size=config.batch_size)
       
        model = IrisModel(input_dim=4, hidden_dim=10, output_dim=3, lr =config.learning_rate )
        #device = torch.device('cuda:0' )
        model = model #.to(device)
        #model.configure_optimizers = lambda: torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        trainer = Trainer(max_epochs=config.max_epochs, accelerator='gpu' , devices= [0,1,2] ) # gpus=1 if torch.cuda.is_available() else 0) #,accelerator='gpu' , devices= 'auto') # [0,1,2,3,4,6] , strategy='ddp' )
        trainer.fit(model, train_dataloaders=train_loader , val_dataloaders=test_loader)

        # for batch in test_loader:
        #     x, y = batch
        #     logits = model(x)
        #     loss = F.cross_entropy(logits, y)
        #     wandb.log({"loss": loss})

sweep_id = wandb.sweep(sweep_config, project="Iris")

wandb.agent(sweep_id, train)