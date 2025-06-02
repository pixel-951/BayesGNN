import wandb


from bayes.training.trainer import ModelTrainer



def train():
    
    
    config = wandb.config
    print(config)
    trainer = ModelTrainer(config)
    trainer.train()
      
        
train()

    
   