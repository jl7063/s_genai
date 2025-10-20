from helper_lib.model import get_model
from helper_lib.trainer import train_model

if __name__=="__main__": 
    train_model(get_model("CNN", 10))
    train_model(model)
    
