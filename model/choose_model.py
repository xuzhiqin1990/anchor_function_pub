from .DNN import myDNN
from .LSTM import myLSTM
from .GPT import myGPT
from .GPT_lightly import myGPT_lightly
from .GPT_specific import myGPT_specific

def get_model(args, device):
    if args.model == 'LSTM':
        model = myLSTM(args, device).to(device)
    elif args.model == 'GPT':
        model = myGPT(args, device).to(device)
    elif args.model == 'DNN':
        model = myDNN(args, device).to(device)
    elif args.model == 'GPT_lightly':
        model = myGPT_lightly(args, device).to(device)
    elif args.model == 'GPT_specific':
        model = myGPT_specific(args, device).to(device)

    return model