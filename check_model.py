from network import Network
from config import *
import torch

path = 'models/insert_cube/model_EGO_AND_BIRDSEYE_TARGET_POSITION_128px_RGB_1000_episodes_2step_lstm_1layers_4seq_transformers_taskloss_10.pth'
# path = 'models/insert_cube/model_BIRDSEYE_POSITION_128px_1000_episodes_lstm_4seq_1layers_1.pth'

if __name__ == "__main__":
    device = torch.device('mps')

    # model = Network(len(ACTIVE_CAMERAS), device).to(device)
    state_dict = torch.load(path)

    # model.load_state_dict(state_dict)
    
    for name, param in state_dict.items():
        print(f"{name}: {param.size()}")




