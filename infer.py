import torch

from data import *
from utils import *
from model import *


def infer(text : str, model_addr = "models\writer_model.h5") :
    infer_token = data.Line2tensor(''.join(i for i in text if i in all_letters))
    infer_tensor = torch.tensor(infer_token)

    infer_model = Writer_classifi(embd_n=embd_n, hidden_size=hidden_size, layer_n=layer_n).eval()
    infer_model.load_state_dict(torch.load(model_addr, map_location=torch.device(device)))

    pred = infer_model(infer_tensor)
    return pred

if __name__ == "__main__" :
    print(infer("""aze on the sad procession of pitiful exiles.
Fully a league it must be to the causeway they have to pass over,
Yet all are hurrying down in the dusty heat of the noonday.
I, in good sooth, would not stir from my place to witness the sorrows
Borne by good, fugitive people, who now, with their rescued possessions,
Driven, alas! from beyond the Rhine, their beautiful country,
Over to us are coming, and through the prosperous corner
Roam of this our luxuriant valley, and traverse its windings.
Well hast thou done, good wife, our son in thus kindly dispatching,
Laden with something to eat and to drink, and with store of old linen,
'Mongst the poor folk to distribute; for giving belongs to the wealthy.
How the youth drives, to be sure! What control he has over the horses!
Makes not our carriage a handsome appe"""))
    