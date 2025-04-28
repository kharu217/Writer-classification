from utils import *
from model import *


def infer(text : str, model_addr = "models\writer_model.h5") :
    infer_token = Line2tensor(''.join(i for i in text if i in all_letters))
    infer_tensor = torch.tensor(infer_token)

    infer_model = Writer_classifi(embd_n=embd_n, hidden_size=hidden_size, layer_n=layer_n).eval()
    infer_model.load_state_dict(torch.load(model_addr, map_location=torch.device(device)))
    with torch.no_grad() :
        pred = infer_model(infer_tensor).flatten()

    pred_dict = dict(zip(writer_list, np.round(pred.tolist(), decimals=3)))
    return pred_dict

def infer_result(text_l : list, addr) :
    file = open(addr, 'w+')
    bar = "|"
    result_txt = ""

    img = Image.new('RGB', (900, 500), color='white')
    draw = ImageDraw.Draw(img)

    for i, text in enumerate(text_l) :
        infer_dict = infer(text)
        result_txt += f"Inference text {i+1} : '{(text[:len(text)//2] + '\n                                ' + text[len(text)//2:]) if len(text) >= 100 else text}'\n"
        result_txt += f"  |   goethe : {bar*int((infer_dict["goethe"])*20)} {infer_dict["goethe"]*100}%\n"
        result_txt += f"  |   hesse  : {bar*int((infer_dict["hesse"])*20)} {infer_dict["hesse"]*100}%\n"
        result_txt += f"  |   kafka  : {bar*int((infer_dict["kafka"])*20)} {infer_dict["kafka"]*100}%\n"
        result_txt += "------------------------------------------------\n\n"

    draw.text((0, 0), result_txt, fill=(0, 0, 0))
    img.save(addr)
    
if __name__ == "__main__" :
    infer_result(["I don’t know whether deluding spirits hover about this region, or whether it is the warm, heavenly fancy in my heart which turns my whole environment into a paradise.",
                  "I see that I have lapsed into raptures, parables and oratory, and have thus forgotten to complete the story of my further doings with the children.",
                  "You have long known my way of settling down, pitching my tent in some spot I like, and lodging there in a modest fashion. Here too I have again come upon a nook that I found attractive.",
                  "We need to recall such facts when we approach a story which is in many respects so foreign to our present modes of thought and expression. Young men today, however greatly they may be influenced by emotion, do not shed “a thousand tears” or impress “a thousand kisses” on a lady’s picture",
                  "For the rest, I like it here very much. Solitude in this paradise is a precious balm to my heart, and this youthful time of year warms with all its fullness my oft-shivering heart. ",],
                'resrc\\goethe_werther.png')
    
    infer_result(["It was my destiny to join in a great experience. Having had the good fortune to belong to the League, I was permitted to be a participant in a unique journey.",
                  " chapels and altars were adorned with flowers; ruins were honored with songs or silent contemplation; the dead were commemorated with music and prayers.",
                  "I see that I am already coming up against one of the greatest obstacles in my account. The heights to which our deeds rose, the spiritual plane of experience to "
                  "that I quickly forgot him. But it happened some time later, when none of us"
                  "thought about him any more, that we heard the inhabitants of several villages and towns through which we passed, talk about this same youth.",
                  "Still another had conceived the idea of capturing a certain snake to which he attributed magical powers and which he called Kundalini. My own journey and life-goal, which had colored my dreams since my late boyhood, was to see the beautiful Princess Fatima and, if possible, to win her love", 
                  "” I asked about the oldest, and she had hardly told me that he was racing around the meadow with some geese when he came running and brought the second boy a hazel switch. I"],
                 'resrc\\hesse_journey.png')
    
    infer_result([", I don’t know why the matter shouldn’t come to the attention of your parents. Your productivity has also been very unsatisfactory recently.",
                  "And he looked over at the alarm clock ticking away by the chest of drawers. ‘Good God,’ he thought. It was half past six, and the hands were going quietly on.",
                  "However, after a similar effort, while he lay there again sighing as before and once again saw his small limbs fighting one another, if anything worse than before,",
                  "Why was Gregor the only one condemned to work in a firm where at the slightest lapse someone immediately attracted the greatest suspicion? ",
                  "‘Mr. Samsa,’ the manager was now shouting, his voice raised, ‘what’s the matter? You are barricading yourself in your room, answer with only a yes and a no,"],
                 "resrc\\kafka_metapho.png")
    