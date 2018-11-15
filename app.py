import os
import sys
from io import BytesIO
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array

from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage
)

app = Flask(__name__)
file_path = "/images"
# 環境変数からchannel_secret・channel_access_tokenを取得
channel_secret = os.environ['DOG_BOT_CHANNEL_SECRET']
channel_access_token = os.environ['DOG_BOT_CHANNEL_ACCESS_TOKEN']

if channel_secret is None:
    print('Specify CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

model = None

def load_model():
    #load json and create model
    global model
    json_file = open("./model/model.json")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    #load weights into new model
    model.load_weights("./model/model_weights.h5")

classes = ['Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih', 'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black', 'Walker_hound', 'English_foxhound', 'redbone', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier', 'Norwich_terrier', 'Yorkshire_terrier', 'wire', 'Lakeland_terrier', 'Sealyham_terrier', 'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 'soft', 'West_Highland_white_terrier', 'Lhasa', 'flat', 'curly', 'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 'German_short', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel', 'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd', 'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog', 'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle', 'standard_poodle', 'Mexican_hairless', 'dingo', 'dhole', 'African_hunting_dog']

if model is None:
    load_model()

def predict(image):
    global model
    img = load_img(image,target_size=(448,448))
    x = img_to_array(img)
    predicts = model.predict(x.reshape([-1,448,448,3]))
    top3 = predicts.argsort()[2::-1]
    top3_acc = predicts.sort()[2::-1]
    return predicts


@app.route("/")
def hello_world():
    return "hello world!"


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    messages = [
        TextSendMessage(text='犬の画像を送ってみて！品種当てちゃうぞ！'),
    ]

    line_bot_api.reply_message(event.reply_token, messages)

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)
    image = BytesIO(message_content.content)

    top3_index, top3_acc = predict(image)

    message = "This dog is:\n" + classes[top3_index[0]] + "  acc."+ str(float(top3_acc[0])) + "\n" + \
              classes[top3_index[1]] + "  acc." + str(float(top3_acc[1])) + "\n" +classes[top3_index[2]] + \
              "  acc." + str(float(top3_acc[2]))
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text=message
        )
    )

    # line_bot_api.reply_message(
    #         event.reply_token,
    #         TextSendMessage(
    #             text="画像を送ったな。"
    #         )
    # )

# @handler.add(MessageEvent, message=ImageMessage)
# def handle_message(event):
#     message_id = event.message.id
#     message_content = line_bot_api.get_message_content(message_id)
#
#     image = BytesIO(message_content.content)
#     top3_index, top3_acc = predict(image)
#
#     message = "This dog is:\n" + classes[top3_index[0]] + "  acc."+ str(float(top3_acc[0])) + "\n" + \
#               classes[top3_index[1]] + "  acc." + str(float(top3_acc[1])) + "\n" +classes[top3_index[2]] + \
#               "  acc." + str(float(top3_acc[2]))
#     line_bot_api.reply_message(
#         event.reply_token,
#         TextSendMessage(
#             text=message
#         )
#     )


if __name__ == "__main__":
    app.run(port=5001)