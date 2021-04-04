import os
import sys
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import efficientnet.tfkeras

from PIL import Image

from flask import Flask, request, abort, render_template, redirect, flash
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    ImageMessage,
)

app = Flask(__name__)
file_path = "/images"
# 環境変数からchannel_secret・channel_access_tokenを取得
channel_secret = os.environ["DOG_BOT_CHANNEL_SECRET"]
channel_access_token = os.environ["DOG_BOT_CHANNEL_ACCESS_TOKEN"]

if channel_secret is None:
    print("Specify CHANNEL_SECRET as environment variable.")
    sys.exit(1)
if channel_access_token is None:
    print("Specify CHANNEL_ACCESS_TOKEN as environment variable.")
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# load model
model = load_model("./model/efficient_net_model.h5")


classes = [
    "Chihuahua",
    "Japanese_spaniel",
    "Maltese_dog",
    "Pekinese",
    "Shih-Tzu",
    "Blenheim_spaniel",
    "papillon",
    "toy_terrier",
    "Rhodesian_ridgeback",
    "Afghan_hound",
    "basset",
    "beagle",
    "bloodhound",
    "bluetick",
    "black-and-tan_coonhound",
    "Walker_hound",
    "English_foxhound",
    "redbone",
    "borzoi",
    "Irish_wolfhound",
    "Italian_greyhound",
    "whippet",
    "Ibizan_hound",
    "Norwegian_elkhound",
    "otterhound",
    "Saluki",
    "Scottish_deerhound",
    "Weimaraner",
    "Staffordshire_bullterrier",
    "American_Staffordshire_terrier",
    "Bedlington_terrier",
    "Border_terrier",
    "Kerry_blue_terrier",
    "Irish_terrier",
    "Norfolk_terrier",
    "Norwich_terrier",
    "Yorkshire_terrier",
    "wire-haired_fox_terrier",
    "Lakeland_terrier",
    "Sealyham_terrier",
    "Airedale",
    "cairn",
    "Australian_terrier",
    "Dandie_Dinmont",
    "Boston_bull",
    "miniature_schnauzer",
    "giant_schnauzer",
    "standard_schnauzer",
    "Scotch_terrier",
    "Tibetan_terrier",
    "silky_terrier",
    "soft",
    "West_Highland_white_terrier",
    "Lhasa",
    "flat",
    "curly",
    "golden_retriever",
    "Labrador_retriever",
    "Chesapeake_Bay_retriever",
    "German_short",
    "vizsla",
    "English_setter",
    "Irish_setter",
    "Gordon_setter",
    "Brittany_spaniel",
    "clumber",
    "English_springer",
    "Welsh_springer_spaniel",
    "cocker_spaniel",
    "Sussex_spaniel",
    "Irish_water_spaniel",
    "kuvasz",
    "schipperke",
    "groenendael",
    "malinois",
    "briard",
    "kelpie",
    "komondor",
    "Old_English_sheepdog",
    "Shetland_sheepdog",
    "collie",
    "Border_collie",
    "Bouvier_des_Flandres",
    "Rottweiler",
    "German_shepherd",
    "Doberman",
    "miniature_pinscher",
    "Greater_Swiss_Mountain_dog",
    "Bernese_mountain_dog",
    "Appenzeller",
    "EntleBucher",
    "boxer",
    "bull_mastiff",
    "Tibetan_mastiff",
    "French_bulldog",
    "Great_Dane",
    "Saint_Bernard",
    "Eskimo_dog",
    "malamute",
    "Siberian_husky",
    "affenpinscher",
    "basenji",
    "pug",
    "Leonberg",
    "Newfoundland",
    "Great_Pyrenees",
    "Samoyed",
    "Pomeranian",
    "chow",
    "keeshond",
    "Brabancon_griffon",
    "Pembroke",
    "Cardigan",
    "toy_poodle",
    "miniature_poodle",
    "standard_poodle",
    "Mexican_hairless",
    "dingo",
    "dhole",
    "African_hunting_dog",
]


def predict(input_image):
    img = Image.open(input_image)
    img = img.resize((224, 224), Image.NEAREST)
    x = img_to_array(img)
    x /= 255
    result = model.predict(x.reshape([-1, 224, 224, 3]))
    predicted = result.argmax()
    return classes[predicted]


UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("ファイルがありません")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("ファイルがありません")
            return redirect(request.url)

        image = BytesIO(file.stream.read())
        pred = predict(image)
        pred_answer = f"この犬は[{pred}]ですね！"

        return render_template("index.html", answer=pred_answer)

    return render_template("index.html", answer="")


@app.route("/callback", methods=["POST"])
def callback():
    # get X-Line-Signature header value
    signature = request.headers["X-Line-Signature"]

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    messages = [
        TextSendMessage(text="犬の画像を送ってみて！品種当てちゃうぞ！"),
    ]

    line_bot_api.reply_message(event.reply_token, messages)


@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)
    image = BytesIO(message_content.content)

    pred = predict(image)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=f"この犬は[{pred}]ですね！"),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
