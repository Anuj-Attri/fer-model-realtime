from keras.models import model_from_json
import numpy as np
import emoji

class FacialExpressionModel(object):

    EMOTELIST = ["Angry", "Nauseated",
                     "Fear", "Happy",
                     "Sad", "Sad",
                     "Calm"]

    def __init__(self, fer_model_file, fer_weights_file):
        # load Model from JSON file
        with open(fer_model_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new Model
        self.loaded_model.load_weights(fer_weights_file)
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTELIST[np.argmax(self.preds)]
