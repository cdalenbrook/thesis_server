import click
from flask import Flask, request
import decisiontree
import utils as utilities
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/make-tree", methods=["POST"])
def handle_data():
    try:
        user_id = request.args.get("id", "default")
        category = request.json["category"]
        data = request.json["data"]
        dev = request.json['dev']
        result = decisiontree.main(categories=category, data=data, dev=dev)

        utilities.store_tree(user_id, result)
        utilities.store_targets(user_id, request.json["category"])
        utilities.store_usertruth(user_id, request.json["data"])
        return "success"
    except Exception as err:
        print(err)
        return "failed", 500


@app.route("/predict-category", methods=["POST"])
def predict():
    try:
        tree_id = request.args.get("id", "default")
        toy_id = request.json["toy_id"]
        tree = utilities.get_tree(tree_id)
        targets = utilities.get_targets(tree_id)
        user_truth = utilities.get_usertruth(tree_id)
        if(tree is None or targets is None or user_truth is None):
            return "Your tree doesn\"t exist yet!", 404
        y_pred = decisiontree.predict(tree, toy_id, user_truth, targets[0])
        print('Prediction: ', targets[y_pred[0]])
        return {"prediction": y_pred.tolist()}
    except Exception as err:
        print(err)
        return "Failed", 500
