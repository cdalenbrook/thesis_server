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

        tree_json = utilities.store_tree(user_id, result)
        return {
            "tree": tree_json,
            "user_truth": data,
            "category": category
        }
    except Exception as err:
        print(err)
        return "failed", 500


@app.route("/predict-category", methods=["POST"])
def predict():
    try:
        tree_json = request.json["tree"]
        toy_id = request.json["toy_id"]
        user_truth = request.json["user_truth"]
        targets = request.json["category"]
        tree = utilities.get_tree(tree_json)
        if(tree is None):
            return "Your tree doesn't exist yet!", 404
        if(targets is None):
            return "You haven't set any targets!", 404
        if(user_truth is None):
            return "You never gave us the truth!", 404
        y_pred = decisiontree.predict(tree, toy_id, user_truth, targets[0])
        print('Prediction: ', targets[y_pred[0]])
        return {"prediction": y_pred.tolist()}
    except Exception as err:
        print(err)
        return "Failed", 500
