import click
from flask import Flask, request
import decisiontree
import utils as utilities
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/make-tree', methods=['POST'])
def handle_data():
    try:
        user_id = request.args.get('id', 'default')
        result = decisiontree.main(
            request.json['category'], request.json['data'])

        utilities.store_tree(user_id, result)
        utilities.store_targets(user_id, request.json['category'])
        utilities.store_usertruth(user_id, request.json['data'])
        return 'success'
    except Exception as err:
        print(err)
        return 'failed', 500


@app.route('/predict-category', methods=['POST'])
def predict():
    try:
        tree_id = request.args.get('id', 'default')
        tree = utilities.get_tree(tree_id)
        target = utilities.get_target(tree_id)
        user_truth = utilities.get_usertruth(tree_id)
        if(tree is None or target is None or user_truth is None):
            return 'Your tree doesn\'t exist yet!', 404
        y_pred = decisiontree.predict(
            tree, request.json['toy_id'], user_truth, target)
        return {'prediction': y_pred.tolist()}
    except Exception as err:
        print(err)
        return 'Failed', 500
