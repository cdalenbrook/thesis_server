import click
from flask import Flask, request
import decisiontree
from utils import store_tree, get_tree, get_target
from flask_cors import CORS

app = Flask(__name__)

CORS(app)


@app.route('/make-tree', methods=['POST'])
def handle_data():
    try:
        result = decisiontree.main(
            request.json['category'], request.json['data'])

        store_tree(result, request.args.get(
            'id', ''), request.json['category'])
        return 'success'
    except Exception as err:
        print(err)
        return 'failed'


@app.route('/predict-category', methods=['POST'])
def predict():
    tree_id = request.args.get('id', '')
    tree = get_tree(tree_id)
    target = get_target(tree_id)
    if(tree is None):
        return 'Your tree doesn\'t exist yet!'
    y_pred = decisiontree.predict(
        tree, request.json['toy_id'], request.json['user_truth'], target)
    return {'prediction': y_pred.tolist()}
