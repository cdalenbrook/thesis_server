# app.py

# Required imports
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from firebase_admin import credentials, firestore, storage, initialize_app
from server import decisiontree, cloud_storage, utilities, cloud_firestore


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
firebase_app = initialize_app(cred)
db = firestore.client(app=firebase_app)
storage_bucket = storage.bucket(cloud_storage.BUCKETNAME)

# Cannot convert an array value in an array value.
@app.route('/create-tree', methods=['POST'])
def createTree():
    """
        createTree() : Creates a decision tree based on the data received
        and dumps the json for the tree in a Firestore document
    """
    try:
        session_id = request.json['id']
        categories = request.json['category']
        training_data = request.json['data']

        if not session_id or not categories or not training_data:
            return "Request malformed.", 400

        print(f'Creating tree for session: {session_id}.')

        tree = decisiontree.create_tree(categories, training_data)
        # tree_image = decisiontree.tree2image(tree, categories)
        # tree_image_url = cloud_storage.upload_image(
        # storage_bucket, f'{session_id}.png', tree_image)
        print(f'Decision tree created.')

        tree_image_url = 'http://via.placeholder.com/150'
        tree_json = utilities.tree2json(tree)

        print(f'Decision tree jsonified.')
        print(f'Tree data type: {type(tree_json)}')

        cloud_firestore.save_tree(
            db,
            session_id,
            tree_json,
            categories,
            training_data,
            tree_image_url
        )

        print(f'Created and saved decision tree: {session_id}')

        return "sucess", 200
    except Exception as e:
        return f"An Error Occured: {e}", 400


@app.route('/predict', methods=['POST'])
def predict():
    """
        predict() : Gets a prediction from a tree fetched from Firestore
        using the session_id provided in the request
    """
    try:
        session_id = request.json['id']
        toy_id = request.json['toy_id']

        if not session_id or not toy_id:
            return jsonify({"success": False, "message": f'Request malformed. id: {session_id} toy_id: {toy_id}'}), 400

        tree_json = cloud_firestore.get_tree(db, session_id=session_id)

        if not tree_json:
            return jsonify({"success": False, "message": f'Tree with id {session_id} does not exist.'}), 400

        print(f'Found tree for session {session_id}')

        tree = utilities.json2tree(tree_json)

        print(f'Deserialized the tree for session {session_id}')

        prediction = decisiontree.predict(tree, toy_id)

        print(
            f'Predicting {prediction} for toyID {toy_id} in session {session_id}')

        return jsonify({"success": prediction is not None, "prediction": prediction}), 200

    except Exception as e:
        return f"An Error Occured: {e}"


port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=port)
