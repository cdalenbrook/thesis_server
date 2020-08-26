from firebase_admin import firestore
from typing import List


def save_tree(db, session_id: str, tree, categories: List[str], training_data: dict, tree_image: str):
    sessions_ref = db.collection('sessions')
    print(tree)
    data = {
        "tree": tree,
        "treeImageUrl": tree_image,
        "training": training_data,
        "categories": categories
    }
    sessions_ref.document(session_id).set(data, merge=True)


def get_tree(db, session_id: str):
    sessions_ref = db.collection('sessions')
    session = sessions_ref.document(session_id).get()
    return session.get('tree')
