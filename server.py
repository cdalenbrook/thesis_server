from flask import Flask, request
import decisiontree
app = Flask(__name__)


@app.route('/', methods=['POST'])
def handle_data():
    try:
        print(request.json['data']['bike_001'])

        print(request.json['category'])
        print(request.json['data'])
        result = decisiontree.main(
            request.json['category'], request.json['data'])
        print(result)
        return 'success'
    except Exception as err:
        print(type(err))
        return 'failed'
