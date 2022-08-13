from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from base import SpellingCorrection

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)
obj = SpellingCorrection()

class SpellingError(Resource):
    def post(self):
        req = request.json
        res = obj(req['text'])
        print(res)
        return {
            'IsSuccessed': True,
            'Message': 'Success',
            'ResultObj':{
                'src': req['text'],
                'result': res
            }
        }

api.add_resource(SpellingError, '/api/spelling_correction/')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8016)