# using flask_restful
from flask import Flask, jsonify, request
from flask_restful import Resource, Api

from sap_gpt.SapGPTClass import SapGpt

# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)

sapGPT = SapGpt().init_prodigy()


# making a class for a particular resource
# the get, post methods correspond to get and post requests
# they are automatically mapped by flask_restful.
# other methods include put, delete, etc.
class SapAPI(Resource):

    # corresponds to the GET request.
    # this function is called whenever there
    # is a GET request for this resource
    def get(self):
        return jsonify({'message': 'hello world'})

    # Corresponds to POST request
    def post(self):
        data = request.get_json()  # status code
        response = sapGPT.answer_with_chain(data["query"])
        return jsonify(response)


api.add_resource(SapAPI, '/')

# driver function
if __name__ == '__main__':
    app.run(debug=True)
