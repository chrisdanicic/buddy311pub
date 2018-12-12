from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import random
from finetune import Classifier
import pandas
from sklearn.model_selection import train_test_split
import time
import re, string

app = Flask(__name__)
cors = CORS(app, resources={r"*":{"origins": "*"}})
api = Api(app)
model = None

class test(Resource):
	def get(self):
		return{'result':'Server is active'}

	def post(self):
		some_json = request.get_json()
		if some_json.get('name') == None:
			print("Did not receive a name field")
		else:
			print("name received: ", some_json.get('name'))
		return {'you sent': some_json}, 201

"""
The class to implement our Buddy311 REST interface
Usage: simply implement the post method that sends in a text in JSON format in
	a field called 'description' and returns the classification of the text
"""

class buddy311(Resource):

	# The POST method which receives POST http requests (data sent in JSON format)
	# and returns the classification
	def post(self):
		global model
		print("Received POST request on Open311 interface")

		# if the classifier has not been loaded then load it
		if model == None:
			model = Classifier(max_length=512, val_interval=3000, verbose = True)
			model = Classifier.load("/root/combined_model_20181021")

		# check if the JSON description has been filled in
		some_json = request.get_json()
		print("Received JSON: ", some_json)
		if some_json.get('description') == None:
			return {'service_code': 'unknown'}
		newTextDescription = some_json.get('description')
		print("received: ", newTextDescription)
		prediction = model.predict([newTextDescription])
		# check if the input data also contained the service code and if so replace it 
		# and return the original message
		if some_json.get('service_code') == None:
			# No service code so just return that
			print("No service code provided, returning one")
			return {'service_code': prediction[0]}
		else:
			print("Service_code was provided so updating it")
			some_json['service_code'] = prediction[0]
			return some_json


"""
This class takes in requests from the google go interface
"""

class dialogflow(Resource):

	def get(self):
		print("Received get request on google interface")
		return{'result':'Server is active'}

	# The POST method which receives POST http requests (data sent in JSON format)
	# and returns the classification
	def post(self):
		global model
		print("Received POST request on google interface")

		# if the classifier has not been loaded then load it
		if model == None:
			model = Classifier(max_length=512, val_interval=3000, verbose = True)
			model = Classifier.load("/root/combined_model_20181021")

		# check if the JSON description has been filled in
		some_json = request.get_json()
		if some_json.get('queryResult') == None:
			print("Empty message text")
			return {'fulfillmentText': 'unknown'}
		queryResult = some_json.get('queryResult')
		if queryResult.get('queryText') == None:
			print("Empty message text")
			return {'fulfillmentText': 'unknown'}
		newTextDescription = queryResult.get('queryText')
		print("received: ", newTextDescription)

		# Predict the classification of the text
		prediction = model.predict([newTextDescription])

		# Return the result
		print("returning: ", prediction[0])
		return {'fulfillmentText': prediction[0]}

# Map URL paths to classes that implement the http requests for these paths
api.add_resource(test, '/')
api.add_resource(buddy311, '/buddy311/v0.1/')
api.add_resource(dialogflow, '/v0.1/assistant')

# Set up SSL keys
context = ('/etc/pki/tls/certs/www_buddy311_org.crt', '/etc/ssl/private/www.buddy311.org.key')

# Enable debugging and set the port to 31102
if __name__ == '__main__':
	app.run(debug=True, port=31102, host='0.0.0.0')#, ssl_context=context) #ssl_context='adhoc')
