from pymongo import MongoClient

        
        
client = MongoClient("mongodb://192.168.0.99:27017/")

db = client["test_database"]
collection = db["test_collection"]


collection.insert_one({"dsr_state_xpos" : 999})


