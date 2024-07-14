from sshtunnel import SSHTunnelForwarder
from pymongo import MongoClient
from pymongo.cursor import CursorType

db_config = {
    'mongo_host': 'yourhost',
    'mongo_db': None, 
    'project': None, 
    'ssh_username': None,
    'ssh_pwd': None,
    'local_addr': '127.0.0.1',
    'local_port': 27017
}

class DBHandler:
    def __init__(self):
        self.server = SSHTunnelForwarder(
            db_config['mongo_host'],
            ssh_username=db_config['ssh_username'],
            ssh_password=db_config['ssh_pwd'],
            remote_bind_address=(db_config['local_addr'], db_config['local_port'])
        )
        self.server.start()
        self.client = MongoClient(db_config['local_addr'], self.server.local_bind_port)
        self.db = self.client[db_config['mongo_db']]
        self.collection = self.db[db_config['project']]

    def insert_item_one(self, data, db_name=db_config['mongo_db'], collection_name=db_config['project']):
        result = self.collection.insert_one(data).inserted_id
        return result

    def insert_item_many(self, datas, db_name=db_config['mongo_db'], collection_name=db_config['project']):
        result = self.collection.insert_many(datas).inserted_ids
        return result

    def find_item_one(self, condition=None, db_name=db_config['mongo_db'], collection_name=db_config['project']):
        result = self.collection.find_one(condition, {"_id": False})
        return result

    def find_item(self, condition=None, db_name=db_config['mongo_db'], collection_name=db_config['project']):
        result = self.collection.find(condition, {"_id": False}, no_cursor_timeout=True, cursor_type=CursorType.EXHAUST)
        return result

    def delete_item_one(self, condition=None, db_name=db_config['mongo_db'], collection_name=db_config['project']):
        result = self.collection.delete_one(condition)
        return result

    def delete_item_many(self, condition=None, db_name=db_config['mongo_db'], collection_name=db_config['project']):
        result = self.collection.delete_many(condition)
        return result

    def update_item_one(self, condition=None, update_value=None, db_name=db_config['mongo_db'], collection_name=db_config['project']):
        result = self.collection.update_one(filter=condition, update=update_value)
        return result

    def update_item_many(self, condition=None, update_value=None, db_name=db_config['mongo_db'], collection_name=db_config['project']):
        result = self.collection.update_many(filter=condition, update=update_value)
        return result

    def text_search(self, text=None, db_name=db_config['mongo_db'], collection_name=db_config['project']):
        result = self.collection.find({"$text": {"$search": text}})
        return result

    def close_connection(self):
        self.server.stop()