import pymongo
import configparser as cfg
import datetime


class Smartparkdb:
    def __init__(self):
        config_reader = cfg.ConfigParser()
        config_reader.read('config.ini')
        print("Python read config : ", config_reader.sections())
        connection_string = config_reader['DB_Connection']['Connection_String']
        self.DB_Name = config_reader['DB_Connection']['DB_Name']
        self.Collection_Name = config_reader['DB_Connection']['Collection_Name']
        try:
            self.Mongodb_obj = pymongo.MongoClient(connection_string,serverSelectionTimeoutMS=6000)
            var_test = self.Mongodb_obj.admin.command('ping')
            if var_test:
                Client= self.Mongodb_obj[self.DB_Name]
                DB_Collection = Client[self.Collection_Name]
                print("Database connected successfully")
            else:
                raise Exception("Invalid Connection string or DB Doesnt Exist")
            # #Var_Db_check = DB_Collection.find_one()
            # if Var_Db_check != None:
            #     print("Database connected successfully")
            # else:
            #     raise Exception("There is no data in collection")

        except Exception as E:
            print(" Exception  raised during db connection \n  Error : ", E)

    def __str__(self):
        return "This variable contains Smartparkdb class object "

    def record_count_old(self):
        var_count = 0
        for i in self.Mongodb_obj[self.DB_Name][self.Collection_Name].find():
            var_count = var_count + 1
        # print("Record Count :",var_count)
        return var_count

    def record_count(self):
        try:
            db_count =self.Mongodb_obj[self.DB_Name][self.Collection_Name].find({}, {'_id': 1}).sort('_id', -1).limit(1)[0]['_id']
        except Exception as E:
            print("No Record Found")
            db_count = 0
        return db_count

    def insert_in_db(self, Entry_DateTime, Vehicle_Type, Vehicle_Number, Vehicle_Status='parked'):
        try:
            db_object = self.Mongodb_obj[self.DB_Name][self.Collection_Name]
            park_id = self.record_count()
            insert_json = {'_id': park_id + 1, 'vehicle_status': Vehicle_Status, 'entry_datetime': Entry_DateTime,
                           'vehicle_type': Vehicle_Type, 'vehicle_number': Vehicle_Number}
            insert_result = db_object.insert_one(insert_json).inserted_id
            print(insert_result)
            return insert_result
        except TypeError as ty_ex:
            print("Unable to insert in DB due to TypeError \n Reason :", ty_ex)
        except Exception as e:
            print("Unable  to insert in db \n  Resaon : ", e)

    def vehicle_entry_check(self, Vehicle_Number):
        try:
            db_object = self.Mongodb_obj[self.DB_Name][self.Collection_Name]
            fetch_result = db_object.find_one({'vehicle_number': Vehicle_Number, 'vehicle_status': 'parked'})
            if fetch_result != None:
                return fetch_result
            else:
                return False
        except Exception as E:
            print("Unable to Fetch vehicle Details \nReason: ",E)

    def update_exit_in_db(self, parked_id, entry_datetime: datetime, vehicle_type, vehicle_number,
                          exit_datetime: datetime):

        if vehicle_type == 'motorcycle' or vehicle_type == 'car':
            db_object = self.Mongodb_obj[self.DB_Name][self.Collection_Name]
            config_reader = cfg.ConfigParser()
            config_reader.read('config.ini')
            Complimentary_Hours = int(config_reader[vehicle_type]['Complimentary_Hours'])
            Day_FixRate = int(config_reader[vehicle_type]['Day_FixRate'])
            Day_PerHour= int(config_reader[vehicle_type]['Day_PerHour'])
            free_parking_duration =int(config_reader[vehicle_type]['Free_Parking_Duration'])

            try:
                fare = 0
                total_difference = exit_datetime-entry_datetime
                parked_duration = divmod(total_difference.total_seconds(),60)
                if parked_duration[0] < free_parking_duration :
                    fare = 0
                    print("Free Parking")
                elif parked_duration[0]/60 <= Complimentary_Hours:
                    fare = Day_FixRate
                else:
                    additional_hours = int(parked_duration[0]/60 -2.0)
                    fare = Day_FixRate + (Day_PerHour * additional_hours)
                print(f"Vehicle Number {vehicle_number} has to pay {fare} and it's duration time {parked_duration} minutes")
                update_json ={'$set':{'vehicle_status':'Done','exit_datetime':exit_datetime,'parking_duration':int(parked_duration[0]),'fare':fare}}
                update_result = db_object.update_one({'_id':parked_id,'vehicle_number':vehicle_number},update_json)
                print(update_result)
                return parked_duration[0]/60,fare
            except Exception as ex:
                print("Exception raised while running fare logic or update exit module \n Reason : ", ex)

        else:
            print("Unknown Vehicle Type")
            return False

    def get_vehicle_count(self):
        try:
            config_reader = cfg.ConfigParser()
            config_reader.read('config.ini')
            Car_Parking_Capacity = int(config_reader['car']['Parking_Capacity'])
            Total_car_visited = self.Mongodb_obj[self.DB_Name][self.Collection_Name].count_documents({'vehicle_type': 'car'})
            #Total_car_visited = self.Mongodb_obj[self.DB_Name][self.Collection_Name].count_documents({'vehicle_type': 'car', 'entry_datetime': {'$gt': datetime.datetime.utcnow()}})
            Total_car_parked = self.Mongodb_obj[self.DB_Name][self.Collection_Name].count_documents({'vehicle_type': 'car','vehicle_status': 'parked'})
            #Total_car_parked = self.Mongodb_obj[self.DB_Name][self.Collection_Name].count_documents({'vehicle_type': 'car', 'vehicle_status': 'parked', 'entry_datetime': {'$gt': datetime.datetime.utcnow()}})
            Car_slot_available = Car_Parking_Capacity - Total_car_parked
            return Total_car_visited, Car_slot_available
        except Exception as E:
            print('Exception while fetching vehicle count :\nReason :',E)

    def close_connection(self):
        try:
            self.Mongodb_obj.close()
            print("DB Close Successfully")
        except Exception as E:
            print("Failure while closing db \nReason :" , E)

if __name__ == '__main__':
    db_object = Smartparkdb()
    print(db_object)
    #db_object.insert_in_db(datetime.datetime.now(),'car','GJ03TY2033','parked')
    # entry_log = db_object.vehicle_entry_check("GJ03TY2033")
    # print(entry_log)
    #print(db_object.record_count())
    count_1 ,count_2 = db_object.get_vehicle_count()
    db_object.close_connection()
    #print(db_object.update_exit_in_db(vehicle_type=entry_log['vehicle_type'], vehicle_number=entry_log['vehicle_number'], exit_datetime= datetime.datetime.now(),parked_id=entry_log['_id'],entry_datetime=entry_log['entry_datetime']))
