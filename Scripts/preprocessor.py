# import libraries
import numpy as np
import pandas as pd
import config
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

class Pipeline:
    def __init__(self,
                 target,
                 leaky_features,
                 data_type_conversion,
                 high_low_cardinality,
                 cat_imp_mode,
                 cat_imp_missing,
                 cat_vars,
                 temp_vars,
                 cont_vars,
                 test_size=0.1,
                 random_state=42):
        # initializing attributes
        self.target = target
        self.leaky_features = leaky_features
        self.data_type_conversion = data_type_conversion
        self.high_low_cardinality = high_low_cardinality
        self.cat_imp_mode = cat_imp_mode
        self.cat_imp_missing = cat_imp_missing
        self.cat_vars = cat_vars
        self.temp_vars = temp_vars
        self.cont_vars = cont_vars
        self.test_size = test_size
        self.random_state = random_state
        
        
        # initializing data splits
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        
        # learned parameters
        self.imp_cat_mode_dict = {}
        self.frequent_labels_dict = {}
        self.encoding_dict = {}
        self.temp_mode_dict = {}
        
        # let's initialize models
        self.scaler = StandardScaler()
        self.lr_model = LinearRegression()
        
    #=================================================================================  
    # Data Cleaning
    
    # function to drop leaky features
    def drop_leaky(self, data):
        data = data.drop(self.leaky_features, axis=1)
        return data
    
    # function to perform data type conversion
    def data_conversion(self, data):
        for var in self.data_type_conversion:
            data[var] = np.where(data[var] == "tbd", np.nan, data[var])
            data[var] = data[var].astype("float")
        return data
            
    # function to drop low and high cardinality features
    def drop_HL_cardinality(self, data):
        data = data.drop(self.high_low_cardinality, axis=1)
        return data
        
    # ===============================================================================
    # Funcions to perform Feature Engineering
    
    # impute categorical values with the mode
    # first let's learn the mode from the training set
    def learn_cat_mode(self, data, features):
        for var in features:
            mode = data[var].mode()[0]
            self.imp_cat_mode_dict[var] = mode
    # now let's define a function to imput the model
    def impute_cat_mode(self, data, feature):
        for var in feature:
            data[var] = data[var].fillna(self.imp_cat_mode_dict[var])
        return data
            
    # impute categorical values with the string "Missing"
    def impute_cat_missing(self, data, features):
        for var in features:
            data[var] = data[var].fillna("Missing")
        return data
            
            
            
    # Removing Rare Labels
    
    # first let's define a function to capture frequent lebels
    
    def find_frequent(self, data, percent, features):
        
        for var in features:
            # first let's assing a temporary series
            temp = data.groupby(var)[var].count()/len(data)
            self.frequent_labels_dict[var] = temp[temp > percent].index

    # now let's remove this rare labels
    def remove_rare(self, data, features):
        for var in features:
            data[var] = np.where(data[var].isin(self.frequent_labels_dict[var]), data[var], "Rare")
        return data
    
    
    # Encoding Categorical Variables
    # first let's define a function to learn encoding dictionaries for each variables
    def learn_encoding(self, data, target, features):
        # first let's order the labels
        for var in features:
            temp = pd.concat([data, target], axis=1)
            ordered_label = temp.groupby(var)[self.target].mean().sort_values().index
            self.encoding_dict[var] = {k:i for i, k in enumerate(ordered_label)}
            
    # now we can use the features learned to encode our data
    def encode_cat(self, data, features):
        for var in features:
            data[var] = data[var].map(self.encoding_dict[var])
        return data
     
     
    #Filling Temporal Variables with mode
    # first let's define a function to learn the mode
    def learn_temp_mode(self,data,features):
        for var in features:
            self.temp_mode_dict[var] = data[var].mode()[0]
            
    
    # now let's define a function to impute mode
    def impute_temp_mode(self, data, feature):
        for var in feature:
            data[var] = data[var].fillna(self.temp_mode_dict[var])
        return data
        
        
    # Extract the Age information
    def extract_age(self,data,feature):
        for var in feature:
            today = datetime.today()
            year = today.year
            data["Age"] = year - data[var]
        return data
            
    # Drop the Temp Vars
    def drop_temp(self,data,feature):
        data = data.drop(feature, axis =1)
        return data
        
    # Drop continous variables with more than 50% missing values
    def drop_cont(self, data, feature):
        data = data.drop(feature, axis = 1)
        return data

        
    #=================================================================================
    # Master Function to Synchronize data preprocessing
    
    def fit(self, data):
        
        # Dropping Leaky Features
        data = self.drop_leaky(data)
        
        # Performing Data type conversion
        data = self.data_conversion(data)
        
        # let's drop high and low cardinality features
        data = self.drop_HL_cardinality(data)
        
        # split the data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data.drop(self.target, axis=1),
                                                            data[self.target],
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)
        
        # imputing categorical missing values with mode
        # first learn the mode
        self.learn_cat_mode(self.x_train, self.cat_imp_mode)
        
        # now apply the mode
        self.x_train = self.impute_cat_mode(self.x_train, self.cat_imp_mode)
        self.x_test = self.impute_cat_mode(self.x_test, self.cat_imp_mode)
        
        # filling categorical missing values with the string missing
        self.x_train = self.impute_cat_missing(self.x_train, self.cat_imp_missing)
        self.x_test = self.impute_cat_missing(self.x_test, self.cat_imp_missing)
        
        # now let's learn the frequent lables using the train set
        self.find_frequent(self.x_train, 0.01, self.cat_vars)
        
        # now let's remove rare labels on both training and testing set
        self.x_train = self.remove_rare(self.x_train, self.cat_vars)
        self.x_test = self.remove_rare(self.x_test, self.cat_vars)
        
        
        # Encoding Categorical Variables
        #first let's learn the encoding
        self.learn_encoding(self.x_train, self.y_train, self.cat_vars)
        
        # now let's apply the encoder
        self.x_train = self.encode_cat(self.x_train, self.cat_vars)
        self.x_test = self.encode_cat(self.x_test, self.cat_vars)
        
        #Imputing Temporal Variables
        #first let's impute the mode
        self.learn_temp_mode(self.x_train, self.temp_vars)
        
        # now let's impute the temp mode
        self.x_train = self.impute_temp_mode(self.x_train, self.temp_vars)
        self.x_test = self.impute_temp_mode(self.x_test, self.temp_vars)
        
        # now let's extract the "Age" INforamtion
        self.x_train = self.extract_age(self.x_train, self.temp_vars)
        self.x_test = self.extract_age(self.x_test, self.temp_vars)
        
        # now let's drop the temp columns
        self.x_train = self.drop_temp(self.x_train, self.temp_vars)
        self.x_test = self.drop_temp(self.x_test, self.temp_vars)
        
        # now let's apply the drop cont function to remvoe continous variables
        # with more than 50% missing values
        self.x_train = self.drop_cont(self.x_train, self.cont_vars)
        self.x_test = self.drop_cont(self.x_test, self.cont_vars)
        
        # now let's perform feature scalling
        # first fit the scaler with the taining set
        self.scaler.fit(self.x_train)
        
        # now transform both the triaing and test sets with the scaler
        self.x_train =pd.DataFrame(self.scaler.transform(self.x_train), columns=self.x_train.columns)
        self.x_test = pd.DataFrame(self.scaler.transform(self.x_test), columns=self.x_test.columns)
        
        # trian the linear regression model
        self.lr_model.fit(self.x_train,self.y_train)
        
        print("Finished Training Successfully")
        
    def transform(self,data):
        
        # Dropping Leaky Features
        data = self.drop_leaky(data)
        
        # Performing Data type conversion
        data = self.data_conversion(data)
        
        # let's drop high and low cardinality features
        data = self.drop_HL_cardinality(data)
        
        # imputing categorical missing values with mode
        data = self.impute_cat_mode(data,self.cat_imp_mode)
        
        # filling categorical missing values with the string missing
        data = self.impute_cat_missing(data, self.cat_imp_missing)
        
        # now let's remove rare labels on both training and testing set
        data = self.remove_rare(data,self.cat_vars)
            
        # Encoding Categorical Variables
        data = self.encode_cat(data,self.cat_vars)
        
        #Imputing Temporal Variables
        data = self.impute_temp_mode(data,self.temp_vars)
        
        # now let's extract the "Age" INforamtion
        data = self.extract_age(data,self.temp_vars)
        
        # now let's drop the temp columns
        data = self.drop_temp(data,self.temp_vars)
        
        # now let's apply the drop cont function to remvoe continous variables
        # with more than 50% missing values
        data = self.drop_cont(data, self.cont_vars)
        
        # now transform scaler transform both the triaing and test sets.
        data = pd.DataFrame(self.scaler.transform(data), columns= data.columns)
        
        return data
    
    def predict(self, data):
        
        #get the data
        data = self.transform(data)
        # perform prediction
        prediction = self.lr_model.predict(data)
        return prediction
    
    def evaluate(self):
        
        # first get the triaining and test predictions
        train_pred = self.lr_model.predict(self.x_train)
        test_pred = self.lr_model.predict(self.x_test)
        
        # perform model  evaluation
        train_mse = mean_absolute_error(self.y_train, train_pred)
        test_mse = mean_absolute_error(self.y_test, test_pred)
        
        print(f"trian mean_absolute_erroris :__{train_mse}")
        print(f"test mean absolute error is__{test_mse}")
                
        