DATA_PATH = "C:\\Users\\yozil\\Desktop\\My projects\\11. Video_Game_Sales_Prediction\\data\\raw data\\video_games_sales.csv"

TARGET = "Global_Sales"

# Data Cleaning

LEAKY_FEATURES = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

DATA_TYPE_CONVERSION = ["User_Score"]

HIGH_LOW_CARDINALITY_FEATURES = ['Name', 'Publisher', 'Developer']


# Feature Engineering
CAT_VARS = ['Platform', 'Genre', 'Rating']

CAT_IMP_MODE = ["Genre"]
CAT_IMP_MISSING = ["Rating"]

TEMPORAL_VARIABLES = ["Year_of_Release"]

CONT_VARS = ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']