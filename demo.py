import pandas as pd
from sqlalchemy import create_engine

# Database connection setup
db_user = 'root'
db_password = 'password123'
db_host = 'localhost'
db_name = 'smart_shopping'

# Create the engine
engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}')
connection = engine.connect()

# Read the TSV file
df = pd.read_csv('walmart_dataset.tsv', sep='\t')

# Handle missing values
df.fillna({'Product_Rating': 0, 'Product_Reviews_Count': 0}, inplace=True)

try:
    # Insert data into the 'products' table
    df.to_sql('products', con=engine, if_exists='append', index=False)
    print("Data inserted successfully!")

except Exception as e:
    print(f"Error inserting data: {e}")
    




