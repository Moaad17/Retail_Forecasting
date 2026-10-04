import os
from io import BytesIO

import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

blob_service_client = BlobServiceClient.from_connection_string(
    connection_string
)

container_client = blob_service_client.get_container_client("storage")

def load_csv(blob_name):
    blob_client = container_client.get_blob_client(blob_name)

    data = blob_client.download_blob().readall()

    return pd.read_csv(BytesIO(data))

train = load_csv("train.csv")
stores = load_csv("stores.csv")
features = load_csv("features.csv")


df = train.merge(
    stores,
    on="Store",
    how="left"
)

df = df.merge(
    features,
    on=["Store", "Date"],
    how="left"
)

markdown_cols = [
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5"
]



df['Date'] = pd.to_datetime(df['Date'])



mask = df["Date"] < pd.Timestamp("2011-11-01")

df.loc[mask, markdown_cols] = (
    df.loc[mask, markdown_cols].fillna(0)
)

df["week"] = df["Date"].dt.isocalendar().week.astype(int)

df["month"] = df["Date"].dt.month


df = df.sort_values(["Store", "Date"])

df["next_week_holiday"] = (
    df.groupby("Store")["IsHoliday_x"]
      .shift(-1)
      .fillna(False)
      .astype(int)
)







df["week_before_holiday"] = df["next_week_holiday"]


df = df.sort_values(["Store", "Dept", "Date"])


df["Weekly_Sales_lag_1"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
      .shift(1)
)

df["Weekly_Sales_lag_52"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
      .shift(52)
)


df["rolling_mean_4"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
      .transform(lambda x: x.shift(1).rolling(4).mean())
)

df = pd.get_dummies(
    df,
    columns=["Type"],
    dtype=int
)

df.to_parquet(
    "data//cleaned//processed.parquet",
    index=False
)

processed_client = (
    blob_service_client
    .get_container_client('processed')
)

with open("data//cleaned//processed.parquet", "rb") as data:

    processed_client.upload_blob(
        name="processed.parquet",
        data=data,
        overwrite=True
    )


print("Preprocessing completed successfully.")
print(f"Final shape: {df.shape}")