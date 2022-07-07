# %%
import pandas as pd
from datetime import timedelta

orders = pd.read_csv("C:\\Users\\2702b\\OneDrive - Asia Pacific University\\Diploma\\Semester 4\\Introducton of Data Analytics\\Assignment\\Datasets\\olist_orders_dataset.csv")

orders = orders.drop(["order_approved_at", "order_delivered_carrier_date", "order_estimated_delivery_date"], axis=1)

orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], format="%Y-%m-%d")
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"], format="%Y-%m-%d")
# %%
orders = orders[orders["order_status"] == "delivered"]
#%%
orders["order_delivered_customer_date"] = orders["order_delivered_customer_date"].fillna(orders["order_delivered_customer_date"].mean())

orders["Delivery_duration"] = orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
orders[orders["Delivery_duration"] < timedelta(days=0)]
# %%
orders = orders.drop(orders[orders["Delivery_duration"] < timedelta(days=0)].index)

orders["Delivery_duration"] = orders["Delivery_duration"].dt.days

# %%
