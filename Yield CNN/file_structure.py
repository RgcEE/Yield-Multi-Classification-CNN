import pickle
from pathlib import Path

pkl_path = Path("Dataset/LSWMD.pkl")
with pkl_path.open("rb") as f:
    obj = pickle.load(f)

print(type(obj))

if hasattr(obj, "columns"):
    print("DataFrame columns:", obj.columns.tolist())
    print("Shape:", obj.shape)
    print("\nFirst row sample:")
    print(obj.iloc[0])
    print("\nfailureType unique values:", obj["failureType"].unique() if "failureType" in obj.columns else "column not found")

elif isinstance(obj, dict):
    print("Dict keys:", list(obj.keys()))