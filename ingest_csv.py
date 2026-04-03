import pandas as pd
import os
import glob
import re
from database import SessionLocal, engine, Base
from models import FoodItem

# Create/Reset Tables
Base.metadata.create_all(bind=engine)

def clean_value(val):
    """Extracts a float from strings like '143.5 g' or '10.93'."""
    if pd.isna(val) or val == 0:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    
    # Regex to find the first number (int or decimal)
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
    return float(match.group()) if match else 0.0

def load_all_csvs_to_db():
    db = SessionLocal()
    print("🧹 Cleaning old data...")
    db.query(FoodItem).delete() 
    db.commit()

    csv_files = glob.glob("./data/*.csv")
    total_count = 0

    for file in csv_files:
        filename = os.path.basename(file)
        print(f"📄 Processing {filename}...")
        df = pd.read_csv(file)
        
        
        # Fill missing numbers with 0, and missing text with an empty string
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("")
                
        headers = df.columns.tolist()

        file_count = 0
        for _, row in df.iterrows():
            try:
                # --- CASE 1: FAST FOOD CSV (Company, Product, Energy...) ---
                if 'Company' in headers and 'Product' in headers:
                    # Combine Company + Product for better searchability
                    name = f"{row['Company']} {row['Product']}".lower()
                    serving = str(row.get('Per Serve', '1 serving'))
                    # Note: Headers might be truncated in Excel, check exact spelling
                    # Using .get() with the likely header names
                    calories = clean_value(row.get('Energy (kcal)', row.get('Energy (kCal)', 0)))
                    carbs = clean_value(row.get('Carbohydrates (g)', 0))
                    protein = clean_value(row.get('Protein (g)', 0))
                    fat = clean_value(row.get('Total Fat (g)', 0))

                # --- CASE 2: INDIAN DIET CSV (Food, Calories...) ---
                elif 'Food' in headers:
                    name = str(row['Food']).lower()
                    serving = "100g/standard" # Assuming all items are per 100g or standard serving
                    calories = clean_value(row.get('Calories', 0))
                    carbs = clean_value(row.get('Carbs', 0))
                    protein = clean_value(row.get('Protein', 0))
                    fat = clean_value(row.get('Fat', 0))

                else:
                    print(f"⚠️ Header mismatch in {filename}. Skipping row.")
                    continue

                food = FoodItem(
                    item_name=name,
                    serving_size=serving,
                    calories=calories,
                    carbs=carbs,
                    protein=protein,
                    fat=fat
                )
                db.add(food)
                file_count += 1
                total_count += 1

            except Exception as e:
                print(f"❌ Error in row: {e}")
                continue
        
        db.commit()
        print(f"   ↳ Added {file_count} items from {filename}")

    print(f"\n✅ SUCCESS! {total_count} items are now in the AI's memory.")
    db.close()

if __name__ == "__main__":
    load_all_csvs_to_db()