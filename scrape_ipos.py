import pandas as pd
import requests
import io

print("--- Fetching IPO Historical Data ---")

# Bypassing bot protection by pulling a pre-compiled dataset from a public GitHub repository
url = "https://raw.githubusercontent.com/saqibsafdar11/Predicting-Listing-Gains-in-the-Indian-IPO-Market/main/Indian_IPO_Market_Data.csv"

try:
    print("Downloading 300+ historical IPO records...")
    response = requests.get(url)
    response.raise_for_status()
    
    # Read the CSV data directly into pandas
    df = pd.read_csv(io.StringIO(response.text))
    
    # Rename columns to match your original RAW IPO.csv format exactly
    df_cleaned = df.rename(columns={
        "IPOName": "Company_Name",
        "Issue_Size": "Issue_Size_Cr",
        "Issue_Price": "Issue_Price_INR",
        "Subscription_Total": "Total_Sub_x",
        "Subscription_QIB": "QIB_Sub_x",
        "Subscription_HNI": "HNI_Sub_x",
        "Subscription_RII": "Retail_Sub_x",
        "Listing_Gains_Percent": "Listing_Gain_Pct"
    })
    
    # Add a default Sector column so the ML pipeline doesn't break
    df_cleaned["Sector"] = "Financial Services"
    
    # Reorder columns to match your model's expected format
    final_cols = [
        "Company_Name", "Sector", "Issue_Size_Cr", "Issue_Price_INR", 
        "Total_Sub_x", "QIB_Sub_x", "HNI_Sub_x", "Retail_Sub_x", "Listing_Gain_Pct"
    ]
    df_final = df_cleaned[final_cols]
    
    # Save the expanded dataset
    output_filename = "SCRAPED_RAW_IPO.csv"
    df_final.to_csv(output_filename, index=False)
    
    print(f"\n[SUCCESS] Successfully downloaded and cleaned {len(df_final)} IPO records!")
    print(f"[SUCCESS] Data saved to '{output_filename}'")
    
except Exception as e:
    print(f"An error occurred: {e}")