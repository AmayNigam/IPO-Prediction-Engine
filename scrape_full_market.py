import time
import pandas as pd
from io import StringIO
import undetected_chromedriver as uc

print("==========================================================")
print("   LAUNCHING STEALTH BROWSER FOR IPOMATRIX (2004-2026)    ")
print("==========================================================")

# Initialize the undetected Chrome browser using your exact version
options = uc.ChromeOptions()
options.add_argument("--start-maximized")
driver = uc.Chrome(options=options, version_main=154)

url = "https://www.ipomatrix.com/report/85/all"

try:
    print(f"Navigating to {url}...")
    driver.get(url)
    
    # IPOMatrix is a React App. We must wait generously for their API to fetch 20 years of data.
    print("Waiting 15 seconds for JavaScript and data tables to fully render...")
    time.sleep(15) 
    
    # Get the raw HTML after React has populated the DOM
    page_html = driver.page_source
    
    # Parse all tables using StringIO
    tables = pd.read_html(StringIO(page_html))
    
    if len(tables) > 0:
        # Sort tables by length to guarantee we grab the massive main data table
        df_full = max(tables, key=len)
        
        output_name = "IPOMATRIX_2004_2026.csv"
        df_full.to_csv(output_name, index=False)
        
        print("\n==========================================================")
        print(f" [BINGO] Successfully extracted {len(df_full)} IPO records from IPOMatrix!")
        print(f" Dataset saved to {output_name}")
        print("==========================================================")
        print(" Columns found:")
        print(list(df_full.columns))
    else:
        print("\n[WARNING] No tables detected. The site might require login or infinite scrolling.")

except Exception as e:
    print(f"\n[ERROR] Something went wrong: {e}")

finally:
    driver.quit()