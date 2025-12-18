import polars as pl
import torch
import sys

def check_missing_photos(df, features_dict):
    all_bids = df["business_id"].to_list()
    existing_bids = set(features_dict.keys())
    
    missing_in_dict = 0
    for bid in all_bids:
        if bid not in existing_bids:
            missing_in_dict += 1
            
    print(f"\n--- RISULTATI ANALISI ---")
    print(f"Ristoranti totali analizzati: {len(all_bids)}")
    print(f"Ristoranti SENZA feature (mancanti): {missing_in_dict}")
    print(f"Ristoranti CON feature: {len(all_bids) - missing_in_dict}")
    
    perc_missing = (missing_in_dict / len(all_bids)) * 100
    print(f"Percentuale dati 'vuoti': {perc_missing:.2f}%")
    print(f"---------------------------\n")

if __name__ == "__main__":
    PATH_JSON = "data/yelp_academic_dataset_business.json"
    PATH_FEATURES = "data/features_grouped.pt"

    try:
        print(f"Caricamento JSON (formato ndjson)...")
        df = pl.read_ndjson(PATH_JSON)
        

        df = df.filter(pl.col("categories").str.contains("Restaurants"))

        print(f"Caricamento Feature Grouped...")
        features_dict = torch.load(PATH_FEATURES)

        check_missing_photos(df, features_dict)
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")