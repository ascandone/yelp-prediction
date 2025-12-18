import torch
import polars as pl
import os
from train_features_MIL import run_MIL

def check_data():
    # Verifica che i file siano al posto giusto
    paths = [
        "data/features_grouped.pt",
        "data/yelp_academic_dataset_business.json"
    ]
    for p in paths:
        if not os.path.exists(p):
            print(f"❌ Errore: Non trovo il file {p}")
            return False
    return True

if __name__ == "__main__":
    if check_data():
        print("✅ File trovati! Caricamento in corso...")
        
        # 1. Carica le features raggruppate
        features_dict = torch.load("data/features_grouped.pt", weights_only=False)
        
        # 2. Carica i metadati dei business
        yelp_df = pl.read_ndjson("data/yelp_academic_dataset_business.json")
        yelp_df = yelp_df.filter(pl.col("categories").str.contains("Restaurants"))
        print(f"Dati caricati: {len(yelp_df)} potenziali ristoranti nel dataset.")

        # 1. Ottieni la lista dei business_id che hanno effettivamente le foto
        bids_con_foto = set(features_dict.keys())

        # 2. Filtra il DataFrame originale (yelp_df)
        initial_len = len(yelp_df)
        yelp_df = yelp_df.filter(pl.col("business_id").is_in(bids_con_foto))

        print(f"\n--- PULIZIA DATI ---")
        print(f"Ristoranti (categoria 'Restaurants'): {initial_len}")
        print(f"Ristoranti rimossi (perché senza foto): {initial_len - len(yelp_df)}")
        print(f"Ristoranti finali per il training: {len(yelp_df)}")
        print(f"--------------------\n")

        import sys
        sys.stdout.flush()
        
        # 3. Avvia il training
        # Qui passiamo i dati alla tua funzione run
        run_MIL(
            features_dict=features_dict,
            yelp_df=yelp_df,
            input_dim=512, # Dimensione CLIP
            epochs=15, 
            max_photos = 10     
        )