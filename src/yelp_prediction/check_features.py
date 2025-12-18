import torch
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


FILE_PATH = BASE_DIR / "data" / "features-clip.pt"

print(f"--- DEBUG ---")
print(f"Lo script si trova in: {BASE_DIR}")
print(f"Sto cercando il file in: {FILE_PATH}")
print(f"--------------\n")

if FILE_PATH.exists():
    print("File trovato! Caricamento in corso...")
    features = torch.load(FILE_PATH, map_location="cpu", mmap=True, weights_only=False)
    
    print(f"Successo!")
    print(f"Numero totale di foto processate: {len(features)}")
    
    primo_id = list(features.keys())[0]
    print(f"Esempio - ID Foto: {primo_id}")
    print(f"Esempio - Dimensione Feature: {features[primo_id].shape}")
else:
    print("ERRORE: Il file non esiste nel percorso indicato.")
    print("Verifica che:")
    print(f"1. Esista una cartella chiamata 'data' dentro {BASE_DIR}")
    print(f"2. Il file si chiami esattamente 'features-clip.pt' (attenzione alle estensioni doppie tipo .pt.pt)")


test_bid = "il-tuo-business-id-qui"
print(features.get(test_bid))



