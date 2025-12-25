import pandas as pd
import yfinance as yf
from pathlib import Path

PROJECT_ROOT = Path("/home/onyxia/work/Gestion-portefeuille/")

# 1. TAUX BCE (Main Refinancing Operations)
print("📥 Téléchargement du taux BCE...")
# Le taux BCE n'est pas directement sur yfinance, on utilise l'Euribor 3M comme proxy
euribor = yf.download("EURIBOR3MD.HA", start="2015-01-01", end="2024-12-31")
euribor = euribor[['Close']].rename(columns={'Close': 'ECB_Rate'})
euribor = euribor.reset_index()
euribor['Date'] = pd.to_datetime(euribor['Date'])
euribor.to_csv(PROJECT_ROOT / "data/raw/ECB_Rate.csv", index=False)
print(f"✅ Taux BCE sauvegardé : {len(euribor)} jours")

# 2. V2X (Volatilité européenne)
print("\n📥 Téléchargement du V2X...")
v2x = yf.download("^V2X", start="2015-01-01", end="2024-12-31")
v2x = v2x[['Close']].rename(columns={'Close': 'V2X'})
v2x = v2x.reset_index()
v2x['Date'] = pd.to_datetime(v2x['Date'])
v2x.to_csv(PROJECT_ROOT / "data/raw/V2X.csv", index=False)
print(f"✅ V2X sauvegardé : {len(v2x)} jours")
