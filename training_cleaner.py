import pandas as pd

# --- CONFIGURAZIONE NOMI FILE ---
file_train = "pxr-challenge_TRAIN.csv"
file_counter = "pxr-challenge_counter-assay_TRAIN.csv"
file_single = "pxr-challenge_single_concentration_TRAIN.csv"
file_output = "training_final_fixed.csv"

# Nomi colonne sensibili
col_err_std = 'pEC50_std.error (-log10(molarity))'

print("⏳ Caricamento dei dataset...")
df_main = pd.read_csv(file_train)
df_counter = pd.read_csv(file_counter)
df_single = pd.read_csv(file_single)

# 1. NORMALIZZAZIONE DEI NOMI (Cruciale per i duplicati e i merge)
# Rimuoviamo spazi vuoti invisibili e mettiamo tutto in maiuscolo
print("🧹 Normalizzazione dei nomi delle molecole...")
for df in [df_main, df_counter, df_single]:
    df['Molecule Name'] = df['Molecule Name'].astype(str).str.strip().str.upper()

# 2. INCROCIO DEI DATI (Merge)
# Prendiamo solo quello che ci serve dagli altri saggi per usarli come filtri
df_counter_sub = df_counter[['Molecule Name', 'pEC50']].rename(columns={'pEC50': 'pEC50_counter'})
df_merged = pd.merge(df_main, df_counter_sub, on='Molecule Name', how='left')

df_single_sub = df_single[['Molecule Name', 'fdr_bh']]
df_merged = pd.merge(df_merged, df_single_sub, on='Molecule Name', how='left')

iniziali = len(df_merged)

# 3. FILTRI DI QUALITÀ BIOLOGICA E STATISTICA
# Filtro A: Errore sperimentale troppo alto (Teniamo l'85% dei dati migliori)
if col_err_std in df_merged.columns:
    soglia_errore = df_merged[col_err_std].quantile(0.85)
    df_pulito = df_merged[df_merged[col_err_std] <= soglia_errore]
    print(f"📉 Filtro Errore Sperimentale: rimosse {iniziali - len(df_pulito)} misurazioni incerte.")
else:
    df_pulito = df_merged.copy()

# Filtro B: Falsi Positivi Statistici (FDR)
pre_fdr = len(df_pulito)
# Teniamo molecole con FDR < 0.05 o quelle che non hanno questo dato (NaN)
df_pulito = df_pulito[(df_pulito['fdr_bh'] < 0.05) | (df_pulito['fdr_bh'].isna())]
print(f"📉 Filtro FDR (Single-Concentration): rimosse {pre_fdr - len(df_pulito)} molecole.")

# Filtro C: Promiscuità (Counter-Assay)
pre_count = len(df_pulito)
# Scartiamo molecole troppo attive nel counter-assay (es. pEC50_counter >= 5.0)
df_pulito = df_pulito[(df_pulito['pEC50_counter'] < 5.0) | (df_pulito['pEC50_counter'].isna())]
print(f"📉 Filtro Specificità (Counter-Assay): rimosse {pre_count - len(df_pulito)} molecole.")

# 4. RIMOZIONE DUPLICATI INTELLIGENTE
print("🔍 Ricerca e rimozione dei duplicati...")
# Ordiniamo prima per Nome, e poi per Errore Standard (dal più piccolo al più grande)
# In questo modo, il primo duplicato che incontra è quello misurato in modo più preciso
if col_err_std in df_pulito.columns:
    df_pulito = df_pulito.sort_values(by=['Molecule Name', col_err_std], ascending=[True, True])

# Rimuoviamo tenendo solo la prima riga (la migliore)
df_pulito = df_pulito.drop_duplicates(subset=['Molecule Name'], keep='first')

# 5. SALVATAGGIO
# Rimuoviamo le colonne filtro per non confondere il calcolo dei descrittori chimici in futuro
df_pulito = df_pulito.drop(columns=['pEC50_counter', 'fdr_bh'])

df_pulito.to_csv(file_output, index=False)
print(f"\n✅ Operazione conclusa! Dataset perfetto con {len(df_pulito)} molecole uniche salvato in '{file_output}'.")