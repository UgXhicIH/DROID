import pandas as pd
import glob

fichiers_xlsx = glob.glob('Train_*.xlsx')
if not fichiers_xlsx:
    print("Aucun fichier dans le répertoire.")
else:
    print(f"{len(fichiers_xlsx)} fichiers trouvés\n")
    for fichier in fichiers_xlsx:
        print(f"---{fichier}---")
        
        df = pd.read_excel(fichier)
        print(f"Taille du dataset: {df.shape}")
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df_shuffled.to_excel(fichier, index=False)
        print(f"Dataset mélangé, sauvegardé: {fichier}\n")
    print("Fichiers mélangés")