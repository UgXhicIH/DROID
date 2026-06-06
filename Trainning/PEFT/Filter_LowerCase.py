# Copyright (c) 2026 Morgan Letoux. All rights reserved.
# This file is part of PROTEOGEN/DROID.
# Unauthorized use, reproduction or distribution is strictly prohibited.
# See LICENSE for details.
import pandas as pd
import glob
print ("Filtre pour : _ * ' - ~ . , ? + chiffres et miniscules")
fichiers_xlsx = glob.glob('Train_*.xlsx')
if not fichiers_xlsx:
    print("Aucun fichier dans le répertoire.")
else:
     for fichier in fichiers_xlsx:
        df = pd.read_excel(fichier)
        print(f"Traitement du fichier : {fichier}")
        lowercase_rows = df[df['sequence'] != df['sequence'].str.upper()]
        if not lowercase_rows.empty:
            print("Lettre miniscule trouvés dans ces lignes:")
            print(lowercase_rows)
        asterisk_rows = df[df['sequence'].str.contains('*', regex=False, na=False)]
        if not asterisk_rows.empty:
            print("\nAstérisque trouvés dans ces lignes:")
            print(asterisk_rows)
        space_rows = df[df['sequence'].str.contains(' ', regex=False, na=False)]
        if not space_rows.empty:
            print("\nEspaces trouvés dans ces lignes:")
            print(space_rows)
        dash_rows = df[df['sequence'].str.contains('-', regex=True, na=False)]
        if not dash_rows.empty:
            print("\nTirets trouvés dans ces lignes:")
            print(dash_rows)
        wave_rows = df[df['sequence'].str.contains('~', regex=True, na=False)]
        if not wave_rows.empty:
            print("\nTildes trouvés dans ces lignes:")
            print(wave_rows)
        point_rows = df[df['sequence'].str.contains('.', regex=False, na=False)]
        if not point_rows.empty:
            print("\nPoints trouvés dans ces lignes:")
            print(point_rows)
        commas_rows = df[df['sequence'].str.contains(',', regex=False, na=False)]
        if not commas_rows.empty:
            print("\nVirgules trouvés dans ces lignes:")
            print(commas_rows)
        underscore_rows = df[df['sequence'].str.contains('_', regex=False, na=False)]
        if not underscore_rows.empty:
            print("\nSoulignements trouvés dans ces lignes:")
            print(underscore_rows)
        interogations_rows = df[df['sequence'].str.contains('?', regex=False, na=False)]
        if not interogations_rows.empty:
            print("\nInterrogation trouvés dans ces lignes:")
            print(interogations_rows)
        quote_rows = df[df['sequence'].str.contains('"', regex=False, na=False)]
        if not quote_rows.empty:
            print("\nGuillemets trouvés dans ces lignes:")
            print(quote_rows)
        asterisk_rows = df[df['sequence'].str.contains('*', regex=False, na=False)]
        if not asterisk_rows.empty:
            print("\nAstérisque trouvés dans ces lignes:")
            print(asterisk_rows)
        number_rows = df[df['sequence'].str.contains('1/2/3/4/5/6/7/8/9/0', regex=True, na=False)]
        if not number_rows.empty:
            print("\nNombres trouvés dans ces lignes:")
            print(number_rows)
        parenthesis_rows = df[df['sequence'].str.contains('(/)', regex=False, na=False) | df['sequence'].str.contains(')', regex=False, na=False)]
        if not parenthesis_rows.empty:
            print("\nParenthèses trouvés dans ces lignes:")
            print(parenthesis_rows)
        slash_rows = df[df['sequence'].str.contains('/', regex=False, na=False)]
        if not slash_rows.empty:
            print("\nSlash trouvés dans ces lignes:")
            print(slash_rows)
print("Fin")
    