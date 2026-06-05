import subprocess
import glob
import sys
import os
import logging
os.makedirs("Logs", exist_ok=True)
fichier_log = f"Logs/Run_Token.log"

logging.basicConfig(
    level=logging.INFO, # Niveau de détail minimum à capturer
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(fichier_log, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

scripts = glob.glob("DoRa_Train_*_650M.py")

logging.info(f"{len(scripts)} scripts trouvés. Début de l'exécution séquentielle...")
python_exe = sys.executable

for script in scripts:
    logging.info(f"\n--- Lancement de : {script} ---")
    subprocess.run([python_exe, script])

logging.info("\nTous les entraînements sont terminés !")