import os
from pathlib import Path

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient


# Charger les variables du fichier .env
load_dotenv()

# Récupérer la connection string
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Nom du container Azure
container_name = "storage"

# Dossier contenant les fichiers CSV
data_dir = Path("data")

# Liste des fichiers CSV à uploader
files = [
    data_dir / "features.csv",
    data_dir / "stores.csv",
    data_dir / "train.csv"
]

# Connexion au Storage Account
blob_service_client = BlobServiceClient.from_connection_string(
    connection_string
)

# Connexion au container "raw"
container_client = blob_service_client.get_container_client(
    container_name
)

# Upload des fichiers
for file_path in files:

    with open(file_path, "rb") as data:
        container_client.upload_blob(
            name=file_path.name,
            data=data,
            overwrite=True
        )

    print(f"{file_path.name} uploaded successfully")