import os
import torch
from data_loader.ocid_loader import OcidDataset
from modular_configs.configs.dataet_config import get_dataset_config

def test_load_item():
    config = get_dataset_config()

    # Setze die erforderlichen Verzeichnisse
    split_file_path = config.split_file_path  # Pfad zum Split-Datei
    root_path = config.root_path  # Pfad zum Root-Verzeichnis des Datasets
    split_name = 'training_0'  # oder 'val', 'test' je nach deinem Split
    transform = None  # oder deine Transformationsfunktion

    # Erstelle die Dataset-Instanz
    dataset = OcidDataset(split_file_path, root_path, split_name, transform)

    # Teste den ersten Eintrag im Dataset
    item = dataset._image_paths[0]  # Nimm den ersten Eintrag
    img, msk, boxes = dataset._load_item(item)

    # Überprüfe die Dimensionen und Typen der geladenen Daten
    assert img is not None, "Bild wurde nicht geladen."
    assert msk is not None, "Maske wurde nicht geladen."
    assert boxes is not None, "Bounding Boxes wurden nicht geladen."
    
    # Überprüfe die Form des Bildes (z.B. C x H x W)
    assert img.ndimension() == 3, "Bild hat nicht die erwartete Dimension (C, H, W)."
    
    # Überprüfe die Form der Maske (z.B. C x H x W oder H x W)
    assert msk.ndimension() in [2, 3], "Maske hat nicht die erwartete Dimension (H, W) oder (C, H, W)."
    
    # Überprüfe die Form der Bounding Boxes (z.B. Anzahl der Kästen, 4 Punkte)
    print(boxes.shape)
    assert boxes.shape[1] == 4, "Jede Bounding Box sollte 4 Punkte haben."

    # Optional: Drucke die geladenen Daten zur Überprüfung
    print(f'Bild: {img.shape}, Maske: {msk.shape}, Bounding Boxes: {boxes}')

# Führe den Test durch
test_load_item()