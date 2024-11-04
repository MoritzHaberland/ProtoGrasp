import numpy as np
import math
from data_loader.ocid_loader import OcidDataset
from modular_configs.configs.dataet_config import get_dataset_config


import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
import time

def test_load_item():
    config = get_dataset_config()

    # Setze die erforderlichen Verzeichnisse
    split_file_path = config.split_file_path  # Pfad zum Split-Datei
    root_path = config.root_path  # Pfad zum Root-Verzeichnis des Datasets
    split_name = 'training_0'  # oder 'val', 'test' je nach deinem Split
    transform = None  # oder deine Transformationsfunktion

    # Erstelle die Dataset-Instanz
    dataset = OcidDataset(split_file_path, root_path, split_name, transform, 18)

    # Teste den ersten Eintrag im Dataset
    item = dataset.__getitem__(0)
    img = item["image"]
    msk = item["mask"]
    boxes = item["boxes_rotated_corners"]
    boxes_unrotated = item["boxes_unrotated"]
    angles = item["angles"]

    # Überprüfe die Dimensionen und Typen der geladenen Daten
    assert img is not None, "Bild wurde nicht geladen."
    assert msk is not None, "Maske wurde nicht geladen."
    assert boxes is not None, "Bounding Boxes wurden nicht geladen."
    
    # Überprüfe die Form des Bildes (z.B. C x H x W)
    assert img.ndimension() == 3, "Bild hat nicht die erwartete Dimension (C, H, W)."
    
    # Überprüfe die Form der Maske (z.B. C x H x W oder H x W)
    assert msk.ndimension() in [2, 3], "Maske hat nicht die erwartete Dimension (H, W) oder (C, H, W)."
    
    # Überprüfe die Form der Bounding Boxes (z.B. Anzahl der Kästen, 4 Punkte)
    #print(boxes.shape)
    assert boxes.shape[1] == 4, "Jede Bounding Box sollte 4 Punkte haben."
    print(img.shape)

    # Optional: Drucke die geladenen Daten zur Überprüfung
    #print(f'Bild: {img.shape}, Maske: {msk.shape}, Bounding Boxes: {boxes}')

    return img, boxes, msk, boxes_unrotated, angles

def visualize_image(img, boxes, boxes_unrotated, angles):
    #plt.figure(figsize=(12, 9))
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())  # Ändere von CxHxW zu HxWxC für die Darstellung
    # Füge ein Rechteck mit den angegebenen Eckpunkten hinzu
    # Füge ein Rechteck mit den angegebenen Eckpunkten hinzu
    box = boxes[10]
    points = box.numpy()
    polygon = patches.Polygon(points, closed=True, linewidth=1, edgecolor='r', facecolor='none')
    ax = plt.gca()
    ax.add_patch(polygon)
    

    box = boxes_unrotated[10]
    xmin, ymin, xmax, ymax = box.numpy() 
    points = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
    polygon = patches.Polygon(points, closed=True, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(polygon)
    
    # unrotate)
    angle_rad = math.radians(angles[10][1])
    rotation_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
                                 [math.sin(angle_rad), math.cos(angle_rad)]])
    
    # Calculate the centroid of the original points
    centroid = np.mean(points, axis=0)

    # Center the points around the centroid
    centered_points = points - centroid

    unrot_points = []
    for point in centered_points:
        unrot_points.append(rotation_matrix.dot(np.array(point)) + centroid)
    polygon = patches.Polygon(unrot_points, closed=True, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(polygon)
    

    plt.savefig("test.jpg")
    




# Führe den Test durch
img, boxes, msk, boxes_unrotated, angles = test_load_item()
visualize_image(img, boxes, boxes_unrotated, angles)