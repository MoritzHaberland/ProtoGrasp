import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn
import torch.nn.functional as F

class GraspDetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GraspDetectionHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        # Sub-Netzwerk für Greiforientierung
        self.orientation_fc = nn.Linear(1024, num_classes + 1)  # +1 für ungültig

        # Sub-Netzwerk für Bounding Box (x, y, w, h)
        self.bounding_box_fc = nn.Linear(1024, 4)  # Vorhersage von x, y, w, h

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        # Vorhersage der Orientierung
        orientation_scores = self.orientation_fc(x)

        # Vorhersage der Bounding Box
        bounding_box = self.bounding_box_fc(x)

        return orientation_scores, bounding_box


class FasterRCNNModel:
    def __init__(self, num_classes, learning_rate=0.005, momentum=0.9, weight_decay=0.0005, grasp_weight=1.0):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self._create_model(num_classes).to(self.device)
        self.optimizer = self._create_optimizer(learning_rate, momentum, weight_decay)

        # Gewichtung für den Grasp Loss
        self.grasp_weight = grasp_weight

    def _create_model(self, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = GraspDetectionHead(in_features, num_classes)  # Angepasster Grasp Detection Head
        return model

    def _create_optimizer(self, learning_rate, momentum, weight_decay):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    def train(self, data_loader, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            for images, targets in data_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Vorhersagen des Modells
                orientation_scores, bounding_boxes = self.model(images)

                # Verlustberechnung
                grasp_loss = self._compute_grasp_loss(orientation_scores, bounding_boxes, targets)

                total_loss = self.grasp_weight * grasp_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}')

    def _compute_grasp_loss(self, orientation_scores, bounding_boxes, targets):
        grasp_loss = 0.0

        # Verlust für die Orientierung
        grasp_loss += self._compute_orientation_loss(orientation_scores, [t['grasp_orientation'] for t in targets])

        # Verlust für die Bounding Box
        grasp_loss += self._compute_bounding_box_loss(bounding_boxes, [t['bounding_box'] for t in targets])

        return grasp_loss

    def _compute_orientation_loss(self, orientation_scores, targets):
        return F.cross_entropy(orientation_scores, targets)

    def _compute_bounding_box_loss(self, bounding_boxes, targets):
        losses = []
        for predicted_box, target_box in zip(bounding_boxes, targets):
            # Berechne Korrekturwerte
            tx = (predicted_box[0] - target_box[0]) / target_box[2]
            ty = (predicted_box[1] - target_box[1]) / target_box[3]
            tw = torch.log(predicted_box[2] / target_box[2])
            th = torch.log(predicted_box[3] / target_box[3])

            # Smooth L1 Loss
            loss = F.smooth_l1_loss(torch.tensor([tx, ty, tw, th]).to(self.device), torch.zeros(4).to(self.device))
            losses.append(loss)

        return sum(losses) / len(losses) if losses else 0.0

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)

    def evaluate(self, data_loader):
        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for images in data_loader:
                images = [img.to(self.device) for img in images]
                predictions = self.model(images)
                all_predictions.append(predictions)
        return all_predictions
