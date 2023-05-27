from abc import ABC, abstractmethod

import torch

from app.config import MODEL_CONFIG, DATALOAD_CONFIG

class PairSampleLabeller(ABC):
  @abstractmethod
  def __init__(self):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", self.device, "\n")
    self.model = self.model.to(self.device)
    self.statistics = {
      "loss" : { "training": [], "validation": [] },
      "accuracy" : { "training": [], "validation": []},
      "f1-macro" : { "training": [], "validation": []},
      "selected": {"loss": 0.0, "accuracy": 0.0, "f1-macro": 0.0}
    }
    self.uuid = uuid.uuid4().hex
    self.parameters = MODEL_CONFIG | self.parameters
    self.model_dir = os.path.join(os.path.dirname(__file__), f'../../models/{self.parameters["common-name"]}')
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
  
  @abstractmethod
  def seed(self, dataloaders, criterion, optimizer, scheduler):
    begin_time = time.time()
    best_model_wts = copy.deepcopy(self.model.state_dict())
    for epoch in range(self.parameters["num-epochs"]):
      epoch_disclaimer = f'Epoch {epoch+1}/{self.parameters["num-epochs"]}:'
      print(epoch_disclaimer + "\n" + "-"*len(epoch_disclaimer))
      # Each epoch has a training and validation phase
      for phase in ['training', 'validation']:
        if phase == 'training':
          self.model.train() # Set model to training mode
        else:
          self.model.eval()  # Set model to evaluate mode
        phase_loss = 0.0
        phase_predictions = []
        phase_labels = []
        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
          inputs = list(map(lambda input: input.to(self.device), inputs))
          labels = labels.to(self.device)
          optimizer.zero_grad()
          with torch.set_grad_enabled(phase == 'training'):
            outputs = self.model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # backward + optimize only if in training phase
            if phase == 'training':
              loss.backward()
              optimizer.step()
          # statistics
          phase_loss += loss.item()*inputs[0].size(0)
          phase_predictions.extend(predictions.tolist())
          phase_labels += labels.tolist()
        if phase == 'training' and scheduler is not None:
          scheduler.step()
        epoch_loss = phase_loss / len(dataloaders[phase].dataset)
        epoch_accuracy = accuracy_score(phase_labels, phase_predictions)
        epoch_f1_macro = f1_score(phase_labels, phase_predictions, average="macro")
        self.statistics["loss"][phase].append(epoch_loss)
        self.statistics["accuracy"][phase].append(epoch_accuracy)
        self.statistics["f1-macro"][phase].append(epoch_f1_macro)
        print(f'{phase.capitalize()}:\tLoss: {epoch_loss:.6f} Accuracy: {epoch_accuracy:.6f} F1-macro: {epoch_f1_macro:.6f}')
        if phase == 'validation':
          if epoch_accuracy > self.statistics["selected"]["accuracy"]: # epoch_f1_macro > self.statistics["selected"]["f1-macro"]
            self.statistics["selected"]["accuracy"] = epoch_accuracy
            self.statistics["selected"]["f1-macro"] = epoch_f1_macro
            best_model_wts = copy.deepcopy(self.model.state_dict())
      print()
    time_elapsed = time.time() - begin_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'> Accuracy: {self.statistics["selected"]["accuracy"]:6f} F1-macro: {self.statistics["selected"]["f1-macro"]:6f}')
    # load best model weights
    self.model.load_state_dict(best_model_wts)