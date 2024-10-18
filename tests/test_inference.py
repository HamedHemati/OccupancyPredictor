import os
from inference.ro_inference.model import load_model
from training.room_occupancy.model import get_model
from training.room_occupancy.data_utils import load_dataset


def test_bestmodel_exists():
    """Test if best model exists."""
    assert os.path.exists("inference/pretrained_models/best_model.bin")


def test_load_model():
    """Test loading model from bin file."""
    model = load_model("inference/pretrained_models/best_model.bin")
    assert model is not None


def test_model_accuracy():
    """Test model accuracy on test dataset."""
    X, y = load_dataset("data/original/datatest.txt")
    model = get_model()
    model.load_model("inference/pretrained_models/best_model.bin")
    pred = model.predict(X)
    corrects = (pred == y).sum()
    accuracy = corrects / len(y)
    assert accuracy > 0.9
