"""Common configuration for the IRMAS instrument identification project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "IRMAS-TrainingData"
DEFAULT_OPENMIC_DIR = PROJECT_ROOT / "data" / "raw" / "openmic-2018"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
FEATURE_CACHE_DIR = OUTPUT_DIR / "cache"


CLASS_CODES = [
    "cel",
    "cla",
    "flu",
    "gac",
    "gel",
    "org",
    "pia",
    "sax",
    "tru",
    "vio",
    "voi",
]

CLASS_NAMES = {
    "cel": "cello",
    "cla": "clarinet",
    "flu": "flute",
    "gac": "acoustic guitar",
    "gel": "electric guitar",
    "org": "organ",
    "pia": "piano",
    "sax": "saxophone",
    "tru": "trumpet",
    "vio": "violin",
    "voi": "voice",
}

NUM_CLASSES = len(CLASS_CODES)
CLASS_TO_INDEX = {class_code: index for index, class_code in enumerate(CLASS_CODES)}
INDEX_TO_CLASS = {index: class_code for class_code, index in CLASS_TO_INDEX.items()}


OPENMIC_OVERLAP_CLASS_NAMES = [
    "cello",
    "clarinet",
    "flute",
    "organ",
    "piano",
    "saxophone",
    "trumpet",
    "violin",
    "voice",
]

OPENMIC_OVERLAP_CLASS_CODES = [
    "cel",
    "cla",
    "flu",
    "org",
    "pia",
    "sax",
    "tru",
    "vio",
    "voi",
]

OPENMIC_OVERLAP_CLASS_TO_IRMAS_INDEX = [
    CLASS_TO_INDEX[class_code] for class_code in OPENMIC_OVERLAP_CLASS_CODES
]


SAMPLE_RATE = 22050
DURATION = 3.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 40
MONO = True

FEATURE_TYPES = ("mel", "mfcc")
DEFAULT_FEATURE_TYPE = "mel"


VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
