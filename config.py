"""
config.py — Central configuration for the Real-Time Fraud Detection Pipeline.
All hyperparameters, paths, and service settings live here.
"""

from pathlib import Path
import logging

# ─────────────────────────────────────────────
# Project root
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Kafka
# ─────────────────────────────────────────────
import os
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_RAW = os.getenv("KAFKA_TOPIC_RAW", "transactions-raw")
KAFKA_TOPIC_RESULTS = os.getenv("KAFKA_TOPIC_RESULTS", "fraud-results")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "fraud-detection-group")
KAFKA_AUTO_OFFSET_RESET = os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest")
KAFKA_MAX_POLL_RECORDS = int(os.getenv("KAFKA_MAX_POLL_RECORDS", 500))
KAFKA_RETRY_ATTEMPTS = int(os.getenv("KAFKA_RETRY_ATTEMPTS", 3))
KAFKA_RETRY_BACKOFF_MS = 1000

# ─────────────────────────────────────────────
# Spark
# ─────────────────────────────────────────────
SPARK_MASTER = "local[*]"          # use local for dev; swap to spark://localhost:7077 for Docker
SPARK_APP_NAME = "FraudDetectionPipeline"
SPARK_CHECKPOINT_DIR = str(ROOT_DIR / "outputs" / "checkpoints")
SPARK_TRIGGER_INTERVAL = "5 seconds"
SPARK_WATERMARK_DELAY = "10 minutes"
SPARK_WINDOW_DURATION = "5 minutes"
SPARK_MAX_OFFSETS_PER_TRIGGER = 1000

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
TRANSACTION_CSV = DATA_DIR / "train_transaction.csv"
IDENTITY_CSV = DATA_DIR / "train_identity.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
GRAPH_DATA_PATH = MODEL_DIR / "graph_data.pt"
GNN_MODEL_PATH = MODEL_DIR / "gnn_model.pt"
TRAINING_CURVES_PATH = MODEL_DIR / "training_curves.png"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
NODE_MAPS_PATH = MODEL_DIR / "node_maps.pkl"
OUTPUTS_DIR = ROOT_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# GNN Hyperparameters
# ─────────────────────────────────────────────
GNN_HIDDEN_CHANNELS = 128
GNN_NUM_LAYERS = 3
GNN_DROPOUT = 0.3
GNN_LEARNING_RATE = 0.001
GNN_EPOCHS = 50
GNN_BATCH_SIZE = 256
GNN_TRAIN_RATIO = 0.70
GNN_VAL_RATIO = 0.15
GNN_TEST_RATIO = 0.15

# ─────────────────────────────────────────────
# Fraud detection
# ─────────────────────────────────────────────
FRAUD_THRESHOLD = 0.3
TIME_COMPRESSION_FACTOR = 1000   # replay 1000x faster than real time
PRODUCER_FAST_DELAY = 0.05       # seconds between messages in fast mode
BURST_MIN_SIZE = 10
BURST_MAX_SIZE = 100
BURST_PAUSE_MIN = 0.5
BURST_PAUSE_MAX = 3.0

# ─────────────────────────────────────────────
# IEEE-CIS feature columns
# ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    "TransactionAmt",
    "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
    "D10", "D11", "D12", "D13", "D14", "D15",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15",
]

CATEGORICAL_FEATURES = [
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

GRAPH_NODE_FEATURES = [
    "TransactionAmt",
    "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
    "D1", "D2", "D3", "D4", "D5",
    "V1", "V2", "V3", "V4", "V5",
]

TARGET_COLUMN = "isFraud"
TRANSACTION_ID_COL = "TransactionID"
TRANSACTION_DT_COL = "TransactionDT"
AMOUNT_COL = "TransactionAmt"
