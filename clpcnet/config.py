from pathlib import Path


###############################################################################
# Configuration
###############################################################################


# Run inference using checkpoint and settings from the original LPCNet
ORIGINAL_LPCNET = False

# Ablations
ABLATE_CREPE = False
ABLATE_PITCH_REPR = False
ABLATE_SAMPLING = False
ABLATE_SAMPLING_TAIL = False

# Settings for using original lpcnet checkpoints
if ORIGINAL_LPCNET:
    ABLATE_CREPE = True
    ABLATE_PITCH_REPR = True
    ABLATE_SAMPLING = True
    ABLATE_SAMPLING_TAIL = False

# Directories
ASSETS_DIR = Path(__file__).parent / 'assets'
DATA_DIR = Path(__file__).parent.parent / 'data'
RUNS_DIR = Path(__file__).parent.parent / 'runs'
CACHE_DIR = RUNS_DIR / 'cache'
CHECKPOINT_DIR = RUNS_DIR / 'checkpoints'
EVAL_DIR = RUNS_DIR / 'eval'
LOG_DIR = RUNS_DIR / 'log'

# Default pretrained checkpoint
if ORIGINAL_LPCNET:
    DEFAULT_CHECKPOINT = ASSETS_DIR / 'checkpoints' / 'original.h5'
else:
    DEFAULT_CHECKPOINT = ASSETS_DIR / 'checkpoints' / 'model.h5'

# Pitch representation
PITCH_BINS = 256
FMAX = 550.  # Hz
# 63 Hz is hard minimum imposed when using non-uniform pitch bins
FMIN = 63. if ABLATE_PITCH_REPR else 50.  # Hz

# DSP parameters
HOPSIZE = 160  # samples
BLOCK_SIZE = 640  # samples
LPC_ORDER = 16
MAX_SAMPLE_VALUE = 32768
PCM_LEVELS = 256
PREEMPHASIS_COEF = 0.85
SAMPLE_RATE = 16000  # Hz

# Training parameters
AVERAGE_STEPS_PER_EPOCH = 436925  # batches per epoch
BATCH_SIZE = 64  # items per batch
FEATURE_CHUNK_SIZE = 15  # frames per item in batch
LEARNING_RATE = 1e-3
PCM_CHUNK_SIZE = HOPSIZE * FEATURE_CHUNK_SIZE  # samples per item in batch
STEPS = 45000000
WEIGHT_DECAY = 5e-5

# Neural network sizes
SPECTRAL_FEATURE_SIZE = 38
EMBEDDING_SIZE = 128
GRU_A_SIZE = 384
GRU_B_SIZE = 16

# Number of features on disk
TOTAL_FEATURE_SIZE = 55

# Feature indices
PITCH_IDX = 36
CORRELATION_IDX = 37
