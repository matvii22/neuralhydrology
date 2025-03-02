import pandas as pd
import numpy as np
from pathlib import Path

# Load hydrometry attributes file
df = pd.read_csv('/Users/matviikotolyk/hydrology_data/CAMELS_GB/CAMELS_GB_hydrometry_attributes.csv')

# Extract gauge IDs
gauge_ids = df['gauge_id'].astype(str).tolist()

# Shuffle the list for random splitting
np.random.seed(42)  # For reproducibility
np.random.shuffle(gauge_ids)


# Create 60/20/20 split
n_total = len(gauge_ids)
n_train = int(0.6 * n_total)
n_val = int(0.2 * n_total)

train_ids = gauge_ids[:n_train]
val_ids = gauge_ids[n_train:n_train+n_val]
test_ids = gauge_ids[n_train+n_val:]

# Create data directory if it doesn't exist
data_dir = Path('neuralhydrology/data')
data_dir.mkdir(exist_ok=True, parents=True)

# Write to basin files
with open(data_dir / 'basins_train.txt', 'w') as f:
    f.write('\n'.join(train_ids))
    
with open(data_dir / 'basins_val.txt', 'w') as f:
    f.write('\n'.join(val_ids))
    
with open(data_dir / 'basins_test.txt', 'w') as f:
    f.write('\n'.join(test_ids))

print(f"Created basin files: {len(train_ids)} training, {len(val_ids)} validation, {len(test_ids)} test basins")