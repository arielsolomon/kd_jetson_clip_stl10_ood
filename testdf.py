
import pandas as pd
import re

# Sample DataFrame with a column containing the patterns
data = {'log': ['Epoch 1 | TRAIN LOSS 0.18785571901104117 | TEST ACC 98.412 |',
                'Epoch 2 | TRAIN LOSS 9.077525606422753e-06 | TEST ACC 99.001 |',
                'Epoch 3 | TRAIN LOSS 0.00123456789 | TEST ACC 97.825 |']}
df = pd.DataFrame(data)

# Define a function to extract values using regular expressions
def extract_values(log):
    match = re.search(r'TRAIN LOSS ([\d\.e-]+) \| TEST ACC ([\d\.]+) \|', log)
    if match:
        train_loss = float(match.group(1))
        test_acc = float(match.group(2))
        return train_loss, test_acc
    else:
        return None, None

# Apply the function to create new columns in the DataFrame
df[['TRAIN LOSS', 'TEST ACC']] = df['log'].apply(lambda x: pd.Series(extract_values(x)))

# Display the resulting DataFrame
print(df)