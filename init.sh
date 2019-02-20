# Create checkpoints directory to save trained models.
mkdir checkpoints

# Download clothing-co-parsing dataset.
git clone https://github.com/bearpaw/clothing-co-parsing.git

# Process the dataset to get only required annotations and images.
python3 ./make_dataset.py

# Cleanup.
# rm -rf ./clothing-co-parsing
