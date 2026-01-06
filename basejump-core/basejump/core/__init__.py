import warnings

# validate_default is being used in incorrectly in llama index, which triggers a warning
warnings.filterwarnings("ignore", message=".*validate_default.*")
