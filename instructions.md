First create an empty directory on computer and copy over all files into it

Then run these commands in the terminal in the root of the empty directory
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# If you're on Linux/Windows with an NVIDIA GPU you can optionally install bitsandbytes manually:
# pip install bitsandbytes==0.43.1

# Log in to Hugging Face. You will need to provide an access token.
huggingface-cli login



Then run the run_experiment.sh shell script

Then run the plot_results.py file



If there's anything wrong with the code just try to ask ai and talk about it in the challenges / solutions too lol
