import os
import configparser
from huggingface_hub import snapshot_download, create_repo, upload_folder

class HuggingFacehub:

    def __init__(self):
        """
        Initializes the HFHub class.
        """
        super(HuggingFacehub, self).__init__()

    def load_hf_token(self, config_path: str = "config.ini") -> str:
        """
        Loads the Hugging Face API token from a .ini configuration file.
        
        Args:
            config_path (str): Path to the .ini file. Default is "config.ini".
        
        Returns:
            str: The Hugging Face API token.
        """
        config = configparser.ConfigParser()
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        config.read(config_path)
        
        if "huggingface" not in config or "token" not in config["huggingface"]:
            raise KeyError("Hugging Face token not found in config file. Ensure it has [huggingface] section with a 'token' key.")
        
        return config["huggingface"]["token"]


    def save_to_hf_hub(self, directory_path: str, repo_name: str, token: str, organization: str = None, private: bool = True):
        """Saves a model from the specified directory to a private Hugging Face Hub repository."""
        
        # Set environment variable (optional)
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        
        # Ensure repo exists (set `private=True` if needed)
        repo_id = f"{organization}/{repo_name}" if organization else repo_name
        create_repo(repo_id, token=token, exist_ok=True, private=private)
        
        # Upload model
        upload_folder(folder_path=directory_path, repo_id=repo_id, commit_message="Upload model", token=token)
        
        self.logger_info(f"Model uploaded to https://huggingface.co/{repo_id} (Private: {private})")


    def load_from_hf_hub(self, directory_path: str, repo_name: str, token: str, organization: str = None):
        """Loads a model from a private Hugging Face Hub repository into the specified directory."""
        
        repo_id = f"{organization}/{repo_name}" if organization else repo_name
        
        # Download model
        snapshot_download(repo_id=repo_id, local_dir=directory_path, token=token)
        
        self.logger_info(f"Model downloaded to {directory_path}")