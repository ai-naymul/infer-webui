from huggingface_hub import snapshot_download
import os
import shutil
class DownloadModel:
    def __init__(self, model_name, model_path, hugging_face_token):
        self.model_name = model_name
        self.model_path = model_path
        self.hugging_face_token = hugging_face_token

    def download_model(self, destination_dir):
        """
            Download the model from huggingface and move the model files to the main models folder
            Cause when we are using the snapshot_download method the models is saved into a snapshot folder
        """
        logs = []
        logs.append("Starting model download...")
        try:
            print(self.model_path)
            snapshot_download(repo_id=self.model_name, token=self.hugging_face_token, cache_dir=self.model_path)
            logs.append("Model downloaded successfully.")
            if os.path.exists(destination_dir):
                subfolders = [f.path for f in os.scandir(self.model_path) if f.is_dir()] 
                for subfolder in subfolders:
                    files_in_subfolder = os.listdir(subfolder)
                    for file in files_in_subfolder:
                        source_file = os.path.join(subfolder, file)
                        destination_file = os.path.join(destination_dir, file)
                        shutil.move(source_file, destination_file)
                        logs.append(f"Moved {file} from {subfolder} into {destination_dir} folder")
        except Exception as e:
            logs.append(f"Error during download: {str(e)}")
        return "\n".join(logs)
