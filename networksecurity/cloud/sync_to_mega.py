from mega import Mega 
import os
from dotenv import load_dotenv
load_dotenv()

MEGA_EMAIL=os.getenv("MEGA_EMAIL")
MEGA_PASS=os.getenv("MEGA_PASS")


def upload_to_mega(folder):
    mega = Mega() 
    m = mega.login(MEGA_EMAIL, MEGA_PASS) 
    m.upload(folder)

# if __name__ == "__main__":
upload_to_mega("/home/ebi/machinelearning/test_ML/MLOPS/networksecurity/requirements.txt")