import os
import zipfile
from mega import Mega

MEGA_URL = "https://mega.nz/file/sOBGiTwR#fjvzvxoLDFDKLv8EtLPBGqPUCo_T_I7y8f14n4eGu94"

def main():
    os.makedirs("data", exist_ok=True)

    mega = Mega()
    m = mega.login()

    print("Downloading from MEGA ...")
    try:
        m.download_url(MEGA_URL)
    except PermissionError:
        pass

    if not os.path.exists("193.zip"):
        raise FileNotFoundError("193.zip was not downloaded. Please download it manually.")

    print("Unzipping to ./data ...")
    with zipfile.ZipFile("193.zip", "r") as z:
        z.extractall("data")

    print("DONE")

if __name__ == "__main__":
    main()
