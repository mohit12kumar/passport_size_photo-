import os
import sys
import urllib.request

def download_file(url, dest_path, label):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"\nDownloading {label} from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 1024 * 1024  # 1 MB
            downloaded = 0
            temp_path = dest_path + ".tmp"
            with open(temp_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    if total_size > 0:
                        percent = min(1.0, downloaded / total_size)
                        sys.stdout.write(f"\rProgress: {percent*100:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
                        sys.stdout.flush()
            print() # new line
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(temp_path, dest_path)
        print(f"[OK] {label} downloaded successfully.")
        return True
    except Exception as e:
        print(f"\n[!] Error downloading {label}: {e}")
        if os.path.exists(dest_path + ".tmp"):
            os.remove(dest_path + ".tmp")
        return False

def main():
    print("=========================================")
    print("       PRE-DOWNLOADING AI MODELS         ")
    print("=========================================")
    
    # 1. Download rembg model
    try:
        import rembg
        from config import REMBG_MODEL
        print(f"\n[1/2] Downloading rembg model '{REMBG_MODEL}'...")
        rembg.new_session(REMBG_MODEL)
        print("[OK] rembg model downloaded successfully.")
    except Exception as e:
        print(f"[!] Warning: Failed to pre-download rembg model: {e}")

    # 2. Download Real-ESRGAN weights
    project_root = os.path.dirname(os.path.abspath(__file__))
    dest_path = os.path.join(project_root, "gfpgan", "weights", "RealESRGAN_x4plus.pth")
    
    print("\n[2/2] Downloading Real-ESRGAN weights...")
    if os.path.isfile(dest_path):
        print(f"[OK] Real-ESRGAN weights already exist at: {dest_path}")
    else:
        download_file(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            dest_path,
            "Real-ESRGAN weights"
        )

    print("\n=========================================")
    print("         Model download complete!         ")
    print("=========================================")

if __name__ == "__main__":
    # Ensure config can be imported from root
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
