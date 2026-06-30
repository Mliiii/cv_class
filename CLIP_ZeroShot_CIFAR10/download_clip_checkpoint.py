from pathlib import Path
from urllib.request import urlopen


URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
)
OUTPUT = Path(__file__).resolve().parent / "checkpoints" / "ViT-B-32.pt"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CLIP ViT-B-32 checkpoint to: {OUTPUT}")
    with urlopen(URL) as response, OUTPUT.open("wb") as f:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r{downloaded / total * 100:6.2f}%", end="", flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
