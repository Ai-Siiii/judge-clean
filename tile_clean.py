"""Third-option inference: tile the page, judge each tile, clean ONLY
tiles the decision head flags. Unflagged tiles are copied pixel-for-pixel,
so art elsewhere is never touched by the cleaner - unlike running LaMa
over a whole page, and unlike waiting for Kiuyha nano/S to catch a bubble.
Usage: tile_clean.py <input.png> <output.png>  (tiles are 64x64)
"""
import sys, torch, numpy as np
from PIL import Image
sys.path.insert(0, ".")
from train import JudgeClean, S
def main(inp, outp, thresh=0.0):
    model = JudgeClean(); model.load_state_dict(torch.load("out/judgeclean-64.pt", map_location="cpu")); model.eval()
    im = Image.open(inp).convert("RGB")
    w, h = im.size
    arr = np.array(im)
    out = arr.copy()
    flagged = []
    with torch.no_grad():
        for y in range(0, h - S + 1, S):
            for x in range(0, w - S + 1, S):
                tile = torch.tensor(arr[y:y+S, x:x+S].transpose(2,0,1)/255.0).unsqueeze(0).float()
                logit, clean = model(tile)
                if float(logit) > thresh:
                    flagged.append((x, y, round(float(logit), 2)))
                    out[y:y+S, x:x+S] = (clean[0].permute(1,2,0).numpy()*255).astype(np.uint8)
    Image.fromarray(out).save(outp)
    print(f"tiles flagged: {len(flagged)} / {((h//S)*(w//S))}")
    print("flagged:", flagged[:20])
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
