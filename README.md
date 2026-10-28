# JudgeClean-64 — a third option beside Kiuyha nano/S

One tiny model (156,324 params) that decides AND cleans in a single
forward pass, per 64x64 tile.

Why: Kiuyha nano and S can still miss text (artistic coloured text,
white-on-light, tiny bubbles, edge tiles). Re-running the whole pipeline
for a miss is expensive and risks smudging art elsewhere. JudgeClean is
the third option: tile the page, judge each tile, clean ONLY flagged
tiles, copy every other tile pixel-for-pixel.

Architecture: one shared CNN encoder, two heads.
- decision head: binary logit, "does this tile contain text?"
- clean head: decoder reconstructing the background without text.

Training (this prototype): fully synthetic data — random backgrounds
(solid, gradient, screentone dots, stripes) + PIL-drawn text. No manga
pages, no copyrighted art. Synthetic results: decision accuracy 1.000,
clean PSNR 21.08 dB on held-out synthetic tiles. That proves the
architecture, NOT real-manga performance. Real manga (coloured text,
white-on-light, tiny bubbles) needs fine-tuning on real missed-text
crops before this replaces anything.

Tile inference (`tile_clean.py`): on a real clean page (900x1000) it
flagged 0/210 tiles, so the output is untouched. On a synthetic page
with text in exactly one tile, it flagged exactly that tile (128,64).

Files: `train.py` (model + synthetic training), `tile_clean.py`
(inference), `out/judgeclean-64.pt` (weights, 617 KB), `out/demo.png`.

Limitations, stated plainly: synthetic-only training, overconfident
logits, clean head is a tiny decoder, not LaMa. Next step is fine-tuning
on real Kiuyha-missed crops, then A/B against Kiuyha nano/S on the same
pages by eye.
