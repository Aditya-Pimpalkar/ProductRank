# assets

Place the demo recording here as `demo.gif`, then uncomment the image line near the top of
the root [README](../README.md) (`![ProductRank demo](assets/demo.gif)`).

## Recording recipe

Record `localhost:3000` (both servers must be running). A good ~15s clip:

1. Type an MS MARCO example query and show the four ranked columns + divergence highlight.
2. Toggle a variant (e.g. Dense vs BM25) to show the ranking difference.
3. Flip the dataset toggle to FiQA and run a financial query.
4. Open the Analytics page to show the metrics table.

Then convert to a repo-friendly GIF (aim < 5 MB):

```bash
# macOS screen capture: Cmd+Shift+5 → record → save clip.mov
ffmpeg -i clip.mov -vf "fps=12,scale=1200:-1:flags=lanczos" -loop 0 demo.gif
# or, higher quality / smaller:  brew install gifski && gifski -o demo.gif --fps 12 --width 1200 clip.mov
```
