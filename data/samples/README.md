# Sample photos for leaf_segmentation (T3.1)

Phone photos used to check that `processing/leaf_segmentation.py` draws correct masks.
They are test material, not research data. Final tuning still needs Arducam photos (T3.1 is BLOCKED-CAM for that).

## What to photograph (aim for 5+ leaflets and 3+ cards)

| File name | What is in the photo |
|---|---|
| `leaflet_01.jpg` ... `leaflet_05.jpg` | One soybean leaflet (detached or still on the plant), lying flat |
| `card_01.jpg` ... `card_03.jpg` | One cut-out reference card |
| `leaflet_tilted_01.jpg` (optional) | A leaflet photographed about 30 degrees off straight-down |

## How to take each photo

1. **Plain background.** Leaflets: white or light-gray matte paper. Cards (black): white paper.
2. **Straight down.** Hold the phone parallel to the table, about 20-30 cm above it.
3. **Ruler in frame.** Lay a ruler flat at the SAME height as the leaf, along one edge of the photo.
   It must not touch or overlap the leaf or card. A clear or white ruler is best (a black ruler can look like a card).
4. **Whole object inside the frame,** with a gap of at least 1 cm to every edge. A mask touching the edge is rejected.
5. **Even light, no flash, no shadow** across the leaf. Do not press the leaf under glass (glare).
6. **One object per photo.**

## After taking them

Copy the files here, then add one line per photo to `scale.csv` (create it):

```
image_file,mm_per_pixel,notes
leaflet_01.jpg,0.085,10 cm on ruler = 1176 px
```

To get `mm_per_pixel`: open the photo in an image viewer that shows pixel coordinates (Paint shows them in the
bottom bar), note the x position of the 0 cm and 10 cm marks, then `mm_per_pixel = 100 / (x_10cm - x_0cm)`.
