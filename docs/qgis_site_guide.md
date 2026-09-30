# Drawing a site in QGIS

This guide describes how the site of the project was drawn. The result is a GeoPackage
`data/site/site.gpkg` with three layers, read by `src.site`. The rules that `src.site` checks are
summarised at the [end of this guide](#what-srcsite-checks).

## 1. Load the LiDAR tiles as a background

QGIS opens `.LAZ` files directly: drag them into the window. On the first load, it creates a `.copc.laz`
index next to each tile. It is slow, which is why `*.copc.laz` is in the `.gitignore`.

In **Properties → Symbology**, choose **Classification** to see the ground, the buildings and class 1
(vegetation and objects). The point cloud becomes the main background, with OpenStreetMap as a complement
for street names.

If the display is slow (127 million points for the four tiles), hide the tiles you are not looking at, or
raise the maximum screen error in **Properties → Rendering** so that QGIS draws fewer points.

## 2. Draw the allowed frame

The site plus its study buffer must stay inside the LiDAR data. The buffer is at most 400 m and is reduced
automatically near the edge of the tiles, down to `intruder.outer_band_m` + 20 m (50 m): below that, the site
is rejected. To avoid going beyond it by mistake, draw that frame first.

- **Processing → Toolbox**, search for "Create layer from extent".
- In the **Extent** field, paste `XXXXXX,XXXXXX,XXXXXX,XXXXXX [EPSG:2950]` (the four tiles
  minus 50 m on each side, or 400 m to keep the full study buffer).
- Run the tool. In the symbology of the new layer, choose no fill and a thick red outline.

The perimeter must stay entirely inside this frame.

## 3. Choose the location

Browse the area in 2D (OSM background with the tiles) and in the 3D view. 

## 4. Create the GeoPackage

**Layer → Create Layer → New GeoPackage Layer**, file `site.gpkg`, CRS **EPSG:2950**:

| Layer | Geometry | Fields |
|---|---|---|
| `perimeter` | polygon | `name` (text), `fence_type` (text) |
| `zones` | polygon | `name` (text), `level` (integer), `weight` (real), `intruder_target` (boolean), `type` (text) |
| `access` | point | `name` (text), `type` (text, with the gate width, e.g. "Sliding gate 4m") |

When adding the second and third layers, choose **Add new layer**, not **Overwrite**.

`type` of a zone is `Building`, `Equipment` or `Periphery`. Every layer must be in EPSG:2950, including
`access` (QGIS sometimes creates point layers in EPSG:4326): `src.site` rejects any other CRS.

## 5. Input constraints on `zones`

Double-click on `zones`, then open the **Attributes Form** tab. For each field:

- `level`: widget **Value Map** with three rows 1 / Periphery, 2 / Sensitive, 3 / Core. Constraints: **Not
  null** and **Enforce not null constraint**.
- `weight`: **Not null** + enforce; expression `"weight" > 0`, then **Enforce expression constraint**.
- `intruder_target`: widget **Checkbox**, default value `false`.
- `name`: **Not null** + enforce.

Click **Apply**, then **OK**.

## 6. Snapping

Show the toolbar: **View → Toolbars → Snapping Toolbar**. Click the magnet (**Enable Snapping**), open the
menu next to it and choose **Advanced Configuration**, then **Edit advanced configuration** to show the
per-layer table:

| Layer | Enabled | Type | Tolerance | Units | Avoid overlap |
|---|---|---|---|---|---|
| `perimeter` | ✔ | Vertex + Segment | 1 | map units (m) | no |
| `zones` | ✔ | Vertex + Segment | 1 | map units (m) | yes |
| `access` | ✔ | Vertex + Segment | 1 | map units (m) | n/a |
| LAZ tiles, red frame | ✘ | | | | |

Do not tick **Avoid overlap** on `perimeter`: the zones glued inside it would be cut so as not to overlap
it, and they would disappear. Then enable **Topological Editing** (the icon next to the magnet).

## 7. Draw the perimeter

1. Select `perimeter` in the Layers panel, then **Toggle Editing** (the pencil).
2. **Add Polygon Feature**: left click for each vertex, inside the red frame, right click to close.
3. Fill the form: `name` = "Site Beta" (or any name), `fence_type` = "Wire mesh fence 2m".
4. **Save Layer Edits**, then **Toggle Editing** to leave the edit mode. One polygon only.

## 8. Create the zones by splitting the perimeter

The zones must exactly partition the perimeter (gaps and overlaps < 0.5 m²). The safest way is to copy the
perimeter and cut it:

1. Select the perimeter polygon (**Select Features**), then **Edit → Copy Features**.
2. Select `zones`, **Toggle Editing**, then **Edit → Paste Features**.
3. Show **View → Toolbars → Advanced Digitizing Toolbar** and choose **Split Features** (scissors).
4. Draw a line that crosses the polygon completely (start and end outside, right click to finish). Repeat
   to obtain the zones: a core, one or two sensitive zones, the rest as periphery.
5. Open the attribute table (F6) and fill each row, for example:

| Zone | level | weight (starting point) | intruder_target |
|---|---|---|---|
| Core | 3 | 10 | true |
| Sensitive | 2 | 3 | false |
| Periphery | 1 | 1 | false |

Exactly one zone of type `Building` or `Equipment` must have `intruder_target` = true.

## 9. Gates

Select `access`, **Toggle Editing**, then **Add Point Feature**. Click on the fence where a gate is: the
snapping glues the point on the edge (it must be within 0.5 m of the fence). Fill `name` and `type`; the
type must contain the width of the gate (e.g. "Gate 1.5m"), since it defines the opening crossed by the
intruder.

## 10. Check

- **Vector → Geometry Tools → Check Validity** on `perimeter` and `zones`: no error expected.
- In the attribute table of `zones`, open the **Field Calculator** and create virtual fields:

| Field | Expression |
|---|---|
| `area` | `area($geometry)` |
| `total_weight` | `"weight" * area($geometry)` |
| `share_pct` | `100 * "weight" * area($geometry) / sum("weight" * area($geometry))` |

`share_pct` is the share of each zone in the objective function: if the periphery weighs much more than
the core, the optimiser will mostly watch the periphery. Use `area($geometry)` (planar area in the layer
projection, MTM zone 8) rather than `$area` (ellipsoidal): it gives the same numbers as geopandas.
Virtual fields are stored in the `.qgz` project only, not in the GeoPackage: derived values do not belong
in the source data, and `src.site` recomputes them.

## 11. Run the chain

Save the GeoPackage as `data/site/site.gpkg`, then:

```bash
python -m src.site          # checks, weighted zones, grid (data/site/)
python -m src.rasters
python -m src.candidates
```

and the rest of the computation chain (see the main README).

## What `src.site` checks

| Layer / field | Rule |
|---|---|
| every layer | CRS EPSG:2950, valid geometries |
| `perimeter` | a single polygon |
| `zones` | partition the perimeter (gap and overlap < 0.5 m²): split the copied perimeter |
| `zones.type` | `Building`, `Equipment` or `Periphery` |
| `zones.weight` | > 0 |
| `zones.intruder_target` | true for **exactly one** `Building` or `Equipment` zone |
| `access` | points snapped on the fence (0.5 m), type with the width ("Gate 1.5m") |
| Extent | site at least 50 m from the edge of the tiles |

## Common errors

| Message | Fix |
|---|---|
| `partition: ... m² of gap` | split the copied perimeter again, with snapping enabled |
| `access: points off the fence` | snap the gate on the fence |
| `access: CRS ... instead of EPSG:2950` | reproject the layer (**Export → Save Features As**, CRS EPSG:2950) |
| `zones: unknown types` | fill `type` in QGIS (`Building`, `Equipment` or `Periphery`) |
| `the site is ... m from the edge of the LiDAR data` | move the site or add the neighbouring tiles to `data/raw/` |
| `width missing from the gate type` | write the width in the type of the gate ("Gate 3m") |
