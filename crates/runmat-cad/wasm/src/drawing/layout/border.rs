//! The sheet itself: its edge, the drawing frame with its zone grid and centring marks, the
//! title block as the compile laid it out, and the projection symbol beside it.
//!
//! - ISO 5457: the frame sits 20 mm in on the left (the filing margin) and 10 mm in elsewhere;
//!   zones are about 50 mm, numbered left to right along the top and bottom and lettered top to
//!   bottom along the sides; centring marks at the middle of each side.
//! - ASME Y14.1: the frame sits 10 mm in all round; zones are numbered right to left from the
//!   lower right corner and lettered bottom to top.

use super::marks::{line, prim, text};
use super::solver::{BoxR, Class};
use super::{xy, Ctx, Shape, Xy};

/// How far a centring mark reaches inside the frame, in mm.
pub const CENTRING_REACH: f64 = 5.0;

/// The parts of the centring marks inside the frame, as segments. A table keeps clear of them.
pub fn centring_marks(size: Xy, f: &BoxR) -> [(Xy, Xy); 4] {
    let (mx, my, r) = (size[0] / 2.0, size[1] / 2.0, CENTRING_REACH);
    [([mx, f.y0], [mx, f.y0 + r]), ([mx, f.y1], [mx, f.y1 - r]), ([f.x0, my], [f.x0 + r, my]), ([f.x1, my], [f.x1 - r, my])]
}

/// The drawing frame's inside, by the standard.
pub fn frame_box(size: Xy, iso: bool) -> BoxR {
    let left = if iso { 20.0 } else { 10.0 };
    BoxR::new(left, 10.0, size[0] - 10.0, size[1] - 10.0)
}

/// How many zones a side of `length` mm gets: about 50 mm each, an even count, at least 2.
fn zones(length: f64) -> usize {
    let n = (length / 50.0).round().max(2.0) as usize;
    n + n % 2
}

fn zone_letter(i: usize) -> String {
    // I and O are left out (ISO 5457, ASME Y14.1): they read as 1 and 0.
    const LETTERS: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ";
    (LETTERS[i % LETTERS.len()] as char).to_string()
}

pub fn draw(ctx: &mut Ctx<'_>) {
    let [w, h] = ctx.size;
    let f = ctx.frame;
    let iso = ctx.std.iso;
    let sheet_id = format!("sheet:{}", ctx.sheet);
    let mut out = vec![
        prim(&sheet_id, "sheet", "Sheet", "sheet", Shape::Rect { x: 0.0, y: 0.0, w, h }),
        prim("", "frame", "Frame", "frame", Shape::Rect { x: f.x0, y: f.y0, w: f.w(), h: f.h() }),
    ];
    let edge = 5.0;
    let label = 2.5;
    // Zone ticks across the margin and their numbers and letters.
    let (nx, ny) = (zones(f.w()), zones(f.h()));
    for i in 0..=nx {
        let x = f.x0 + f.w() * i as f64 / nx as f64;
        if i > 0 && i < nx {
            out.push(prim("", "frame", "Frame", "frame", line(vec![x, f.y0, x, edge])));
            out.push(prim("", "frame", "Frame", "frame", line(vec![x, f.y1, x, h - edge])));
        }
        if i < nx {
            let cx = x + f.w() / nx as f64 / 2.0;
            let n = if iso { i + 1 } else { nx - i };
            for y in [(edge + f.y0) / 2.0 - label / 2.0, (f.y1 + h - edge) / 2.0 - label / 2.0] {
                out.push(prim("", "label", "Frame", "muted", text(cx, y, &n.to_string(), label, "middle")));
            }
        }
    }
    for j in 0..=ny {
        let y = f.y0 + f.h() * j as f64 / ny as f64;
        if j > 0 && j < ny {
            out.push(prim("", "frame", "Frame", "frame", line(vec![f.x0, y, edge, y])));
            out.push(prim("", "frame", "Frame", "frame", line(vec![f.x1, y, w - edge, y])));
        }
        if j < ny {
            let cy = y + f.h() / ny as f64 / 2.0 - label / 2.0;
            // ISO letters run top to bottom; ASME bottom to top.
            let letter = zone_letter(if iso { ny - 1 - j } else { j });
            for x in [(edge + f.x0) / 2.0, (f.x1 + w - edge) / 2.0] {
                out.push(prim("", "label", "Frame", "muted", text(x, cy, &letter, label, "middle")));
            }
        }
    }
    out.push(prim("", "frame", "Frame", "frame", Shape::Rect { x: edge, y: edge, w: w - 2.0 * edge, h: h - 2.0 * edge }));
    // Centring marks: from the trimmed edge to `CENTRING_REACH` inside the frame, at each side's
    // middle.
    let (mx, my) = (w / 2.0, h / 2.0);
    let r = CENTRING_REACH;
    for pts in [[mx, 0.0, mx, f.y0 + r], [mx, h, mx, f.y1 - r], [0.0, my, f.x0 + r, my], [w, my, f.x1 - r, my]] {
        out.push(prim("", "frame", "Frame", "thick", line(pts.to_vec())));
    }
    ctx.push(out);
    title_block(ctx);
}

/// The title block's cells, label and value text and logo outlines (`title_layouts`, ISO 7200),
/// and the projection symbol in its cell.
fn title_block(ctx: &mut Ctx<'_>) {
    let layout = &ctx.data["title_layouts"][&ctx.sheet];
    let mut out = Vec::new();
    let mut area: Option<BoxR> = None;
    let mut grow = |b: BoxR| area = Some(area.map_or(b, |a| a.union(&b)));
    for cell in layout["cells"].as_array().into_iter().flatten() {
        let (Some(at), Some(size)) = (xy(&cell["at"]), xy(&cell["size"])) else { continue };
        out.push(prim("title", "frame", "Frame", "frame", Shape::Rect { x: at[0], y: at[1], w: size[0], h: size[1] }));
        grow(BoxR::new(at[0], at[1], at[0] + size[0], at[1] + size[1]));
        for (key, style) in [("label", "muted"), ("value", "note")] {
            let t = &cell[key];
            if let (Some(s), Some(p)) = (t["text"].as_str().filter(|s| !s.is_empty()), xy(&t["at"])) {
                out.push(prim("title", "label", "Frame", style, text(p[0], p[1], s, t["height"].as_f64().unwrap_or(2.5), "start")));
            }
        }
        for l in cell["logo"].as_array().into_iter().flatten() {
            let points: Vec<f64> = l.as_array().into_iter().flatten().filter_map(xy).flatten().collect();
            if points.len() >= 6 {
                out.push(prim("title", "frame", "Frame", "frame", Shape::Polyline { points, closed: true }));
            }
        }
    }
    if let Some(cell) = layout["projection_cell"].as_array().filter(|c| c.len() == 2) {
        if let (Some(at), Some(size)) = (xy(&cell[0]), xy(&cell[1])) {
            out.push(prim("title", "frame", "Frame", "frame", Shape::Rect { x: at[0], y: at[1], w: size[0], h: size[1] }));
            grow(BoxR::new(at[0], at[1], at[0] + size[0], at[1] + size[1]));
            let third = ctx.doc()["sheets"].as_array().and_then(|s| s.iter().find(|s| s["name"] == ctx.sheet.as_str())).is_none_or(|s| s["projection"] != "first_angle");
            out.extend(ctx.std.projection_symbol([at[0] + 4.0, at[1] + size[1] / 2.0 - ctx.std.text], third));
        }
    }
    if let Some(a) = area {
        ctx.occ.add(a, "title", Class::Table);
    }
    ctx.push(out);
}
