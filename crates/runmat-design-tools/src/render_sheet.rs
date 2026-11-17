//! `design.render` of a `.drawing`: one sheet, or one view of it, as the Desktop's sheet draws
//! it, to a PNG.
//!
//! The picture is drawn from the compile's sheet layout (`layouts`, runmat-cad-wasm
//! `drawing::layout`), the layout the Desktop's sheet and every export (PDF, DXF, DWG) draw.
//! What the agent sees is therefore what the person sees: the views' lines in their line types
//! (visible wide, hidden dashed, tangent thin, centerlines chained), the dimensions, callouts,
//! notes, tables, title block and frame. Lines take the drafting standard's widths; text is set
//! in the renderer's bitmap font at the layout's size, each character 0.6 of the height wide.
//!
//! The answer's `notes` say what the sheet holds, where each view sits, and what Check drawing
//! (`drawing.check`) finds, so a problem too small to see at the picture's size still reads.

use crate::scene::{RenderInput, RenderOutput};
use crate::{text_of, Files, ToolError};
use base64::Engine;
use serde_json::Value;

/// The picture's longest side when the call gives no size: a sheet at A3 then draws about
/// 3.7 px per mm, so 3.5 mm figures are 13 px high.
const DEFAULT_LONG_SIDE: u32 = 1568;
const MAX_PIXELS: u32 = 4096;
/// How far past a framed view's own lines the picture reaches (sheet mm): its dimension rows,
/// label and callouts.
const FRAME_MARGIN: f64 = 22.0;

/// A box on the sheet: min x, min y, max x, max y (mm, y up).
type Bx = [f64; 4];

pub fn render(input: &RenderInput, files: &Files) -> Result<RenderOutput, ToolError> {
    let path = &input.path;
    let text = text_of(files, path)?;
    let compiled: Value = serde_json::from_str(&runmat_cad_wasm::drawing::compile_json(
        path, text, files,
    ))
    .map_err(|e| ToolError::new("internal", format!("the drawing's compile isn't JSON: {e}")))?;
    let data = &compiled["projection"]["data"];
    let layouts = data["layouts"]
        .as_object()
        .filter(|l| !l.is_empty())
        .ok_or_else(|| {
            ToolError::new(
                "not_compiled",
                format!("{path} has no sheet to draw yet (check it with design.check)"),
            )
        })?;
    let sheet = match &input.sheet {
        Some(name) => name.clone(),
        None => data["document"]["sheets"][0]["name"]
            .as_str()
            .or_else(|| layouts.keys().next().map(String::as_str))
            .unwrap_or("1")
            .to_owned(),
    };
    let layout = layouts.get(&sheet).ok_or_else(|| {
        ToolError::invalid(format!(
            "{path} has no sheet `{sheet}`; its sheets are {}",
            layouts.keys().cloned().collect::<Vec<_>>().join(", ")
        ))
    })?;
    let size = [
        layout["size"][0].as_f64().unwrap_or(420.0),
        layout["size"][1].as_f64().unwrap_or(297.0),
    ];
    let prims: Vec<&Value> = layout["primitives"]
        .as_array()
        .into_iter()
        .flatten()
        .collect();
    let mut notes = Vec::new();
    // Where each view sits: the box of its own lines and label.
    let mut views: Vec<(String, Bx)> = Vec::new();
    for p in &prims {
        let Some(name) = p["id"].as_str().and_then(|id| id.strip_prefix("view:")) else {
            continue;
        };
        let Some(b) = shape_box(&p["shape"]) else {
            continue;
        };
        match views.iter_mut().find(|(n, _)| n == name) {
            Some((_, have)) => *have = union(*have, b),
            None => views.push((name.to_owned(), b)),
        }
    }
    let area: Bx = match &input.frame {
        Some(name) => {
            let (_, b) = views.iter().find(|(n, _)| n == name).ok_or_else(|| {
                ToolError::invalid(format!(
                    "sheet {sheet} of {path} has no view `{name}`; its views are {}",
                    views
                        .iter()
                        .map(|(n, _)| n.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?;
            [
                (b[0] - FRAME_MARGIN).max(0.0),
                (b[1] - FRAME_MARGIN).max(0.0),
                (b[2] + FRAME_MARGIN).min(size[0]),
                (b[3] + FRAME_MARGIN).min(size[1]),
            ]
        }
        None => [0.0, 0.0, size[0], size[1]],
    };
    let aspect = (area[2] - area[0]).max(1.0) / (area[3] - area[1]).max(1.0);
    let (width, height) = match (input.view.width, input.view.height) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => (w, (w as f64 / aspect).round() as u32),
        (None, Some(h)) => ((h as f64 * aspect).round() as u32, h),
        (None, None) if aspect >= 1.0 => (
            DEFAULT_LONG_SIDE,
            (DEFAULT_LONG_SIDE as f64 / aspect).round() as u32,
        ),
        (None, None) => (
            (DEFAULT_LONG_SIDE as f64 * aspect).round() as u32,
            DEFAULT_LONG_SIDE,
        ),
    };
    let (width, height) = (width.clamp(16, MAX_PIXELS), height.clamp(16, MAX_PIXELS));
    let dark = input.view.theme.as_deref() == Some("dark");
    let mut canvas = Canvas::new(width, height, area);
    for p in &prims {
        canvas.draw(p);
    }
    let (paper, ink): ([u8; 3], [u8; 3]) = if dark {
        ([18, 20, 26], [222, 226, 232])
    } else {
        ([255, 255, 255], [20, 22, 28])
    };
    let png =
        runmat_plot::export::cpu_surface::encode_png_bytes(width, height, &canvas.rgba(paper, ink))
            .map_err(|e| ToolError::new("internal", e))?;
    notes.push(format!(
        "Sheet {sheet} ({:.0} × {:.0} mm){}, drawn from the layout the Desktop's sheet and the PDF draw; {:.2} px per mm.",
        size[0],
        size[1],
        input
            .frame
            .as_ref()
            .map(|v| format!(", framed on the {v} view"))
            .unwrap_or_default(),
        canvas.scale
    ));
    for (name, b) in &views {
        notes.push(format!(
            "View {name}: x {:.1}–{:.1}, y {:.1}–{:.1} mm on the sheet (y up).",
            b[0], b[2], b[1], b[3]
        ));
    }
    let mut values = serde_json::Map::new();
    values.insert("sheet".into(), Value::String(sheet.clone()));
    match runmat_cad_wasm::analyze::drawing_check::findings(path, text, files, &values) {
        Ok((found, _)) if found.is_empty() => notes.push("Check drawing: no findings.".into()),
        Ok((found, _)) => {
            notes.push(format!(
                "Check drawing finds {} (the sheet's findings; design.check lists them with their items):",
                found.len()
            ));
            for f in found.iter().take(12) {
                notes.push(format!("- {} {}: {}", f.severity, f.rule, f.message));
            }
        }
        Err(e) => notes.push(format!("Check drawing didn't run: {e}")),
    }
    for n in layout["notes"].as_array().into_iter().flatten() {
        if let Some(n) = n.as_str() {
            notes.push(n.to_owned());
        }
    }
    Ok(RenderOutput {
        path: path.clone(),
        mime_type: "image/png".into(),
        png_base64: base64::engine::general_purpose::STANDARD.encode(&png),
        width,
        height,
        renderer: "sheet".into(),
        camera: Value::Null,
        triangles: 0,
        boxed_occurrences: 0,
        notes,
    })
}

fn union(a: Bx, b: Bx) -> Bx {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

/// The box a shape covers on the sheet.
fn shape_box(shape: &Value) -> Option<Bx> {
    let f = |k: &str| shape[k].as_f64();
    match shape["type"].as_str()? {
        "polyline" => {
            let pts = floats(&shape["points"]);
            let mut b: Option<Bx> = None;
            for c in pts.chunks_exact(2) {
                let p = [c[0], c[1], c[0], c[1]];
                b = Some(b.map_or(p, |b| union(b, p)));
            }
            b
        }
        "rect" => Some([f("x")?, f("y")?, f("x")? + f("w")?, f("y")? + f("h")?]),
        "circle" => {
            let (x, y, r) = (f("x")?, f("y")?, f("r")?);
            Some([x - r, y - r, x + r, y + r])
        }
        "text" => {
            let corners = text_corners(shape)?;
            let xs = corners.map(|p| p[0]);
            let ys = corners.map(|p| p[1]);
            Some([
                xs.iter().cloned().fold(f64::MAX, f64::min),
                ys.iter().cloned().fold(f64::MAX, f64::min),
                xs.iter().cloned().fold(f64::MIN, f64::max),
                ys.iter().cloned().fold(f64::MIN, f64::max),
            ])
        }
        _ => None,
    }
}

/// Text's frame: its origin (the anchor point on the baseline), its run direction and up
/// direction (unit, sheet mm), the run's start along it, its width and its height.
struct TextFrame {
    origin: [f64; 2],
    run: [f64; 2],
    up: [f64; 2],
    start: f64,
    width: f64,
    size: f64,
}

fn text_frame(shape: &Value) -> Option<TextFrame> {
    let (x, y, size) = (
        shape["x"].as_f64()?,
        shape["y"].as_f64()?,
        shape["size"].as_f64()?,
    );
    let n = shape["text"].as_str()?.chars().count() as f64;
    let width = n * size * 0.6;
    let start = match shape["anchor"].as_str() {
        Some("middle") => -width / 2.0,
        Some("end") => -width,
        _ => 0.0,
    };
    let (s, c) = shape["rotation"]
        .as_f64()
        .unwrap_or(0.0)
        .to_radians()
        .sin_cos();
    Some(TextFrame {
        origin: [x, y],
        run: [c, s],
        up: [-s, c],
        start,
        width,
        size,
    })
}

/// The four corners of text's box: from 0.2 of its height below the baseline to its top.
fn text_corners(shape: &Value) -> Option<[[f64; 2]; 4]> {
    let t = text_frame(shape)?;
    let at = |u: f64, v: f64| {
        [
            t.origin[0] + t.run[0] * u + t.up[0] * v,
            t.origin[1] + t.run[1] * u + t.up[1] * v,
        ]
    };
    let (lo, hi) = (-0.2 * t.size, 1.05 * t.size);
    Some([
        at(t.start, lo),
        at(t.start + t.width, lo),
        at(t.start, hi),
        at(t.start + t.width, hi),
    ])
}

/// A coverage raster of one ink on one paper: every mark adds coverage (the most any mark
/// gives a pixel), so crossing lines don't darken where they meet.
struct Canvas {
    w: usize,
    h: usize,
    cover: Vec<f32>,
    /// Pixels per sheet mm, and the sheet point at the picture's top left.
    scale: f64,
    x0: f64,
    y1: f64,
}

impl Canvas {
    fn new(width: u32, height: u32, area: Bx) -> Canvas {
        let (w, h) = (width as usize, height as usize);
        let scale = (width as f64 / (area[2] - area[0]).max(1e-6))
            .min(height as f64 / (area[3] - area[1]).max(1e-6));
        // Centre the area in the picture.
        let x0 = area[0] - (width as f64 / scale - (area[2] - area[0])) / 2.0;
        let y1 = area[3] + (height as f64 / scale - (area[3] - area[1])) / 2.0;
        Canvas {
            w,
            h,
            cover: vec![0.0; w * h],
            scale,
            x0,
            y1,
        }
    }

    fn px(&self, p: [f64; 2]) -> [f64; 2] {
        [(p[0] - self.x0) * self.scale, (self.y1 - p[1]) * self.scale]
    }

    fn mm(&self, q: [f64; 2]) -> [f64; 2] {
        [q[0] / self.scale + self.x0, self.y1 - q[1] / self.scale]
    }

    fn add(&mut self, x: usize, y: usize, c: f32) {
        let i = y * self.w + x;
        if c > self.cover[i] {
            self.cover[i] = c.min(1.0);
        }
    }

    fn rgba(&self, paper: [u8; 3], ink: [u8; 3]) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.w * self.h * 4);
        for c in &self.cover {
            let c = *c as f64;
            for k in 0..3 {
                out.push((paper[k] as f64 + (ink[k] as f64 - paper[k] as f64) * c).round() as u8);
            }
            out.push(255);
        }
        out
    }

    /// The pixel range a box (in pixels) covers, clipped to the picture.
    fn span(&self, lo: [f64; 2], hi: [f64; 2]) -> Option<(usize, usize, usize, usize)> {
        let x0 = lo[0].floor().max(0.0);
        let y0 = lo[1].floor().max(0.0);
        let x1 = hi[0].ceil().min(self.w as f64 - 1.0);
        let y1 = hi[1].ceil().min(self.h as f64 - 1.0);
        (x0 <= x1 && y0 <= y1).then_some((x0 as usize, y0 as usize, x1 as usize, y1 as usize))
    }

    /// A straight stroke `width` px wide, with round ends, antialiased by distance.
    fn segment(&mut self, a: [f64; 2], b: [f64; 2], width: f64) {
        let (a, b) = (self.px(a), self.px(b));
        let r = (width / 2.0).max(0.5);
        let lo = [a[0].min(b[0]) - r - 1.0, a[1].min(b[1]) - r - 1.0];
        let hi = [a[0].max(b[0]) + r + 1.0, a[1].max(b[1]) + r + 1.0];
        let Some((x0, y0, x1, y1)) = self.span(lo, hi) else {
            return;
        };
        let d = [b[0] - a[0], b[1] - a[1]];
        let len2 = d[0] * d[0] + d[1] * d[1];
        for y in y0..=y1 {
            for x in x0..=x1 {
                let p = [x as f64 + 0.5, y as f64 + 0.5];
                let t = if len2 > 0.0 {
                    (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let q = [a[0] + d[0] * t - p[0], a[1] + d[1] * t - p[1]];
                let dist = (q[0] * q[0] + q[1] * q[1]).sqrt();
                let c = (r + 0.5 - dist).clamp(0.0, 1.0);
                if c > 0.0 {
                    self.add(x, y, c as f32);
                }
            }
        }
    }

    /// A polyline (sheet mm) stroked `width` mm wide in a dash pattern (mm, on and off), or
    /// solid.
    fn stroke(&mut self, pts: &[[f64; 2]], width_mm: f64, dash: &[f64]) {
        let width = (width_mm * self.scale).max(1.0);
        if dash.is_empty() {
            for w in pts.windows(2) {
                self.segment(w[0], w[1], width);
            }
            return;
        }
        let period: f64 = dash.iter().sum();
        let mut phase = 0.0;
        for w in pts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            let mut s = 0.0;
            while s < len {
                // Where in the pattern this point falls, and how far to the next change.
                let mut at = phase % period;
                let mut k = 0;
                while at >= dash[k] {
                    at -= dash[k];
                    k = (k + 1) % dash.len();
                }
                let step = (dash[k] - at).min(len - s);
                if k % 2 == 0 {
                    let p = |u: f64| {
                        [
                            a[0] + (b[0] - a[0]) * u / len,
                            a[1] + (b[1] - a[1]) * u / len,
                        ]
                    };
                    self.segment(p(s), p(s + step), width);
                }
                s += step;
                phase += step;
            }
        }
    }

    /// Fills the region inside `rings` (sheet mm, even-odd), sampled 2 × 2 per pixel.
    fn fill(&mut self, rings: &[Vec<[f64; 2]>]) {
        let px: Vec<Vec<[f64; 2]>> = rings
            .iter()
            .map(|r| r.iter().map(|p| self.px(*p)).collect())
            .collect();
        let all = px.iter().flatten();
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for p in all {
            lo = [lo[0].min(p[0]), lo[1].min(p[1])];
            hi = [hi[0].max(p[0]), hi[1].max(p[1])];
        }
        let Some((x0, y0, x1, y1)) = self.span(lo, hi) else {
            return;
        };
        for y in y0..=y1 {
            for x in x0..=x1 {
                let mut n = 0;
                for (dx, dy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                    let q = [x as f64 + dx, y as f64 + dy];
                    if px.iter().filter(|r| inside(r, q)).count() % 2 == 1 {
                        n += 1;
                    }
                }
                if n > 0 {
                    self.add(x, y, n as f32 / 4.0);
                }
            }
        }
    }

    /// Text in the bitmap font, scaled to the layout's box: each character 0.6 of the height
    /// wide, its 8 rows from 0.15 of the height below the baseline to 0.85 above.
    fn text(&mut self, shape: &Value) {
        let (Some(t), Some(chars)) = (text_frame(shape), shape["text"].as_str()) else {
            return;
        };
        let glyphs: Vec<[u8; 8]> = chars.chars().map(glyph).collect();
        let Some(corners) = text_corners(shape) else {
            return;
        };
        let cs = corners.map(|c| self.px(c));
        let lo = [
            cs.iter().map(|c| c[0]).fold(f64::MAX, f64::min),
            cs.iter().map(|c| c[1]).fold(f64::MAX, f64::min),
        ];
        let hi = [
            cs.iter().map(|c| c[0]).fold(f64::MIN, f64::max),
            cs.iter().map(|c| c[1]).fold(f64::MIN, f64::max),
        ];
        let Some((x0, y0, x1, y1)) = self.span(lo, hi) else {
            return;
        };
        let cell = 0.6 * t.size;
        let (bottom, top) = (-0.15 * t.size, 0.85 * t.size);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let mut n = 0;
                for (dx, dy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                    let m = self.mm([x as f64 + dx, y as f64 + dy]);
                    let rel = [m[0] - t.origin[0], m[1] - t.origin[1]];
                    let u = rel[0] * t.run[0] + rel[1] * t.run[1] - t.start;
                    let v = rel[0] * t.up[0] + rel[1] * t.up[1];
                    if u < 0.0 || u >= t.width || v < bottom || v >= top {
                        continue;
                    }
                    let i = (u / cell) as usize;
                    let col = (((u - i as f64 * cell) / cell) * 8.0) as u32;
                    let row = (((top - v) / (top - bottom)) * 8.0) as usize;
                    if glyphs
                        .get(i)
                        .is_some_and(|g| (g[row.min(7)] >> col.min(7)) & 1 == 1)
                    {
                        n += 1;
                    }
                }
                if n > 0 {
                    self.add(x, y, n as f32 / 4.0);
                }
            }
        }
    }

    /// One layout primitive, as the Desktop's sheet draws it.
    fn draw(&mut self, p: &Value) {
        let style = p["style"].as_str().unwrap_or("");
        // A model edge a drawing command picks draws only when hovered or selected.
        if matches!(style, "pick" | "outline" | "sheet") {
            return;
        }
        let width = p["width"].as_f64().unwrap_or(match style {
            "visible" => 0.5,
            "thick" => 0.7,
            "frame" => 0.35,
            _ => 0.25,
        });
        let dash: &[f64] = match style {
            "hidden" | "benddown" => &[3.0, 1.5],
            "center" | "bendup" => &[8.0, 1.5, 1.0, 1.5],
            _ => &[],
        };
        let shape = &p["shape"];
        let f = |k: &str| shape[k].as_f64().unwrap_or(0.0);
        match shape["type"].as_str().unwrap_or("") {
            "polyline" => {
                let mut pts: Vec<[f64; 2]> = floats(&shape["points"])
                    .chunks_exact(2)
                    .map(|c| [c[0], c[1]])
                    .collect();
                if shape["closed"] == true {
                    if let Some(first) = pts.first().copied() {
                        pts.push(first);
                    }
                }
                if style == "arrow" || style == "fill" {
                    self.fill(&[pts]);
                } else {
                    self.stroke(&pts, width, dash);
                }
            }
            "polygon" => {
                let rings: Vec<Vec<[f64; 2]>> = shape["rings"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|r| floats(r).chunks_exact(2).map(|c| [c[0], c[1]]).collect())
                    .collect();
                self.fill(&rings);
            }
            "rect" => {
                let (x, y, w, h) = (f("x"), f("y"), f("w"), f("h"));
                let pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]];
                if style == "arrow" || style == "fill" {
                    self.fill(&[pts.to_vec()]);
                } else {
                    self.stroke(&pts, width, dash);
                }
            }
            "circle" => {
                let (x, y, r) = (f("x"), f("y"), f("r"));
                let pts: Vec<[f64; 2]> = (0..=72)
                    .map(|i| {
                        let a = i as f64 / 72.0 * std::f64::consts::TAU;
                        [x + r * a.cos(), y + r * a.sin()]
                    })
                    .collect();
                if style == "arrow" || style == "fill" {
                    self.fill(&[pts]);
                } else {
                    self.stroke(&pts, width, dash);
                }
            }
            "text" => self.text(shape),
            _ => {}
        }
    }
}

/// Whether `q` is inside the ring (even-odd crossing test).
fn inside(ring: &[[f64; 2]], q: [f64; 2]) -> bool {
    let mut odd = false;
    let n = ring.len();
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        if (a[1] > q[1]) != (b[1] > q[1]) {
            let x = a[0] + (q[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if q[0] < x {
                odd = !odd;
            }
        }
    }
    odd
}

/// A character's bitmap. The diameter sign draws as the slashed O it is written like, and a
/// character the font lacks as `?`.
fn glyph(c: char) -> [u8; 8] {
    use runmat_plot::export::cpu_surface::glyph_bitmap;
    let c = match c {
        '⌀' => 'Ø',
        '−' => '-',
        '‘' | '’' => '\'',
        '“' | '”' => '"',
        '–' | '—' => '-',
        other => other,
    };
    glyph_bitmap(c)
        .or_else(|| glyph_bitmap('?'))
        .unwrap_or([0; 8])
}
