//! Laying out drawings and flat patterns as [`Sheet`]s, which the DXF and DWG writers share.
//!
//! A drawing sheet carries the frame and the title block (a block, inserted once), each
//! view's visible edges (`VISIBLE`, continuous) and hidden edges (`HIDDEN`, dashed) at the
//! sheet scale, center marks and center lines (`CENTER`), section hatching (`HATCH`, ANSI31),
//! section and detail labels, view labels, the linear dimensions whose anchors the drawing
//! resolved (`DIMENSIONS`), and the production callouts (`NOTES`): feature control frames on
//! leaders, datum symbols, ISO 1302 finish marks, hole callouts and diameter dimensions on
//! leaders, notes blocks, and the revision table. A flat pattern carries its outline at 1:1 (`OUTLINE`),
//! bend lines with their notes, and its overall width and height as dimensions.

use crate::formats::sheet::{standard_linetypes, Block, Dimension, DimensionKind, Entity, Layer, Shape, Sheet};
use crate::{Files, ToolError};
use runmat_cad::drawing::flat_pattern::{FlatPatternSheet, SheetCurve};
use runmat_cad::model::sheet_metal::BendDirection;
use serde_json::Value;

use super::floats;

/// Sheet sizes in millimetres, landscape.
fn sheet_size(name: &str) -> [f64; 2] {
    match name.to_ascii_uppercase().as_str() {
        "A0" => [1189.0, 841.0],
        "A1" => [841.0, 594.0],
        "A2" => [594.0, 420.0],
        "A4" => [297.0, 210.0],
        "B" => [431.8, 279.4],
        "C" => [558.8, 431.8],
        "D" => [863.6, 558.8],
        _ => [420.0, 297.0],
    }
}

fn layer(name: &str, color: i16, linetype: &str) -> Layer {
    Layer { name: name.into(), color, linetype: linetype.into() }
}

/// View axes (u right, v up) per orientation, as the Desktop's sheet projects model points.
fn axes(orientation: &str) -> ([f64; 3], [f64; 3]) {
    match orientation {
        "back" => ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        "top" => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        "bottom" => ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        "right" => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        "left" => ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
        _ => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    }
}

/// A view's orientation, following projected, section and detail views to their parents.
fn orientation(views: &[Value], name: &str, depth: usize) -> String {
    let Some(view) = views.iter().find(|v| v["name"].as_str() == Some(name)) else {
        return "front".into();
    };
    let kind = &view["kind"];
    match kind["view"].as_str() {
        Some("standard") => kind["orientation"].as_str().unwrap_or("front").into(),
        Some("projected") => match kind["direction"].as_str() {
            Some("up") => "top".into(),
            Some("down") => "bottom".into(),
            Some("left") => "left".into(),
            Some("right") => "right".into(),
            _ => "front".into(),
        },
        _ if depth < 8 => kind["from"].as_str().map(|f| orientation(views, f, depth + 1)).unwrap_or_else(|| "front".into()),
        _ => "front".into(),
    }
}

/// One drawing sheet, laid out. Returns the sheet's name, the sheet and notes.
pub(super) fn drawing_sheet(
    path: &str,
    text: &str,
    files: &Files,
    sheet: Option<&str>,
    configuration: Option<&str>,
) -> Result<(String, Sheet, Vec<String>), ToolError> {
    let compiled: Value =
        serde_json::from_str(&runmat_cad_wasm::drawing::compile_json_in(path, text, files, configuration)).unwrap_or(Value::Null);
    let data = &compiled["projection"]["data"];
    let doc = &data["document"];
    let sheets = doc["sheets"].as_array().cloned().unwrap_or_default();
    let Some(sheet) = sheets.iter().find(|s| sheet.is_none_or(|n| s["name"].as_str() == Some(n))) else {
        return Err(ToolError::invalid(format!("{path} has no sheet {}", sheet.unwrap_or(""))));
    };
    let sheet_name = sheet["name"].as_str().unwrap_or("1").to_owned();
    let format = doc["formats"]
        .as_array()
        .and_then(|f| f.iter().find(|f| f["name"] == sheet["format"]))
        .cloned()
        .unwrap_or(Value::Null);
    let [w, h] = sheet_size(format["size"]["name"].as_str().unwrap_or("A3"));
    let ratio = floats(&sheet["scale"]);
    let scale = if ratio.len() == 2 && ratio[1] != 0.0 { ratio[0] / ratio[1] } else { 1.0 };
    let mut out = Sheet {
        paper: [w, h],
        layers: vec![
            layer("FRAME", 7, "CONTINUOUS"),
            layer("VISIBLE", 7, "CONTINUOUS"),
            layer("HIDDEN", 8, "DASHED"),
            layer("CENTER", 4, "CENTER"),
            layer("HATCH", 8, "CONTINUOUS"),
            layer("DIMENSIONS", 3, "CONTINUOUS"),
            layer("NOTES", 3, "CONTINUOUS"),
        ],
        linetypes: standard_linetypes(),
        ..Sheet::default()
    };
    let mut notes = Vec::new();
    out.rect("FRAME", 10.0, 10.0, w - 20.0, h - 20.0);
    // The title block: one block, inserted at the frame's lower right corner.
    let rows = data["title_blocks"][&sheet_name].as_array().cloned().unwrap_or_default();
    if !rows.is_empty() {
        let mut block = Sheet::default();
        for (i, row) in rows.iter().enumerate() {
            let y = i as f64 * 7.0;
            block.rect("FRAME", -120.0, y, 120.0, 7.0);
            block.text("FRAME", [-118.0, y + 2.2], 2.2, row[0].as_str().unwrap_or(""));
            block.text("FRAME", [-80.0, y + 2.2], 2.6, row[1].as_str().unwrap_or(""));
        }
        out.blocks.push(Block { name: "TITLE_BLOCK".into(), base: [0.0, 0.0], entities: block.entities });
        out.add("FRAME", Shape::Insert { block: "TITLE_BLOCK".into(), at: [w - 10.0, 10.0], scale: 1.0, rotation: 0.0 });
    }
    let all_views = doc["views"].as_array().cloned().unwrap_or_default();
    let views: Vec<&Value> = all_views.iter().filter(|v| v["sheet"].as_str() == Some(sheet_name.as_str())).collect();
    let texts = &data["text"];
    for (n, view) in views.iter().enumerate() {
        let name = view["name"].as_str().unwrap_or("view");
        let hlr = &data["hlr"][name];
        let own = floats(&view["scale"]);
        let s = if own.len() == 2 && own[1] != 0.0 { own[0] / own[1] } else { scale };
        let runs = |key: &str| -> Vec<Vec<f64>> { hlr[key].as_array().into_iter().flatten().map(floats).collect() };
        let (visible, hidden) = (runs("visible"), runs("hidden"));
        if visible.is_empty() && hidden.is_empty() {
            notes.push(format!("view {name} has no edges (its model isn't in the workspace or doesn't compile)"));
        }
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for run in visible.iter().chain(&hidden) {
            for p in run.chunks_exact(2) {
                lo = [lo[0].min(p[0]), lo[1].min(p[1])];
                hi = [hi[0].max(p[0]), hi[1].max(p[1])];
            }
        }
        let centre = if lo[0].is_finite() { [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0] } else { [0.0, 0.0] };
        let at = floats(&view["at"]);
        let origin = if at.len() == 2 {
            [at[0], at[1]]
        } else {
            [w * (0.3 + 0.38 * (n % 2) as f64), h * (0.62 - 0.37 * (n / 2 % 2) as f64)]
        };
        let place = |p: [f64; 2]| [origin[0] + (p[0] - centre[0]) * s, origin[1] + (p[1] - centre[1]) * s];
        let pt = |v: &Value| {
            let f = floats(v);
            place([f.first().copied().unwrap_or(0.0), f.get(1).copied().unwrap_or(0.0)])
        };
        for (layer, set) in [("VISIBLE", &visible), ("HIDDEN", &hidden)] {
            for run in set.iter() {
                let points: Vec<[f64; 2]> = run.chunks_exact(2).map(|p| place([p[0], p[1]])).collect();
                if points.len() == 2 {
                    out.line(layer, points[0], points[1]);
                } else if points.len() > 2 {
                    out.add(layer, Shape::Polyline { points, closed: false });
                }
            }
        }
        // Section hatching: each region a closed loop.
        let loops: Vec<Vec<[f64; 2]>> = hlr["hatch"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|l| floats(l).chunks_exact(2).map(|p| place([p[0], p[1]])).collect::<Vec<_>>())
            .filter(|l| l.len() >= 3)
            .collect();
        if !loops.is_empty() {
            out.add("HATCH", Shape::Hatch { loops, pattern: "ANSI31".into(), angle: 0.0, scale: 1.0 });
        }
        for c in hlr["centers"].as_array().into_iter().flatten() {
            let r = c["radius"].as_f64().unwrap_or(0.0) * s;
            if c["kind"] == "mark" {
                let p = pt(&c["at"]);
                let k = r + 2.0;
                out.line("CENTER", [p[0] - k, p[1]], [p[0] + k, p[1]]);
                out.line("CENTER", [p[0], p[1] - k], [p[0], p[1] + k]);
            } else {
                out.line("CENTER", pt(&c["from"]), pt(&c["to"]));
            }
        }
        let half_h = if lo[1].is_finite() { (hi[1] - lo[1]) * s / 2.0 } else { 10.0 };
        if let Some(detail) = hlr.get("detail").filter(|d| d.is_object()) {
            let r = detail["radius"].as_f64().unwrap_or(0.0) * s;
            out.add("NOTES", Shape::Circle { center: pt(&detail["center"]), radius: r });
        }
        let shown = s;
        let scale_text = if shown >= 1.0 { format!("{shown}:1") } else { format!("1:{}", (1.0 / shown).round()) };
        let label = match view["kind"]["label"].as_str() {
            Some(l) if view["kind"]["view"] == "section" => format!("SECTION {l}-{l}  SCALE {scale_text}"),
            Some(l) if view["kind"]["view"] == "detail" => format!("DETAIL {l}  SCALE {scale_text}"),
            _ => format!("{}  SCALE {scale_text}", name.to_uppercase()),
        };
        out.text("NOTES", [origin[0], origin[1] - half_h - 7.0], 3.0, &label);
        let (u, v) = axes(&orientation(&all_views, name, 0));
        let project = |c: &[f64]| -> [f64; 2] {
            let d = |a: [f64; 3]| a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
            place([d(u), d(v)])
        };
        if lo[0].is_finite() {
            let box_ = (place(lo), place(hi));
            let marks: Vec<([f64; 2], f64)> = hlr["centers"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|c| c["kind"] == "mark")
                .map(|c| (pt(&c["at"]), c["radius"].as_f64().unwrap_or(0.0) * s))
                .collect();
            let anchor = |id: &str| data["anchors"][id][0].as_array().map(|_| project(&floats(&data["anchors"][id][0])));
            annotations(&mut out, name, box_, &marks, &anchor, s, doc, data, [w, h]);
        }
        // Linear dimensions between the drawing's resolved anchors, stacked above the view.
        let mut row = 0;
        for a in doc["annotations"].as_array().into_iter().flatten() {
            let kind = &a["kind"];
            if a["view"].as_str() != Some(name) || kind["annotation"] != "dimension" || kind["style"] != "linear" {
                continue;
            }
            let id = a["id"].as_str().unwrap_or("");
            let pts: Vec<Vec<f64>> = data["anchors"][id].as_array().into_iter().flatten().map(floats).collect();
            if pts.len() != 2 || pts.iter().any(|p| p.len() != 3) {
                continue;
            }
            let (p, q) = (project(&pts[0]), project(&pts[1]));
            let (dx, dy) = ((q[0] - p[0]).abs(), (q[1] - p[1]).abs());
            if dx.max(dy) < 1e-6 {
                continue;
            }
            row += 1;
            let gap = 8.0 * row as f64;
            let (line, rotation, measured) = if dx >= dy {
                ([p[0], origin[1] + half_h + gap], 0.0, dx)
            } else {
                let half_w = if lo[0].is_finite() { (hi[0] - lo[0]) * s / 2.0 } else { 10.0 };
                ([origin[0] + half_w + gap, p[1]], std::f64::consts::FRAC_PI_2, dy)
            };
            let text_at = if rotation == 0.0 { [(p[0] + q[0]) / 2.0, line[1] + 1.5] } else { [line[0] + 1.5, (p[1] + q[1]) / 2.0] };
            out.dimension(
                "DIMENSIONS",
                Dimension {
                    kind: DimensionKind::Linear { from: p, to: q, line, rotation },
                    text_at,
                    measurement: measured / s,
                    text: texts[id].as_str().map(str::to_owned),
                    block: String::new(),
                },
            );
        }
    }
    for table in doc["tables"].as_array().into_iter().flatten() {
        if table["sheet"].as_str() != Some(sheet_name.as_str()) || table["kind"]["table"] != "revision" {
            continue;
        }
        let rows = table["kind"]["rows"].as_array().cloned().unwrap_or_default();
        let at = floats(&table["at"]);
        let corner = if at.len() == 2 && (at[0] != 0.0 || at[1] != 0.0) { [at[0], at[1]] } else { [w - 10.0, h - 10.0] };
        revision_table(&mut out, &rows, corner);
    }
    // Cut-list tables: where the drawing puts them, else on the title block's top edge,
    // right-aligned, as the Desktop's sheet places a parts list.
    let block_h = rows.len() as f64 * 7.0;
    for table in doc["tables"].as_array().into_iter().flatten() {
        if table["sheet"].as_str() != Some(sheet_name.as_str()) || table["kind"]["table"] != "cut_list" {
            continue;
        }
        let data = &data["cut_list_tables"][table["name"].as_str().unwrap_or("")];
        let Some(body) = data["rows"].as_array() else { continue };
        let columns: Vec<(String, bool)> = data["columns"].as_array().into_iter().flatten().map(|c| (c["title"].as_str().unwrap_or("").to_uppercase(), c["align"] == "right")).collect();
        let cells: Vec<Vec<String>> = body.iter().map(|r| r["cells"].as_array().into_iter().flatten().map(|c| c.as_str().unwrap_or("").to_owned()).collect()).collect();
        let at = floats(&table["at"]);
        let placed = at.len() == 2 && (at[0] != 0.0 || at[1] != 0.0);
        let (tw, th) = grid_size(&columns, &cells);
        let top_left = if placed { [at[0], at[1]] } else { [w - 10.0 - tw, 10.0 + block_h + th] };
        grid_table(&mut out, &columns, &cells, top_left);
    }
    Ok((sheet_name, out, notes))
}

/// Callout proportions, as the Desktop's sheet draws them: 3.5 mm text, 3 x 1 mm arrowheads.
const TEXT: f64 = 3.5;
const ARROW: f64 = 3.0;

fn text_width(text: &str, size: f64) -> f64 {
    text.chars().count() as f64 * size * 0.6
}

/// A filled arrowhead with its tip at `tip`, pointing along `dir` (unit).
fn arrowhead(out: &mut Sheet, tip: [f64; 2], dir: [f64; 2]) {
    let (n, back) = ([-dir[1], dir[0]], [tip[0] - dir[0] * ARROW, tip[1] - dir[1] * ARROW]);
    let loop_ = vec![tip, [back[0] + n[0] * 0.5, back[1] + n[1] * 0.5], [back[0] - n[0] * 0.5, back[1] - n[1] * 0.5]];
    out.add("NOTES", Shape::Hatch { loops: vec![loop_], pattern: "SOLID".into(), angle: 0.0, scale: 1.0 });
}

/// A feature control frame's cells from `at` (lower left): the symbol cell square, the others
/// as wide as their text. Returns its width.
fn control_frame(out: &mut Sheet, at: [f64; 2], cells: &[&str]) -> f64 {
    let h = 2.0 * TEXT;
    let mut x = at[0];
    for (i, cell) in cells.iter().enumerate() {
        let w = if i == 0 { h } else { h.max(cell.chars().count() as f64 * TEXT * 0.62 + 3.0) };
        out.rect("NOTES", x, at[1], w, h);
        let tw = text_width(cell, TEXT);
        out.text("NOTES", [x + (w - tw) / 2.0, at[1] + (h - TEXT) / 2.0 + 0.4], TEXT, cell);
        x += w;
    }
    x - at[0]
}

/// A datum feature symbol: the filled triangle on `on`, a stem along `away`, the boxed letter.
fn datum_symbol(out: &mut Sheet, on: [f64; 2], away: [f64; 2], letter: &str) {
    let (b, tri) = (2.0 * TEXT, 1.8);
    let n = [-away[1], away[0]];
    let apex = [on[0] + away[0] * tri * 1.6, on[1] + away[1] * tri * 1.6];
    out.add("NOTES", Shape::Hatch { loops: vec![vec![[on[0] + n[0] * tri, on[1] + n[1] * tri], [on[0] - n[0] * tri, on[1] - n[1] * tri], apex]], pattern: "SOLID".into(), angle: 0.0, scale: 1.0 });
    let stem = [on[0] + away[0] * (tri + 4.0), on[1] + away[1] * (tri + 4.0)];
    out.line("NOTES", apex, stem);
    let c = [stem[0] + away[0] * b / 2.0, stem[1] + away[1] * b / 2.0];
    out.rect("NOTES", c[0] - b / 2.0, c[1] - b / 2.0, b, b);
    out.text("NOTES", [c[0] - text_width(letter, TEXT) / 2.0, c[1] - TEXT / 2.0 + 0.3], TEXT, letter);
}

/// An ISO 1302 surface texture symbol standing on a leader's end `elbow`, the arrow on `to`.
fn surface_finish(out: &mut Sheet, to: [f64; 2], elbow: [f64; 2], text: &str) {
    let h = 1.4 * TEXT;
    let t = (std::f64::consts::PI / 6.0).tan();
    let len = ((to[0] - elbow[0]).powi(2) + (to[1] - elbow[1]).powi(2)).sqrt().max(1e-9);
    out.line("NOTES", elbow, to);
    arrowhead(out, to, [(to[0] - elbow[0]) / len, (to[1] - elbow[1]) / len]);
    let [x, y] = elbow;
    let bar = text_width(text, TEXT) + 2.0;
    out.add("NOTES", Shape::Polyline { points: vec![[x - h * t, y + h], [x, y], [x + 2.0 * h * t, y + 2.0 * h], [x + 2.0 * h * t + bar, y + 2.0 * h]], closed: false });
    out.text("NOTES", [x + 2.0 * h * t + 1.0, y + h + 0.6], TEXT, text);
}

/// A leader from `tip` (with its arrow) to `elbow`, a shoulder and the text on it.
fn leader(out: &mut Sheet, tip: [f64; 2], elbow: [f64; 2], text: &str) {
    let len = ((tip[0] - elbow[0]).powi(2) + (tip[1] - elbow[1]).powi(2)).sqrt().max(1e-9);
    arrowhead(out, tip, [(tip[0] - elbow[0]) / len, (tip[1] - elbow[1]) / len]);
    let right = elbow[0] >= tip[0];
    let run = 6.0 + text_width(text, TEXT);
    let end = [elbow[0] + if right { run } else { -run }, elbow[1]];
    out.add("NOTES", Shape::Polyline { points: vec![tip, elbow, end], closed: false });
    let x = if right { elbow[0] + 3.0 } else { end[0] + 3.0 };
    out.text("NOTES", [x, elbow[1] + 0.8], TEXT, text);
}

/// The leadered callouts of one view, placed as the Desktop's sheet places them: feature control
/// frames clear of the view on the feature's side with a leader to it (stacking outward), datum
/// symbols hung on the frame of the same feature or else on the datum surface's outline, ISO 1302
/// finish marks on leaders, hole callouts and diameter dimensions on 45° leaders from the hole,
/// and notes blocks at their sheet position.
#[allow(clippy::too_many_arguments)]
fn annotations(
    out: &mut Sheet,
    view: &str,
    box_: ([f64; 2], [f64; 2]),
    marks: &[([f64; 2], f64)],
    anchor: &dyn Fn(&str) -> Option<[f64; 2]>,
    s: f64,
    doc: &Value,
    data: &Value,
    paper: [f64; 2],
) {
    let (lo, hi) = box_;
    let texts = &data["text"];
    let radius = |id: &str| data["radii"][id].as_f64().map(|r| r * s);
    let mut rows = [0usize; 2];
    let mut frames: Vec<([f64; 2], [f64; 4], bool)> = Vec::new();
    let mine: Vec<&Value> = doc["annotations"].as_array().into_iter().flatten().filter(|a| a["view"].as_str() == Some(view)).collect();
    // Frames first, so a datum symbol on the same feature hangs on its frame.
    for a in mine.iter().filter(|a| a["kind"]["annotation"] == "gdt") {
        let id = a["id"].as_str().unwrap_or("");
        let Some(p) = anchor(id) else { continue };
        let text = texts[id].as_str().unwrap_or(id).to_owned();
        let cells: Vec<&str> = text.split('|').collect();
        let h = 2.0 * TEXT;
        let below = p[1] - lo[1] < hi[1] - p[1];
        let row = &mut rows[below as usize];
        let y = if below { lo[1] - 6.0 - h - *row as f64 * (h + 3.0) } else { hi[1] + 6.0 + *row as f64 * (h + 3.0) };
        *row += 1;
        let mut probe = Sheet::default();
        let width = control_frame(&mut probe, [0.0, 0.0], &cells);
        let inward = if p[0] > (lo[0] + hi[0]) / 2.0 { -1.0 } else { 1.0 };
        let x = (if inward > 0.0 { p[0] + 4.0 } else { p[0] - 4.0 - width }).min(hi[0] - width).max(lo[0]);
        control_frame(out, [x, y], &cells);
        let from = [if inward > 0.0 { x } else { x + width }, y + h / 2.0];
        let toward = if below { 1.0 } else { -1.0 };
        let r = radius(id).unwrap_or(0.0);
        let tip = if r > 0.0 { [p[0] + inward * r * std::f64::consts::FRAC_1_SQRT_2, p[1] - toward * r * std::f64::consts::FRAC_1_SQRT_2] } else { p };
        out.add("NOTES", Shape::Polyline { points: vec![from, [tip[0], from[1]], [tip[0], tip[1] - toward * ARROW]], closed: false });
        arrowhead(out, tip, [0.0, toward]);
        frames.push((p, [x, y, x + width, y + h], below));
    }
    for a in &mine {
        let id = a["id"].as_str().unwrap_or("");
        let kind = &a["kind"];
        let text = texts[id].as_str().unwrap_or(id).to_owned();
        match kind["annotation"].as_str() {
            Some("datum_symbol") => {
                let letter = kind["label"].as_str().unwrap_or("A");
                let Some(p) = anchor(id) else { continue };
                if let Some((_, f, below)) = frames.iter().find(|(q, _, _)| (q[0] - p[0]).hypot(q[1] - p[1]) < 0.05) {
                    datum_symbol(out, [f[0] + TEXT, if *below { f[1] } else { f[3] }], [0.0, if *below { -1.0 } else { 1.0 }], letter);
                } else {
                    // On the outline's side nearest the anchor, the stem leaving the view.
                    let room = [(hi[1] - p[1], 0), (p[1] - lo[1], 1), (p[0] - lo[0], 2), (hi[0] - p[0], 3)];
                    let side = room.iter().min_by(|a, b| a.0.total_cmp(&b.0)).map_or(0, |r| r.1);
                    let (on, away) = match side {
                        0 => ([p[0], hi[1]], [0.0, 1.0]),
                        1 => ([p[0], lo[1]], [0.0, -1.0]),
                        2 => ([lo[0], p[1]], [-1.0, 0.0]),
                        _ => ([hi[0], p[1]], [1.0, 0.0]),
                    };
                    datum_symbol(out, on, away, letter);
                }
            }
            Some("surface_finish") => {
                let Some(p) = anchor(id) else { continue };
                let r = radius(id).unwrap_or(0.0);
                let tip = if r > 0.0 { [p[0] + r * std::f64::consts::FRAC_1_SQRT_2, p[1] + r * std::f64::consts::FRAC_1_SQRT_2] } else { p };
                let up = p[1] - lo[1] >= hi[1] - p[1] || r > 0.0;
                let elbow = [tip[0] - 9.0, if up { tip[1].max(hi[1]) + 9.0 } else { tip[1].min(lo[1]) - 14.0 }];
                surface_finish(out, tip, elbow, &text);
            }
            Some("note") => {
                let at = floats(&a["at"]);
                let at = if at.len() == 2 && (at[0] != 0.0 || at[1] != 0.0) { [at[0], at[1]] } else { [hi[0] + 6.0, hi[1]] };
                for (i, line) in text.split('\n').enumerate() {
                    let size = if i == 0 { TEXT } else { TEXT * 0.8 };
                    out.text("NOTES", [at[0], at[1] - (i + 1) as f64 * TEXT * 1.7], size, line);
                }
            }
            Some("hole_callout") | Some("dimension") if kind["annotation"] == "hole_callout" || matches!(kind["style"].as_str(), Some("diameter" | "radius")) => {
                // The hole whose drawn diameter the text names, else the view's first.
                let named = text.split('⌀').nth(1).and_then(|t| t.split(|c: char| !(c.is_ascii_digit() || c == '.')).next()).and_then(|t| t.parse::<f64>().ok());
                let hole = named
                    .and_then(|d| marks.iter().find(|(_, r)| (r / s - d / 2.0).abs() < 1e-3))
                    .or(marks.first());
                let Some(&(c, r)) = hole else {
                    out.add("NOTES", Shape::MText { at: [hi[0] + 4.0, hi[1]], height: 2.5, width: 40.0, text });
                    continue;
                };
                // Out of the view the short way at 45°, the elbow clear of the outline.
                let sx = if hi[0] - c[0] <= c[0] - lo[0] { 1.0 } else { -1.0 };
                let sy = if c[1] - lo[1] <= hi[1] - c[1] { -1.0 } else { 1.0 };
                let tip = [c[0] + sx * r * std::f64::consts::FRAC_1_SQRT_2, c[1] + sy * r * std::f64::consts::FRAC_1_SQRT_2];
                let y = if sy < 0.0 { lo[1] - TEXT - 4.0 } else { hi[1] + TEXT + 4.0 };
                let mut elbow = [tip[0] + sx * (y - tip[1]).abs(), y];
                let mut tip = tip;
                // A shoulder that would run past the frame turns to the hole's other side.
                let run = 6.0 + text_width(&text, TEXT);
                if (sx > 0.0 && elbow[0] + run > paper[0] - 10.0) || (sx < 0.0 && elbow[0] - run < 10.0) {
                    tip[0] = 2.0 * c[0] - tip[0];
                    elbow[0] = 2.0 * c[0] - elbow[0];
                }
                leader(out, tip, elbow, &text);
            }
            _ => {}
        }
    }
}

/// An ASME Y14.35 revision table (REV, DESCRIPTION, DATE, APPROVED) with its top-right corner
/// at `top_right`: a title row, the column headers and one row per revision, oldest first.
fn revision_table(out: &mut Sheet, rows: &[Value], top_right: [f64; 2]) {
    let (size, row, pad) = (2.5, 7.0, 1.5);
    let longest = rows.iter().map(|r| r["description"].as_str().unwrap_or("").chars().count()).max().unwrap_or(0).max("DESCRIPTION".len());
    let widths = [12.0, (longest as f64 * size * 0.6 + 2.0 * pad).max(50.0), 24.0, 30.0];
    let width: f64 = widths.iter().sum();
    let height = (rows.len() + 2) as f64 * row;
    let (x0, y1) = (top_right[0] - width, top_right[1]);
    out.rect("FRAME", x0, y1 - height, width, height);
    out.text("FRAME", [x0 + (width - text_width("REVISIONS", size)) / 2.0, y1 - row + (row - size) / 2.0], size, "REVISIONS");
    let header = ["REV", "DESCRIPTION", "DATE", "APPROVED"].map(str::to_owned);
    let cells: Vec<[String; 4]> = std::iter::once(header)
        .chain(rows.iter().map(|r| ["rev", "description", "date", "by"].map(|k| r[k].as_str().unwrap_or("").to_owned())))
        .collect();
    for (i, cells) in cells.iter().enumerate() {
        let y = y1 - (i + 2) as f64 * row;
        out.line("FRAME", [x0, y + row], [x0 + width, y + row]);
        let mut x = x0;
        for (c, cell) in cells.iter().enumerate() {
            if c > 0 {
                out.line("FRAME", [x, y], [x, y + row]);
            }
            let tx = if c == 0 { x + (widths[c] - text_width(cell, size)) / 2.0 } else { x + pad };
            out.text("FRAME", [tx, y + (row - size) / 2.0], size, cell);
            x += widths[c];
        }
    }
}

/// Parts-list proportions, as the Desktop's sheet draws them: 2.5 mm text in 7 mm rows.
const GRID: (f64, f64, f64) = (2.5, 7.0, 1.5);

fn grid_widths(columns: &[(String, bool)], cells: &[Vec<String>]) -> Vec<f64> {
    let (size, _, pad) = GRID;
    columns
        .iter()
        .enumerate()
        .map(|(i, (title, _))| {
            let longest = cells.iter().map(|r| r.get(i).map_or(0, |c| c.chars().count())).chain([title.chars().count()]).max().unwrap_or(0);
            longest as f64 * size * 0.6 + 2.0 * pad
        })
        .collect()
}

/// A table's [width, height] (mm): a header row and one row per line.
fn grid_size(columns: &[(String, bool)], cells: &[Vec<String>]) -> (f64, f64) {
    (grid_widths(columns, cells).iter().sum(), (cells.len() + 1) as f64 * GRID.1)
}

/// A header row over one row per line, with its top-left corner at `top_left`; right-aligned
/// columns (numbers) end a pad inside their rule.
fn grid_table(out: &mut Sheet, columns: &[(String, bool)], cells: &[Vec<String>], top_left: [f64; 2]) {
    let (size, row, pad) = GRID;
    let widths = grid_widths(columns, cells);
    let (width, height) = grid_size(columns, cells);
    let [x0, y0] = top_left;
    out.rect("FRAME", x0, y0 - height, width, height);
    let header: Vec<String> = columns.iter().map(|(t, _)| t.clone()).collect();
    for (n, line) in std::iter::once(&header).chain(cells.iter()).enumerate() {
        let top = y0 - n as f64 * row;
        if n > 0 {
            out.line("FRAME", [x0, top], [x0 + width, top]);
        }
        let mut x = x0;
        for (i, w) in widths.iter().enumerate() {
            let text = line.get(i).map(String::as_str).unwrap_or("");
            if !text.is_empty() {
                let tx = if columns[i].1 { x + w - pad - text_width(text, size) } else { x + pad };
                out.text("FRAME", [tx, top - row / 2.0 - size / 2.0], size, text);
            }
            x += w;
            if i + 1 < widths.len() {
                out.line("FRAME", [x, y0], [x, y0 - height]);
            }
        }
    }
}

/// A part's flat pattern as a sheet at 1:1, when it has one.
pub(super) fn flat_pattern(path: &str, text: &str) -> Result<Sheet, ToolError> {
    let part = runmat_cad_wasm::runtime::regenerate_part(path, text)
        .map_err(|_| ToolError::new("not_compiled", format!("{path} has syntax errors")))?;
    // The outline as cut: from the flat body (holes and cuts in the flats included).
    if let Some((_, layout)) = runmat_cad_wasm::exchange::flat::flat_layout(&part) {
        return Ok(flat_pattern_sheet(&runmat_cad::drawing::flat_pattern::flat_pattern_sheet(&layout)));
    }
    Err(ToolError::new(
        "unsupported",
        format!("{path} has no flat pattern; DXF and DWG export drawings and sheet-metal flat patterns (add a flat_pattern feature)"),
    ))
}

/// The sheet of a flat pattern: the outline, bend lines (up and down on their own layers)
/// with their notes, and the overall width and height dimensioned.
pub fn flat_pattern_sheet(sheet: &FlatPatternSheet) -> Sheet {
    let mut out = Sheet {
        paper: [sheet.max[0] - sheet.min[0], sheet.max[1] - sheet.min[1]],
        layers: vec![
            layer("OUTLINE", 7, "CONTINUOUS"),
            layer("BEND_UP", 3, "DASHED"),
            layer("BEND_DOWN", 1, "DASHED"),
            layer("BEND_NOTES", 8, "CONTINUOUS"),
            layer("DIMENSIONS", 3, "CONTINUOUS"),
        ],
        linetypes: standard_linetypes(),
        ..Sheet::default()
    };
    for curve in sheet.outline.iter().flatten() {
        let shape = match curve {
            SheetCurve::Line { start, end } => Shape::Line { a: *start, b: *end },
            SheetCurve::Arc { center, radius, start_angle, end_angle } => {
                Shape::Arc { center: *center, radius: *radius, start: start_angle.to_radians(), end: end_angle.to_radians() }
            }
            SheetCurve::Circle { center, radius } => Shape::Circle { center: *center, radius: *radius },
            SheetCurve::Polyline { points } => Shape::Polyline { points: points.clone(), closed: false },
        };
        out.entities.push(Entity { layer: "OUTLINE".into(), shape });
    }
    let height = (sheet.max[0] - sheet.min[0]).max(sheet.max[1] - sheet.min[1]).max(1.0) / 60.0;
    for b in &sheet.bend_lines {
        let layer = if matches!(b.direction, BendDirection::Up) { "BEND_UP" } else { "BEND_DOWN" };
        out.line(layer, b.start, b.end);
        let mid = [(b.start[0] + b.end[0]) / 2.0, (b.start[1] + b.end[1]) / 2.0];
        let rotation = (b.end[1] - b.start[1]).atan2(b.end[0] - b.start[0]);
        out.add("BEND_NOTES", Shape::Text { at: mid, height, rotation, text: b.note.clone() });
    }
    let (lo, hi) = (sheet.min, sheet.max);
    if hi[0] - lo[0] > 1e-9 {
        let line = [lo[0], lo[1] - 10.0];
        out.dimension(
            "DIMENSIONS",
            Dimension {
                kind: DimensionKind::Linear { from: lo, to: [hi[0], lo[1]], line, rotation: 0.0 },
                text_at: [(lo[0] + hi[0]) / 2.0, lo[1] - 8.5],
                measurement: hi[0] - lo[0],
                text: None,
                block: String::new(),
            },
        );
    }
    if hi[1] - lo[1] > 1e-9 {
        out.dimension(
            "DIMENSIONS",
            Dimension {
                kind: DimensionKind::Linear { from: [hi[0], lo[1]], to: hi, line: [hi[0] + 10.0, lo[1]], rotation: std::f64::consts::FRAC_PI_2 },
                text_at: [hi[0] + 11.5, (lo[1] + hi[1]) / 2.0],
                measurement: hi[1] - lo[1],
                text: None,
                block: String::new(),
            },
        );
    }
    out
}
