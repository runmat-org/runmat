//! Laying out drawings and flat patterns as [`Sheet`]s, which the DXF and DWG writers share.
//!
//! A drawing sheet carries the frame and the title block (a block, inserted once), each
//! view's visible edges (`VISIBLE`, continuous) and hidden edges (`HIDDEN`, dashed) at the
//! sheet scale, center marks and center lines (`CENTER`), section hatching (`HATCH`, ANSI31),
//! section and detail labels, view labels, and the linear dimensions whose anchors the
//! drawing resolved (`DIMENSIONS`). A flat pattern carries its outline at 1:1 (`OUTLINE`),
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
        if lo[0].is_finite() {
            let box_ = (place(lo), place(hi));
            let mark = hlr["centers"].as_array().into_iter().flatten().find(|c| c["kind"] == "mark").map(|c| pt(&c["at"]));
            annotations(&mut out, name, box_, mark, doc, texts);
        }
        // Linear dimensions between the drawing's resolved anchors, stacked above the view.
        let (u, v) = axes(&orientation(&all_views, name, 0));
        let project = |c: &[f64]| -> [f64; 2] {
            let d = |a: [f64; 3]| a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
            place([d(u), d(v)])
        };
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
    Ok((sheet_name, out, notes))
}

/// Hole callouts and feature control frames as paragraph text by their view, datum symbols as
/// inserts of a `DATUM_<label>` block (the label boxed, on a leg).
fn annotations(out: &mut Sheet, view: &str, box_: ([f64; 2], [f64; 2]), mark: Option<[f64; 2]>, doc: &Value, texts: &Value) {
    let (lo, hi) = box_;
    let mut datums = 0;
    for a in doc["annotations"].as_array().into_iter().flatten() {
        if a["view"].as_str() != Some(view) {
            continue;
        }
        let id = a["id"].as_str().unwrap_or("");
        let text = texts[id].as_str().unwrap_or(id).to_owned();
        match a["kind"]["annotation"].as_str() {
            Some("hole_callout") => {
                let at = mark.map(|m| [m[0] + 6.0, m[1] + 8.0]).unwrap_or([hi[0] + 4.0, hi[1]]);
                out.add("NOTES", Shape::MText { at, height: 2.5, width: 40.0, text });
            }
            Some("gdt") => {
                let cells = text.replace('|', " | ");
                out.add("NOTES", Shape::MText { at: [lo[0] + (hi[0] - lo[0]) * 0.55, hi[1] + 9.0], height: 2.5, width: 40.0, text: cells });
            }
            Some("datum_symbol") => {
                let label = a["kind"]["label"].as_str().unwrap_or("A");
                let name = format!("DATUM_{label}");
                if !out.blocks.iter().any(|b| b.name == name) {
                    let mut b = Sheet::default();
                    b.rect("NOTES", -3.0, -9.0, 6.0, 6.0);
                    b.line("NOTES", [0.0, -3.0], [0.0, 0.0]);
                    b.text("NOTES", [-1.2, -7.3], 2.5, label);
                    out.blocks.push(Block { name: name.clone(), base: [0.0, 0.0], entities: b.entities });
                }
                datums += 1;
                out.add("NOTES", Shape::Insert { block: name, at: [lo[0] + 6.0 * datums as f64, lo[1]], scale: 1.0, rotation: 0.0 });
            }
            _ => {}
        }
    }
}

/// A part's flat pattern as a sheet at 1:1, when it has one.
pub(super) fn flat_pattern(path: &str, text: &str) -> Result<Sheet, ToolError> {
    let part = runmat_cad_wasm::runtime::regenerate_part(path, text)
        .map_err(|_| ToolError::new("not_compiled", format!("{path} has syntax errors")))?;
    for record in part.result.state.records.values() {
        if let runmat_cad::regen::records::FeatureRecord::FlatPattern { layout, .. } = record {
            return Ok(flat_pattern_sheet(&runmat_cad::drawing::flat_pattern::flat_pattern_sheet(layout)));
        }
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
