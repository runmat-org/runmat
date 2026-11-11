//! Drawings and flat patterns as [`Sheet`]s, which the PDF, DXF and DWG writers share.
//!
//! A drawing sheet draws the layout the compile made (`layouts`, runmat-cad-wasm
//! `drawing::layout`), the one the Desktop's sheet draws: the frame with its zones, the title
//! block (a block, inserted once), each view's visible edges (`VISIBLE`, continuous) and hidden
//! edges (`HIDDEN`, dashed), center marks and lines (`CENTER`), section hatching (`HATCH`),
//! view labels, linear dimensions as dimension entities (`DIMENSIONS`), the callouts on their
//! leaders and the notes (`NOTES`), and the tables (`TABLES`). A flat pattern carries its
//! outline at 1:1 (`OUTLINE`), bend lines with their notes, and its overall width and height as
//! dimensions.

use crate::formats::sheet::{standard_linetypes, Block, Dimension, DimensionKind, Entity, Layer, Shape, Sheet};
use crate::{Files, ToolError};
use runmat_cad::drawing::flat_pattern::{FlatPatternSheet, SheetCurve};
use runmat_cad::model::sheet_metal::BendDirection;
use serde_json::Value;

use super::floats;

fn layer(name: &str, color: i16, linetype: &str) -> Layer {
    Layer { name: name.into(), color, linetype: linetype.into() }
}

/// The export layer a layout primitive draws on.
fn layer_of(p: &Value, dimension: bool) -> &'static str {
    match p["style"].as_str().unwrap_or("") {
        "visible" | "thick" => "VISIBLE",
        "hidden" | "benddown" => "HIDDEN",
        "center" | "bendup" => "CENTER",
        "hatch" => "HATCH",
        _ if p["layer"] == "Frame" => "FRAME",
        _ if dimension => "DIMENSIONS",
        _ if p["kind"] == "table" => "TABLES",
        _ => "NOTES",
    }
}

fn pairs(points: &[f64]) -> Vec<[f64; 2]> {
    points.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
}

/// A layout primitive as sheet shapes: a rect as a closed polyline, a filled region as a
/// solid hatch, text from its insertion point at the left baseline.
fn shapes_of(p: &Value) -> Vec<Shape> {
    let s = &p["shape"];
    let n = |k: &str| s[k].as_f64().unwrap_or(0.0);
    match s["type"].as_str().unwrap_or("") {
        "rect" => {
            let (x, y, w, h) = (n("x"), n("y"), n("w"), n("h"));
            vec![Shape::Polyline { points: vec![[x, y], [x + w, y], [x + w, y + h], [x, y + h]], closed: true }]
        }
        "circle" => vec![Shape::Circle { center: [n("x"), n("y")], radius: n("r") }],
        "polyline" => {
            let pts = pairs(&floats(&s["points"]));
            let closed = s["closed"].as_bool().unwrap_or(false);
            match pts.len() {
                0 | 1 => Vec::new(),
                2 if !closed => vec![Shape::Line { a: pts[0], b: pts[1] }],
                _ => vec![Shape::Polyline { points: pts, closed }],
            }
        }
        "polygon" => {
            let loops: Vec<Vec<[f64; 2]>> = s["rings"].as_array().into_iter().flatten().map(|r| pairs(&floats(r))).filter(|l| l.len() >= 3).collect();
            if loops.is_empty() { Vec::new() } else { vec![Shape::Hatch { loops, pattern: "SOLID".into(), angle: 0.0, scale: 1.0 }] }
        }
        "text" => {
            let text = s["text"].as_str().unwrap_or("").to_owned();
            let height = n("size");
            let rotation = n("rotation").to_radians();
            let w = text.chars().count() as f64 * height * 0.6;
            let back = match s["anchor"].as_str() {
                Some("middle") => w / 2.0,
                Some("end") => w,
                _ => 0.0,
            };
            let (sin, cos) = rotation.sin_cos();
            vec![Shape::Text { at: [n("x") - back * cos, n("y") - back * sin], height, rotation, text }]
        }
        _ => Vec::new(),
    }
}

/// One drawing sheet, as the compile laid it out (`layouts`: the same primitives the Desktop's
/// sheet draws). The title block is one block inserted at the sheet origin; each linear
/// dimension is a native dimension entity whose block holds the lines, arrows and figure the
/// layout drew for it; everything else draws on its layer. Returns the sheet's name, the sheet
/// and notes (the layout's findings among them).
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
    let sheets = data["document"]["sheets"].as_array().cloned().unwrap_or_default();
    let Some(sheet) = sheets.iter().find(|s| sheet.is_none_or(|n| s["name"].as_str() == Some(n))) else {
        // A drawing that doesn't compile has no sheets: say why, not which sheet is missing.
        let errors: Vec<String> = compiled["diagnostics"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|d| d["severity"] == "error")
            .filter_map(|d| d["message"].as_str().map(str::to_owned))
            .collect();
        if !errors.is_empty() {
            return Err(ToolError::invalid(format!("{path} does not compile: {}", errors.join("; "))));
        }
        return Err(ToolError::invalid(match sheet {
            Some(name) => format!("{path} has no sheet {name}"),
            None => format!("{path} has no sheets"),
        }));
    };
    let sheet_name = sheet["name"].as_str().unwrap_or("1").to_owned();
    let layout = &data["layouts"][&sheet_name];
    let size = floats(&layout["size"]);
    let mut out = Sheet {
        paper: [size.first().copied().unwrap_or(420.0), size.get(1).copied().unwrap_or(297.0)],
        layers: vec![
            layer("FRAME", 7, "CONTINUOUS"),
            layer("VISIBLE", 7, "CONTINUOUS"),
            layer("HIDDEN", 8, "DASHED"),
            layer("CENTER", 4, "CENTER"),
            layer("HATCH", 8, "CONTINUOUS"),
            layer("DIMENSIONS", 3, "CONTINUOUS"),
            layer("NOTES", 3, "CONTINUOUS"),
            layer("TABLES", 7, "CONTINUOUS"),
        ],
        linetypes: standard_linetypes(),
        ..Sheet::default()
    };
    let mut notes: Vec<String> = layout["notes"].as_array().into_iter().flatten().filter_map(|n| n.as_str().map(str::to_owned)).collect();
    notes.extend(layout["findings"].as_array().into_iter().flatten().filter_map(|f| f["message"].as_str().map(str::to_owned)));
    let dims: Vec<&Value> = layout["dimensions"].as_array().into_iter().flatten().collect();
    let dim_ids: Vec<String> = dims.iter().map(|d| format!("annotation:{}", d["id"].as_str().unwrap_or(""))).collect();
    let mut title = Sheet::default();
    let mut blocks: Vec<Sheet> = dims.iter().map(|_| Sheet::default()).collect();
    for p in layout["primitives"].as_array().into_iter().flatten() {
        // The sheet's paper, the edge picks and the glyph lines of symbols exports write as text.
        let (kind, style) = (p["kind"].as_str().unwrap_or(""), p["style"].as_str().unwrap_or(""));
        if kind == "sheet" || style == "pick" || kind == "glyph" && style != "glyphtext" {
            continue;
        }
        let id = p["id"].as_str().unwrap_or("");
        let dim = dim_ids.iter().position(|d| d == id);
        let layer = layer_of(p, dim.is_some());
        for shape in shapes_of(p) {
            match dim {
                _ if id == "title" => title.add("FRAME", shape),
                Some(k) => blocks[k].add("DIMENSIONS", shape),
                None => out.add(layer, shape),
            }
        }
    }
    if !title.entities.is_empty() {
        out.blocks.push(Block { name: "TITLE_BLOCK".into(), base: [0.0, 0.0], entities: title.entities });
        out.add("FRAME", Shape::Insert { block: "TITLE_BLOCK".into(), at: [0.0, 0.0], scale: 1.0, rotation: 0.0 });
    }
    for (d, block) in dims.iter().zip(blocks) {
        let pair = |k: &str| { let f = floats(&d[k]); [f.first().copied().unwrap_or(0.0), f.get(1).copied().unwrap_or(0.0)] };
        let name = format!("*D{}", out.blocks.iter().filter(|b| b.name.starts_with("*D")).count() + 1);
        out.blocks.push(Block { name: name.clone(), base: [0.0, 0.0], entities: block.entities });
        out.add(
            "DIMENSIONS",
            Shape::Dimension(Dimension {
                kind: DimensionKind::Linear { from: pair("from"), to: pair("to"), line: pair("line"), rotation: d["rotation"].as_f64().unwrap_or(0.0) },
                text_at: pair("text_at"),
                measurement: d["measurement"].as_f64().unwrap_or(0.0),
                text: d["text"].as_str().map(str::to_owned),
                block: name,
            }),
        );
    }
    Ok((sheet_name, out, notes))
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
