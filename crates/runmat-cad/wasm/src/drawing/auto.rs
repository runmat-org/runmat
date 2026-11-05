//! `drawing.auto`: a drawing of a part laid out to drafting practice, written into the open
//! `.drawing` as new statements.
//!
//! - **Scale.** The largest preferred scale (ISO 5455: 5:1, 2:1, 1:1, 1:2, 1:5, 1:10) at which
//!   the front, top and right views with their dimension room fit the sheet above its title
//!   block.
//! - **Views.** Front, top (projected up) and right (projected right), third angle, spaced by
//!   the room their dimensions take; a section A–A through the first hole, cut across the top
//!   view, beside it.
//! - **Annotations.** The overall size (width and thickness on the front view, depth on the
//!   top view) between the part's outermost planar faces, and a hole callout on the top view
//!   for every hole.
//!
//! A drawing that has a sheet keeps it and gains the views; an empty file gets an A3 format
//! with a title block and its sheet first.

use super::values::{cylinder, face};
use crate::edit::model;
use crate::runtime::{compile_part, resolve_path, Files};
use runmat_cad::syntax::edit::{EditFailure, SourceChange};
use runmat_cad::syntax::parse_drawing;
use runmat_design_syntax::edit::TextEdit;
use runmat_types::Span;
use serde_json::{Map, Value};

/// Preferred scales, largest first.
const SCALES: &[(u32, u32)] = &[(5, 1), (2, 1), (1, 1), (1, 2), (1, 5), (1, 10)];

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

/// The planar face whose outward normal is `axis`·`sign` and that lies furthest that way.
fn outermost(display: &Value, axis: usize, sign: f64) -> Option<String> {
    let mut best: Option<(f64, String)> = None;
    for body in display["data"]["bodies"].as_array()? {
        for f in body["faces"].as_array().into_iter().flatten() {
            let n = floats(&f["surface"]["normal"]);
            let Some(reference) = f["reference"].as_str() else {
                continue;
            };
            if f["surface"]["surface"] != "plane" || n.len() != 3 || (n[axis] * sign) < 0.999 {
                continue;
            }
            let pos = floats(&f["positions"]);
            let at = pos.iter().skip(axis).step_by(3).sum::<f64>() / (pos.len() / 3).max(1) as f64;
            if best.as_ref().is_none_or(|(b, _)| at * sign > *b) {
                best = Some((at * sign, reference.to_owned()));
            }
        }
    }
    best.map(|(_, r)| r)
}

/// Every hole: a cylindrical face, named, turning all the way round.
fn holes(display: &Value) -> Vec<String> {
    let mut out = Vec::new();
    for body in display["data"]["bodies"].as_array().into_iter().flatten() {
        for f in body["faces"].as_array().into_iter().flatten() {
            let Some(reference) = f["reference"].as_str() else {
                continue;
            };
            if cylinder(&f["surface"]).is_some()
                && reference.contains(".face.wall")
                && face(display, reference).is_some()
            {
                out.push(reference.to_owned());
            }
        }
    }
    out
}

fn stem(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    let s: String = name
        .split('.')
        .next()
        .unwrap_or("drawing")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if s.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
        s
    } else {
        format!("d_{s}")
    }
}

/// Runs `drawing.auto` on the drawing `path`.
pub fn auto(
    path: &str,
    text: &str,
    files: &Files,
    values: &Map<String, Value>,
) -> Result<(SourceChange, String), EditFailure> {
    let model_file = values
        .get("model")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("bracket.part")
        .to_owned();
    if !model_file.ends_with(".part") {
        return Err(model(
            "auto-drawing lays out parts in this build; pick a .part",
        ));
    }
    let at = resolve_path(path, &model_file, files);
    let source = files
        .get(&at)
        .ok_or_else(|| model(format!("{model_file} isn't in the workspace")))?;
    let part =
        compile_part(&at, source).map_err(|_| model(format!("{model_file} doesn't compile")))?;
    let display = serde_json::json!({ "kind": "part", "data": part.display });
    let b = &display["data"]["bounds"];
    let (lo, hi) = (floats(&b["min"]), floats(&b["max"]));
    if lo.len() != 3 || hi.len() != 3 {
        return Err(model(format!("{model_file} has no geometry to draw")));
    }
    let size = [0, 1, 2].map(|i| (hi[i] - lo[i]) * 1000.0);
    // A3 inside its frame: 400 × 277 mm, the title block's 40 mm along the bottom.
    let (room_w, room_h) = (400.0 - 70.0, 277.0 - 40.0 - 70.0);
    let (sn, sd) = *SCALES
        .iter()
        .find(|(n, d)| {
            let s = *n as f64 / *d as f64;
            (size[0] + size[1]) * s <= room_w && (size[2] + size[1]) * s <= room_h
        })
        .unwrap_or(&(1, 10));
    let s = sn as f64 / sd as f64;
    let (x, y, z) = (size[0] * s, size[1] * s, size[2] * s);
    // Front low left with dimension room under and left of it; top above, right beside.
    let (gap, left, bottom) = (30.0, 10.0 + 40.0, 10.0 + 40.0 + 30.0);
    let front = [left + x / 2.0, bottom + z / 2.0];
    let top = [front[0], front[1] + z / 2.0 + gap + y / 2.0];
    let right = [front[0] + x / 2.0 + gap + y / 2.0, front[1]];
    let section = [right[0] + x / 2.0 - y / 2.0, top[1]];
    let file = format!("model = {{ file = \"{model_file}\" }}");
    let parsed = parse_drawing(path, text);
    let document = parsed.document.as_ref();
    let taken: Vec<String> = document
        .map(|d| {
            d.views
                .iter()
                .map(|v| v.name.clone())
                .chain(d.annotations.iter().map(|a| a.id.clone()))
                .collect()
        })
        .unwrap_or_default();
    let fresh = |stem: &str| -> String {
        if !taken.iter().any(|t| t == stem) {
            return stem.to_owned();
        }
        (2..)
            .map(|n| format!("{stem}{n}"))
            .find(|t| !taken.contains(t))
            .expect("a free name")
    };
    let sheet = document
        .and_then(|d| d.sheets.first())
        .map(|s| s.name.clone())
        .unwrap_or_else(|| "1".into());
    let mut lines: Vec<String> = Vec::new();
    if document.is_none_or(|d| d.sheets.is_empty()) {
        lines.push("    format a3 = {\n        size = { size = iso, name = A3 },\n        title_block = [\n            { label = \"Part number\", source = { source = property, name = part_number } },\n            { label = \"Material\", source = { source = property, name = material } },\n            { label = \"Scale\", source = { source = scale } },\n            { label = \"Sheet\", source = { source = sheet_number } }\n        ]\n    }\n".into());
        lines.push(format!("    sheet \"{sheet}\" = {{ format = a3, scale = [{sn}, {sd}], projection = third_angle }}\n"));
    }
    let (vf, vt, vr) = (fresh("front"), fresh("top"), fresh("right"));
    let p = |v: [f64; 2]| format!("[{}, {}]", v[0].round(), v[1].round());
    lines.push(format!("    view {vf} = standard(orientation = front, sheet = \"{sheet}\", {file}, at = {}, display = hidden_lines)", p(front)));
    lines.push(format!("    view {vt} = projected(from = {vf}, direction = up, sheet = \"{sheet}\", {file}, at = {}, display = hidden_lines)", p(top)));
    lines.push(format!("    view {vr} = projected(from = {vf}, direction = right, sheet = \"{sheet}\", {file}, at = {}, display = hidden_lines)", p(right)));
    let holes = holes(&display);
    if let Some(first) = holes.first() {
        let va = fresh("a_a");
        lines.push(format!("    view {va} = section(from = {vt}, plane = {first}, label = A, sheet = \"{sheet}\", {file}, at = {}, display = hidden_lines)", p(section)));
    }
    let mut dims = 0;
    for (axis, view, name) in [
        (0usize, &vf, "width"),
        (2, &vf, "thickness"),
        (1, &vt, "depth"),
    ] {
        if let (Some(a), Some(b)) = (
            outermost(&display, axis, -1.0),
            outermost(&display, axis, 1.0),
        ) {
            lines.push(format!("    annotation {} = dimension(style = linear, references = [{a}, {b}], view = {view})", fresh(name)));
            dims += 1;
        }
    }
    for (i, hole) in holes.iter().enumerate() {
        lines.push(format!(
            "    annotation {} = hole_callout(hole = {hole}, view = {vt})",
            fresh(&format!("hole{}", i + 1))
        ));
    }
    let body = lines.join("\n");
    let (new, edit) = match document {
        Some(_) => {
            let close = text
                .rfind('}')
                .ok_or_else(|| model("the drawing has no closing brace"))?;
            let insert = format!("\n{body}\n");
            let mut new = text.to_owned();
            new.insert_str(close, &insert);
            (
                new,
                TextEdit::replace(
                    Span {
                        start: close,
                        end: close,
                    },
                    insert,
                ),
            )
        }
        None if text.trim().is_empty() => {
            let new = format!(
                "runmat 0.7\nstd 1\n\ndrawing {} {{\n{body}\n}}\n",
                stem(path)
            );
            (
                new.clone(),
                TextEdit::replace(
                    Span {
                        start: 0,
                        end: text.len(),
                    },
                    new,
                ),
            )
        }
        None => return Err(EditFailure::Invalid(parsed.diagnostics.clone())),
    };
    if parse_drawing(path, &new).document.is_none() {
        return Err(model(
            "the auto-drawing didn't read back; this is a bug in the layout",
        ));
    }
    let summary = format!(
        "Drew {model_file} at {sn}:{sd}: 3 views{}, {dims} dimensions, {} hole callout{}",
        if holes.is_empty() {
            ""
        } else {
            " and section A-A"
        },
        holes.len(),
        if holes.len() == 1 { "" } else { "s" }
    );
    Ok((
        SourceChange {
            edits: vec![edit],
            source: new,
            reprinted: false,
        },
        summary,
    ))
}
