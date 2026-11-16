//! `drawing.check`: a drawing checked against the models it shows and against drafting rules.
//! The drawing is compiled and laid out as the sheet draws it, and each rule reads that result.
//! Every finding names the view, annotation, table or sheet it is about, so selecting its row
//! selects that item on the sheet and in the Outline.
//!
//! | Rule | Finds | Source |
//! |---|---|---|
//! | Reference | an annotation whose model references no longer resolve (a dimension to a face the model lost), and the drawing's own reference errors | ISO 129-1: a dimension refers to a feature of the part |
//! | Model | a view whose model file is missing, doesn't compile or builds no solid, so the sheet draws no model lines for it | |
//! | Title block | an empty title-block field, and a block without a title or an identification number | ISO 7200 identification fields |
//! | Overlap | views that overlap, a table over a view, annotation texts drawn over each other or over a view's lines, and what the layout reports (no room, crowded, not aligned) | ISO 128-1 legibility; ISO 129-1 values not crossed |
//! | Scale | a sheet or view scale outside the preferred scales, a projected view at another scale than its parent, a detail view that doesn't enlarge | ISO 5455 (ASME Y14.1 inch scales on inch ASME sheets), ISO 5456-2 |

use super::{Analysis, Row};
use crate::runtime::Files;
use runmat_cad::drawing::standard::DrawingStandard;
use runmat_cad::drawing::units::DrawingUnits;
use runmat_cad::drawing::{DrawingDocument, ViewKind};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

/// One finding: its rule, the Outline item it names, its text and severity.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub rule: &'static str,
    pub item: String,
    pub message: String,
    pub severity: &'static str,
}

/// The standard the check reads scales by: `values.standard` (`iso`, `asme`), else the sheets'.
fn standard_of(document: &DrawingDocument, values: &Map<String, Value>) -> DrawingStandard {
    match values.get("standard").and_then(Value::as_str) {
        Some("iso") => DrawingStandard::Iso,
        Some("asme") => DrawingStandard::Asme,
        _ => document
            .formats
            .iter()
            .map(|f| f.standard)
            .find(|s| *s == DrawingStandard::Asme)
            .unwrap_or_default(),
    }
}

/// Whether `n:m` is a preferred scale. ISO 5455: 1, 2 and 5 times a power of ten, enlarging or
/// reducing. Inch drawings to ASME Y14.1 also use the binary ratios (1:4, 1:8, 1:16, 4:1, 8:1),
/// and architectural sheets in feet and inches the inch-to-foot scales (1:12 … 1:384).
pub fn preferred_scale((n, m): (u32, u32), inch: bool, feet: bool) -> bool {
    let g = gcd(n, m).max(1);
    let (n, m) = (n / g, m / g);
    let decade = |v: u32| {
        let mut v = v;
        while v >= 10 && v.is_multiple_of(10) {
            v /= 10;
        }
        matches!(v, 1 | 2 | 5)
    };
    if n == 1 && decade(m) || m == 1 && decade(n) {
        return true;
    }
    if inch && (n == 1 && matches!(m, 4 | 8 | 16 | 32) || m == 1 && matches!(n, 4 | 8 | 16)) {
        return true;
    }
    feet && n == 1 && matches!(m, 12 | 16 | 24 | 48 | 96 | 192 | 384)
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn ratio(s: (u32, u32)) -> f64 {
    s.0 as f64 / s.1.max(1) as f64
}

fn scale_text(s: (u32, u32)) -> String {
    format!("{}:{}", s.0, s.1)
}

/// The Outline item a name stands for in `document`.
fn item_of(document: &DrawingDocument, name: &str, sheet: &str) -> String {
    if name.contains(':') {
        return name.to_owned();
    }
    if document.views.iter().any(|v| v.name == name) {
        return format!("view:{name}");
    }
    if document.annotations.iter().any(|a| a.id == name) {
        return format!("annotation:{name}");
    }
    if document.tables.iter().any(|t| t.name == name) {
        return format!("table:{name}");
    }
    format!("sheet:{}", if name.is_empty() { sheet } else { name })
}

/// How the results table names an item: `front view`, `width annotation`, `sheet 1`.
fn label_of(item: &str) -> String {
    match item.split_once(':') {
        Some(("view", n)) => format!("{n} view"),
        Some(("annotation", n)) => format!("{n} annotation"),
        Some(("table", n)) => format!("{n} table"),
        Some(("sheet", n)) => format!("sheet {n}"),
        _ => item.to_owned(),
    }
}

/// A box on the sheet: min x, min y, max x, max y (mm).
type Bx = [f64; 4];

fn grow(b: &mut Option<Bx>, x: f64, y: f64) {
    let n = b.get_or_insert([x, y, x, y]);
    *n = [n[0].min(x), n[1].min(y), n[2].max(x), n[3].max(y)];
}

/// How far two boxes overlap along x and y (negative when apart).
fn overlap(a: &Bx, b: &Bx) -> (f64, f64) {
    (
        a[2].min(b[2]) - a[0].max(b[0]),
        a[3].min(b[3]) - a[1].max(b[1]),
    )
}

/// Whether the segment a→b runs inside box `g` (Liang–Barsky clipping).
fn crosses_box(a: [f64; 2], b: [f64; 2], g: &Bx) -> bool {
    if g[0] >= g[2] || g[1] >= g[3] {
        return false;
    }
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for (p, d, lo, hi) in [
        (a[0], b[0] - a[0], g[0], g[2]),
        (a[1], b[1] - a[1], g[1], g[3]),
    ] {
        if d.abs() < 1e-12 {
            if p < lo || p > hi {
                return false;
            }
        } else {
            let (u, v) = ((lo - p) / d, (hi - p) / d);
            t0 = t0.max(u.min(v));
            t1 = t1.min(u.max(v));
        }
    }
    t1 - t0 > 1e-9
}

/// A text primitive's box, as the layout measures it (`drawing::layout::solver::text_box`):
/// each figure 0.6 of the height wide, from just below the baseline to just above the cap
/// height, turned with the text.
fn text_box(shape: &Value) -> Option<Bx> {
    let (x, y, size) = (
        shape["x"].as_f64()?,
        shape["y"].as_f64()?,
        shape["size"].as_f64()?,
    );
    let w = shape["text"].as_str()?.chars().count() as f64 * size * 0.6;
    let start = match shape["anchor"].as_str() {
        Some("middle") => -w / 2.0,
        Some("end") => -w,
        _ => 0.0,
    };
    let (lo, hi) = (-size * 0.2, size * 1.05);
    let quarter = shape["rotation"]
        .as_f64()
        .is_some_and(|r| ((r.abs() % 180.0) - 90.0).abs() < 1.0);
    Some(if quarter {
        [x - hi, y + start, x - lo, y + start + w]
    } else {
        [x + start, y + lo, x + start + w, y + hi]
    })
}

/// The overlap rules on one laid-out sheet.
fn overlaps(sheet: &str, layout: &Value, out: &mut Vec<Finding>) {
    let mut geometry: BTreeMap<String, Option<Bx>> = BTreeMap::new();
    let mut texts: Vec<(String, Bx)> = Vec::new();
    // Each view's model lines (visible, hidden, tangent), as polylines on the sheet.
    let mut lines: Vec<(String, Vec<f64>)> = Vec::new();
    for p in layout["primitives"].as_array().into_iter().flatten() {
        let (Some(id), Some(kind)) = (p["id"].as_str(), p["kind"].as_str()) else {
            continue;
        };
        let shape = &p["shape"];
        let ty = shape["type"].as_str().unwrap_or("");
        if ty == "text" {
            if kind == "annotation" && id.starts_with("annotation:") {
                if let Some(b) = text_box(shape) {
                    texts.push((id.to_owned(), b));
                }
            }
            continue;
        }
        // A view's model lines and a table's grid: their extent on the sheet.
        let owner = if (kind == "view" && id.starts_with("view:")) || id.starts_with("table:") {
            id.to_owned()
        } else {
            continue;
        };
        if kind == "view"
            && ty == "polyline"
            && matches!(p["style"].as_str(), Some("visible" | "hidden" | "tangent"))
        {
            if let Some(a) = shape["points"].as_array() {
                lines.push((owner.clone(), a.iter().filter_map(Value::as_f64).collect()));
            }
        }
        let b = geometry.entry(owner).or_default();
        let nums: Vec<f64> = match ty {
            "polyline" => shape["points"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_f64).collect())
                .unwrap_or_default(),
            "rect" => {
                let f = |k: &str| shape[k].as_f64().unwrap_or(0.0);
                vec![f("x"), f("y"), f("x") + f("w"), f("y") + f("h")]
            }
            _ => Vec::new(),
        };
        for c in nums.chunks(2) {
            if let [x, y] = c {
                grow(b, *x, *y);
            }
        }
    }
    let boxes: Vec<(&String, Bx)> = geometry
        .iter()
        .filter_map(|(id, b)| Some((id, (*b)?)))
        .collect();
    // Views and tables: more than half a millimetre into each other both ways.
    for (i, (a, ba)) in boxes.iter().enumerate() {
        for (b, bb) in &boxes[i + 1..] {
            let (dx, dy) = overlap(ba, bb);
            if dx > 0.5 && dy > 0.5 {
                let both = format!("{} and {}", label_of(a), label_of(b));
                out.push(Finding {
                    rule: "Overlap",
                    item: (*a).clone(),
                    message: format!(
                        "On sheet {sheet}, the {both} overlap by {dx:.1} × {dy:.1} mm; move one clear"
                    ),
                    severity: "error",
                });
            }
        }
    }
    // Annotation texts drawn over a view's model lines (a dimension figure on a hole).
    for (a, ba) in &texts {
        // A hair inside the box: a line that only touches its edge doesn't cross the text.
        let g = [ba[0] + 0.2, ba[1] + 0.2, ba[2] - 0.2, ba[3] - 0.2];
        let mut said: Vec<&String> = Vec::new();
        for (view, pts) in &lines {
            if said.contains(&view) {
                continue;
            }
            let crosses = pts
                .chunks_exact(2)
                .collect::<Vec<_>>()
                .windows(2)
                .any(|w| crosses_box([w[0][0], w[0][1]], [w[1][0], w[1][1]], &g));
            if crosses {
                said.push(view);
                out.push(Finding {
                    rule: "Overlap",
                    item: a.clone(),
                    message: format!(
                        "The text of the {} is drawn over the {}'s lines; move it clear of the model",
                        label_of(a),
                        label_of(view)
                    ),
                    severity: "warning",
                });
            }
        }
    }
    // Annotation texts of different annotations drawn over each other.
    let mut said: Vec<(String, String)> = Vec::new();
    for (i, (a, ba)) in texts.iter().enumerate() {
        for (b, bb) in &texts[i + 1..] {
            if a == b || said.iter().any(|(x, y)| x == a && y == b) {
                continue;
            }
            let (dx, dy) = overlap(ba, bb);
            if dx > 0.2 && dy > 0.2 {
                said.push((a.clone(), b.clone()));
                out.push(Finding {
                    rule: "Overlap",
                    item: a.clone(),
                    message: format!(
                        "The texts of the {} and the {} are drawn over each other; move one",
                        label_of(a),
                        label_of(b)
                    ),
                    severity: "warning",
                });
            }
        }
    }
}

/// What the layout found, as overlap findings; those it settled itself are notes.
fn layout_findings(
    document: &DrawingDocument,
    sheet: &str,
    layout: &Value,
    out: &mut Vec<Finding>,
) {
    for f in layout["findings"].as_array().into_iter().flatten() {
        let code = f["code"].as_str().unwrap_or("");
        let settled = matches!(code, "view_moved" | "note_moved" | "duplicate_dimension");
        out.push(Finding {
            rule: "Overlap",
            item: item_of(document, f["item"].as_str().unwrap_or(""), sheet),
            message: f["message"].as_str().unwrap_or("").to_owned(),
            severity: if settled { "info" } else { "warning" },
        });
    }
}

/// The scale rules.
fn scales(document: &DrawingDocument, standard: DrawingStandard, out: &mut Vec<Finding>) {
    let units = |sheet: &str| {
        document
            .sheets
            .iter()
            .find(|s| s.name == sheet)
            .and_then(|s| document.formats.iter().find(|f| f.name == s.format))
            .map(|f| f.units)
            .unwrap_or_default()
    };
    let preferred = |s: (u32, u32), sheet: &str| {
        let u = units(sheet);
        let feet = u == DrawingUnits::FtIn;
        let inch = standard == DrawingStandard::Asme && !u.is_mm();
        preferred_scale(s, inch, feet)
    };
    let which = if standard == DrawingStandard::Asme {
        "an ASME Y14.1"
    } else {
        "an ISO 5455"
    };
    for s in &document.sheets {
        if !preferred(s.scale, &s.name) {
            out.push(Finding {
                rule: "Scale",
                item: format!("sheet:{}", s.name),
                message: format!(
                    "Sheet {} is at {}, which isn't {which} scale (1:2, 1:5, 1:10, 2:1, 5:1 …)",
                    s.name,
                    scale_text(s.scale)
                ),
                severity: "warning",
            });
        }
    }
    let drawn = |name: &str| -> Option<(u32, u32)> {
        let v = document.view(name)?;
        v.scale.or_else(|| {
            document
                .sheets
                .iter()
                .find(|s| s.name == v.sheet)
                .map(|s| s.scale)
        })
    };
    for v in &document.views {
        let item = format!("view:{}", v.name);
        if let Some(own) = v.scale {
            if !preferred(own, &v.sheet) {
                out.push(Finding {
                    rule: "Scale",
                    item: item.clone(),
                    message: format!(
                        "The {} view is at {}, which isn't {which} scale",
                        v.name,
                        scale_text(own)
                    ),
                    severity: "warning",
                });
            }
        }
        let (Some(here), Some(parent)) = (drawn(&v.name), v.kind.parent()) else {
            continue;
        };
        let Some(there) = drawn(parent) else { continue };
        match &v.kind {
            ViewKind::Projected { .. } if (ratio(here) - ratio(there)).abs() > 1e-9 => {
                out.push(Finding {
                    rule: "Scale",
                    item,
                    message: format!(
                        "The {} view is projected from {parent} but drawn at {} where {parent} is at {}: views in projection share one scale (ISO 5456-2)",
                        v.name,
                        scale_text(here),
                        scale_text(there)
                    ),
                    severity: "warning",
                });
            }
            ViewKind::Detail { .. } if ratio(here) <= ratio(there) + 1e-9 => {
                out.push(Finding {
                    rule: "Scale",
                    item,
                    message: format!(
                        "The {} detail is at {}, no larger than {parent} at {}: a detail view enlarges",
                        v.name,
                        scale_text(here),
                        scale_text(there)
                    ),
                    severity: "warning",
                });
            }
            _ => {}
        }
    }
}

/// The title-block rules: each sheet's filled rows (a logo cell holds a picture, not text).
fn title_blocks(document: &DrawingDocument, data: &Value, out: &mut Vec<Finding>) {
    let pictures: Vec<&str> = document
        .formats
        .iter()
        .flat_map(|f| &f.title_block)
        .filter(|t| matches!(t.source, runmat_cad::drawing::FieldSource::Image { .. }))
        .map(|t| t.label.as_str())
        .collect();
    let Some(blocks) = data["title_blocks"].as_object() else {
        return;
    };
    for (sheet, rows) in blocks {
        let rows: Vec<(String, String)> = rows
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|r| {
                Some((
                    r[0].as_str()?.trim().to_owned(),
                    r[1].as_str()?.trim().to_owned(),
                ))
            })
            .collect();
        if rows.is_empty() {
            continue;
        }
        let item = format!("sheet:{sheet}");
        for (label, text) in &rows {
            // The sheet prints a dash for a field the model doesn't fill.
            if (text.is_empty() || text == "—" || text == "-")
                && !pictures.contains(&label.as_str())
            {
                out.push(Finding {
                    rule: "Title block",
                    item: item.clone(),
                    message: format!(
                        "Sheet {sheet}'s title block has no {label}: fill it in, or set the property it reads in the model"
                    ),
                    severity: "warning",
                });
            }
        }
        let has = |words: &[&str]| {
            rows.iter().any(|(l, _)| {
                let l = l.to_lowercase();
                words.iter().any(|w| l.contains(w))
            })
        };
        if !has(&["title"]) {
            out.push(Finding {
                rule: "Title block",
                item: item.clone(),
                message: format!("Sheet {sheet}'s title block has no Title field (ISO 7200)"),
                severity: "warning",
            });
        }
        if !has(&["number", "identification", "drawing no", "part no"]) {
            out.push(Finding {
                rule: "Title block",
                item,
                message: format!(
                    "Sheet {sheet}'s title block has no identification number, such as a part or drawing number (ISO 7200)"
                ),
                severity: "warning",
            });
        }
    }
}

/// The reference and model rules.
fn references(document: &DrawingDocument, answer: &Value, data: &Value, out: &mut Vec<Finding>) {
    for a in &document.annotations {
        if data["resolved"][&a.id] == Value::Bool(false) {
            let kind = serde_json::to_value(&a.kind)
                .ok()
                .and_then(|k| k["annotation"].as_str().map(|s| s.replace('_', " ")))
                .unwrap_or_else(|| "annotation".into());
            out.push(Finding {
                rule: "Reference",
                item: format!("annotation:{}", a.id),
                message: format!(
                    "The {} {kind} on the {} view refers to geometry the model no longer has; reattach it or delete it",
                    a.id, a.view
                ),
                severity: "error",
            });
        }
    }
    // The drawing's own reference errors (a view drawn from a view that's gone, an occurrence
    // path the assembly lost), each on the item it names.
    for d in data["diagnostics"].as_array().into_iter().flatten() {
        let severity = match d["severity"].as_str() {
            Some("error") => "error",
            Some("warning") => "warning",
            _ => continue,
        };
        let name = d["site"]["subject"]
            .as_object()
            .and_then(|o| o.values().find_map(|v| v["name"].as_str()))
            .unwrap_or("");
        // A title-block field the model doesn't fill is the title-block rule's.
        let message = d["message"].as_str().unwrap_or("");
        if name.starts_with("sheet.") || message.starts_with("title block") {
            continue;
        }
        // The item the message names first: `annotation `lost` on view `top``.
        let named = message
            .split('`')
            .skip(1)
            .step_by(2)
            .find(|w| !w.contains(':') && item_of(document, w, "") != format!("sheet:{w}"));
        let item = item_of(document, named.unwrap_or(name), "");
        // An unresolved annotation is reported once, above.
        if item.starts_with("annotation:") && out.iter().any(|f| f.item == item) {
            continue;
        }
        let item = if item == "sheet:" {
            document
                .sheets
                .first()
                .map(|s| format!("sheet:{}", s.name))
                .unwrap_or_default()
        } else {
            item
        };
        out.push(Finding {
            rule: "Reference",
            item,
            message: d["message"].as_str().unwrap_or("").to_owned(),
            severity,
        });
    }
    // A model the compile couldn't read: each view showing it.
    for p in answer["diagnostics"].as_array().into_iter().flatten() {
        let message = p["message"].as_str().unwrap_or("");
        for v in &document.views {
            let file = v.model.file.as_str();
            if message.starts_with(&format!("{file} ")) {
                out.push(Finding {
                    rule: "Model",
                    item: format!("view:{}", v.name),
                    message: format!("The {} view no longer shows its model: {message}", v.name),
                    severity: "error",
                });
            }
        }
    }
}

/// Every finding on the drawing at `path`.
pub fn findings(
    path: &str,
    text: &str,
    files: &Files,
    values: &Map<String, Value>,
) -> Result<(Vec<Finding>, DrawingStandard), String> {
    let scope = crate::runtime::module_scope(path, text, files);
    let parsed = runmat_cad::syntax::parse_drawing_in(path, text, &scope);
    // A drawing that doesn't read (a view naming a model that isn't there) is checked no
    // further: each reason is a finding.
    let Some(document) = parsed.document else {
        let found = parsed
            .diagnostics
            .iter()
            .map(|d| Finding {
                rule: "Model",
                item: String::new(),
                message: format!("The drawing doesn't read: {}", d.message),
                severity: "error",
            })
            .collect();
        return Ok((found, DrawingStandard::default()));
    };
    let mut all = files.clone();
    all.insert(path.to_owned(), text.to_owned());
    let answer: Value = serde_json::from_str(&crate::drawing::compile_json(path, text, &all))
        .map_err(|e| format!("the drawing's compile isn't JSON: {e}"))?;
    let data = &answer["projection"]["data"];
    let standard = standard_of(&document, values);
    let mut out = Vec::new();
    references(&document, &answer, data, &mut out);
    title_blocks(&document, data, &mut out);
    for s in &document.sheets {
        let layout = &data["layouts"][&s.name];
        overlaps(&s.name, layout, &mut out);
        layout_findings(&document, &s.name, layout, &mut out);
    }
    // A view the sheet draws no model lines for no longer shows its model (a part that
    // doesn't build, or builds no solid).
    for v in &document.views {
        let item = format!("view:{}", v.name);
        if out.iter().any(|f| f.rule == "Model" && f.item == item) {
            continue;
        }
        let drawn = data["layouts"][&v.sheet]["primitives"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|p| {
                p["id"] == item.as_str() && p["kind"] == "view" && p["shape"]["type"] == "polyline"
            });
        if !drawn {
            out.push(Finding {
                rule: "Model",
                item,
                message: format!(
                    "The {} view draws no model lines: {} has no solid to show (it doesn't build, or builds nothing)",
                    v.name,
                    v.model.file.as_str()
                ),
                severity: "error",
            });
        }
    }
    scales(&document, standard, &mut out);
    // Errors first, then warnings, then notes; each rule's findings together.
    let rank = |s: &str| match s {
        "error" => 0,
        "warning" => 1,
        _ => 2,
    };
    out.sort_by_key(|f| rank(f.severity));
    Ok((out, standard))
}

/// `drawing.check` as the results table under the sheet.
pub fn analysis(
    path: &str,
    text: &str,
    files: &Files,
    values: &Map<String, Value>,
) -> Result<Analysis, String> {
    let (found, standard) = findings(path, text, files, values)?;
    let count = |s: &str| found.iter().filter(|f| f.severity == s).count();
    let (errors, warnings, notes) = (count("error"), count("warning"), count("info"));
    let plural = |n: usize, one: &str| format!("{n} {one}{}", if n == 1 { "" } else { "s" });
    let summary = if found.is_empty() {
        "No findings: the references resolve, the title block is filled, nothing overlaps and the scales are standard".to_owned()
    } else {
        [(errors, "error"), (warnings, "warning"), (notes, "note")]
            .iter()
            .filter(|(n, _)| *n > 0)
            .map(|(n, w)| plural(*n, w))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let rows = found
        .iter()
        .map(|f| Row {
            cells: vec![label_of(&f.item), f.rule.to_owned(), f.message.clone()],
            items: vec![f.item.clone()],
            severity: Some(f.severity),
            legend: None,
            reference: None,
        })
        .collect();
    Ok(Analysis {
        title: "Drawing check".into(),
        summary,
        columns: vec!["Item".into(), "Rule".into(), "Finding".into()],
        rows,
        problems: Vec::new(),
        data: json!({
            "findings": found.iter().map(|f| json!({
                "rule": f.rule, "item": f.item, "message": f.message, "severity": f.severity,
            })).collect::<Vec<_>>(),
        }),
        settings: vec![format!(
            "Standard: {}",
            if standard == DrawingStandard::Asme {
                "ASME"
            } else {
                "ISO"
            }
        )],
        ..Analysis::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PART: &str = "runmat 0.7\nstd 1\n\npart bracket {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output bracket = { body = feature.body.body, part_number = \"BR-100\" }\n}\n";

    const PATH: &str = "/p/bracket.drawing";

    fn files() -> Files {
        [("/p/bracket.part".to_string(), PART.to_string())].into()
    }

    fn sheet(body: &str, scale: &str) -> String {
        format!(
            "runmat 0.7\nstd 1\n\ndrawing bracket {{\n    format a3 = {{\n        size = {{ size = iso, name = A3 }},\n        title_block = [\n            {{ label = \"Title\", source = {{ source = literal, text = \"Bracket\" }} }},\n            {{ label = \"Part number\", source = {{ source = property, name = part_number }} }},\n            {{ label = \"Material\", source = {{ source = property, name = material }} }}\n        ]\n    }}\n    sheet \"1\" = {{ format = a3, scale = {scale}, projection = third_angle }}\n{body}}}\n"
        )
    }

    const FRONT: &str = "    view front = standard(orientation = front, sheet = \"1\", model = { part = bracket }, at = [100, 100], display = visible_lines)\n";

    fn check(text: &str) -> Vec<Finding> {
        findings(PATH, text, &files(), &Map::new()).unwrap().0
    }

    fn rules(found: &[Finding]) -> Vec<(&'static str, String)> {
        found.iter().map(|f| (f.rule, f.item.clone())).collect()
    }

    #[test]
    fn a_clean_drawing_has_only_its_empty_material_field() {
        let text = sheet(
            &format!("{FRONT}    annotation width = dimension(style = linear, references = [feature.body.face.side(s1.l4), feature.body.face.side(s1.l2)], view = front)\n"),
            "[1, 1]",
        );
        let found = check(&text);
        // The part sets no material: that field is empty, and nothing else is found.
        assert_eq!(
            rules(&found),
            vec![("Title block", "sheet:1".to_string())],
            "{found:#?}"
        );
        assert!(
            found[0].message.contains("no Material"),
            "{}",
            found[0].message
        );
        let a = analysis(PATH, &text, &files(), &Map::new()).unwrap();
        assert_eq!(a.columns, vec!["Item", "Rule", "Finding"]);
        assert_eq!(a.rows[0].items, vec!["sheet:1"]);
        assert_eq!(a.summary, "1 warning");
    }

    #[test]
    fn a_dimension_to_a_lost_face_is_an_error_on_that_dimension() {
        // s1.l9 isn't a line of the sketch: the face it names doesn't exist.
        let text = sheet(
            &format!("{FRONT}    annotation width = dimension(style = linear, references = [feature.body.face.side(s1.l4), feature.body.face.side(s1.l9)], view = front)\n"),
            "[1, 1]",
        );
        let found = check(&text);
        let lost = found
            .iter()
            .find(|f| f.item == "annotation:width")
            .unwrap_or_else(|| panic!("{found:#?}"));
        assert_eq!((lost.rule, lost.severity), ("Reference", "error"));
        // Reported once, on the dimension: the index's own diagnostic for it isn't repeated.
        assert_eq!(
            found.iter().filter(|f| f.rule == "Reference").count(),
            1,
            "{found:#?}"
        );
        assert_eq!(found[0].severity, "error", "errors come first");
    }

    #[test]
    fn a_model_that_doesnt_build_or_isnt_there_is_a_model_finding() {
        // The part names a sketch it doesn't have, so it doesn't build: the view is out of date.
        let mut files = files();
        files.insert(
            "/p/broken.part".into(),
            "runmat 0.7\nstd 1\n\npart broken {\n    feature body = extrude(profile = sketch.nope, distance = 10 mm)\n}\n".into(),
        );
        let text = sheet(
            "    view front = standard(orientation = front, sheet = \"1\", model = { part = broken }, at = [100, 100], display = visible_lines)\n",
            "[1, 1]",
        );
        let found = findings(PATH, &text, &files, &Map::new()).unwrap().0;
        assert!(
            found
                .iter()
                .any(|f| f.rule == "Model" && f.severity == "error"),
            "{found:#?}"
        );
        // A model that isn't in the folder at all: the drawing doesn't read.
        let text = sheet(
            "    view front = standard(orientation = front, sheet = \"1\", model = { part = gone }, at = [100, 100], display = visible_lines)\n",
            "[1, 1]",
        );
        let found = check(&text);
        assert!(
            found
                .iter()
                .any(|f| f.rule == "Model" && f.message.contains("gone")),
            "{found:#?}"
        );
    }

    #[test]
    fn overlapping_views_and_scales_off_the_standard_are_found() {
        // The top view sits on the front; the sheet is at 1:3; the projected top at 2:1 on a
        // 1:3 front; a detail at 1:3, no larger than its parent.
        let body = format!(
            "{FRONT}    view top = projected(from = front, direction = up, sheet = \"1\", model = {{ part = bracket }}, at = [105, 102], scale = [2, 1], display = visible_lines)\n    view d = detail(from = front, center = edges_between(feature.body.face.side(s1.l1), feature.body.face.end), radius = 4 mm, label = D, sheet = \"1\", model = {{ part = bracket }}, at = [300, 200], display = visible_lines)\n"
        );
        let found = check(&sheet(&body, "[1, 3]"));
        let has = |rule: &str, item: &str, words: &str| {
            found
                .iter()
                .any(|f| f.rule == rule && f.item == item && f.message.contains(words))
        };
        // The layout moves the top view off the front, and the check reports the move.
        assert!(has("Overlap", "view:top", "moved"), "{found:#?}");
        assert!(
            has("Scale", "sheet:1", "isn't an ISO 5455 scale"),
            "{found:#?}"
        );
        assert!(has("Scale", "view:top", "share one scale"), "{found:#?}");
        assert!(
            has("Scale", "view:d", "a detail view enlarges"),
            "{found:#?}"
        );
    }

    #[test]
    fn preferred_scales_follow_iso_5455_and_inch_practice() {
        for s in [
            (1, 1),
            (1, 2),
            (1, 5),
            (1, 10),
            (1, 20),
            (1, 50),
            (1, 100),
            (2, 1),
            (5, 1),
            (10, 1),
            (2, 2),
        ] {
            assert!(preferred_scale(s, false, false), "{s:?}");
        }
        for s in [(1, 3), (3, 1), (1, 4), (1, 25), (3, 2)] {
            assert!(!preferred_scale(s, false, false), "{s:?}");
        }
        assert!(preferred_scale((1, 4), true, false));
        assert!(preferred_scale((1, 48), false, true));
        assert!(!preferred_scale((1, 48), true, false));
    }

    #[test]
    fn title_block_rules_need_a_title_and_a_number() {
        let text = "runmat 0.7\nstd 1\n\ndrawing bracket {\n    format a3 = { size = { size = iso, name = A3 }, title_block = [{ label = \"Material\", source = { source = literal, text = \"Steel\" } }] }\n    sheet \"1\" = { format = a3, scale = [1, 1], projection = third_angle }\n    view front = standard(orientation = front, sheet = \"1\", model = { part = bracket }, at = [100, 100], display = visible_lines)\n}\n";
        let found = check(text);
        assert!(
            found.iter().any(|f| f.message.contains("no Title field")),
            "{found:#?}"
        );
        assert!(
            found
                .iter()
                .any(|f| f.message.contains("identification number")),
            "{found:#?}"
        );
    }

    #[test]
    fn overlapping_views_tables_and_annotation_texts_on_a_laid_out_sheet() {
        let line = |id: &str, kind: &str, pts: [f64; 4]| json!({ "id": id, "kind": kind, "shape": { "type": "polyline", "points": pts } });
        let text = |id: &str, x: f64, y: f64, t: &str| json!({ "id": id, "kind": "annotation", "shape": { "type": "text", "x": x, "y": y, "text": t, "size": 3.5, "anchor": "middle" } });
        let layout = json!({ "primitives": [
            line("view:front", "view", [0.0, 0.0, 50.0, 30.0]),
            line("view:top", "view", [40.0, 20.0, 90.0, 60.0]),
            line("view:right", "view", [200.0, 0.0, 240.0, 30.0]),
            line("table:bom", "table", [230.0, 10.0, 300.0, 40.0]),
            text("annotation:width", 120.0, 100.0, "50"),
            text("annotation:depth", 121.0, 100.5, "30"),
            text("annotation:hole", 160.0, 100.0, "Ø6"),
        ] });
        let mut out = Vec::new();
        overlaps("1", &layout, &mut out);
        let said: Vec<(&str, &str, &str)> = out
            .iter()
            .map(|f| (f.rule, f.item.as_str(), f.severity))
            .collect();
        assert_eq!(
            said,
            vec![
                ("Overlap", "table:bom", "error"),
                ("Overlap", "view:front", "error"),
                ("Overlap", "annotation:width", "warning"),
            ],
            "{out:#?}"
        );
        assert!(
            out[1]
                .message
                .contains("front view and top view overlap by 10.0 × 10.0 mm"),
            "{}",
            out[1].message
        );
    }

    /// A figure set on a view's model lines (a dimension's 50 on a hole) is an overlap finding;
    /// one beside the lines, or over a centerline only, isn't.
    #[test]
    fn annotation_text_over_a_views_model_lines_is_found() {
        let line = |style: &str, pts: Vec<f64>| json!({ "id": "view:side", "kind": "view", "style": style, "shape": { "type": "polyline", "points": pts } });
        let text = |id: &str, x: f64, y: f64| json!({ "id": id, "kind": "annotation", "style": "dimtext", "shape": { "type": "text", "x": x, "y": y, "text": "50", "size": 3.5, "anchor": "middle" } });
        // A hole of radius 3.25 at (100, 100), as a polygon, and the view's outline.
        let hole: Vec<f64> = (0..=36)
            .flat_map(|i| {
                let a = i as f64 / 36.0 * std::f64::consts::TAU;
                [100.0 + 3.25 * a.cos(), 100.0 + 3.25 * a.sin()]
            })
            .collect();
        let layout = json!({ "primitives": [
            line("visible", hole),
            line("visible", vec![75.0, 70.0, 125.0, 70.0, 125.0, 130.0, 75.0, 130.0, 75.0, 70.0]),
            line("center", vec![90.0, 110.0, 110.0, 110.0]),
            text("annotation:height", 100.0, 99.0),
            text("annotation:clear", 100.0, 85.0),
            text("annotation:on_center", 100.0, 109.0),
        ] });
        let mut out = Vec::new();
        overlaps("1", &layout, &mut out);
        let said: Vec<(&str, &str)> = out.iter().map(|f| (f.item.as_str(), f.severity)).collect();
        assert_eq!(said, vec![("annotation:height", "warning")], "{out:#?}");
        assert!(
            out[0]
                .message
                .contains("height annotation is drawn over the side view's lines"),
            "{}",
            out[0].message
        );
    }
}
