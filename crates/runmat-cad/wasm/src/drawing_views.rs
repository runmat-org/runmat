//! Drawing view edits from the sheet, on the selected views (`view:<name>`):
//!
//! - `drawing.view-scale` sets the view's scale (`1:2`, `2:1` …), or `sheet` to take the
//!   sheet's scale again;
//! - `drawing.view-display` sets how the view draws: hidden lines removed, hidden lines shown
//!   dashed, or shaded;
//! - `drawing.edit-view` edits one view's properties together: scale, orientation (a standard
//!   view's; a projected, section or detail view follows its parent), display style, tangent
//!   edges (thin, visible or hidden; `TangentEdges`), label and place on the sheet. A projected view that drew at its parent's scale keeps drawing at it,
//!   so the views in projection stay at one scale (ISO 5456-2). The view's dimensions and
//!   annotations name it, so they move and scale with it.
//!
//! Each is written as the smallest change to the drawing's text.

use crate::edit::model;
use crate::runtime::Files;
use runmat_cad::drawing::{
    Display, DrawingDocument, Orientation, ProjectDirection, TangentEdges, ViewKind,
};
use runmat_cad::syntax::edit::{diff_source_in, EditFailure, SourceChange};
use runmat_cad::syntax::{parse_drawing_in, DRAWING};
use serde_json::{Map, Value};

fn views(selection: &[String]) -> Vec<&str> {
    selection
        .iter()
        .filter_map(|s| s.strip_prefix("view:"))
        .collect()
}

fn scale(word: &str) -> Result<Option<(u32, u32)>, EditFailure> {
    if word == "sheet" {
        return Ok(None);
    }
    let (a, b) = word
        .split_once(':')
        .ok_or_else(|| model(format!("`{word}` isn't a scale (write it as 1:2)")))?;
    match (a.trim().parse::<u32>(), b.trim().parse::<u32>()) {
        (Ok(a), Ok(b)) if a > 0 && b > 0 => Ok(Some((a, b))),
        _ => Err(model(format!("`{word}` isn't a scale (write it as 1:2)"))),
    }
}

fn display(word: &str) -> Result<Display, EditFailure> {
    Ok(match word {
        "hidden_removed" => Display::VisibleLines,
        "hidden_visible" => Display::HiddenLines,
        "shaded" => Display::Shaded,
        other => return Err(model(format!("there is no `{other}` display style"))),
    })
}

/// Runs a view edit; `None` when `command` isn't one.
pub fn view_edit(
    path: &str,
    text: &str,
    files: &Files,
    command: &str,
    values: &Map<String, Value>,
    selection: &[String],
) -> Option<Result<(SourceChange, String), EditFailure>> {
    let word = |key: &str| values.get(key).and_then(Value::as_str).unwrap_or("");
    type ViewEdit = dyn Fn(&mut runmat_cad::drawing::View) -> Result<(), EditFailure>;
    let set: Box<ViewEdit> = match command {
        "drawing.view-scale" => {
            let s = match scale(word("scale")) {
                Ok(s) => s,
                Err(e) => return Some(Err(e)),
            };
            Box::new(move |v| {
                v.scale = s;
                Ok(())
            })
        }
        "drawing.view-display" => {
            let d = match display(word("style")) {
                Ok(d) => d,
                Err(e) => return Some(Err(e)),
            };
            Box::new(move |v| {
                v.display = d;
                Ok(())
            })
        }
        "drawing.edit-view" => return Some(edit_view(path, text, files, values, selection)),
        _ => return None,
    };
    Some((|| {
        let names = views(selection);
        if names.is_empty() {
            return Err(model("select the views to change"));
        }
        let scope = crate::runtime::module_scope(path, text, files);
        let parsed = parse_drawing_in(path, text, &scope);
        let mut document: DrawingDocument = parsed
            .document
            .ok_or_else(|| EditFailure::Invalid(parsed.diagnostics.clone()))?;
        for name in &names {
            let view = document
                .views
                .iter_mut()
                .find(|v| v.name == *name)
                .ok_or_else(|| model(format!("the drawing has no view `{name}`")))?;
            set(view)?;
        }
        let change = diff_source_in(&DRAWING, path, text, &document, &parsed.uses, &[], &scope)?;
        let what = if command == "drawing.view-scale" {
            "scale"
        } else {
            "display style"
        };
        Ok((change, format!("Set the {what} of {}", names.join(", "))))
    })())
}

fn orientation(word: &str) -> Result<Orientation, EditFailure> {
    Ok(match word {
        "front" => Orientation::Front,
        "back" => Orientation::Back,
        "top" => Orientation::Top,
        "bottom" => Orientation::Bottom,
        "left" => Orientation::Left,
        "right" => Orientation::Right,
        "isometric" => Orientation::Isometric,
        "dimetric" => Orientation::Dimetric,
        "trimetric" => Orientation::Trimetric,
        other => return Err(model(format!("there is no `{other}` orientation"))),
    })
}

/// What a view looks along: a standard view's orientation, a projected view's direction from
/// its parent's front, and for a view cut or enlarged from another, that view's.
pub(crate) fn seen(document: &DrawingDocument, name: &str) -> Option<Orientation> {
    let mut at = name;
    for _ in 0..16 {
        let view = document.view(at)?;
        match &view.kind {
            ViewKind::Standard { orientation, .. } => return Some(*orientation),
            ViewKind::Projected { direction, .. } => {
                return Some(match direction {
                    ProjectDirection::Up => Orientation::Top,
                    ProjectDirection::Down => Orientation::Bottom,
                    ProjectDirection::Left => Orientation::Left,
                    ProjectDirection::Right => Orientation::Right,
                })
            }
            kind => at = kind.parent()?,
        }
    }
    None
}

/// The tangent edges the view `name` draws when it doesn't say: its orientation's default.
fn tangent_default(document: &DrawingDocument, name: &str) -> TangentEdges {
    let views = serde_json::to_value(&document.views).unwrap_or(Value::Null);
    let mut unset = views.clone();
    if let Some(v) = unset
        .as_array_mut()
        .and_then(|vs| vs.iter_mut().find(|v| v["name"] == name))
        .and_then(Value::as_object_mut)
    {
        v.remove("tangent_edges");
    }
    crate::hlr::tangent_edges(&unset, name)
}

/// `drawing.edit-view`: the selected view's scale, orientation, display style, label and place.
/// A field left empty keeps what the view has.
fn edit_view(
    path: &str,
    text: &str,
    files: &Files,
    values: &Map<String, Value>,
    selection: &[String],
) -> Result<(SourceChange, String), EditFailure> {
    let word = |key: &str| {
        values
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or("")
    };
    let names = views(selection);
    let [name] = names.as_slice() else {
        return Err(model("select one view to edit"));
    };
    let scope = crate::runtime::module_scope(path, text, files);
    let parsed = parse_drawing_in(path, text, &scope);
    let mut document: DrawingDocument = parsed
        .document
        .ok_or_else(|| EditFailure::Invalid(parsed.diagnostics.clone()))?;
    let before = document
        .view(name)
        .cloned()
        .ok_or_else(|| model(format!("the drawing has no view `{name}`")))?;
    let mut changed: Vec<&str> = Vec::new();
    let mut view = before.clone();
    if !word("scale").is_empty() {
        view.scale = scale(word("scale"))?;
        if view.scale != before.scale {
            changed.push("scale");
        }
    }
    if !word("orientation").is_empty() {
        let wanted = orientation(word("orientation"))?;
        match &mut view.kind {
            ViewKind::Standard { orientation, .. } => {
                if *orientation != wanted {
                    *orientation = wanted;
                    changed.push("orientation");
                }
            }
            kind => {
                if seen(&document, name) != Some(wanted) {
                    let parent = kind.parent().unwrap_or("its parent").to_owned();
                    return Err(model(format!(
                        "{name} is drawn from {parent}, so it looks the way {parent} sets; change {parent}'s orientation, or add a model view"
                    )));
                }
            }
        }
    }
    if !word("display").is_empty() {
        view.display = display(word("display"))?;
        if view.display != before.display {
            changed.push("display style");
        }
    }
    if !word("tangent_edges").is_empty() {
        let wanted = serde_json::from_value::<TangentEdges>(Value::from(word("tangent_edges")))
            .map_err(|_| {
                model(format!(
                    "there is no `{}` tangent edge style (thin, show or hide)",
                    word("tangent_edges")
                ))
            })?;
        if view
            .tangent_edges
            .unwrap_or_else(|| tangent_default(&document, name))
            != wanted
        {
            view.tangent_edges = Some(wanted);
            changed.push("tangent edges");
        }
    }
    if let Some(label) = values.get("label").and_then(Value::as_str) {
        let label = label.trim().to_uppercase();
        match &mut view.kind {
            ViewKind::Section { label: l, .. }
            | ViewKind::Detail { label: l, .. }
            | ViewKind::Auxiliary { label: l, .. } => {
                if label.is_empty() {
                    return Err(model(format!(
                        "{name} needs a label letter: its cutting line or circle carries it"
                    )));
                }
                if *l != label {
                    *l = label;
                    changed.push("label");
                }
            }
            // Other views print a title under them: the label replaces it, and an empty label
            // gives back the one the view's kind prints.
            _ => {
                let title = (!label.is_empty()).then_some(label);
                if view.title != title {
                    view.title = title;
                    changed.push("label");
                }
            }
        }
    }
    let to = crate::features::picks_of(values, "at")
        .iter()
        .find_map(|p| match crate::drawing_edits::parse_pick(p) {
            Ok(crate::drawing_edits::SheetPick::Point(p)) => Some(p),
            _ => None,
        });
    if let Some(to) = to {
        let to = to.map(|v| (v * 10.0).round() / 10.0);
        if to != before.at {
            view.at = to;
            changed.push("place");
        }
    }
    if changed.is_empty() {
        return Err(model(format!("{name} already has these properties")));
    }
    // Projected views drawn at their parent's scale keep drawing at it.
    let followers: Vec<String> = document
        .views
        .iter()
        .filter(|v| {
            matches!(&v.kind, ViewKind::Projected { from, .. } if from.as_str() == *name)
                && v.scale == before.scale
        })
        .map(|v| v.name.clone())
        .collect();
    for v in document.views.iter_mut() {
        if v.name == *name {
            *v = view.clone();
        } else if followers.contains(&v.name) {
            v.scale = view.scale;
        }
    }
    let change = diff_source_in(&DRAWING, path, text, &document, &parsed.uses, &[], &scope)?;
    let mut summary = format!("Edited {name}: {}", changed.join(", "));
    if view.scale != before.scale && !followers.is_empty() {
        summary.push_str(&format!(" ({} follow its scale)", followers.join(", ")));
    }
    Ok((change, summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PART: &str = "runmat 0.7\nstd 1\n\npart bracket {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n}\n";

    fn drawing() -> (Files, String) {
        let files: Files = [("/p/bracket.part".to_string(), PART.to_string())].into();
        let text = r#"runmat 0.7
std 1

drawing bracket {
    format a3 = { size = { size = iso, name = A3 }, title_block = [] }
    sheet "1" = { format = a3, scale = [1, 1], projection = third_angle }
    view front = standard(orientation = front, sheet = "1", model = { part = bracket }, at = [100, 100], display = visible_lines)
}
"#.to_string();
        (files, text)
    }

    #[test]
    fn a_selected_view_takes_a_scale_and_a_display_style() {
        let (files, text) = drawing();
        let parsed = parse_drawing_in(
            "/p/bracket.drawing",
            &text,
            &crate::runtime::module_scope("/p/bracket.drawing", &text, &files),
        );
        assert!(
            parsed.document.is_some(),
            "fixture: {:?}",
            parsed.diagnostics
        );
        let sel = vec!["view:front".to_string()];
        let values: Map<String, Value> = serde_json::from_str(r#"{"scale":"1:2"}"#).unwrap();
        let (change, summary) = view_edit(
            "/p/bracket.drawing",
            &text,
            &files,
            "drawing.view-scale",
            &values,
            &sel,
        )
        .unwrap()
        .unwrap();
        assert!(
            change.source.contains("scale = [1, 2]"),
            "{}",
            change.source
        );
        assert_eq!(summary, "Set the scale of front");
        let values: Map<String, Value> =
            serde_json::from_str(r#"{"style":"hidden_visible"}"#).unwrap();
        let (change, _) = view_edit(
            "/p/bracket.drawing",
            &change.source,
            &files,
            "drawing.view-display",
            &values,
            &sel,
        )
        .unwrap()
        .unwrap();
        assert!(change.source.contains("hidden_lines"), "{}", change.source);
        // Back to the sheet's scale.
        let values: Map<String, Value> = serde_json::from_str(r#"{"scale":"sheet"}"#).unwrap();
        let (back, _) = view_edit(
            "/p/bracket.drawing",
            &change.source,
            &files,
            "drawing.view-scale",
            &values,
            &sel,
        )
        .unwrap()
        .unwrap();
        let view_line = back
            .source
            .lines()
            .find(|l| l.contains("view front"))
            .unwrap()
            .to_owned();
        assert!(!view_line.contains("scale ="), "{}", back.source);
        assert!(view_edit(
            "/p/bracket.drawing",
            &text,
            &files,
            "drawing.view-scale",
            &values,
            &[]
        )
        .unwrap()
        .is_err());
    }

    /// The bracket's front view, the top view projected from it, a section through the front,
    /// and a width dimension on the front.
    fn family() -> (Files, String) {
        let (files, _) = drawing();
        let text = r#"runmat 0.7
std 1

drawing bracket {
    format a3 = { size = { size = iso, name = A3 }, title_block = [] }
    sheet "1" = { format = a3, scale = [1, 1], projection = third_angle }
    view front = standard(orientation = front, sheet = "1", model = { part = bracket }, at = [100, 100], display = visible_lines)
    view top = projected(from = front, direction = up, sheet = "1", model = { part = bracket }, at = [100, 180], display = visible_lines)
    view a_a = section(from = top, plane = origin.yz, label = A, sheet = "1", model = { part = bracket }, at = [260, 180], display = visible_lines)

    annotation width = dimension(style = linear, references = [feature.body.face.side(s1.l4), feature.body.face.side(s1.l2)], view = front)
}
"#
        .to_string();
        (files, text)
    }

    fn edit(
        text: &str,
        files: &Files,
        view: &str,
        values: &str,
    ) -> Result<(String, String), EditFailure> {
        let values: Map<String, Value> = serde_json::from_str(values).unwrap();
        let (change, summary) = view_edit(
            "/p/bracket.drawing",
            text,
            files,
            "drawing.edit-view",
            &values,
            &[format!("view:{view}")],
        )
        .expect("edit-view is a view edit")?;
        assert!(!change.reprinted, "edit-view reprinted the drawing");
        Ok((change.source, summary))
    }

    fn document(text: &str, files: &Files) -> DrawingDocument {
        let scope = crate::runtime::module_scope("/p/bracket.drawing", text, files);
        parse_drawing_in("/p/bracket.drawing", text, &scope)
            .document
            .expect("the edited drawing reads back")
    }

    /// The sheet box of everything the layout drew for `id` (`view:front`, `annotation:width`).
    fn drawn_box(text: &str, files: &Files, id: &str) -> [f64; 4] {
        let mut all = files.clone();
        all.insert("/p/bracket.drawing".into(), text.to_owned());
        let out: Value = serde_json::from_str(&crate::drawing::compile_json(
            "/p/bracket.drawing",
            text,
            &all,
        ))
        .unwrap();
        let prims = out["projection"]["data"]["layouts"]["1"]["primitives"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut b = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for p in prims.iter().filter(|p| p["id"] == id) {
            let sh = &p["shape"];
            let mut xy: Vec<f64> = sh["points"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_f64).collect())
                .unwrap_or_default();
            if let (Some(x), Some(y)) = (sh["x"].as_f64(), sh["y"].as_f64()) {
                xy.extend([x, y]);
            }
            for c in xy.chunks(2) {
                if let [x, y] = c {
                    b = [b[0].min(*x), b[1].min(*y), b[2].max(*x), b[3].max(*y)];
                }
            }
        }
        assert!(b[0].is_finite(), "nothing drawn for {id}");
        b
    }

    #[test]
    fn edit_view_writes_scale_orientation_display_label_and_place_and_its_dimensions_follow() {
        let (files, text) = family();
        let front0 = drawn_box(&text, &files, "view:front");
        let width0 = drawn_box(&text, &files, "annotation:width");

        // Place: the view and its dimension move by the same amount.
        let (moved, summary) = edit(&text, &files, "front", r#"{"at":["point:140,90"]}"#).unwrap();
        assert_eq!(summary, "Edited front: place");
        assert_eq!(
            document(&moved, &files).view("front").unwrap().at,
            [140.0, 90.0]
        );
        let front1 = drawn_box(&moved, &files, "view:front");
        let width1 = drawn_box(&moved, &files, "annotation:width");
        let dx = front1[0] - front0[0];
        assert!((dx - 40.0).abs() < 1e-6, "front moved {dx}");
        assert!(
            ((width1[0] - width0[0]) - dx).abs() < 1e-6,
            "the width dimension stayed behind"
        );

        // Scale: the projected top view drew at the front's scale and keeps doing so; the
        // section, drawn from the top, keeps its own (the sheet's). The dimension grows with it.
        let (scaled, summary) = edit(&moved, &files, "front", r#"{"scale":"2:1"}"#).unwrap();
        assert_eq!(summary, "Edited front: scale (top follow its scale)");
        let d = document(&scaled, &files);
        assert_eq!(d.view("front").unwrap().scale, Some((2, 1)));
        assert_eq!(d.view("top").unwrap().scale, Some((2, 1)));
        assert_eq!(d.view("a_a").unwrap().scale, None);
        let width2 = drawn_box(&scaled, &files, "annotation:width");
        assert!(
            (width2[2] - width2[0]) > 1.5 * (width1[2] - width1[0]),
            "the width dimension didn't scale with its view: {width1:?} → {width2:?}"
        );

        // Orientation, display style and label (a title for a standard view).
        let (turned, summary) = edit(
            &scaled,
            &files,
            "front",
            r#"{"orientation":"back","display":"hidden_visible","label":"Rear elevation"}"#,
        )
        .unwrap();
        assert_eq!(summary, "Edited front: orientation, display style, label");
        let d = document(&turned, &files);
        let front = d.view("front").unwrap();
        assert!(matches!(
            front.kind,
            ViewKind::Standard {
                orientation: Orientation::Back,
                ..
            }
        ));
        assert_eq!(front.display, Display::HiddenLines);
        assert_eq!(front.title.as_deref(), Some("REAR ELEVATION"));
        // The projected top now looks along the back view's projection: it is still drawn
        // from front, and its orientation can't be set on its own.
        let err = edit(&turned, &files, "top", r#"{"orientation":"front"}"#).unwrap_err();
        assert!(err.to_string().contains("is drawn from front"), "{err}");
        // Its own orientation, as the panel starts with it, is no change.
        assert!(edit(&turned, &files, "top", r#"{"orientation":"top"}"#)
            .unwrap_err()
            .to_string()
            .contains("already has these properties"));

        // A section's label is its letter; it can't be left empty.
        let (relabelled, _) = edit(&turned, &files, "a_a", r#"{"label":"b"}"#).unwrap();
        assert!(matches!(
            &document(&relabelled, &files).view("a_a").unwrap().kind,
            ViewKind::Section { label, .. } if label == "B"
        ));
        assert!(edit(&turned, &files, "a_a", r#"{"label":""}"#).is_err());
        // An empty label gives a standard view its kind's title back.
        let (plain, _) = edit(&relabelled, &files, "front", r#"{"label":""}"#).unwrap();
        assert_eq!(document(&plain, &files).view("front").unwrap().title, None);

        // One view at a time; a bad scale is refused.
        let values: Map<String, Value> = serde_json::from_str(r#"{"scale":"2:1"}"#).unwrap();
        let two = ["view:front".to_string(), "view:top".to_string()];
        assert!(view_edit(
            "/p/bracket.drawing",
            &text,
            &files,
            "drawing.edit-view",
            &values,
            &two
        )
        .unwrap()
        .is_err());
        assert!(edit(&text, &files, "front", r#"{"scale":"two"}"#).is_err());
    }
}
