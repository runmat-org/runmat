//! Drawing view edits from the sheet, on the selected views (`view:<name>`):
//!
//! - `drawing.view-scale` sets the view's scale (`1:2`, `2:1` …), or `sheet` to take the
//!   sheet's scale again;
//! - `drawing.view-display` sets how the view draws: hidden lines removed, hidden lines shown
//!   dashed, or shaded.
//!
//! Each is written as the smallest change to the drawing's text.

use crate::edit::model;
use crate::runtime::Files;
use runmat_cad::drawing::{Display, DrawingDocument};
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
    let set: Box<dyn Fn(&mut runmat_cad::drawing::View) -> Result<(), EditFailure>> = match command
    {
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
}
