//! Each sheet's title block as drawn ([`runmat_cad::drawing::title_layout`]), with the outlines
//! of its logo fields read from the workspace's SVG files. The compile puts it in the
//! projection as `title_layouts`, and the Desktop's sheet and the exports draw it as given.

use crate::runtime::{resolve_path, Diagnostic, Files};
use runmat_cad::drawing::logo::{outline, Outline};
use runmat_cad::drawing::title_layout::sheet_layouts;
use runmat_cad::drawing::{DrawingDocument, FieldSource};
use serde_json::Value;
use std::collections::BTreeMap;

/// The SVG files a drawing's text names (string literals ending in `.svg`), from its folder.
pub fn image_files(path: &str, text: &str) -> Vec<String> {
    let folder = path.rsplit_once('/').map_or("", |(f, _)| f);
    text.split('"')
        .skip(1)
        .step_by(2)
        .filter(|s| s.to_ascii_lowercase().ends_with(".svg") && !s.contains('\n'))
        .map(|s| {
            if s.starts_with('/') {
                s.to_owned()
            } else {
                format!("{folder}/{s}")
            }
        })
        .collect()
}

/// Sheet name → its title layout with each logo field's outline read from its SVG file; a
/// file that is missing or isn't an SVG with shapes is reported as a warning and its cell
/// stays empty.
pub fn title_layouts(
    path: &str,
    document: &DrawingDocument,
    rows: &BTreeMap<String, Vec<(String, String)>>,
    files: &Files,
    problems: &mut Vec<Diagnostic>,
) -> Value {
    let mut outlines: BTreeMap<String, Outline> = BTreeMap::new();
    for format in &document.formats {
        for field in &format.title_block {
            let FieldSource::Image { file } = &field.source else {
                continue;
            };
            if outlines.contains_key(file) {
                continue;
            }
            let at = resolve_path(path, file, files);
            match files
                .get(&at)
                .ok_or_else(|| format!("{file} isn't in the workspace"))
                .and_then(|svg| outline(svg))
            {
                Ok(o) => {
                    outlines.insert(file.clone(), o);
                }
                Err(e) => problems.push(Diagnostic {
                    path: path.to_owned(),
                    start: 0,
                    end: 0,
                    severity: "warning".into(),
                    message: format!("title block `{}`: {e}, so its logo is empty", field.label),
                }),
            }
        }
    }
    serde_json::to_value(sheet_layouts(document, rows, |_, file| outlines.get(file)))
        .unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOGO_DRAWING: &str = "runmat 0.7\nstd 1\n\ndrawing plate {\n    format a3 = {\n        size = { size = iso, name = A3 },\n        title_block = [\n            { label = \"Title\", source = { source = literal, text = \"Plate\" } },\n            { label = \"\", source = { source = image, file = \"runmat-logo.svg\" } }\n        ]\n    }\n    sheet \"1\" = { format = a3, scale = [1, 1], projection = third_angle }\n}\n";

    #[test]
    fn a_logo_field_draws_the_svg_outline_in_its_cell_and_a_missing_file_warns() {
        assert_eq!(
            image_files("/p/plate.drawing", LOGO_DRAWING),
            vec!["/p/runmat-logo.svg".to_owned()]
        );
        let mut files = Files::new();
        files.insert(
            "/p/runmat-logo.svg".into(),
            include_str!("../../../tests/fixtures/runmat-logo.svg").into(),
        );
        let out: Value = serde_json::from_str(&super::super::compile_json(
            "/p/plate.drawing",
            LOGO_DRAWING,
            &files,
        ))
        .unwrap();
        let data = &out["projection"]["data"];
        let cells = data["title_layouts"]["1"]["cells"].as_array().unwrap();
        assert_eq!(cells.len(), 2, "{data}");
        assert_eq!(cells[0]["value"]["text"], "Plate");
        let loops = cells[1]["logo"].as_array().unwrap();
        assert!(loops.len() >= 12, "{} loops", loops.len());
        assert_eq!(
            out["diagnostics"].as_array().map(Vec::len),
            Some(0),
            "{out}"
        );

        let out: Value = serde_json::from_str(&super::super::compile_json(
            "/p/plate.drawing",
            LOGO_DRAWING,
            &Files::new(),
        ))
        .unwrap();
        let warning = out["diagnostics"][0]["message"].as_str().unwrap_or("");
        assert!(
            warning.contains("runmat-logo.svg isn't in the workspace"),
            "{out}"
        );
    }
}
