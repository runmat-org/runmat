//! DXF export: drawing sheets (frame, title block, visible and hidden edges at the sheet
//! scale) and sheet-metal flat patterns.

use crate::{Files, ToolError};
use serde_json::Value;
use std::fmt::Write;

use super::floats;

struct Dxf {
    body: String,
}

impl Dxf {
    fn line(&mut self, layer: &str, a: [f64; 2], b: [f64; 2]) {
        let _ = write!(
            self.body,
            "0\nLINE\n8\n{layer}\n10\n{:.4}\n20\n{:.4}\n30\n0.0\n11\n{:.4}\n21\n{:.4}\n31\n0.0\n",
            a[0], a[1], b[0], b[1]
        );
    }
    fn text(&mut self, layer: &str, at: [f64; 2], height: f64, text: &str) {
        let _ = write!(
            self.body,
            "0\nTEXT\n8\n{layer}\n10\n{:.4}\n20\n{:.4}\n30\n0.0\n40\n{height:.3}\n1\n{text}\n",
            at[0], at[1]
        );
    }
    fn rect(&mut self, layer: &str, x: f64, y: f64, w: f64, h: f64) {
        self.line(layer, [x, y], [x + w, y]);
        self.line(layer, [x + w, y], [x + w, y + h]);
        self.line(layer, [x + w, y + h], [x, y + h]);
        self.line(layer, [x, y + h], [x, y]);
    }
    fn finish(self, layers: &[(&str, u8, &str)]) -> String {
        let mut s = String::from(
            "0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n9\n$INSUNITS\n70\n4\n0\nENDSEC\n0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLTYPE\n70\n2\n0\nLTYPE\n2\nCONTINUOUS\n70\n0\n3\nSolid line\n72\n65\n73\n0\n40\n0.0\n0\nLTYPE\n2\nDASHED\n70\n0\n3\n__ __ __\n72\n65\n73\n2\n40\n4.5\n49\n3.0\n49\n-1.5\n0\nENDTAB\n0\nTABLE\n2\nLAYER\n",
        );
        let _ = write!(s, "70\n{}\n", layers.len());
        for (name, color, ltype) in layers {
            let _ = write!(s, "0\nLAYER\n2\n{name}\n70\n0\n62\n{color}\n6\n{ltype}\n");
        }
        s.push_str("0\nENDTAB\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n");
        s.push_str(&self.body);
        s.push_str("0\nENDSEC\n0\nEOF\n");
        s
    }
}

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

/// A drawing sheet as DXF: the frame and title block, each view's visible edges (`VISIBLE`,
/// continuous) and hidden edges (`HIDDEN`, dashed) at the sheet's scale, and view labels.
pub(super) fn drawing_dxf(
    path: &str,
    text: &str,
    files: &Files,
    sheet: Option<&str>,
) -> Result<(String, String, Vec<String>), ToolError> {
    let compiled: Value =
        serde_json::from_str(&runmat_cad_wasm::drawing::compile_json(path, text, files))
            .unwrap_or(Value::Null);
    let data = &compiled["projection"]["data"];
    let doc = &data["document"];
    let sheets = doc["sheets"].as_array().cloned().unwrap_or_default();
    let Some(sheet) = sheets
        .iter()
        .find(|s| sheet.is_none_or(|n| s["name"].as_str() == Some(n)))
    else {
        return Err(ToolError::invalid(format!(
            "{path} has no sheet {}",
            sheet.unwrap_or("")
        )));
    };
    let sheet_name = sheet["name"].as_str().unwrap_or("1").to_owned();
    let format = doc["formats"]
        .as_array()
        .and_then(|f| f.iter().find(|f| f["name"] == sheet["format"]))
        .cloned()
        .unwrap_or(Value::Null);
    let size_name = format["size"]["name"].as_str().unwrap_or("A3");
    let [w, h] = sheet_size(size_name);
    let ratio = floats(&sheet["scale"]);
    let scale = if ratio.len() == 2 && ratio[1] != 0.0 {
        ratio[0] / ratio[1]
    } else {
        1.0
    };
    let mut dxf = Dxf {
        body: String::new(),
    };
    let mut notes = Vec::new();
    dxf.rect("FRAME", 10.0, 10.0, w - 20.0, h - 20.0);
    let rows = data["title_blocks"][&sheet_name]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for (i, row) in rows.iter().enumerate() {
        let y = 10.0 + i as f64 * 7.0;
        let x = w - 130.0;
        dxf.rect("FRAME", x, y, 120.0, 7.0);
        dxf.text(
            "FRAME",
            [x + 2.0, y + 2.2],
            2.2,
            row[0].as_str().unwrap_or(""),
        );
        dxf.text(
            "FRAME",
            [x + 40.0, y + 2.2],
            2.6,
            row[1].as_str().unwrap_or(""),
        );
    }
    let views: Vec<&Value> = doc["views"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|v| v["sheet"].as_str() == Some(sheet_name.as_str()))
        .collect();
    for (n, view) in views.iter().enumerate() {
        let name = view["name"].as_str().unwrap_or("view");
        let hlr = &data["hlr"][name];
        let own = floats(&view["scale"]);
        let s = if own.len() == 2 && own[1] != 0.0 {
            own[0] / own[1]
        } else {
            scale
        } * 1e3;
        let runs = |key: &str| -> Vec<Vec<f64>> {
            hlr[key]
                .as_array()
                .into_iter()
                .flatten()
                .map(floats)
                .collect()
        };
        let (visible, hidden) = (runs("visible"), runs("hidden"));
        if visible.is_empty() && hidden.is_empty() {
            notes.push(format!(
                "view {name} has no edges (its model isn't in the workspace or doesn't compile)"
            ));
        }
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for run in visible.iter().chain(&hidden) {
            for p in run.chunks_exact(2) {
                lo = [lo[0].min(p[0]), lo[1].min(p[1])];
                hi = [hi[0].max(p[0]), hi[1].max(p[1])];
            }
        }
        let centre = if lo[0].is_finite() {
            [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0]
        } else {
            [0.0, 0.0]
        };
        let at = floats(&view["at"]);
        let origin = if at.len() == 2 {
            [at[0], at[1]]
        } else {
            [
                w * (0.3 + 0.38 * (n % 2) as f64),
                h * (0.62 - 0.37 * (n / 2 % 2) as f64),
            ]
        };
        let place = |p: &[f64]| {
            [
                origin[0] + (p[0] - centre[0]) * s,
                origin[1] + (p[1] - centre[1]) * s,
            ]
        };
        for (layer, set) in [("VISIBLE", &visible), ("HIDDEN", &hidden)] {
            for run in set.iter() {
                for seg in run.chunks_exact(2).collect::<Vec<_>>().windows(2) {
                    dxf.line(layer, place(seg[0]), place(seg[1]));
                }
            }
        }
        let half_h = if lo[1].is_finite() {
            (hi[1] - lo[1]) * s / 2.0
        } else {
            10.0
        };
        let shown = s / 1e3;
        let label = format!(
            "{}  SCALE {}",
            name.to_uppercase(),
            if shown >= 1.0 {
                format!("{shown}:1")
            } else {
                format!("1:{}", (1.0 / shown).round())
            }
        );
        dxf.text("NOTES", [origin[0], origin[1] - half_h - 7.0], 3.0, &label);
    }
    let out = dxf.finish(&[
        ("FRAME", 7, "CONTINUOUS"),
        ("VISIBLE", 7, "CONTINUOUS"),
        ("HIDDEN", 8, "DASHED"),
        ("NOTES", 3, "CONTINUOUS"),
    ]);
    Ok((sheet_name, out, notes))
}

/// The DXF of a part's flat pattern, when it has one.
pub(super) fn flat_pattern_dxf(path: &str, text: &str) -> Result<String, ToolError> {
    let part = runmat_cad_wasm::runtime::regenerate_part(path, text)
        .map_err(|_| ToolError::new("not_compiled", format!("{path} has syntax errors")))?;
    for record in part.result.state.records.values() {
        if let runmat_cad::regen::records::FeatureRecord::FlatPattern { layout, .. } = record {
            return Ok(runmat_cad::drawing::flat_pattern::flat_pattern_sheet(layout).to_dxf());
        }
    }
    Err(ToolError::new(
        "unsupported",
        format!("{path} has no flat pattern; DXF exports drawings and sheet-metal flat patterns (add a flat_pattern feature)"),
    ))
}
