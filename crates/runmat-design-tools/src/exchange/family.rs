//! Product families: one export per configuration of a part (its STEP), of a drawing's part
//! (every sheet as DXF), and BOMs as CSV, so a family's documents come from one model in one
//! call.

use super::sheet::drawing_sheet;
use super::exact_step::exact_step_in;
use super::mesh::model_scene_in;
use super::step::step;
use super::{binary_file, stem, text_file, ExportedFile};
use crate::{kind_of, text_of, DesignKind, Files, ToolError};
use serde_json::Value;

/// The named configurations of the part at `path` (empty when it has none or doesn't parse).
pub fn part_configurations(path: &str, files: &Files) -> Vec<String> {
    let Ok(text) = text_of(files, path) else { return Vec::new() };
    runmat_cad::syntax::parse_part(path, text)
        .document
        .map(|d| d.configurations.iter().map(|c| c.name.to_string()).collect())
        .unwrap_or_default()
}

/// The part a drawing's views show first, resolved against the drawing's folder.
fn drawn_part(path: &str, text: &str, files: &Files) -> Option<String> {
    runmat_cad_wasm::drawing::dependencies(path, text).into_iter().find(|p| p.ends_with(".part") && files.contains_key(p))
}

fn suffix(c: &Option<String>) -> String {
    c.as_ref().map(|c| format!("-{c}")).unwrap_or_default()
}

/// The default configuration, then each named one.
fn family(names: Vec<String>) -> Vec<Option<String>> {
    std::iter::once(None).chain(names.into_iter().map(Some)).collect()
}

/// STEP of every configuration of a part.
pub fn part_family_step(path: &str, files: &Files, notes: &mut Vec<String>) -> Result<Vec<ExportedFile>, ToolError> {
    let name = stem(path);
    let mut out = Vec::new();
    for c in family(part_configurations(path, files)) {
        let file = format!("{name}{}.step", suffix(&c));
        match exact_step_in(path, files, c.as_deref())? {
            Some(bytes) => out.push(text_file(file, "model/step", String::from_utf8_lossy(&bytes).into_owned())),
            None => {
                let meshes = model_scene_in(path, files, c.as_deref())?.world_meshes();
                out.push(text_file(file, "model/step", step(&meshes, &name)));
            }
        }
    }
    notes.push(format!("{} configurations of {path}", out.len()));
    Ok(out)
}

/// Every sheet of a drawing as DXF (or DWG), once per configuration of the part it shows.
pub fn drawing_family(path: &str, files: &Files, dwg: bool, notes: &mut Vec<String>) -> Result<Vec<ExportedFile>, ToolError> {
    let text = text_of(files, path)?;
    let name = stem(path);
    let configs = drawn_part(path, text, files).map(|p| part_configurations(&p, files)).unwrap_or_default();
    let sheets: Vec<String> = runmat_cad::syntax::parse_drawing(path, text)
        .document
        .map(|d| d.sheets.iter().map(|s| s.name.clone()).collect())
        .unwrap_or_default();
    let mut out = Vec::new();
    for c in family(configs) {
        for sheet in &sheets {
            let (sheet, laid, n) = drawing_sheet(path, text, files, Some(sheet), c.as_deref())?;
            notes.extend(n);
            out.push(if dwg {
                binary_file(format!("{name}{}-{sheet}.dwg", suffix(&c)), "image/vnd.dwg", &crate::formats::dwg::write(&laid))
            } else {
                text_file(format!("{name}{}-{sheet}.dxf", suffix(&c)), "image/vnd.dxf", super::dxf::write(&laid))
            });
        }
    }
    Ok(out)
}

fn csv_cell(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}

/// An assembly's flat BOM as CSV: item, part number, file, configuration, material, quantity.
pub fn bom_csv(path: &str, files: &Files) -> Result<ExportedFile, ToolError> {
    let text = text_of(files, path)?;
    if kind_of(path)? != DesignKind::Assembly {
        return Err(ToolError::new("unsupported", format!("a BOM exports from an assembly; {path} is not one")));
    }
    let (index, _, _) = runmat_cad_wasm::runtime::compile_assembly(path, text, files).map_err(|d| {
        ToolError::new("not_compiled", format!("{path} doesn't compile: {}", d.first().map(|d| d.message.as_str()).unwrap_or("")))
    })?;
    let mut s = String::from("item,part_number,file,configuration,material,quantity\n");
    for (i, row) in index["bom"]["rows"].as_array().into_iter().flatten().enumerate() {
        let item = &row["item"];
        let file = match &item["definition"] {
            Value::String(f) => f.clone(),
            d => d["file"].as_str().or_else(|| d["path"].as_str()).map(str::to_owned).unwrap_or_else(|| d.to_string()),
        };
        let cells = [
            (i + 1).to_string(),
            item["part_number"].as_str().unwrap_or("").to_owned(),
            file,
            item["configuration"].as_str().unwrap_or("").to_owned(),
            item["material"].as_str().unwrap_or("").to_owned(),
            row["quantity"].as_u64().unwrap_or(0).to_string(),
        ];
        s.push_str(&cells.iter().map(|c| csv_cell(c)).collect::<Vec<_>>().join(","));
        s.push('\n');
    }
    Ok(text_file(format!("{}-bom.csv", stem(path)), "text/csv", s))
}
