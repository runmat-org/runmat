//! The properties a drawing reads from the parts it shows: a part's output properties for its
//! own views and title block, and each BOM item's definition properties for parts lists
//! (`description`, `finish`, `vendor` …).
use crate::runtime::{resolve_path, Files};
use runmat_cad::assembly::bom::Bom;
use runmat_cad::drawing::bom_table::material_name;
use runmat_cad::model::PartDocument;
use runmat_cad::syntax::parse_part;
use serde_json::Value;
use std::collections::BTreeMap;

/// A part's output properties as text: `part_number`, `material` (by its display name, as a
/// parts list names it) and every declared property.
pub fn output_properties(document: &PartDocument) -> BTreeMap<String, String> {
    let mut props = BTreeMap::new();
    let Some(output) = document.parts.first() else {
        return props;
    };
    if let Some(pn) = &output.part_number {
        props.insert("part_number".to_owned(), pn.clone());
    }
    if let Some(material) = &output.material {
        let text = serde_json::to_value(material).ok().and_then(|v| match v {
            Value::String(s) => Some(material_name(&s)),
            other => other.get("name").and_then(Value::as_str).map(str::to_owned),
        });
        if let Some(text) = text {
            props.insert("material".to_owned(), text);
        }
    }
    for p in &output.properties {
        let value = serde_json::to_value(&p.value).unwrap_or(Value::Null);
        let text = value
            .as_str()
            .map(str::to_owned)
            .or_else(|| value.get("value").map(|v| v.to_string().trim_matches('"').to_owned()))
            .unwrap_or_else(|| value.to_string());
        props.insert(p.name.as_str().to_owned(), text);
    }
    props
}

/// Each BOM item's definition properties, by the definition's file as the BOM names it. Part
/// files are parsed (not regenerated): properties are declared text.
pub fn item_properties(assembly: &str, bom: &Bom, files: &Files) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for row in &bom.rows {
        let file = row.item.definition.file.as_str().to_owned();
        if out.contains_key(&file) || !file.ends_with(".part") {
            continue;
        }
        let at = resolve_path(assembly, &file, files);
        let Some(source) = files.get(&at) else {
            continue;
        };
        if let Some(document) = parse_part(&at, source).document {
            out.insert(file, output_properties(&document));
        }
    }
    out
}
