//! The properties a drawing reads from the parts it shows: a part's output properties for its
//! own views and title block, and each BOM item's definition properties for parts lists
//! (`description`, `finish`, `vendor` …); a circuit board's description comes from its compiled
//! board.
use crate::runtime::{resolve_path, Files};
use runmat_cad::assembly::bom::Bom;
use runmat_cad::drawing::bom_table::{identifier_words, material_name};
use runmat_cad::ids::ConfigurationName;
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
            .or_else(|| {
                value
                    .get("value")
                    .map(|v| v.to_string().trim_matches('"').to_owned())
            })
            .unwrap_or_else(|| value.to_string());
        props.insert(p.name.as_str().to_owned(), text);
    }
    props
}

/// The key [`item_properties`] files an item's properties under: its definition's file, and
/// for a named configuration other than the default `<file>|<configuration>`.
pub fn item_key(file: &str, configuration: &str) -> String {
    if configuration == "default" {
        file.to_owned()
    } else {
        format!("{file}|{configuration}")
    }
}

/// The named configuration a BOM configuration label starts with (`port`, `port (side -1)`).
fn label_configuration(label: &str) -> Option<ConfigurationName> {
    let name = label.split(" (").next().unwrap_or(label);
    (name != "default" && !name.contains(' ')).then(|| ConfigurationName::new(name))
}

/// Each BOM item's definition properties, by [`item_key`]: a configuration that sets its own
/// description or properties has them. Part files are parsed (not regenerated): properties are
/// declared text.
pub fn item_properties(
    assembly: &str,
    bom: &Bom,
    files: &Files,
) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for row in &bom.rows {
        let file = row.item.definition.file.as_str().to_owned();
        let key = item_key(&file, &row.item.configuration);
        if out.contains_key(&key) {
            continue;
        }
        let at = resolve_path(assembly, &file, files);
        // A circuit board: its description from the compiled board.
        if file.ends_with(".circuit") || file.ends_with(".circuit.layout") {
            if let Ok(description) = crate::board::board_description(&at, files) {
                out.insert(
                    file,
                    BTreeMap::from([("description".to_owned(), description)]),
                );
            }
            continue;
        }
        if !file.ends_with(".part") {
            continue;
        }
        let Some(source) = files.get(&at) else {
            continue;
        };
        if let Some(document) = parse_part(&at, source).document {
            let configuration = label_configuration(&row.item.configuration)
                .filter(|name| document.configuration(name).is_some());
            let document = document.in_configuration(configuration.as_ref());
            let mut props = output_properties(&document);
            if let Some(description) = description_of(&document, &props) {
                props.insert("description".to_owned(), description);
            }
            out.insert(key, props);
        }
    }
    out
}

/// What a parts list's Description column reads for a part: its `description` property, else
/// its output's label (`label = "Cold plate"`) or `title` property, else its output's name as
/// words (`body_top` → "Body top", `esc_heatsink` → "ESC heatsink").
pub fn description_of(document: &PartDocument, props: &BTreeMap<String, String>) -> Option<String> {
    let given = |k: &str| props.get(k).filter(|v| !v.trim().is_empty()).cloned();
    if let Some(d) = given("description") {
        return Some(d);
    }
    let output = document.parts.first()?;
    if let Some(label) = output.label.as_ref().filter(|l| !l.trim().is_empty()) {
        return Some(label.clone());
    }
    given("title").or_else(|| words(output.name.as_str()))
}

/// An identifier as words, acronyms upper case: `body_top` → "Body top", `esc_heatsink` →
/// "ESC heatsink".
fn words(name: &str) -> Option<String> {
    let words = identifier_words(name, true);
    (!words.is_empty()).then_some(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A part with no description or label reads its output's name, acronyms upper case.
    #[test]
    fn an_undescribed_part_reads_its_name_with_acronyms_upper_case() {
        let source = "runmat 0.7\nstd 1\n\npart esc_heatsink {\n    sketch s on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = 5 mm)\n    }\n    feature base = extrude(profile = sketch.s, distance = 2 mm)\n    output esc_heatsink = { body = feature.base.body }\n}\n";
        let document = parse_part("/p/esc_heatsink.part", source)
            .document
            .expect("parses");
        assert_eq!(
            description_of(&document, &BTreeMap::new()).as_deref(),
            Some("ESC heatsink")
        );
    }
}
