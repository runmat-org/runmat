//! Hovers, completion and signature help from the word registries, on the L-bracket.

use super::*;

const L_BRACKET: &str = include_str!("../../../../runmat-cad/tests/fixtures/notes/l_bracket.part");

fn analysis() -> DesignDocumentAnalysis {
    let uri = Url::parse("file:///p/l_bracket.part").unwrap();
    DesignDocumentAnalysis::analyze(&uri, Dialect::Part, L_BRACKET, &DesignFiles::new(), &DesignValues::new())
}

fn at(needle: &str, into: usize) -> Position {
    let offset = L_BRACKET.find(needle).unwrap_or_else(|| panic!("{needle}")) + into;
    offset_to_position(L_BRACKET, offset)
}

fn hover(a: &DesignDocumentAnalysis, needle: &str, into: usize) -> String {
    match a.hover(&at(needle, into)).map(|h| h.contents) {
        Some(HoverContents::Markup(m)) => m.value,
        other => panic!("no hover at {needle}: {other:?}"),
    }
}

#[test]
fn keywords_operations_arguments_values_and_units_hover() {
    let a = analysis();
    let keyword = hover(&a, "feature body", 2);
    assert!(keyword.contains("A modeling step"), "{keyword}");
    let operation = hover(&a, "extrude(", 2);
    assert!(operation.starts_with("```runmat-part\nextrude(profile, distance"), "{operation}");
    assert!(operation.contains("- `distance` length (mm): How far the extrude goes"), "{operation}");
    let argument = hover(&a, "diameter = 5.5 mm", 2);
    assert!(argument.contains("**`diameter`** · length (mm) · argument of `hole`"), "{argument}");
    let flag = hover(&a, "through_all\n", 2);
    assert!(flag.contains("argument of `hole`"), "{flag}");
    let unit = hover(&a, "mm)", 0);
    assert!(unit.contains("millimeter · length"), "{unit}");
    assert!(unit.contains("1 mm = 0.001 m"), "{unit}");
    let primitive = hover(&a, "line(start", 1);
    assert!(primitive.contains("A straight sketch line"), "{primitive}");
    let point = hover(&a, "point(15 mm, 25 mm, 3 mm)", 1);
    assert!(point.contains("point(x, y, z)"), "{point}");
    let role = hover(&a, "side(profile.p3)", 1);
    assert!(role.contains("reference path"), "{role}");
}

#[test]
fn materials_hover_with_their_library_properties() {
    let a = analysis();
    let text = hover(&a, "Aluminum 6061\"", 3);
    assert!(text.contains("**Aluminum 6061-T6** · material `aluminum_6061_t6`"), "{text}");
    assert!(text.contains("- Density: 2700 kg/m³"), "{text}");
    assert!(text.contains("- Yield strength:"), "{text}");
}

#[test]
fn reference_hovers_show_the_entities_the_host_supplied() {
    let facts = serde_json::json!({
        "base_holes/face": [1, [{ "entity": "face", "surface": "planar", "area": 0.0021,
            "centroid": [0.03, 0.0215, 0.003], "normal": [0.0, 0.0, 1.0] }]]
    });
    let a = analysis().with_entities(Some(&facts));
    let text = hover(&a, "face = feature.body.face.side(profile.p3)", 20);
    assert!(text.contains("**Resolves to** 1 entity"), "{text}");
    assert!(text.contains("- planar face, 2100 mm², normal (0, 0, 1)"), "{text}");
}

#[test]
fn reference_hovers_resolve_on_the_regenerated_part() {
    let a = analysis();
    let text = hover(&a, "face = feature.body.face.side(profile.p3)", 20);
    assert!(text.contains("**Resolves to** 1 entity in the regenerated part\n- planar face"), "{text}");
}

#[test]
fn completion_offers_arguments_values_and_materials() {
    let a = analysis();
    let text = "    feature cut = extrude(profile = sketch.profile, ";
    let source = format!("{}{text}\n}}\n", &L_BRACKET[..L_BRACKET.rfind('}').unwrap()]);
    let uri = Url::parse("file:///p/l_bracket.part").unwrap();
    let edited = DesignDocumentAnalysis::analyze(&uri, Dialect::Part, &source, &DesignFiles::new(), &DesignValues::new());
    let end = source.find(text).unwrap() + text.len();
    let items = edited.completion_items(&offset_to_position(&source, end));
    let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
    assert!(labels.contains(&"distance") && labels.contains(&"mode"), "{labels:?}");
    assert!(!labels.contains(&"profile"), "written arguments are left out: {labels:?}");
    let mode = format!("{}    feature cut = extrude(profile = sketch.profile, distance = 2 mm, mode = \n}}\n", &L_BRACKET[..L_BRACKET.rfind('}').unwrap()]);
    let edited = DesignDocumentAnalysis::analyze(&uri, Dialect::Part, &mode, &DesignFiles::new(), &DesignValues::new());
    let items = edited.completion_items(&offset_to_position(&mode, mode.find("mode = ").unwrap() + 7));
    let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
    assert_eq!(labels, vec!["new", "add", "cut", "intersect"]);
    let material = a.completion_items(&at("Aluminum 6061\"", 0));
    assert!(material.iter().any(|i| i.label == "Aluminum 6061-T6"), "{material:?}");
}

#[test]
fn signature_help_marks_the_argument_being_written() {
    let a = analysis();
    let help = a.signature_help(&at("diameter = 5.5 mm", 4)).expect("help");
    let signature = &help.signatures[0];
    assert!(signature.label.starts_with("hole(face, at, diameter"), "{}", signature.label);
    assert_eq!(help.active_parameter, Some(2));
}

/// The five constructs the UI flow hovers in l_bracket.part; `--nocapture` prints their text.
#[test]
fn five_l_bracket_constructs_hover() {
    let a = analysis();
    for (needle, into) in [
        ("feature body", 2),
        ("extrude(", 2),
        ("diameter = 5.5 mm", 2),
        ("mm)", 0),
        ("face = feature.body.face.side(profile.p3)", 20),
    ] {
        let text = hover(&a, needle, into);
        assert!(!text.is_empty());
        eprintln!("=== {needle}\n{text}\n");
    }
}
