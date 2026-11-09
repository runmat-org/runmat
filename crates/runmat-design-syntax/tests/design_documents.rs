//! The registry-driven document engine over a small serde model.

use runmat_design_syntax::document::{
    parse_document, print_document, Collection, DocumentGrammar, DocumentModel, LinkGrammar,
    Tagged, UseItem,
};
use runmat_design_syntax::json::{Json, PlainData};
use runmat_design_syntax::Dialect;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Header {
    language: String,
    std: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Input {
    name: String,
    kind: String,
    default: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "feature", rename_all = "snake_case")]
enum Kind {
    Extrude {
        profile: String,
        depth: f64,
        #[serde(default)]
        reverse: bool,
    },
    Fillet {
        edges: Vec<String>,
        radius: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Feature {
    id: String,
    kind: Kind,
    #[serde(default)]
    suppressed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Material {
    body: String,
    material: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "item", rename_all = "snake_case")]
enum Local {
    Feature { feature: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Link {
    kind: String,
    from: Local,
    to: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Doc {
    path: String,
    header: Header,
    inputs: Vec<Input>,
    features: Vec<Feature>,
    #[serde(default)]
    materials: Vec<Material>,
    #[serde(default)]
    links: Vec<Link>,
    #[serde(default)]
    closed: bool,
    notes: Vec<String>,
}

const GRAMMAR: DocumentGrammar = DocumentGrammar {
    dialect: Dialect::Part,
    declaration: "part",
    name_field: None,
    path_field: Some("path"),
    header_field: Some("header"),
    collections: &[
        Collection {
            word: "input",
            field: "inputs",
            id: Some("name"),
            type_field: Some("kind"),
            value_field: Some("default"),
            tagged: None,
            site: "input",
            unique: true,
        },
        Collection {
            word: "feature",
            field: "features",
            id: Some("id"),
            type_field: None,
            value_field: None,
            tagged: Some(Tagged {
                field: "kind",
                tag: "feature",
                extras: &["suppressed"],
            }),
            site: "feature",
            unique: true,
        },
        Collection {
            word: "material",
            field: "materials",
            id: None,
            type_field: None,
            value_field: None,
            tagged: None,
            site: "material",
            unique: true,
        },
    ],
    links: Some(LinkGrammar {
        field: "links",
        kind: "kind",
        from: "from",
        from_tag: "item",
        to: "to",
        site: "link",
    }),
};

struct Model;

impl DocumentModel for Model {
    fn normalize(&self, json: &Json) -> Option<Json> {
        let doc: Doc = json.decode().ok()?;
        Json::from_serialize(&doc).ok()
    }
}

fn doc() -> Doc {
    Doc {
        path: "parts/plate.part".into(),
        header: Header {
            language: "0.7".into(),
            std: "1".into(),
        },
        inputs: vec![
            Input {
                name: "width".into(),
                kind: "length".into(),
                default: 80.0,
                maximum: None,
            },
            Input {
                name: "depth".into(),
                kind: "length".into(),
                default: 8.5,
                maximum: Some(20.0),
            },
        ],
        features: vec![
            Feature {
                id: "body".into(),
                kind: Kind::Extrude {
                    profile: "sketch.profile".into(),
                    depth: 8.0,
                    reverse: false,
                },
                suppressed: false,
            },
            Feature {
                id: "round".into(),
                kind: Kind::Fillet {
                    edges: vec!["e1".into(), "e2".into()],
                    radius: 2.0,
                },
                suppressed: true,
            },
        ],
        materials: vec![Material {
            body: "body".into(),
            material: "aluminum".into(),
        }],
        links: vec![Link {
            kind: "verifies".into(),
            from: Local::Feature {
                feature: "body".into(),
            },
            to: "checks/mass.fea#check.mass".into(),
        }],
        closed: false,
        notes: vec![],
    }
}

fn print(doc: &Doc, uses: &[UseItem]) -> String {
    let json = Json::from_serialize(doc).unwrap();
    print_document(&GRAMMAR, &Model, &json, uses, "plate", &PlainData).unwrap()
}

#[test]
fn a_model_prints_and_parses_back_to_the_same_model() {
    let uses = vec![UseItem {
        symbol: Some(("material".into(), "aluminum".into())),
        from: "./materials".into(),
        alias: Some("al".into()),
        revision: Some("R3".into()),
        span: Default::default(),
    }];
    let text = print(&doc(), &uses);
    let expected = r#"runmat 0.7
std 1

use material aluminum from "./materials" @ "R3" as al

part plate {
    input width : length = 80
    input depth : length = { default = 8.5, maximum = 20 }

    feature body = extrude(profile = "sketch.profile", depth = 8)
    feature round = fillet(edges = [e1, e2], radius = 2, suppressed)

    material = { body = body, material = aluminum }

    link verifies feature.body to "checks/mass.fea#check.mass"

    notes = []
}
"#;
    assert_eq!(text, expected);
    let parsed = parse_document(&GRAMMAR, "parts/plate.part", &text, &PlainData);
    assert!(parsed.diagnostics.is_empty(), "{:#?}", parsed.diagnostics);
    let back: Doc = parsed.json.decode().unwrap();
    assert_eq!(back, doc());
    assert_eq!(parsed.uses.len(), 1);
    assert_eq!(
        UseItem {
            span: Default::default(),
            ..parsed.uses[0].clone()
        },
        uses[0]
    );
    let at = |key: &str| {
        let span = parsed.spans[key];
        &text[span.start..span.end]
    };
    assert_eq!(
        at("feature.round"),
        "feature round = fillet(edges = [e1, e2], radius = 2, suppressed)"
    );
    // The record form and named flags read as the same model.
    let old = text
        .replace(
            "fillet(edges = [e1, e2], radius = 2, suppressed)",
            "fillet { edges = [e1, e2], radius = 2, suppressed = true }",
        )
        .replace("depth = 8)", "depth = 8, reverse = false)");
    let parsed_old = parse_document(&GRAMMAR, "parts/plate.part", &old, &PlainData);
    assert!(
        parsed_old.diagnostics.is_empty(),
        "{:#?}",
        parsed_old.diagnostics
    );
    assert_eq!(parsed_old.json.decode::<Doc>().unwrap(), doc());
    assert_eq!(at("feature.round#edges[1]"), "e2");
    assert_eq!(at("feature.round#radius"), "2");
    assert_eq!(at("input.depth#maximum"), "20");
    assert_eq!(at("input.width#default"), "80");
    assert_eq!(at("material[0]#material"), "aluminum");
    assert_eq!(at("notes"), "notes = []");
    assert_eq!(
        parsed.span_of_json_path("features[1].kind.edges[0]"),
        parsed.spans["feature.round#edges[0]"]
    );
}

#[test]
fn model_errors_point_at_the_text_that_wrote_them() {
    let text = "part plate {\n    input width : length = \"wide\"\n    notes = []\n}\n";
    let parsed = parse_document(&GRAMMAR, "plate.part", text, &PlainData);
    assert!(parsed.diagnostics.is_empty(), "{:#?}", parsed.diagnostics);
    let (path, message) = parsed.json.decode::<Doc>().unwrap_err();
    assert_eq!(path, "inputs[0].default");
    let diagnostic = parsed.model_error(&path, &message);
    assert_eq!(
        &text[diagnostic.span.start..diagnostic.span.end],
        "\"wide\""
    );
}

#[test]
fn misplaced_statements_are_diagnosed() {
    let text = "part other {\n    widget w = 1\n    feature f = 3\n    path = \"x\"\n    feature f = extrude { profile = p, depth = 1 }\n    feature f = extrude { profile = p, depth = 1 }\n}\n";
    let parsed = parse_document(&GRAMMAR, "plate.part", text, &PlainData);
    let messages: Vec<&str> = parsed
        .diagnostics
        .iter()
        .map(|d| d.message.as_str())
        .collect();
    assert!(
        messages.iter().any(|m| m.contains("named `other`")),
        "{messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("`widget` does not start")),
        "{messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("expected `feature … = { … }`")),
        "{messages:?}"
    );
    assert!(
        messages.iter().any(|m| m.contains("`path` is not set")),
        "{messages:?}"
    );
    assert!(
        messages.iter().any(|m| m.contains("declared twice")),
        "{messages:?}"
    );
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Check {
    name: String,
    rule: String,
    limit: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Checked {
    path: String,
    checks: Vec<Check>,
}

struct CheckedModel;

impl DocumentModel for CheckedModel {
    fn normalize(&self, json: &Json) -> Option<Json> {
        let model: Checked = json.decode().ok()?;
        Json::from_serialize(&model).ok()
    }
}

const CHECKED: DocumentGrammar = DocumentGrammar {
    dialect: Dialect::Part,
    declaration: "part",
    name_field: None,
    path_field: Some("path"),
    header_field: None,
    collections: &[Collection {
        word: "check",
        field: "checks",
        id: Some("name"),
        type_field: None,
        value_field: None,
        tagged: None,
        site: "check",
        unique: true,
    }],
    links: None,
};

#[test]
fn a_check_reads_in_its_block_form() {
    let text = "part p {\n    check mass {\n        rule = below\n        limit = 2.5\n    }\n    check span = { rule = above, limit = 1 }\n}\n";
    let parsed = parse_document(&CHECKED, "p.part", text, &PlainData);
    assert!(parsed.diagnostics.is_empty(), "{:#?}", parsed.diagnostics);
    let doc: Checked = parsed.json.decode().unwrap();
    assert_eq!(
        doc.checks[0],
        Check {
            name: "mass".into(),
            rule: "below".into(),
            limit: 2.5
        }
    );
    assert_eq!(doc.checks[1].limit, 1.0);
    let at = |key: &str| {
        let span = parsed.spans[key];
        &text[span.start..span.end]
    };
    assert_eq!(at("check.mass#limit"), "2.5");
    assert!(at("check.mass").starts_with("check mass {"));
    let printed = print_document(
        &CHECKED,
        &CheckedModel,
        &Json::from_serialize(&doc).unwrap(),
        &[],
        "p",
        &PlainData,
    )
    .unwrap();
    // Checks print in the block form, one field per line.
    assert!(
        printed.contains("    check mass {\n        rule = below\n        limit = 2.5\n    }\n"),
        "{printed}"
    );
    assert!(printed.contains("    check span {\n"), "{printed}");
    let back = parse_document(&CHECKED, "p.part", &printed, &PlainData);
    assert_eq!(back.json.decode::<Checked>().unwrap(), doc);
    let bad = "part p {\n    check mass {\n        rule below\n    }\n}\n";
    assert!(!parse_document(&CHECKED, "p.part", bad, &PlainData)
        .diagnostics
        .is_empty());
}
