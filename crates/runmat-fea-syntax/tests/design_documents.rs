//! The registry-driven document engine over a small serde model.

use runmat_fea_syntax::document::{
    parse_document, print_document, Collection, DocumentGrammar, DocumentModel, LinkGrammar,
    Tagged, UseItem,
};
use runmat_fea_syntax::json::{Json, PlainData};
use runmat_fea_syntax::Dialect;
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

    feature body = extrude { profile = "sketch.profile", depth = 8 }
    feature round = fillet { edges = [e1, e2], radius = 2, suppressed = true }

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
        "feature round = fillet { edges = [e1, e2], radius = 2, suppressed = true }"
    );
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
