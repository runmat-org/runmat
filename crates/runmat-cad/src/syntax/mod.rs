//! The `.part`, `.assembly` and `.drawing` front ends on the shared design grammar.
//!
//! Each front end is a registry ([`PART`], [`ASSEMBLY`], [`DRAWING`]) for the shared
//! document engine ([`runmat_design_syntax::document`]): which statement word fills which model
//! collection. The engine parses a source into the model through serde and prints a model
//! back; [`value::CadValues`] gives [`ValueExpr`](crate::value::ValueExpr) its text forms.
//! Every entry and every value inside it is placed in a [`SpanMap`] under its [`Site`] key
//! (`feature.round`, `feature.round#edges[1]`), so diagnostics and structured edits point at
//! the text that wrote them.
//!
//! ```text
//! runmat 0.7
//! std 1
//!
//! use material aluminum from "./materials"
//!
//! part bracket {
//!     input width : length = 80 mm
//!     let half = input.width / 2
//!
//!     sketch s1 on origin.xy { … }
//!     feature body = extrude(profile = sketch.s1, distance = 10 mm)
//!     region mount : face = { select = [ … ], expect = exactly_one }
//!     publish hardpoint = region { region = mount }
//!     link verifies feature.body to "checks/mass.fea#check.mass_budget"
//! }
//! ```
//!
//! Round trip: `parse(print(model)) == model` for every model the printer accepts.
//!
//! `std 1` is the first released form of these dialects. Each entry has one written form (the
//! call form, a sketch block, a `use` statement); later forms will take a new `std` number
//! and files keep the number they were written for.
//!
//! [`Site`]: crate::diagnostics::Site

pub mod edit;
pub mod forms;
pub mod occurrence;
pub mod projection;
pub mod refs;
pub mod sketch;
pub mod value;

use crate::assembly::AssemblyDocument;
use crate::diagnostics::{SourceSpan, SpanMap};
use crate::drawing::DrawingDocument;
use crate::ids::FilePath;
use crate::model::PartDocument;
use crate::scope::UseDecl;
use runmat_design_syntax::document::{
    parse_document, print_document, Collection, DocumentGrammar, DocumentModel, DocumentSyntax,
    LinkGrammar, PrintError, Tagged, UseItem,
};
use runmat_design_syntax::json::Json;
use runmat_design_syntax::{Dialect, FeaSyntaxDiagnostic};
use serde::de::DeserializeOwned;
use serde::Serialize;
use value::CadValues;

const fn named(word: &'static str, field: &'static str, site: &'static str) -> Collection {
    Collection {
        word,
        field,
        id: Some("name"),
        type_field: None,
        value_field: None,
        tagged: None,
        site,
        unique: true,
    }
}

const INPUT: Collection = Collection {
    type_field: Some("kind"),
    value_field: Some("default"),
    ..named("input", "inputs", "input")
};

const LET: Collection = Collection {
    type_field: Some("kind"),
    value_field: Some("value"),
    ..named("let", "lets", "let")
};

const LINKS: LinkGrammar = LinkGrammar {
    field: "links",
    kind: "kind",
    from: "from",
    from_tag: "item",
    to: "to",
    site: "link",
};

/// `.part`: one part, its inputs, feature history, outputs, regions, publishes, materials,
/// links and configurations.
pub const PART: DocumentGrammar = DocumentGrammar {
    dialect: Dialect::Part,
    declaration: "part",
    name_field: None,
    path_field: Some("path"),
    header_field: Some("header"),
    collections: &[
        INPUT,
        LET,
        Collection {
            id: Some("id"),
            tagged: Some(Tagged {
                field: "kind",
                tag: "feature",
                extras: &["suppressed", "frozen"],
            }),
            ..named("feature", "features", "feature")
        },
        named("output", "parts", "part"),
        Collection {
            type_field: Some("kind"),
            ..named("region", "regions", "region")
        },
        Collection {
            tagged: Some(Tagged {
                field: "target",
                tag: "publishes",
                extras: &["auto"],
            }),
            ..named("publish", "publishes", "publish")
        },
        Collection {
            id: None,
            ..named("material", "materials", "material")
        },
        named("configuration", "configurations", "configuration"),
        named(
            "configuration_table",
            "configuration_tables",
            "configuration_table",
        ),
        named("feature_def", "feature_defs", "feature_def"),
        // Semantic PMI: `datum A = { on = feature.body.face.start }`,
        // `tolerance top_flat = { characteristic = flatness, on = [ … ], tolerance = 0.05 mm }`.
        Collection {
            id: Some("label"),
            ..named("datum", "datums", "datum")
        },
        named("tolerance", "tolerances", "tolerance"),
        named("size", "sizes", "size"),
        named("location", "locations", "location"),
    ],
    links: Some(LINKS),
};

/// `.assembly`: occurrences, connectors, mates, patterns, fasteners, publishes and
/// configurations of one assembly.
pub const ASSEMBLY: DocumentGrammar = DocumentGrammar {
    dialect: Dialect::Assembly,
    declaration: "assembly",
    name_field: Some("name"),
    path_field: Some("path"),
    header_field: Some("header"),
    collections: &[
        INPUT,
        LET,
        // Alternatives of a 150% assembly share a name.
        Collection {
            unique: false,
            ..named("occurrence", "occurrences", "occurrence")
        },
        named("connector", "connectors", "connector"),
        named("mate", "mates", "mate"),
        // `pattern ribs = linear(seeds = [rib], direction = [1, 0, 0], spacing = 50 mm, count = 4)`
        Collection {
            tagged: Some(Tagged {
                field: "layout",
                tag: "layout",
                extras: &["seeds", "skip", "when"],
            }),
            ..named("pattern", "patterns", "pattern")
        },
        named("mirror", "mirrors", "mirror"),
        named("fasteners", "fasteners", "fasteners"),
        named("envelope", "envelopes", "envelope"),
        named("publish", "publishes", "publish"),
        named("configuration", "configurations", "configuration"),
        named("property", "properties", "property"),
        named("exploded", "exploded", "exploded"),
    ],
    links: Some(LINKS),
};

/// `.drawing`: sheet formats, sheets, views, annotations and tables.
pub const DRAWING: DocumentGrammar = DocumentGrammar {
    dialect: Dialect::Drawing,
    declaration: "drawing",
    name_field: None,
    path_field: Some("path"),
    header_field: Some("header"),
    collections: &[
        named("format", "formats", "format"),
        named("sheet", "sheets", "sheet"),
        // `view front = standard(orientation = front, sheet = "1", model = …, at = [100, 100])`
        Collection {
            tagged: Some(Tagged {
                field: "kind",
                tag: "view",
                extras: &[
                    "sheet", "model", "at", "scale", "display", "exclude", "exploded",
                ],
            }),
            ..named("view", "views", "view")
        },
        // `annotation d1 = dimension(style = linear, references = […], view = front, at = [0, 0])`
        Collection {
            id: Some("id"),
            tagged: Some(Tagged {
                field: "kind",
                tag: "annotation",
                extras: &["view", "at"],
            }),
            ..named("annotation", "annotations", "annotation")
        },
        Collection {
            tagged: Some(Tagged {
                field: "kind",
                tag: "table",
                extras: &["sheet", "at"],
            }),
            ..named("table", "tables", "table")
        },
    ],
    links: None,
};

/// A parsed design file.
#[derive(Debug, Clone, PartialEq)]
pub struct Parsed<T> {
    /// The model, when the source is free of errors.
    pub document: Option<T>,
    pub uses: Vec<UseDecl>,
    /// Site key → where it is written.
    pub spans: SpanMap,
    pub diagnostics: Vec<FeaSyntaxDiagnostic>,
    /// The engine's output, for edits and projections.
    pub syntax: DocumentSyntax,
}

impl<T> Parsed<T> {
    pub fn is_valid(&self) -> bool {
        self.document.is_some() && self.diagnostics.is_empty()
    }
}

struct Model<T>(std::marker::PhantomData<T>);

impl<T: Serialize + DeserializeOwned> DocumentModel for Model<T> {
    fn normalize(&self, json: &Json) -> Option<Json> {
        let model: T = json.decode().ok()?;
        Json::from_serialize(&model).ok()
    }
}

fn use_decl(item: &UseItem) -> UseDecl {
    UseDecl {
        symbol: item.symbol.clone(),
        from: item.from.clone(),
        alias: item.alias.clone(),
        revision: item.revision.clone(),
    }
}

fn use_item(decl: &UseDecl) -> UseItem {
    UseItem {
        symbol: decl.symbol.clone(),
        from: decl.from.clone(),
        alias: decl.alias.clone(),
        revision: decl.revision.clone(),
        span: Default::default(),
    }
}

/// Parses `source` with any of the CAD grammars.
pub fn grammar_parse<T: DeserializeOwned>(
    grammar: &DocumentGrammar,
    path: &str,
    source: &str,
) -> Parsed<T> {
    let syntax = parse_document(grammar, path, source, &CadValues);
    let mut diagnostics = syntax.diagnostics.clone();
    let document = if diagnostics.is_empty() {
        match syntax.json.decode::<T>() {
            Ok(document) => Some(document),
            Err((json_path, message)) => {
                diagnostics.push(syntax.model_error(&json_path, &message));
                None
            }
        }
    } else {
        None
    };
    let file = FilePath::new(path);
    let spans = SpanMap {
        spans: syntax
            .spans
            .iter()
            .map(|(key, span)| {
                (
                    key.clone(),
                    SourceSpan {
                        file: file.clone(),
                        start: span.start as u32,
                        end: span.end as u32,
                    },
                )
            })
            .collect(),
    };
    Parsed {
        document,
        uses: syntax.uses.iter().map(use_decl).collect(),
        spans,
        diagnostics,
        syntax,
    }
}

/// Prints a model with any of the CAD grammars. A grammar without a name field names the
/// declaration after the file stem.
pub fn print_with<T: Serialize + DeserializeOwned>(
    grammar: &DocumentGrammar,
    document: &T,
    uses: &[UseDecl],
) -> Result<String, PrintError> {
    let json = Json::from_serialize(document).map_err(|error| PrintError(error.to_string()))?;
    let path = grammar
        .path_field
        .and_then(|field| json.get(field))
        .and_then(Json::as_str)
        .unwrap_or_default()
        .to_string();
    let uses: Vec<UseItem> = uses.iter().map(use_item).collect();
    print_document(
        grammar,
        &Model::<T>(std::marker::PhantomData),
        &json,
        &uses,
        &grammar.dialect.stem(&path),
        &CadValues,
    )
}

pub fn parse_part(path: &str, source: &str) -> Parsed<PartDocument> {
    grammar_parse(&PART, path, source)
}

pub fn print_part(document: &PartDocument, uses: &[UseDecl]) -> Result<String, PrintError> {
    print_with(&PART, document, uses)
}

pub fn parse_assembly(path: &str, source: &str) -> Parsed<AssemblyDocument> {
    grammar_parse(&ASSEMBLY, path, source)
}

pub fn print_assembly(document: &AssemblyDocument, uses: &[UseDecl]) -> Result<String, PrintError> {
    print_with(&ASSEMBLY, document, uses)
}

pub fn parse_drawing(path: &str, source: &str) -> Parsed<DrawingDocument> {
    grammar_parse(&DRAWING, path, source)
}

pub fn print_drawing(document: &DrawingDocument, uses: &[UseDecl]) -> Result<String, PrintError> {
    print_with(&DRAWING, document, uses)
}

/// The front end's registry for a dialect.
pub fn grammar(dialect: Dialect) -> Option<&'static DocumentGrammar> {
    match dialect {
        Dialect::Part => Some(&PART),
        Dialect::Assembly => Some(&ASSEMBLY),
        Dialect::Drawing => Some(&DRAWING),
        _ => None,
    }
}
