//! The syntax projection: a versioned JSON outline of a parsed source, for indexes, external
//! tools and fixtures. It lists the version header, every file-level item and every statement
//! with its kind, words, name and spans. Each front end adds its own model projection beside
//! it (`runmat-cad`'s `syntax::projection`).
//!
//! The format is `runmat.design-syntax`, version [`SYNTAX_PROJECTION_VERSION`]. A field is
//! only added in a new version; a reader that knows version 1 reads every later version.

use crate::ast::{self, Headed, Item, Statement};
use crate::tree::SyntaxNode;
use crate::SourceSyntax;
use runmat_types::Span;
use serde_json::{json, Value};

pub const SYNTAX_PROJECTION_FORMAT: &str = "runmat.design-syntax";
pub const SYNTAX_PROJECTION_VERSION: u32 = 1;

fn span(span: Span) -> Value {
    json!([span.start, span.end])
}

fn kind_name(node: &SyntaxNode) -> Value {
    serde_json::to_value(node.kind()).unwrap_or(Value::Null)
}

/// The outline of a source of any dialect.
pub fn syntax_projection(syntax: &SourceSyntax) -> Value {
    let file = syntax.file();
    let items: Vec<Value> = file.items().iter().map(item).collect();
    json!({
        "format": SYNTAX_PROJECTION_FORMAT,
        "version": SYNTAX_PROJECTION_VERSION,
        "dialect": syntax.dialect,
        "language": file.version(false),
        "std": file.version(true),
        "items": items,
        "diagnostics": syntax.diagnostics.iter().map(|d| json!({
            "kind": d.kind,
            "message": d.message,
            "span": span(d.span),
        })).collect::<Vec<_>>(),
    })
}

fn item(item: &Item) -> Value {
    match item {
        Item::Use(declaration) => json!({
            "kind": "use",
            "symbol": declaration.symbol().map(|(kind, name)| json!([kind.text(), name.text()])),
            "source": match declaration.source() {
                Some(ast::UseSource::Path(path)) => json!({ "path": path }),
                Some(ast::UseSource::Package(segments)) => json!({ "package": segments }),
                None => Value::Null,
            },
            "pin": declaration.pin(),
            "alias": declaration.alias().map(|alias| alias.text()),
            "span": span(declaration.span()),
        }),
        Item::Import(import) => json!({
            "kind": "import",
            "import_kind": import.kind(),
            "name": import.name().map(|name| name.text()),
            "path": import.path(),
            "span": span(import.span()),
        }),
        Item::Declaration(declaration) => json!({
            "kind": "declaration",
            "word": declaration.keyword(),
            "name": declaration.name().map(|name| name.text()),
            "type": declaration.type_text(),
            "span": span(declaration.span()),
            "statements": declaration
                .block()
                .map(|block| block.statements().iter().map(statement).collect::<Vec<_>>())
                .unwrap_or_default(),
        }),
        Item::Error(node) => json!({ "kind": "error", "span": span(node.trimmed_span()) }),
    }
}

fn statement(statement: &Statement) -> Value {
    let mut out = json!({
        "kind": kind_name(statement.syntax()),
        "span": span(statement.span()),
    });
    let object = out.as_object_mut().expect("an object");
    let mut set = |key: &str, value: Value| {
        if !value.is_null() {
            object.insert(key.to_string(), value);
        }
    };
    match statement {
        Statement::Value(value) => {
            set("word", json!(value.keyword()));
            set("name", json!(value.name().map(|n| n.text())));
            set("type", json!(value.type_text()));
            set("value", json!(value.value().map(|v| span(v.span()))));
        }
        Statement::Assignment(assignment) => {
            set("target", json!(assignment.target()));
            set("value", json!(assignment.value().map(|v| span(v.span()))));
        }
        Statement::Constrain(constrain) => {
            set(
                "value",
                json!(constrain.expression().map(|v| span(v.span()))),
            );
            set("equals", json!(constrain.equals().map(|v| span(v.span()))));
        }
        Statement::When(when) => {
            set("condition", json!(when.condition().map(|v| span(v.span()))));
            set("statements", block(when.block()));
        }
        Statement::Use(declaration) => {
            set("use", item(&Item::Use(declaration.clone())));
        }
        Statement::Publish(publish) => {
            set("name", json!(publish.name().map(|n| n.text())));
            set("value", json!(publish.value().map(|v| span(v.span()))));
        }
        Statement::Check(check) => {
            set("name", json!(check.name().map(|n| n.text())));
            set("value", json!(check.value().map(|v| span(v.span()))));
            set("statements", block(check.block()));
        }
        Statement::Link(link) => {
            set("link_kind", json!(link.kind().map(|n| n.text())));
            set("from", json!(link.from().map(|v| span(v.span()))));
            set("to", json!(link.to().map(|v| span(v.span()))));
        }
        Statement::Solved(solved) => set("statements", block(solved.block())),
        Statement::Block(statement) => {
            set("word", json!(statement.keyword()));
            set("name", json!(statement.name().map(|n| n.text())));
            set("type", json!(statement.type_text()));
            set(
                "clauses",
                json!(statement
                    .clauses()
                    .iter()
                    .map(|clause| json!({ "word": clause.word(), "span": span(clause.span()) }))
                    .collect::<Vec<_>>()),
            );
            set("statements", block(statement.block()));
        }
        Statement::Command(command) => {
            set("word", json!(command.word()));
            set("value", json!(command.argument().map(|v| span(v.span()))));
        }
        Statement::Fea(_) => {}
    }
    out
}

fn block(block: Option<ast::Block>) -> Value {
    match block {
        Some(block) => Value::Array(block.statements().iter().map(statement).collect()),
        None => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_source, Dialect};

    #[test]
    fn the_projection_outlines_items_and_nested_statements() {
        let source = "runmat 0.7\nstd 1\nuse \"./lib\" as lib\npart p {\n    input w : length = 8 mm\n    construct {\n        plane base = origin.xy\n    }\n    link verifies feature.body to \"a.fea#check.c\"\n}\n";
        let projection = syntax_projection(&parse_source(Dialect::Part, source));
        assert_eq!(projection["format"], SYNTAX_PROJECTION_FORMAT);
        assert_eq!(projection["version"], 1);
        assert_eq!(projection["dialect"], "part");
        assert_eq!(projection["language"], "0.7");
        assert_eq!(projection["items"][0]["alias"], "lib");
        let part = &projection["items"][1];
        assert_eq!(part["name"], "p");
        assert_eq!(part["statements"][0]["kind"], "value_statement");
        assert_eq!(part["statements"][0]["type"], "length");
        assert_eq!(part["statements"][1]["statements"][0]["name"], "base");
        assert_eq!(part["statements"][2]["link_kind"], "verifies");
        assert_eq!(projection["diagnostics"], json!([]));
    }
}
