//! Assembly occurrences as `use` statements.
//!
//! ```text
//! use part bracket from "parts/bracket.part" as left {
//!     configuration = named(name = light)
//!     fixed = false
//!     overrides = [ … ]
//! }
//! use assembly gearbox from "gearbox.assembly" @ "R3" as gb
//! ```
//!
//! The kind word is the definition's kind and the name after it is the part the file
//! defines (the file's stem when the file defines one part). `as` names the occurrence;
//! `@ "<revision>"` pins it to a released revision. The block holds every other occurrence
//! field that is not at its default, as `<field> = <value>`.

use crate::refs::reference::XrefContext;
use runmat_design_syntax::ast::{self, Statement, UseSource};
use runmat_design_syntax::document::BlockEntry;
use runmat_design_syntax::json::{
    is_identifier, lower_field, print_field, quote, Json, LowerError, ValueSyntax, INDENT,
};
use runmat_types::Span;

const KINDS: &[&str] = &["part", "assembly", "circuit"];

fn stem(file: &str) -> &str {
    let name = file.rsplit(['/', '\\']).next().unwrap_or(file);
    name.split('.').next().unwrap_or(name)
}

fn error(span: Span, message: impl Into<String>) -> LowerError {
    LowerError::new(span, message)
}

/// The occurrence as a `use` statement, or `None` for the record form.
pub fn print(name: &str, fields: &[(String, Json)], hook: &dyn ValueSyntax) -> Option<String> {
    let source = fields.iter().find(|(k, _)| k == "source")?.1.clone();
    let definition = source.get("definition")?;
    let kind = definition.get("kind")?.as_str()?;
    let file = definition.get("file")?.as_str()?;
    if !KINDS.contains(&kind) || !is_identifier(name) {
        return None;
    }
    let symbol = match definition.get("part") {
        Some(Json::String(part)) if part != stem(file) && is_identifier(part) => part.clone(),
        Some(_) => return None,
        None if is_identifier(stem(file)) => stem(file).to_string(),
        None => return None,
    };
    if let Json::Object(entries) = definition {
        if entries
            .iter()
            .any(|(k, _)| !["kind", "file", "part"].contains(&k.as_str()))
        {
            return None;
        }
    }
    let pin = match source.get("context") {
        None => String::new(),
        Some(context) => match context.decode::<XrefContext>().ok()? {
            XrefContext::Live => String::new(),
            XrefContext::Pinned { revision } => format!(" @ {}", quote(&revision.0)),
        },
    };
    let mut out = format!("use {kind} {symbol} from {}{pin} as {name}", quote(file));
    let rest: Vec<&(String, Json)> = fields.iter().filter(|(k, _)| k != "source").collect();
    if !rest.is_empty() {
        out.push_str(" {\n");
        for (key, value) in rest {
            if !is_identifier(key) {
                return None;
            }
            let text = print_field(key, value, hook, None, 1);
            out.push_str(&format!("{INDENT}{key} = {text}\n"));
        }
        out.push('}');
    }
    Some(out)
}

/// Reads `use <kind> <name> from "<file>" [@ "<pin>"] as <occurrence> [{ … }]`.
pub fn lower(
    declaration: &ast::UseDeclaration,
    hook: &dyn ValueSyntax,
) -> Result<BlockEntry, LowerError> {
    let span = declaration.span();
    let (kind, symbol) = declaration.symbol().ok_or_else(|| {
        error(
            span,
            "an occurrence is `use <kind> <name> from \"<file>\" as <name>`",
        )
    })?;
    let kind = kind.text();
    if !KINDS.contains(&kind.as_str()) {
        return Err(error(
            span,
            format!("`{kind}` is not part, assembly or circuit"),
        ));
    }
    let file = match declaration.source() {
        Some(UseSource::Path(path)) => path,
        _ => return Err(error(span, "an occurrence reads a quoted file path")),
    };
    let alias = declaration
        .alias()
        .ok_or_else(|| error(span, "an occurrence needs `as <name>`"))?
        .text();
    let symbol = symbol.text();
    let mut definition = vec![
        ("kind".to_string(), Json::String(kind)),
        ("file".to_string(), Json::String(file.clone())),
    ];
    if symbol != stem(&file) {
        definition.push(("part".into(), Json::String(symbol)));
    }
    let mut source = vec![("definition".to_string(), Json::Object(definition))];
    if let Some(revision) = declaration.pin() {
        source.push((
            "context".into(),
            Json::Object(vec![
                ("context".into(), Json::String("pinned".into())),
                ("revision".into(), Json::String(revision)),
            ]),
        ));
    }
    let source = Json::Object(source);
    let mut fields = vec![("source".to_string(), source.clone())];
    let mut written = vec![("source".to_string(), source, span)];
    let mut spans = Vec::new();
    let head_end = declaration
        .overrides()
        .map_or(span.end, |block| block.span().start);
    let value_span = declaration.overrides().map(|block| block.span());
    if let Some(block) = declaration.overrides() {
        for statement in block.statements() {
            let at = statement.span();
            let Statement::Assignment(assignment) = &statement else {
                return Err(error(
                    at,
                    "an occurrence's block holds `<field> = <value>` lines",
                ));
            };
            let target = assignment.target();
            let (Some(value), [key]) = (assignment.value(), target.as_slice()) else {
                return Err(error(at, "expected `<field> = <value>`"));
            };
            if key == "source" || key == "name" || fields.iter().any(|(k, _)| k == key) {
                return Err(error(at, format!("`{key}` cannot be set here")));
            }
            let item = lower_field(key, &value, hook, None, key, &mut spans)?;
            written.push((key.clone(), item.clone(), value.span()));
            fields.push((key.clone(), item));
        }
    }
    Ok(BlockEntry {
        word: "occurrence".into(),
        name: Some(alias),
        fields,
        spans,
        written,
        head_end,
        value_span,
    })
}
