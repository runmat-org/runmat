//! Design values as data: the bridge between design-language expressions and JSON.
//!
//! Front ends build their models through serde, so a model's authored data has a JSON form.
//! This module prints a JSON value as design-language text and lowers a parsed expression back
//! to JSON, losslessly for every value the printer writes:
//!
//! | JSON | Text |
//! |---|---|
//! | `null` | `none` |
//! | `true`, `false` | `true`, `false` |
//! | a number | the number (`-3`, `2.5`) |
//! | a string | a bare word when it reads back as one (`exactly_one`), else a quoted string |
//! | an array | `[a, b]` |
//! | an object | `{ key = value }`, keys bare or quoted |
//!
//! A dialect's own value forms (quantities, tolerances, references, calls) go through a
//! [`ValueSyntax`] hook, which sees each value first. [`Json`] keeps object keys in the order
//! the model serialises them, so printed records follow the struct's field order whatever
//! features `serde_json` is built with.

use crate::ast::{self, Expr};
use crate::lexical::{tokenize_design, FeaTokenKind};
use runmat_types::Span;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A JSON value whose objects keep their key order.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// The serialised form of `value`, keys in serialisation order.
    pub fn from_serialize<T: Serialize + ?Sized>(value: &T) -> serde_json::Result<Json> {
        let text = serde_json::to_string(value)?;
        serde_json::from_str(&text)
    }

    pub fn to_value(&self) -> serde_json::Value {
        match self {
            Json::Null => serde_json::Value::Null,
            Json::Bool(value) => serde_json::Value::Bool(*value),
            Json::Number(value) => serde_json::Value::Number(value.clone()),
            Json::String(value) => serde_json::Value::String(value.clone()),
            Json::Array(items) => {
                serde_json::Value::Array(items.iter().map(Json::to_value).collect())
            }
            Json::Object(entries) => serde_json::Value::Object(
                entries
                    .iter()
                    .map(|(key, value)| (key.clone(), value.to_value()))
                    .collect(),
            ),
        }
    }

    /// Deserialises a model from this value; the error names the JSON path that failed.
    pub fn decode<T: serde::de::DeserializeOwned>(&self) -> Result<T, (String, String)> {
        let value = self.to_value();
        serde_path_to_error::deserialize(value).map_err(|error| {
            let path = error.path().to_string();
            (
                if path == "." { String::new() } else { path },
                error.into_inner().to_string(),
            )
        })
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(value) => Some(value),
            _ => None,
        }
    }

    /// Whether this is `null`, `false`, `0`, `""`, `[]` or `{}`.
    pub fn is_empty_value(&self) -> bool {
        match self {
            Json::Null | Json::Bool(false) => true,
            Json::String(value) => value.is_empty(),
            Json::Array(items) => items.is_empty(),
            Json::Object(entries) => entries.is_empty(),
            _ => false,
        }
    }
}

impl Serialize for Json {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self {
            Json::Null => serializer.serialize_unit(),
            Json::Bool(value) => serializer.serialize_bool(*value),
            Json::Number(value) => value.serialize(serializer),
            Json::String(value) => serializer.serialize_str(value),
            Json::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Json::Object(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct JsonVisitor;
        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = Json;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_none<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Json, D::Error> {
                <Json as Deserialize>::deserialize(d)
            }
            fn visit_bool<E>(self, value: bool) -> Result<Json, E> {
                Ok(Json::Bool(value))
            }
            fn visit_i64<E>(self, value: i64) -> Result<Json, E> {
                Ok(Json::Number(value.into()))
            }
            fn visit_u64<E>(self, value: u64) -> Result<Json, E> {
                Ok(Json::Number(value.into()))
            }
            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Json, E> {
                serde_json::Number::from_f64(value)
                    .map(Json::Number)
                    .ok_or_else(|| E::custom("a non-finite number"))
            }
            fn visit_str<E>(self, value: &str) -> Result<Json, E> {
                Ok(Json::String(value.to_string()))
            }
            fn visit_string<E>(self, value: String) -> Result<Json, E> {
                Ok(Json::String(value))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Json::Array(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut entries = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, Json>()? {
                    entries.push((key, value));
                }
                Ok(Json::Object(entries))
            }
        }
        deserializer.deserialize_any(JsonVisitor)
    }
}

/// A value that could not be lowered to data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LowerError {
    pub span: Span,
    pub message: String,
}

impl LowerError {
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
        }
    }
}

/// Where each lowered value came from: a JSON path (`edges[1]`, `extent.distance`) and the
/// span of the expression that wrote it.
pub type PathSpans = Vec<(String, Span)>;

/// A dialect's value forms. Both methods see every value before the generic rules.
pub trait ValueSyntax {
    /// Lowers `expr` when it is one of the dialect's value forms; `None` leaves it to the
    /// generic rules.
    fn lower(&self, expr: &Expr) -> Option<Result<Json, LowerError>>;

    /// Prints `value` when it is one of the dialect's value forms; `None` leaves it to the
    /// generic rules.
    fn print(&self, value: &Json) -> Option<String>;

    /// The tag key of the internally tagged values the field `key` holds. Such a value is
    /// written only in the call form: `extent = blind(distance = 10 mm)` for the model value
    /// `{"extent": "blind", "distance": 10 mm}`, and `mode = cut` for `{"mode": "cut"}`. `scope` is the tag of the entry whose
    /// arguments are read (`extrude`), or `None` inside nested values.
    fn tag_field(&self, _scope: Option<&str>, _key: &str) -> Option<&'static str> {
        None
    }

    /// Lowers the value of the field `key` when the field has a form of its own (a plane
    /// written as a face reference); `None` leaves it to the other rules.
    fn lower_keyed(&self, _key: &str, _expr: &Expr) -> Option<Result<Json, LowerError>> {
        None
    }

    /// Prints the value of the field `key` in the field's own form; the inverse of
    /// [`ValueSyntax::lower_keyed`].
    fn print_keyed(&self, _key: &str, _value: &Json) -> Option<String> {
        None
    }

    /// The front end's statement forms, for the document engine.
    fn entry_forms(&self) -> Option<&dyn crate::document::EntryForms> {
        None
    }

    /// The workspace path of the document being printed, when the printer knows it: values
    /// that name other files print relative to it.
    fn document_path(&self) -> Option<&str> {
        None
    }
}

/// No value forms beyond plain data.
pub struct PlainData;

impl ValueSyntax for PlainData {
    fn lower(&self, _expr: &Expr) -> Option<Result<Json, LowerError>> {
        None
    }

    fn print(&self, _value: &Json) -> Option<String> {
        None
    }
}

/// Lowers an expression to data. `path` names the value (empty at the top); every lowered
/// value's path and span is pushed to `spans`.
pub fn lower_expr(
    expr: &Expr,
    hook: &dyn ValueSyntax,
    path: &str,
    spans: &mut PathSpans,
) -> Result<Json, LowerError> {
    spans.push((path.to_string(), expr.span()));
    if let Some(result) = hook.lower(expr) {
        return result;
    }
    match expr {
        Expr::Number(number) => number_json(&number.text(), false, expr.span()),
        Expr::Unary(unary) if unary.is_negative() => match unary.operand() {
            Some(Expr::Number(number)) => number_json(&number.text(), true, expr.span()),
            _ => Err(LowerError::new(
                expr.span(),
                "only a number can be negated here",
            )),
        },
        Expr::String(string) => Ok(Json::String(string.value())),
        Expr::Reference(reference) => {
            let segments = reference.segments();
            match segments.as_slice() {
                [word] if word == "true" => Ok(Json::Bool(true)),
                [word] if word == "false" => Ok(Json::Bool(false)),
                [word] if word == "none" => Ok(Json::Null),
                [word] => Ok(Json::String(word.clone())),
                _ => Err(LowerError::new(
                    expr.span(),
                    format!(
                        "`{}` is a reference; a plain value is expected here",
                        segments.join(".")
                    ),
                )),
            }
        }
        Expr::Group(group) => match group.inner() {
            Some(inner) => lower_expr(&inner, hook, path, spans),
            None => Err(LowerError::new(expr.span(), "an empty group")),
        },
        Expr::List(list) => {
            let mut items = Vec::new();
            for (index, item) in list.items().iter().enumerate() {
                items.push(lower_expr(item, hook, &format!("{path}[{index}]"), spans)?);
            }
            Ok(Json::Array(items))
        }
        Expr::Record(record) => lower_record(record, hook, path, spans).map(Json::Object),
        _ => Err(LowerError::new(
            expr.span(),
            "this expression is not a value here",
        )),
    }
}

/// Lowers a record's entries; duplicate keys are an error.
pub fn lower_record(
    record: &ast::Record,
    hook: &dyn ValueSyntax,
    path: &str,
    spans: &mut PathSpans,
) -> Result<Vec<(String, Json)>, LowerError> {
    let mut entries: Vec<(String, Json)> = Vec::new();
    for entry in record.entries() {
        let Some(name) = entry.name() else {
            return Err(LowerError::new(entry.span(), "a record entry needs a name"));
        };
        let key = name.text();
        if entries.iter().any(|(existing, _)| *existing == key) {
            return Err(LowerError::new(
                name.span(),
                format!("`{key}` is given twice"),
            ));
        }
        let Some(value) = entry.value() else {
            return Err(LowerError::new(
                entry.span(),
                format!("`{key}` needs a value"),
            ));
        };
        let child = join_path(path, &key);
        let lowered = lower_field(&key, &value, hook, None, &child, spans)?;
        entries.push((key, lowered));
    }
    Ok(entries)
}

/// Lowers the value of the field `key`: its own form, the call form of a tagged value, or a
/// plain value.
pub fn lower_field(
    key: &str,
    expr: &Expr,
    hook: &dyn ValueSyntax,
    scope: Option<&str>,
    path: &str,
    spans: &mut PathSpans,
) -> Result<Json, LowerError> {
    if let Some(result) = hook.lower_keyed(key, expr) {
        spans.push((path.to_string(), expr.span()));
        return result;
    }
    if let Some(tag) = hook.tag_field(scope, key) {
        if let Some(variant) = bare_word(expr) {
            spans.push((path.to_string(), expr.span()));
            return Ok(Json::Object(vec![(tag.to_string(), Json::String(variant))]));
        }
        if let Expr::Call(call) = expr {
            if let Some(variant) = call.callee().as_ref().and_then(bare_word) {
                spans.push((path.to_string(), expr.span()));
                let mut entries = vec![(tag.to_string(), Json::String(variant))];
                for (key, value) in lower_arguments(call, hook, None, path, spans)? {
                    if entries.iter().any(|(existing, _)| *existing == key) {
                        return Err(LowerError::new(
                            expr.span(),
                            format!("`{key}` is given twice"),
                        ));
                    }
                    entries.push((key, value));
                }
                return Ok(Json::Object(entries));
            }
        }
    }
    lower_expr(expr, hook, path, spans)
}

/// A single word that is not a literal (`cut`, `blind`).
fn bare_word(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Reference(reference) => match reference.segments().as_slice() {
            [word] if !NOT_BARE.contains(&word.as_str()) => Some(word.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Lowers a call's arguments as record entries: `name = value`, and a lone word as a flag
/// (`through_all` is `through_all = true`). `scope` is the tag the call writes (see
/// [`ValueSyntax::tag_field`]).
pub fn lower_arguments(
    call: &ast::Call,
    hook: &dyn ValueSyntax,
    scope: Option<&str>,
    path: &str,
    spans: &mut PathSpans,
) -> Result<Vec<(String, Json)>, LowerError> {
    let mut entries: Vec<(String, Json)> = Vec::new();
    for argument in call.arguments() {
        let (key, value) = match argument {
            ast::Argument::Named(named) => {
                let Some(name) = named.name() else {
                    return Err(LowerError::new(named.span(), "an argument needs a name"));
                };
                let key = name.text();
                let Some(value) = named.value() else {
                    return Err(LowerError::new(
                        named.span(),
                        format!("`{key}` needs a value"),
                    ));
                };
                let child = join_path(path, &key);
                let lowered = lower_field(&key, &value, hook, scope, &child, spans)?;
                (key, lowered)
            }
            ast::Argument::Positional(expr) => match bare_word(&expr) {
                Some(word) => {
                    spans.push((join_path(path, &word), expr.span()));
                    (word, Json::Bool(true))
                }
                None => {
                    return Err(LowerError::new(
                        expr.span(),
                        "an argument is `name = value`, or a word that turns an option on",
                    ))
                }
            },
        };
        if entries.iter().any(|(existing, _)| *existing == key) {
            return Err(LowerError::new(
                call.span(),
                format!("`{key}` is given twice"),
            ));
        }
        entries.push((key, value));
    }
    Ok(entries)
}

/// `a` + `b` → `a.b`; an empty parent gives `b`.
pub fn join_path(parent: &str, key: &str) -> String {
    if parent.is_empty() {
        key.to_string()
    } else {
        format!("{parent}.{key}")
    }
}

fn number_json(text: &str, negative: bool, span: Span) -> Result<Json, LowerError> {
    let text = if negative {
        format!("-{text}")
    } else {
        text.to_string()
    };
    let integral = !text.contains(['.', 'e', 'E']);
    if integral {
        if let Ok(value) = text.parse::<i64>() {
            return Ok(Json::Number(value.into()));
        }
        if let Ok(value) = text.parse::<u64>() {
            return Ok(Json::Number(value.into()));
        }
    }
    text.parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map(Json::Number)
        .ok_or_else(|| LowerError::new(span, format!("`{text}` is not a number")))
}

/// Prints a number so that it reads back as the same number.
pub fn number_text(number: &serde_json::Number) -> String {
    if let Some(value) = number.as_i64() {
        return value.to_string();
    }
    if let Some(value) = number.as_u64() {
        return value.to_string();
    }
    format_f64(number.as_f64().unwrap_or_default())
}

/// The shortest text that parses back to `value` (Rust's `Display` is round-trip exact).
pub fn format_f64(value: f64) -> String {
    if value == 0.0 && value.is_sign_negative() {
        return "-0.0".into();
    }
    let magnitude = value.abs();
    if magnitude != 0.0 && !(1e-6..1e21).contains(&magnitude) {
        return format!("{value:e}");
    }
    format!("{value}")
}

/// Words that read back as something other than a string.
const NOT_BARE: &[&str] = &["true", "false", "none", "as", "volatile"];

/// Whether `text` lexes as one word (an identifier or keyword), so it can be printed without
/// quotes and reads back as the same string.
pub fn is_bare_word(text: &str) -> bool {
    !NOT_BARE.contains(&text) && is_bare_key(text)
}

/// Whether `text` can be a call's argument name, a flag or a callee: one word (an identifier
/// or a keyword) that is not a literal.
pub fn is_identifier(text: &str) -> bool {
    is_bare_word(text)
}

/// Whether `text` can be a record key without quotes (keywords are allowed there).
pub fn is_bare_key(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    let stream = tokenize_design(text);
    let tokens: Vec<_> = stream
        .tokens
        .iter()
        .filter(|token| token.kind != FeaTokenKind::EndOfFile)
        .collect();
    stream.diagnostics.is_empty()
        && tokens.len() == 1
        && matches!(
            tokens[0].kind,
            FeaTokenKind::Identifier | FeaTokenKind::Keyword(_)
        )
        && tokens[0].span.end == text.len()
}

/// A double-quoted string literal that decodes to `text`.
pub fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// Line width the printer aims for before it breaks a collection over lines.
pub const PRINT_WIDTH: usize = 92;

/// The indentation unit the printers write.
pub const INDENT: &str = "    ";

/// Prints `value` as design text at nesting `depth` (continuation lines are indented one
/// [`INDENT`] per level).
pub fn print_json(value: &Json, hook: &dyn ValueSyntax, depth: usize) -> String {
    if let Some(text) = hook.print(value) {
        return text;
    }
    match value {
        Json::Null => "none".into(),
        Json::Bool(value) => value.to_string(),
        Json::Number(number) => number_text(number),
        Json::String(text) if is_bare_word(text) => text.clone(),
        Json::String(text) => quote(text),
        Json::Array(items) => {
            let printed: Vec<String> = items
                .iter()
                .map(|item| print_json(item, hook, depth + 1))
                .collect();
            let inline = format!("[{}]", printed.join(", "));
            if fits(&inline, depth) {
                return inline;
            }
            let inner = INDENT.repeat(depth + 1);
            let mut out = String::from("[\n");
            for (index, item) in printed.iter().enumerate() {
                out.push_str(&inner);
                out.push_str(item);
                if index + 1 < printed.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&INDENT.repeat(depth));
            out.push(']');
            out
        }
        Json::Object(entries) => print_record(entries, hook, depth),
    }
}

/// Prints record entries as `{ key = value, … }` (or one entry per line when long).
pub fn print_record(entries: &[(String, Json)], hook: &dyn ValueSyntax, depth: usize) -> String {
    if entries.is_empty() {
        return "{}".into();
    }
    let printed: Vec<String> = entries
        .iter()
        .map(|(key, value)| {
            let text = print_field(key, value, hook, None, depth + 1);
            format!("{} = {text}", print_key(key))
        })
        .collect();
    wrap(&printed, "{ ", " }", "{", "}", depth)
}

/// `open items close` on one line when it fits, else one item per line.
fn wrap(
    printed: &[String],
    open: &str,
    close: &str,
    open_long: &str,
    close_long: &str,
    depth: usize,
) -> String {
    let inline = format!("{open}{}{close}", printed.join(", "));
    if fits(&inline, depth) {
        return inline;
    }
    // Entries are separated by commas on every line: a number at the end of a line would
    // otherwise take the next key as its unit.
    let inner = INDENT.repeat(depth + 1);
    let mut out = format!("{open_long}\n");
    for (index, entry) in printed.iter().enumerate() {
        out.push_str(&inner);
        out.push_str(entry);
        if index + 1 < printed.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(&INDENT.repeat(depth));
    out.push_str(close_long);
    out
}

/// Prints the value of the field `key` so that [`lower_field`] reads it back.
pub fn print_field(
    key: &str,
    value: &Json,
    hook: &dyn ValueSyntax,
    scope: Option<&str>,
    depth: usize,
) -> String {
    if let Some(text) = hook.print_keyed(key, value) {
        return text;
    }
    let Some(tag) = hook.tag_field(scope, key) else {
        return print_json(value, hook, depth);
    };
    if let Json::Object(entries) = value {
        if let Some(((first, Json::String(variant)), rest)) = entries.split_first() {
            if first == tag
                && is_identifier(variant)
                && rest.iter().all(|(k, _)| k != tag && is_identifier(k))
            {
                if rest.is_empty() {
                    return variant.clone();
                }
                return format!("{variant}{}", print_arguments(rest, hook, None, depth));
            }
        }
    }
    // Anything else must not read back as the call form.
    let text = print_json(value, hook, depth);
    if let Json::String(string) = value {
        if is_bare_word(&text) {
            return quote(string);
        }
    }
    let callee_end = text
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(text.len());
    if (callee_end > 0 && text[callee_end..].starts_with('(')) || is_bare_word(&text) {
        return format!("({text})");
    }
    text
}

/// Prints call arguments, `(name = value, flag)`; a `true` field whose name is a word is
/// written as the word alone. Every key must be an identifier ([`is_identifier`]).
pub fn print_arguments(
    entries: &[(String, Json)],
    hook: &dyn ValueSyntax,
    scope: Option<&str>,
    depth: usize,
) -> String {
    let printed: Vec<String> = entries
        .iter()
        .map(|(key, value)| {
            let plain =
                hook.print_keyed(key, value).is_none() && hook.tag_field(scope, key).is_none();
            if plain && *value == Json::Bool(true) && is_identifier(key) {
                return key.clone();
            }
            format!(
                "{} = {}",
                key,
                print_field(key, value, hook, scope, depth + 1)
            )
        })
        .collect();
    wrap(&printed, "(", ")", "(", ")", depth)
}

pub fn print_key(key: &str) -> String {
    if is_bare_key(key) {
        key.to_string()
    } else {
        quote(key)
    }
}

fn fits(text: &str, depth: usize) -> bool {
    !text.contains('\n') && depth * INDENT.len() + text.len() <= PRINT_WIDTH
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_source, Dialect};

    fn lower_text(text: &str) -> Json {
        let source = format!("part p {{\n    x = {text}\n}}\n");
        let syntax = parse_source(Dialect::Part, &source);
        assert!(
            syntax.diagnostics.is_empty(),
            "{text}: {:#?}",
            syntax.diagnostics
        );
        let ast::Item::Declaration(part) = &syntax.file().items()[0] else {
            panic!()
        };
        use ast::Headed;
        let ast::Statement::Assignment(assignment) = &part.block().unwrap().statements()[0] else {
            panic!()
        };
        lower_expr(
            &assignment.value().unwrap(),
            &PlainData,
            "",
            &mut Vec::new(),
        )
        .unwrap()
    }

    #[test]
    fn plain_data_round_trips_through_text() {
        let value: serde_json::Value = serde_json::json!({
            "name": "exactly_one",
            "quoted": "two words",
            "keyword": "material",
            "flag": true,
            "off": false,
            "nothing": null,
            "count": 3,
            "negative": -4,
            "big": 18446744073709551615u64,
            "ratio": 0.1,
            "tiny": 1e-300,
            "escaped": "a \"b\" \\ c\nd",
            "list": [1, 2.5, "x", [], {}],
            "two words": { "inner": [{ "a": 1 }] },
            "long": ["aaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbb", "cccccccccccccccccccccccccc", "dddddddddddddddd"],
        });
        let json: Json = serde_json::from_value(value.clone()).unwrap();
        let text = print_json(&json, &PlainData, 1);
        let back = lower_text(&text);
        assert_eq!(back.to_value(), value, "{text}");
        assert_eq!(back, json, "{text}");
    }

    #[test]
    fn json_keeps_the_serialisation_order_of_keys() {
        #[derive(Serialize)]
        struct S {
            zeta: u8,
            alpha: u8,
        }
        let json = Json::from_serialize(&S { zeta: 1, alpha: 2 }).unwrap();
        assert_eq!(print_json(&json, &PlainData, 0), "{ zeta = 1, alpha = 2 }");
    }
}
