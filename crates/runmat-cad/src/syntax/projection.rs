//! The model projection of a design file: its model in JSON, its `use` lines and the span of
//! every site, versioned. Servers, indexes and external tools read this instead of parsing
//! the text; the syntax outline beside it is [`runmat_fea_syntax::projection`].
//!
//! The format is `runmat.design-model`, version [`MODEL_PROJECTION_VERSION`]. A field is only
//! added in a new version; a reader that knows version 1 reads every later version.

use super::Parsed;
use runmat_fea_syntax::Dialect;
use serde::Serialize;
use serde_json::{json, Value};

pub const MODEL_PROJECTION_FORMAT: &str = "runmat.design-model";
pub const MODEL_PROJECTION_VERSION: u32 = 1;

/// The projection of a parsed file. `document` is `null` when the file has errors; the
/// diagnostics say why.
pub fn model_projection<T: Serialize>(dialect: Dialect, path: &str, parsed: &Parsed<T>) -> Value {
    let spans: serde_json::Map<String, Value> = parsed
        .spans
        .spans
        .iter()
        .map(|(key, span)| (key.clone(), json!([span.start, span.end])))
        .collect();
    json!({
        "format": MODEL_PROJECTION_FORMAT,
        "version": MODEL_PROJECTION_VERSION,
        "dialect": dialect,
        "path": path,
        "document": parsed.document.as_ref().map(|document| serde_json::to_value(document).unwrap_or(Value::Null)),
        "uses": parsed.uses,
        "spans": spans,
        "diagnostics": parsed.diagnostics.iter().map(|d| json!({
            "kind": d.kind,
            "message": d.message,
            "span": [d.span.start, d.span.end],
        })).collect::<Vec<_>>(),
    })
}
