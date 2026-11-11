//! Input values for one run: a study, a sweep point or a script sets a part's or an
//! assembly's inputs (`thickness = "3 mm"`) and regenerates it, without editing the file.
//!
//! [`set_part_inputs`] and [`set_assembly_inputs`] return the design's source with those inputs'
//! defaults replaced, which every regenerate, export and mass-property path then reads as usual.
//! A name the design doesn't declare is an error that lists the inputs it does declare. A value
//! is written in the design language (`3 mm`, `0.4`); it must be a literal, so `input.width * 2`
//! is refused.

use crate::modules::ModuleScope;
use crate::syntax::edit::{set_value, ConfigurationScope, ValueEdit};
use crate::syntax::{parse_assembly_in, parse_part, print_assembly};
use crate::value::{InputDecl, ValueExpr};

/// Why a run's inputs can't be set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunInputError {
    /// The design doesn't parse.
    Invalid(String),
    /// `name` isn't one of the design's inputs; `declared` lists them.
    Unknown { name: String, declared: Vec<String> },
    /// The value text isn't a literal value.
    Value { name: String, message: String },
}

impl std::fmt::Display for RunInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunInputError::Invalid(m) => write!(f, "the design doesn't parse: {m}"),
            RunInputError::Unknown { name, declared } => write!(
                f,
                "`{name}` is not an input of this design; its inputs are {}",
                if declared.is_empty() {
                    "none".to_string()
                } else {
                    declared.join(", ")
                }
            ),
            RunInputError::Value { name, message } => write!(f, "`{name}`: {message}"),
        }
    }
}

impl std::error::Error for RunInputError {}

/// The value the design-language text `text` writes (`3 mm`, `0.4`, `true`).
pub fn literal(name: &str, text: &str) -> Result<ValueExpr, RunInputError> {
    let bad = |message: String| RunInputError::Value {
        name: name.into(),
        message,
    };
    if text.contains('\n') || text.contains('}') {
        return Err(bad(format!("`{text}` is not one value")));
    }
    let probe = format!("part run_value {{\n    input v : real = {text}\n}}\n");
    let parsed = parse_part("run_value.part", &probe);
    let value = parsed
        .document
        .filter(|_| parsed.diagnostics.is_empty())
        .and_then(|d| d.inputs.into_iter().next())
        .map(|i| i.default)
        .ok_or_else(|| bad(format!("`{text}` is not a value")))?;
    if references_anything(&value) {
        return Err(bad(format!(
            "`{text}` names other values; a run value is a literal"
        )));
    }
    Ok(value)
}

fn references_anything(value: &ValueExpr) -> bool {
    let json = serde_json::to_value(value).unwrap_or_default();
    fn walk(v: &serde_json::Value) -> bool {
        match v {
            serde_json::Value::Object(o) => {
                matches!(
                    o.get("form").and_then(|f| f.as_str()),
                    Some("input" | "let" | "dynamic" | "reference" | "published")
                ) || o.values().any(walk)
            }
            serde_json::Value::Array(a) => a.iter().any(walk),
            _ => false,
        }
    }
    walk(&json)
}

fn find<'a>(inputs: &'a mut [InputDecl], name: &str) -> Result<&'a mut InputDecl, RunInputError> {
    let declared: Vec<String> = inputs.iter().map(|i| i.name.to_string()).collect();
    inputs
        .iter_mut()
        .find(|i| i.name.as_str() == name)
        .ok_or(RunInputError::Unknown {
            name: name.into(),
            declared,
        })
}

/// The part source at `path` with each `(input, value text)` of `set` as that input's default.
/// The rest of the text is kept as written.
pub fn set_part_inputs(
    path: &str,
    source: &str,
    set: &[(String, String)],
) -> Result<String, RunInputError> {
    let mut text = source.to_string();
    for (name, value) in set {
        let parsed = parse_part(path, &text);
        let mut document = parsed.document.ok_or_else(|| {
            RunInputError::Invalid(
                parsed
                    .diagnostics
                    .first()
                    .map(|d| d.message.clone())
                    .unwrap_or_default(),
            )
        })?;
        find(&mut document.inputs, name)?;
        let edit = ValueEdit {
            value: literal(name, value)?,
            text: value.clone(),
            scope: ConfigurationScope::All,
        };
        text = set_value(&text, path, name, &edit)
            .map_err(|e| RunInputError::Invalid(e.to_string()))?
            .source;
    }
    Ok(text)
}

/// The assembly source at `path` (read in `scope`) with each `(input, value text)` of `set` as
/// that input's default, printed back in the module form.
pub fn set_assembly_inputs(
    path: &str,
    source: &str,
    scope: &ModuleScope,
    set: &[(String, String)],
) -> Result<String, RunInputError> {
    if set.is_empty() {
        return Ok(source.to_string());
    }
    let parsed = parse_assembly_in(path, source, scope);
    let mut document = parsed.document.ok_or_else(|| {
        RunInputError::Invalid(
            parsed
                .diagnostics
                .first()
                .map(|d| d.message.clone())
                .unwrap_or_default(),
        )
    })?;
    for (name, value) in set {
        find(&mut document.inputs, name)?.default = literal(name, value)?;
    }
    print_assembly(&document, &parsed.uses).map_err(|e| RunInputError::Invalid(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLATE: &str = "part plate {\n    input width : length = 80 mm   // from the enclosure\n    input thickness : length = 3 mm\n}\n";

    #[test]
    fn a_run_sets_input_defaults_and_keeps_the_rest_of_the_text() {
        let set = vec![("thickness".to_string(), "4.5 mm".to_string())];
        let text = set_part_inputs("plate.part", PLATE, &set).unwrap();
        assert_eq!(text, PLATE.replace("= 3 mm", "= 4.5 mm"));
        let d = parse_part("plate.part", &text).document.unwrap();
        assert_eq!(d.inputs[1].default, literal("thickness", "4.5 mm").unwrap());
    }

    #[test]
    fn an_unknown_input_lists_the_declared_ones_and_a_run_value_is_a_literal() {
        let e =
            set_part_inputs("plate.part", PLATE, &[("depth".into(), "1 mm".into())]).unwrap_err();
        assert_eq!(
            e.to_string(),
            "`depth` is not an input of this design; its inputs are width, thickness"
        );
        let e = set_part_inputs(
            "plate.part",
            PLATE,
            &[("width".into(), "input.thickness * 2".into())],
        )
        .unwrap_err();
        assert!(matches!(e, RunInputError::Value { .. }), "{e}");
        let e = set_part_inputs(
            "plate.part",
            PLATE,
            &[("width".into(), "1 mm }\n    input x : real = 2".into())],
        )
        .unwrap_err();
        assert!(matches!(e, RunInputError::Value { .. }), "{e}");
    }

    #[test]
    fn an_assembly_run_sets_its_inputs() {
        let source = "assembly rig {\n    input pitch : length = 500 mm\n}\n";
        let scope = ModuleScope::from_files([("rig.assembly", source)]);
        let text = set_assembly_inputs(
            "rig.assembly",
            source,
            &scope,
            &[("pitch".into(), "450 mm".into())],
        )
        .unwrap();
        let d = parse_assembly_in("rig.assembly", &text, &scope)
            .document
            .unwrap();
        assert_eq!(d.inputs[0].default, literal("pitch", "450 mm").unwrap());
    }
}
