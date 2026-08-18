//! MATLAB-compatible legacy `genvarname` builtin.

use std::collections::HashSet;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, CellArray, CharArray,
    ResolveContext, StringArray, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::identifiers::{
    is_matlab_keyword, is_valid_varname, MATLAB_NAME_LENGTH_MAX,
};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "genvarname";

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varname",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Valid variable name text with the same container form as the input.",
}];

const INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text, string array, character array, or cell array of text scalars.",
}];

const INPUTS_EXCLUSIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text, string array, character array, or cell array of text scalars.",
    },
    BuiltinParamDescriptor {
        name: "exclusions",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Names that generated output must not match.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "varname = genvarname(str)",
        inputs: &INPUTS_ONE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "varname = genvarname(str, exclusions)",
        inputs: &INPUTS_EXCLUSIONS,
        outputs: &OUTPUT,
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GENVARNAME.INVALID_INPUT",
    identifier: Some("RunMat:genvarname:InvalidInput"),
    when: "Input is not text-compatible or cell elements are not text scalars.",
    message: "genvarname: input must be text or a cell array of text",
};

const ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GENVARNAME.INVALID_OPTION",
    identifier: Some("RunMat:genvarname:InvalidOption"),
    when: "More than one exclusions argument is supplied.",
    message: "genvarname: expected str or str,exclusions",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GENVARNAME.INTERNAL",
    identifier: Some("RunMat:genvarname:Internal"),
    when: "Internal string, cell, or character array construction failed.",
    message: "genvarname: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_INPUT, ERROR_INVALID_OPTION, ERROR_INTERNAL];

pub const GENVARNAME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const GENVARNAME_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "genvarname accepts host text names and exclusions only. All eight integer classes and provider-resident numeric values reject without implicit text conversion, gather, or provider access.",
    };

#[runtime_builtin(
    name = "genvarname",
    category = "strings/core",
    summary = "Construct valid, unique MATLAB variable names from text.",
    keywords = "genvarname,variable name,identifier,legacy,makeValidName,makeUniqueStrings",
    accel = "gather",
    type_resolver(genvarname_type),
    descriptor(crate::builtins::strings::core::genvarname::GENVARNAME_DESCRIPTOR),
    integer_audit(crate::builtins::strings::core::genvarname::GENVARNAME_INTEGER_AUDIT),
    builtin_path = "crate::builtins::strings::core::genvarname"
)]
async fn genvarname_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(error(
            &ERROR_INVALID_OPTION,
            "genvarname: expected one or two input arguments",
        ));
    }

    reject_numeric_text_input(&value, "input")?;
    if let Some(value) = rest.first() {
        reject_numeric_text_input(value, "exclusions")?;
    }

    let input = gather_if_needed_async(&value)
        .await
        .map_err(remap_genvarname_flow)?;
    let exclusions = if let Some(value) = rest.first() {
        let host = gather_if_needed_async(value)
            .await
            .map_err(remap_genvarname_flow)?;
        text_list(&host, TextListContext::Exclusions)?
    } else {
        Vec::new()
    };
    genvarname_value(input, exclusions)
}

fn reject_numeric_text_input(value: &Value, role: &str) -> BuiltinResult<()> {
    if numeric_or_resident(value) || contains_numeric_or_resident(value) {
        return Err(error(
            &ERROR_INVALID_INPUT,
            format!("genvarname: {role} must contain text only"),
        ));
    }
    Ok(())
}

fn numeric_or_resident(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::SparseTensor(_)
            | Value::LogicalArray(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_)
            | Value::Symbolic(_)
            | Value::GpuTensor(_)
    )
}

fn contains_numeric_or_resident(value: &Value) -> bool {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .any(|value| numeric_or_resident(value) || contains_numeric_or_resident(value)),
        _ => false,
    }
}

fn genvarname_value(value: Value, exclusions: Vec<String>) -> BuiltinResult<Value> {
    let mut used = exclusions.into_iter().collect::<HashSet<_>>();
    match value {
        Value::String(text) => Ok(Value::String(unique_name(&text, &mut used))),
        Value::StringArray(array) => {
            let StringArray { data, shape, .. } = array;
            let out = data
                .iter()
                .map(|text| unique_name(text, &mut used))
                .collect::<Vec<_>>();
            StringArray::new(out, shape)
                .map(Value::StringArray)
                .map_err(|err| error(&ERROR_INTERNAL, format!("genvarname: {err}")))
        }
        Value::CharArray(chars) if chars.rows <= 1 => {
            let text = chars.data.iter().collect::<String>();
            Ok(Value::CharArray(CharArray::new_row(&unique_name(
                &text, &mut used,
            ))))
        }
        Value::CharArray(_) => Err(error(
            &ERROR_INVALID_INPUT,
            "genvarname: character array input must be a character vector; use a cell array for multiple names",
        )),
        Value::Cell(cell) => {
            let CellArray { data, shape, .. } = cell;
            let mut out = Vec::with_capacity(data.len());
            for value in &data {
                let text = text_scalar(value, TextListContext::Input)?;
                out.push(Value::CharArray(CharArray::new_row(&unique_name(
                    &text, &mut used,
                ))));
            }
            CellArray::new_with_shape(out, shape)
                .map(Value::Cell)
                .map_err(|err| error(&ERROR_INTERNAL, format!("genvarname: {err}")))
        }
        other => Err(error(
            &ERROR_INVALID_INPUT,
            format!("genvarname: unsupported input type {other:?}"),
        )),
    }
}

#[derive(Copy, Clone)]
enum TextListContext {
    Input,
    Exclusions,
}

fn text_list(value: &Value, context: TextListContext) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(chars) if chars.rows <= 1 => {
            Ok(vec![chars.data.iter().collect::<String>()])
        }
        Value::CharArray(_) => Err(error(
            &ERROR_INVALID_INPUT,
            "genvarname: multiple names must be supplied as a string array or cell array",
        )),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|value| text_scalar(value, context))
            .collect(),
        other => Err(error(
            &ERROR_INVALID_INPUT,
            format!("genvarname: exclusions must be text, got {other:?}"),
        )),
    }
}

fn text_scalar(value: &Value, context: TextListContext) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows <= 1 => Ok(chars.data.iter().collect()),
        _ => {
            let message = match context {
                TextListContext::Input => {
                    "genvarname: cell elements must be character vectors or string scalars"
                }
                TextListContext::Exclusions => {
                    "genvarname: exclusions cell elements must be text scalars"
                }
            };
            Err(error(&ERROR_INVALID_INPUT, message))
        }
    }
}

fn unique_name(input: &str, used: &mut HashSet<String>) -> String {
    let base = normalize_base_name(input);
    let mut candidate = valid_base_name(&base);
    if !used.contains(&candidate) && is_valid_varname(&candidate) {
        used.insert(candidate.clone());
        return candidate;
    }

    for idx in 1usize.. {
        let suffix = idx.to_string();
        candidate = fit_with_suffix(&base, &suffix);
        if is_valid_varname(&candidate) && used.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("unbounded suffix search must find a unique variable name")
}

fn valid_base_name(base: &str) -> String {
    if is_valid_varname(base) {
        return base.to_string();
    }
    if is_matlab_keyword(base) {
        return keyword_to_variable_name(base);
    }
    fit_with_suffix(base, "1")
}

fn normalize_base_name(input: &str) -> String {
    let mut out = String::new();
    let mut capitalize_next_alpha = false;

    for ch in input.chars() {
        if ch.is_whitespace() {
            capitalize_next_alpha = true;
            continue;
        }
        if ch == '_' || ch.is_alphanumeric() {
            if capitalize_next_alpha && ch.is_alphabetic() {
                out.extend(ch.to_uppercase());
            } else {
                out.push(ch);
            }
            capitalize_next_alpha = false;
        } else {
            out.push_str(&format!("0x{:X}", ch as u32));
            capitalize_next_alpha = false;
        }
    }

    if out
        .chars()
        .next()
        .map(|ch| !ch.is_alphabetic())
        .unwrap_or(true)
    {
        out.insert(0, 'x');
    }
    truncate_chars(&out, MATLAB_NAME_LENGTH_MAX)
}

fn fit_with_suffix(base: &str, suffix: &str) -> String {
    if suffix.chars().count() >= MATLAB_NAME_LENGTH_MAX {
        let mut out = String::from("x");
        out.push_str(&truncate_chars(
            suffix,
            MATLAB_NAME_LENGTH_MAX.saturating_sub(1),
        ));
        return out;
    }
    let prefix_len = MATLAB_NAME_LENGTH_MAX - suffix.chars().count();
    let mut out = truncate_chars(base, prefix_len);
    if out.is_empty() || !out.chars().next().is_some_and(|ch| ch.is_alphabetic()) {
        out.insert(0, 'x');
        out = truncate_chars(&out, prefix_len);
    }
    out.push_str(suffix);
    out
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

fn keyword_to_variable_name(keyword: &str) -> String {
    let mut chars = keyword.chars();
    let Some(first) = chars.next() else {
        return "x".to_string();
    };
    let mut out = String::from("x");
    out.extend(first.to_uppercase());
    out.extend(chars);
    truncate_chars(&out, MATLAB_NAME_LENGTH_MAX)
}

fn genvarname_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::String) => Type::String,
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn error(descriptor: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_genvarname_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn call(value: Value, rest: Vec<Value>) -> Value {
        block_on(genvarname_builtin(value, rest)).expect("genvarname")
    }

    #[test]
    fn char_scalar_normalizes_invalid_characters_and_whitespace() {
        let out = call(Value::CharArray(CharArray::new_row("1 a-b")), Vec::new());
        assert_eq!(out, Value::CharArray(CharArray::new_row("x1A0x2Db")));
    }

    #[test]
    fn string_array_preserves_shape_and_makes_names_unique() {
        let input = StringArray::new(
            vec!["A".to_string(), "A".to_string(), "A".to_string()],
            vec![1, 3],
        )
        .unwrap();
        let out = call(Value::StringArray(input), Vec::new());
        let Value::StringArray(array) = out else {
            panic!("expected string array")
        };
        assert_eq!(array.shape, vec![1, 3]);
        assert_eq!(array.data, vec!["A", "A1", "A2"]);
    }

    #[test]
    fn exclusions_are_reserved_before_input_names() {
        let exclusions = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("A")),
                Value::String("A1".to_string()),
            ],
            1,
            2,
        )
        .unwrap();
        let out = call(
            Value::String("A".to_string()),
            vec![Value::Cell(exclusions)],
        );
        assert_eq!(out, Value::String("A2".to_string()));
    }

    #[test]
    fn cellstr_preserves_cell_shape_and_returns_char_rows() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("for")),
                Value::CharArray(CharArray::new_row("has space")),
            ],
            2,
            1,
        )
        .unwrap();
        let out = call(Value::Cell(cell), Vec::new());
        let Value::Cell(cell) = out else {
            panic!("expected cell array")
        };
        assert_eq!(cell.shape, vec![2, 1]);
        assert_eq!(cell.data[0], Value::CharArray(CharArray::new_row("xFor")));
        assert_eq!(
            cell.data[1],
            Value::CharArray(CharArray::new_row("hasSpace"))
        );
    }

    #[test]
    fn multirow_char_array_is_not_a_multiple_name_container() {
        let input =
            CharArray::new(vec!['a', ' ', ' ', 'a', ' ', ' ', '1', ' ', ' '], 3, 3).unwrap();
        assert!(block_on(genvarname_builtin(Value::CharArray(input), Vec::new())).is_err());
    }

    #[test]
    fn truncates_to_name_length_max_and_keeps_suffix_inside_limit() {
        let long = "a".repeat(MATLAB_NAME_LENGTH_MAX + 10);
        let input = StringArray::new(vec![long.clone(), long], vec![1, 2]).unwrap();
        let out = call(Value::StringArray(input), Vec::new());
        let Value::StringArray(array) = out else {
            panic!("expected string array")
        };
        assert_eq!(array.data[0].chars().count(), MATLAB_NAME_LENGTH_MAX);
        assert_eq!(array.data[1].chars().count(), MATLAB_NAME_LENGTH_MAX);
        assert_ne!(array.data[0], array.data[1]);
        assert!(array.data[1].ends_with('1'));
    }

    #[test]
    fn rejects_non_text_input_and_non_scalar_cell_elements() {
        assert!(block_on(genvarname_builtin(Value::Num(1.0), Vec::new())).is_err());
        let bad = CellArray::new(
            vec![Value::StringArray(
                StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap(),
            )],
            1,
            1,
        )
        .unwrap();
        assert!(block_on(genvarname_builtin(Value::Cell(bad), Vec::new())).is_err());
    }

    #[test]
    fn integer_audit_rejects_all_integer_classes_and_resident_numeric_before_gather() {
        assert_eq!(
            GENVARNAME_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        for value in [
            Value::Int(runmat_builtins::IntValue::I8(1)),
            Value::Int(runmat_builtins::IntValue::I16(1)),
            Value::Int(runmat_builtins::IntValue::I32(1)),
            Value::Int(runmat_builtins::IntValue::I64(1)),
            Value::Int(runmat_builtins::IntValue::U8(1)),
            Value::Int(runmat_builtins::IntValue::U16(1)),
            Value::Int(runmat_builtins::IntValue::U32(1)),
            Value::Int(runmat_builtins::IntValue::U64(1)),
        ] {
            let error = block_on(genvarname_builtin(value.clone(), Vec::new()))
                .expect_err("integer name input");
            assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
            let error = block_on(genvarname_builtin(
                Value::String("name".into()),
                vec![value],
            ))
            .expect_err("integer exclusion input");
            assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
        }

        let resident = || {
            Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
                shape: vec![1, 1],
                device_id: u32::MAX,
                buffer_id: u64::MAX,
                descriptor: Default::default(),
            })
        };
        let error =
            block_on(genvarname_builtin(resident(), Vec::new())).expect_err("resident name input");
        assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
        let error = block_on(genvarname_builtin(
            Value::String("name".into()),
            vec![resident()],
        ))
        .expect_err("resident exclusion input");
        assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
    }
}
