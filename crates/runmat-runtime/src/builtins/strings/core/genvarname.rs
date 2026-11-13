//! MATLAB-compatible legacy `genvarname` builtin.

use runmat_builtins::{
    BuiltinErrorDescriptor, GENVARNAME_DESCRIPTOR, GENVARNAME_ERROR_INTERNAL as ERROR_INTERNAL,
    GENVARNAME_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT,
    GENVARNAME_ERROR_INVALID_OPTION as ERROR_INVALID_OPTION, GENVARNAME_ERROR_TOO_MANY_OUTPUTS,
};

use std::collections::HashSet;

use runmat_value::{
    decode_utf16_code_units, CellArray, CharArray, StringArray, Utf16CodeUnit, Value,
};

use crate::builtins::common::identifiers::{
    is_matlab_keyword, is_valid_varname, MATLAB_NAME_LENGTH_MAX,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "genvarname";

#[runmat_macros::runtime_builtin(
    name = "genvarname",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::genvarname"
)]
async fn genvarname_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &GENVARNAME_DESCRIPTOR,
        &GENVARNAME_ERROR_TOO_MANY_OUTPUTS,
        "genvarname",
    )?;
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

    let exclusions = if let Some(value) = rest.first() {
        text_list(value, TextListContext::Exclusions)?
    } else {
        Vec::new()
    };
    genvarname_value(value, exclusions)
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
            .data()
            .iter()
            .any(|value| numeric_or_resident(value) || contains_numeric_or_resident(value)),
        Value::Struct(value) => value
            .fields
            .values()
            .any(|value| numeric_or_resident(value) || contains_numeric_or_resident(value)),
        Value::StructArray(array) => array
            .any_value(|value| numeric_or_resident(value) || contains_numeric_or_resident(value)),
        _ => false,
    }
}

fn genvarname_value(value: Value, exclusions: Vec<String>) -> BuiltinResult<Value> {
    let mut used = exclusions.into_iter().collect::<HashSet<_>>();
    match value {
        Value::String(text) => Ok(Value::string(unique_name(required_name_text(&text)?, &mut used))),
        Value::StringArray(array) => {
let (data, shape, _, _) = array.into_parts();
            let out = data
                .iter()
                .map(|text| required_name_text(text).map(|text| unique_name(text, &mut used)))
                .collect::<BuiltinResult<Vec<_>>>()?;
            StringArray::new(out, shape.to_vec())
                .map(Value::StringArray)
                .map_err(|err| error(&ERROR_INTERNAL, format!("genvarname: {err}")))
        }
        Value::CharArray(chars) if chars.rows <= 1 && chars.shape.iter().skip(2).all(|dimension| *dimension == 1) => {
            let text = decode_chars(chars.data())?;
            Ok(Value::CharArray(CharArray::new_row(&unique_name(
                &text, &mut used,
            ))))
        }
        Value::CharArray(_) => Err(error(
            &ERROR_INVALID_INPUT,
            "genvarname: character array input must be a character vector; use a cell array for multiple names",
        )),
        Value::Cell(cell) => {
            let shape = cell.shape().to_vec();
            let data = cell.into_column_major().map_err(|err| error(&ERROR_INTERNAL, err))?;
            let mut out = Vec::with_capacity(data.len());
            for value in &data {
                let text = text_scalar(value, TextListContext::Input)?;
                out.push(Value::CharArray(CharArray::new_row(&unique_name(
                    &text, &mut used,
                ))));
            }
            CellArray::from_column_major(out, shape)
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
        Value::String(text) => Ok(vec![required_name_text(text)?.to_owned()]),
        Value::StringArray(array) => Ok(array
            .data()
            .iter()
            .map(|text| required_name_text(text).map(str::to_owned))
            .collect::<BuiltinResult<Vec<_>>>()?),
        Value::CharArray(chars)
            if chars.rows <= 1 && chars.shape.iter().skip(2).all(|dimension| *dimension == 1) =>
        {
            Ok(vec![decode_chars(chars.data())?])
        }
        Value::CharArray(_) => Err(error(
            &ERROR_INVALID_INPUT,
            "genvarname: multiple names must be supplied as a string array or cell array",
        )),
        Value::Cell(cell) => cell
            .data()
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
        Value::String(text) => Ok(required_name_text(text)?.to_owned()),
        Value::StringArray(array) if array.data().len() == 1 => {
            Ok(required_name_text(&array.data()[0])?.to_owned())
        }
        Value::CharArray(chars)
            if chars.rows <= 1 && chars.shape.iter().skip(2).all(|dimension| *dimension == 1) =>
        {
            decode_chars(chars.data())
        }
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

fn error(descriptor: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn decode_chars(units: &[Utf16CodeUnit]) -> BuiltinResult<String> {
    decode_utf16_code_units(units).map_err(|source| {
        let mut builder = build_runtime_error("genvarname: malformed UTF-16 character input")
            .with_builtin(BUILTIN_NAME)
            .with_source(source);
        if let Some(identifier) = ERROR_INVALID_INPUT.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    })
}

fn required_name_text(text: &runmat_value::StringElement) -> BuiltinResult<&str> {
    text.as_str().ok_or_else(|| {
        error(
            &ERROR_INVALID_INPUT,
            "genvarname: names must not be missing",
        )
    })
}

#[cfg(test)]
pub(crate) async fn genvarname_builtin_source(
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    genvarname_builtin(
        value,
        crate::call::BuiltinArguments::from_borrowed(&rest, None)?,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{BuiltinIntegerAuditKind, GENVARNAME_INTEGER_AUDIT};

    #[test]
    fn duplicate_suffixes_follow_column_major_order_in_nd_cells() {
        let cell = CellArray::from_column_major(vec![Value::from("A"); 8], vec![2, 2, 2]).unwrap();
        let Value::Cell(result) = call(Value::Cell(cell), vec![]) else {
            panic!("cell output")
        };
        assert_eq!(result.shape, vec![2, 2, 2]);
        let values = result.into_column_major().unwrap();
        for (index, value) in values.iter().enumerate() {
            let expected = if index == 0 {
                "A".to_string()
            } else {
                format!("A{index}")
            };
            assert_eq!(*value, Value::CharArray(CharArray::new_row(&expected)));
        }
    }

    #[test]
    fn nd_character_input_and_excess_outputs_reject() {
        let chars =
            CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap();
        let error =
            block_on(genvarname_builtin_source(Value::CharArray(chars), vec![])).unwrap_err();
        assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = block_on(genvarname_builtin_source(Value::from("A"), vec![])).unwrap_err();
        assert_eq!(
            error.identifier(),
            GENVARNAME_ERROR_TOO_MANY_OUTPUTS.identifier
        );
    }

    fn call(value: Value, rest: Vec<Value>) -> Value {
        block_on(genvarname_builtin_source(value, rest)).expect("genvarname")
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
        assert_eq!(array.data(), vec!["A", "A1", "A2"]);
    }

    #[test]
    fn exclusions_are_reserved_before_input_names() {
        let exclusions = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("A")),
                Value::String("A1".to_string().into()),
            ],
            1,
            2,
        )
        .unwrap();
        let out = call(
            Value::String("A".to_string().into()),
            vec![Value::Cell(exclusions)],
        );
        assert_eq!(out, Value::String("A2".to_string().into()));
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
        assert_eq!(cell.data()[0], Value::CharArray(CharArray::new_row("xFor")));
        assert_eq!(
            cell.data()[1],
            Value::CharArray(CharArray::new_row("hasSpace"))
        );
    }

    #[test]
    fn multirow_char_array_is_not_a_multiple_name_container() {
        let input = CharArray::from_unicode_scalars(
            vec!['a', ' ', ' ', 'a', ' ', ' ', '1', ' ', ' '],
            3,
            3,
        )
        .unwrap();
        assert!(block_on(genvarname_builtin_source(
            Value::CharArray(input),
            Vec::new()
        ))
        .is_err());
    }

    #[test]
    fn truncates_to_name_length_max_and_keeps_suffix_inside_limit() {
        let long = "a".repeat(MATLAB_NAME_LENGTH_MAX + 10);
        let input = StringArray::new(vec![long.clone(), long], vec![1, 2]).unwrap();
        let out = call(Value::StringArray(input), Vec::new());
        let Value::StringArray(array) = out else {
            panic!("expected string array")
        };
        assert_eq!(
            array.data()[0]
                .as_str()
                .expect("nonmissing generated name")
                .chars()
                .count(),
            MATLAB_NAME_LENGTH_MAX
        );
        assert_eq!(
            array.data()[1]
                .as_str()
                .expect("nonmissing generated name")
                .chars()
                .count(),
            MATLAB_NAME_LENGTH_MAX
        );
        assert_ne!(array.data()[0], array.data()[1]);
        assert!(array.data()[1]
            .as_str()
            .expect("nonmissing generated name")
            .ends_with('1'));
    }

    #[test]
    fn rejects_non_text_input_and_non_scalar_cell_elements() {
        assert!(block_on(genvarname_builtin_source(Value::Num(1.0), Vec::new())).is_err());
        let bad = CellArray::new(
            vec![Value::StringArray(
                StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap(),
            )],
            1,
            1,
        )
        .unwrap();
        assert!(block_on(genvarname_builtin_source(Value::Cell(bad), Vec::new())).is_err());
    }

    #[test]
    fn integer_audit_rejects_all_integer_classes_and_resident_numeric_before_gather() {
        assert_eq!(
            GENVARNAME_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        for value in [
            Value::Int(runmat_value::IntValue::I8(1)),
            Value::Int(runmat_value::IntValue::I16(1)),
            Value::Int(runmat_value::IntValue::I32(1)),
            Value::Int(runmat_value::IntValue::I64(1)),
            Value::Int(runmat_value::IntValue::U8(1)),
            Value::Int(runmat_value::IntValue::U16(1)),
            Value::Int(runmat_value::IntValue::U32(1)),
            Value::Int(runmat_value::IntValue::U64(1)),
        ] {
            let error = block_on(genvarname_builtin_source(value.clone(), Vec::new()))
                .expect_err("integer name input");
            assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
            let error = block_on(genvarname_builtin_source(
                Value::String("name".into()),
                vec![value],
            ))
            .expect_err("integer exclusion input");
            assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
        }

        let resident = || {
            Value::GpuTensor(
                runmat_accelerate_api::test_support::TestAllocationOwner::new()
                    .allocate_zeroed(
                        vec![1, 1],
                        runmat_accelerate_api::NumericElementType::F64,
                        runmat_accelerate_api::GpuTensorStorage::Real,
                    )
                    .expect("admit unregistered resident fixture")
                    .handle,
            )
        };
        let error = block_on(genvarname_builtin_source(resident(), Vec::new()))
            .expect_err("resident name input");
        assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
        let error = block_on(genvarname_builtin_source(
            Value::String("name".into()),
            vec![resident()],
        ))
        .expect_err("resident exclusion input");
        assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
    }
}
