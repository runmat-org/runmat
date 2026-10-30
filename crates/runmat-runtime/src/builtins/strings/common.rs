//! Shared helpers for string builtins.
mod borrowed_output;
mod case_output;
mod text_input;
mod text_layout;
pub(crate) use text_layout::has_unsupported_character_pages;

pub(crate) use borrowed_output::{
    transform_scalar_borrowed, transform_string_array_borrowed,
    transform_string_array_borrowed_indexed,
};
pub(crate) use case_output::{
    transform_character_case, transform_scalar_case, transform_string_array_case, TextCase,
};
pub(crate) use text_input::{scalar_text_input, text_input_admission, TextInput};

use runmat_value::{decode_utf16_code_units, CharArray, Utf16CodeUnit, Utf16DecodeError};

use runmat_value::Value;

pub(crate) fn is_multirow_char_array(value: &Value) -> bool {
    matches!(value, Value::CharArray(array) if array.rows > 1)
}

pub(crate) fn cell_contains_string(value: &Value) -> bool {
    match value {
        Value::Cell(cell) => cell.data().iter().any(|value| {
            matches!(value, Value::String(_) | Value::StringArray(_)) || cell_contains_string(value)
        }),
        _ => false,
    }
}

/// Whether an otherwise valid text cell uses RunMat's string-scalar cell extension.
/// Invalid elements remain for the caller's cell-element validator to diagnose.
pub(crate) fn cell_has_valid_string_scalar(value: &Value) -> bool {
    let Value::Cell(cell) = value else {
        return false;
    };
    let mut has_string = false;
    for element in cell.data() {
        match element {
            Value::String(_) => has_string = true,
            Value::StringArray(array) if array.data().len() == 1 => has_string = true,
            Value::CharArray(chars) if chars.rows <= 1 => {}
            _ => return false,
        }
    }
    has_string
}

/// Missing is a typed absence, never inferred from a string's display text.
#[inline]
pub(crate) fn is_missing_string(text: &runmat_value::StringElement) -> bool {
    text.is_missing()
}

/// Collect a row from a [`CharArray`] into a `String`.
#[inline]
pub(crate) fn char_row_to_string(
    array: &CharArray,
    row: usize,
) -> Result<String, Utf16DecodeError> {
    char_row_to_string_slice(array.data(), array.cols, row)
}

/// Collect a row from a character slice laid out in row-major order.
#[inline]
pub(crate) fn char_row_to_string_slice(
    data: &[Utf16CodeUnit],
    cols: usize,
    row: usize,
) -> Result<String, Utf16DecodeError> {
    let start = row * cols;
    let end = start + cols;
    decode_utf16_code_units(&data[start..end])
}

/// MATLAB treats these four Unicode whitespace characters as significant at
/// text boundaries; RunMat extension mode retains the broader Rust predicate.
pub(crate) fn trimmable_whitespace(c: char, strict: bool) -> bool {
    c.is_whitespace()
        && (!strict || !matches!(c, '\u{0085}' | '\u{00A0}' | '\u{2007}' | '\u{202F}'))
}

/// Return `true` for a direct numeric text operand or a recursively nested resident value.
///
/// Text builtins use this before any gather so unsupported resident values cannot trigger provider
/// access merely to discover that their class is not part of the public text contract. Host cell
/// elements remain for each builtin's container validator so it can report the precise argument or
/// element error instead of collapsing every malformed cell into a top-level type error.
pub(crate) fn contains_numeric_or_resident_text_input(value: &Value) -> bool {
    match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::LogicalArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::Symbolic(_)
        | Value::GpuTensor(_) => true,
        Value::Cell(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::Object(_)
        | Value::Closure(_)
        | Value::OutputList(_) => contains_resident_text_input(value),
        _ => false,
    }
}

/// Return `true` when a value or supported aggregate contains provider-resident data.
pub(crate) fn contains_resident_text_input(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(cell) => cell.data().iter().any(contains_resident_text_input),
        Value::Struct(value) => value.fields.values().any(contains_resident_text_input),
        Value::StructArray(array) => array.any_value(contains_resident_text_input),
        Value::Object(value) => value
            .properties()
            .values()
            .any(contains_resident_text_input),
        Value::Closure(value) => value.captures.iter().any(contains_resident_text_input),
        Value::OutputList(values) => values.iter().any(contains_resident_text_input),
        _ => false,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn missing_discriminant_does_not_reserve_literal_text() {
        assert!(is_missing_string(&runmat_value::StringElement::Missing));
        for text in ["<missing>", "<Missing>", "<MISSING>", ""] {
            assert!(!is_missing_string(&runmat_value::StringElement::text(text)));
        }
    }

    #[test]
    fn typed_missing_subject_and_literal_text_operation_matrix() {
        use runmat_value::StringElement;
        let run = |name: &str, arguments: Vec<Value>| {
            futures::executor::block_on(crate::call_builtin_async(name, &arguments))
                .unwrap()
                .value()
                .clone()
        };
        for source in [
            StringElement::text("<missing>"),
            StringElement::text("<Missing>"),
            StringElement::text(""),
            StringElement::Missing,
        ] {
            for (name, expected) in [
                ("lower", source.as_str().map(str::to_lowercase)),
                ("upper", source.as_str().map(str::to_uppercase)),
                ("strtrim", source.as_str().map(str::to_owned)),
                ("append", source.as_str().map(str::to_owned)),
            ] {
                assert_eq!(
                    run(name, vec![Value::String(source.clone())]),
                    Value::String(
                        expected
                            .map(StringElement::text)
                            .unwrap_or(StringElement::Missing)
                    ),
                    "{name}({source:?})"
                );
            }
            let expected = source
                .as_str()
                .map(|text| text.replace("missing", "present"));
            for name in ["erase", "replace", "strrep"] {
                let mut arguments = vec![Value::String(source.clone()), Value::string("missing")];
                let replacement = if name == "erase" { "" } else { "present" };
                if name != "erase" {
                    arguments.push(Value::string(replacement));
                }
                let expected = if name == "erase" {
                    source.as_str().map(|text| text.replace("missing", ""))
                } else {
                    expected.clone()
                };
                assert_eq!(
                    run(name, arguments),
                    Value::String(
                        expected
                            .map(StringElement::text)
                            .unwrap_or(StringElement::Missing)
                    ),
                    "{name}({source:?})"
                );
            }
            let length = run("strlength", vec![Value::String(source.clone())]);
            let Value::Num(length) = length else {
                panic!("scalar length")
            };
            match source.as_str() {
                Some(text) => assert_eq!(length, text.chars().count() as f64),
                None => assert!(length.is_nan()),
            }
            let converted = run("convertStringsToChars", vec![Value::String(source.clone())]);
            let expected_chars = match source.as_str() {
                Some(text) if !text.is_empty() => CharArray::new_row(text),
                _ => CharArray::from_code_units(Vec::new(), 0, 0).unwrap(),
            };
            assert_eq!(converted, Value::CharArray(expected_chars));
            assert_eq!(
                run(
                    "strcmp",
                    vec![Value::String(source.clone()), Value::String(source.clone())]
                ),
                Value::Bool(!source.is_missing())
            );
        }
        for source in [Value::Num(f64::NAN), Value::Complex(1.0, f64::NAN)] {
            let Value::StringArray(output) = run("string", vec![source]) else {
                panic!("string conversion")
            };
            assert_eq!(output.data(), &[StringElement::Missing]);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_row_collection_supports_row_major_storage() {
        let chars = runmat_value::encode_utf16_code_units("ABCDEF");
        assert_eq!(char_row_to_string_slice(&chars, 3, 0).unwrap(), "ABC");
        assert_eq!(char_row_to_string_slice(&chars, 3, 1).unwrap(), "DEF");
    }

    #[test]
    fn char_row_collection_uses_code_unit_offsets_and_refuses_malformed_utf16() {
        let chars = vec![
            Utf16CodeUnit::from_raw(0xD83D),
            Utf16CodeUnit::from_raw(0xDE42),
            Utf16CodeUnit::from_raw(0xD800),
        ];
        assert_eq!(char_row_to_string_slice(&chars, 2, 0).unwrap(), "\u{1F642}");
        assert_eq!(
            char_row_to_string_slice(&chars, 1, 2),
            Err(Utf16DecodeError {
                index: 0,
                unpaired_surrogate: 0xD800,
            })
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn text_admission_leaves_host_cell_elements_to_container_validation() {
        let numeric = Value::Int(runmat_value::IntValue::U64(u64::MAX));
        assert!(contains_numeric_or_resident_text_input(&numeric));
        let nested = Value::Cell(
            runmat_value::CellArray::new(vec![Value::String("ok".into()), numeric], 1, 2).unwrap(),
        );
        assert!(!contains_numeric_or_resident_text_input(&nested));
        assert!(!contains_numeric_or_resident_text_input(&Value::String(
            "text".into()
        )));
    }

    #[test]
    fn text_admission_detects_residency_nested_in_typed_struct_arrays() {
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let resident = Value::GpuTensor(allocation.handle.clone());
        let mut direct_struct = runmat_value::UnboundedStructValueBuilder::new();
        direct_struct.insert("payload", resident.clone());
        assert!(contains_numeric_or_resident_text_input(&Value::Struct(
            direct_struct.finish()
        )));
        let mut object = runmat_value::ObjectInstance::new("TextFixture".to_string());
        object.replace_property_unbounded("payload".to_string(), resident.clone());
        assert!(contains_numeric_or_resident_text_input(&Value::Object(
            object
        )));
        let mut first = runmat_value::UnboundedStructValueBuilder::new();
        first.insert("payload", resident);
        let mut second = runmat_value::UnboundedStructValueBuilder::new();
        second.insert("payload", Value::String("host".into()));
        let second = second.finish();
        let first = first.finish();
        let array = runmat_value::StructArray::new(vec![first, second], vec![2, 1]).unwrap();
        let value = Value::StructArray(array);
        assert!(contains_resident_text_input(&value));
        assert!(contains_numeric_or_resident_text_input(&value));
        drop(value);
        assert_eq!(allocation.probe.retirements(), 0);
        allocation
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        drop(allocation.handle);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}
