//! Shared helpers for string builtins.
use runmat_value::CharArray;

use runmat_value::Value;

pub(crate) fn is_multirow_char_array(value: &Value) -> bool {
    matches!(value, Value::CharArray(array) if array.rows > 1)
}

pub(crate) fn cell_contains_string(value: &Value) -> bool {
    match value {
        Value::Cell(cell) => cell.data.iter().any(|value| {
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
    for element in &cell.data {
        match element {
            Value::String(_) => has_string = true,
            Value::StringArray(array) if array.data.len() == 1 => has_string = true,
            Value::CharArray(chars) if chars.rows <= 1 => {}
            _ => return false,
        }
    }
    has_string
}

/// Canonical display for missing string scalars in MATLAB-compatible output.
const MISSING_SENTINEL: &str = "<missing>";

/// Return `true` when the provided text represents a missing string scalar.
#[inline]
pub(crate) fn is_missing_string(text: &str) -> bool {
    text.eq_ignore_ascii_case(MISSING_SENTINEL)
}

/// Convert text to lowercase while preserving MATLAB's `<missing>` sentinel.
#[inline]
pub(crate) fn lowercase_preserving_missing(text: String) -> String {
    if is_missing_string(&text) {
        MISSING_SENTINEL.to_string()
    } else {
        text.to_lowercase()
    }
}

/// Convert text to uppercase while preserving MATLAB's `<missing>` sentinel.
#[inline]
pub(crate) fn uppercase_preserving_missing(text: String) -> String {
    if is_missing_string(&text) {
        MISSING_SENTINEL.to_string()
    } else {
        text.to_uppercase()
    }
}

/// Collect a row from a [`CharArray`] into a `String`.
#[inline]
pub(crate) fn char_row_to_string(array: &CharArray, row: usize) -> String {
    char_row_to_string_slice(&array.data, array.cols, row)
}

/// Collect a row from a character slice laid out in row-major order.
#[inline]
pub(crate) fn char_row_to_string_slice(data: &[char], cols: usize, row: usize) -> String {
    let start = row * cols;
    let end = start + cols;
    data[start..end].iter().collect()
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
        Value::Cell(cell) => cell.data.iter().any(contains_resident_text_input),
        Value::Struct(value) => value.fields.values().any(contains_resident_text_input),
        Value::StructArray(array) => array.any_value(contains_resident_text_input),
        Value::Object(value) => value.properties.values().any(contains_resident_text_input),
        Value::Closure(value) => value.captures.iter().any(contains_resident_text_input),
        Value::OutputList(values) => values.iter().any(contains_resident_text_input),
        _ => false,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn detects_missing_strings_case_insensitively() {
        assert!(is_missing_string("<missing>"));
        assert!(is_missing_string("<Missing>"));
        assert!(!is_missing_string("<missing value>"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lowercase_preserves_missing() {
        assert_eq!(
            lowercase_preserving_missing("<missing>".to_string()),
            "<missing>"
        );
        assert_eq!(lowercase_preserving_missing("RunMat".to_string()), "runmat");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uppercase_preserves_missing() {
        assert_eq!(
            uppercase_preserving_missing("<missing>".to_string()),
            "<missing>"
        );
        assert_eq!(uppercase_preserving_missing("RunMat".to_string()), "RUNMAT");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_row_collection_supports_row_major_storage() {
        let chars: Vec<char> = vec!['A', 'B', 'C', 'D', 'E', 'F'];
        assert_eq!(char_row_to_string_slice(&chars, 3, 0), "ABC");
        assert_eq!(char_row_to_string_slice(&chars, 3, 1), "DEF");
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
        let mut direct_struct = runmat_value::StructValue::new();
        direct_struct.insert("payload", resident.clone());
        assert!(contains_numeric_or_resident_text_input(&Value::Struct(
            direct_struct
        )));
        let mut object = runmat_value::ObjectInstance::new("TextFixture".to_string());
        object
            .properties
            .insert("payload".to_string(), resident.clone());
        assert!(contains_numeric_or_resident_text_input(&Value::Object(
            object
        )));
        let mut first = runmat_value::StructValue::new();
        first.insert("payload", resident);
        let mut second = runmat_value::StructValue::new();
        second.insert("payload", Value::String("host".into()));
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
