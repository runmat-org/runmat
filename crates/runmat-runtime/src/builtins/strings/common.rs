//! Shared helpers for string builtins.
use runmat_builtins::CharArray;

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
}
