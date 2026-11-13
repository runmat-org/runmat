use super::*;
use runmat_value::{CharArray, StringArray};
use std::sync::Arc;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn paged_filename_and_option_name_fail_before_file_creation() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/tabular",
    ));
    let _guard = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir("/tabular").expect("cwd");

    let paged_path =
        CharArray::from_unicode_scalars_with_shape("a.csvb.csv".chars().collect(), vec![1, 5, 2])
            .expect("paged path");
    let path_error = block_on(super::super::writematrix_builtin(
        None,
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::CharArray(
            paged_path,
        )]),
    ))
    .expect_err("paged filename");
    assert!(path_error.message().contains("character vector"));
    assert!(provider.read_project_path("/tabular/a.csv").is_err());

    let paged_name = CharArray::from_unicode_scalars_with_shape(
        "DelimiterDelimiter".chars().collect(),
        vec![1, 9, 2],
    )
    .expect("paged option name");
    let option_error = block_on(super::super::writematrix_builtin(
        None,
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("out.csv"),
            Value::CharArray(paged_name),
            Value::from(";"),
        ]),
    ))
    .expect_err("paged option");
    assert!(option_error.message().contains("option name"));
    assert!(provider.read_project_path("/tabular/out.csv").is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn singleton_page_char_filename_remains_admitted() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/tabular",
    ));
    let _guard = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir("/tabular").expect("cwd");
    let path =
        CharArray::from_unicode_scalars_with_shape("out.csv".chars().collect(), vec![1, 7, 1])
            .expect("singleton page");
    let returned = block_on(super::super::writematrix_builtin(
        None,
        Value::Num(3.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::CharArray(path)]),
    ))
    .expect("valid character path");
    assert!(matches!(returned, Value::Num(2.0)));
    assert_eq!(
        provider
            .read_project_path("/tabular/out.csv")
            .expect("file"),
        b"3\n"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn large_text_cell_quotes_across_chunk_boundary_without_payload_mirror() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/tabular",
    ));
    let _guard = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir("/tabular").expect("cwd");
    let left = "a".repeat(4095);
    let right = "b".repeat(4095);
    let text = format!("{left}\"{right}");
    let matrix = StringArray::new_2d(vec![text], 1, 1).expect("string matrix");
    block_on(super::super::writematrix_builtin(
        None,
        Value::StringArray(matrix),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("large.csv")]),
    ))
    .expect("chunked text write");
    let expected = format!("\"{left}\"\"{right}\"\n");
    assert_eq!(
        provider
            .read_project_path("/tabular/large.csv")
            .expect("file"),
        expected.into_bytes()
    );
}
