use super::*;
use futures::executor::block_on;
use runmat_value::Tensor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, MutexGuard};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    path: PathBuf,
    _owner: fs::ProviderGuard,
    _lock: MutexGuard<'static, ()>,
}

impl std::ops::Deref for Fixture {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        &self.path
    }
}

impl AsRef<Path> for Fixture {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

fn write_fixture(ext: &str, contents: &str) -> Fixture {
    let lock = fs::provider_override_lock();
    let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let path = PathBuf::from(format!("/import/fixture-{unique}.{ext}"));
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir("/import"));
    provider
        .write_project_path(&path, contents.as_bytes())
        .expect("write fixture");
    let owner = fs::replace_provider(provider);
    Fixture {
        path,
        _owner: owner,
        _lock: lock,
    }
}

fn struct_field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Struct(st) = value else {
        panic!("expected struct");
    };
    st.fields
        .get(name)
        .unwrap_or_else(|| panic!("missing {name}"))
}

fn tensor_data(value: &Value) -> (Vec<f64>, Vec<usize>) {
    let Value::Tensor(tensor) = value else {
        panic!("expected tensor");
    };
    (tensor.materialize_f64(), tensor.shape().to_vec())
}

fn cell_text(value: &Value, row: usize, col: usize) -> String {
    let Value::Cell(cell) = value else {
        panic!("expected cell");
    };
    let Value::String(runmat_value::StringElement::Text(text)) =
        cell.get(row, col).expect("cell value")
    else {
        panic!("expected string cell");
    };
    text.to_string()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn importdata_descriptor_covers_core_forms() {
    let labels: Vec<&str> = IMPORTDATA_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"A = importdata(filename)"));
    assert!(labels.contains(&"A = importdata(filename, delimiterIn)"));
    assert!(labels.contains(&"A = importdata(filename, delimiterIn, headerlinesIn)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn paged_character_controls_reject_before_file_access() {
    let paged_name = runmat_value::CharArray::from_unicode_scalars_with_shape(
        "ab".chars().collect(),
        vec![1, 1, 2],
    )
    .expect("paged filename");
    let paged_delimiter = runmat_value::CharArray::from_unicode_scalars_with_shape(
        ",;".chars().collect(),
        vec![1, 1, 2],
    )
    .expect("paged delimiter");
    assert!(resolve_path(&Value::CharArray(paged_name)).is_err());
    assert!(parse_delimiter_arg(&Value::CharArray(paged_delimiter)).is_err());
    let trailing = runmat_value::CharArray::from_unicode_scalars_with_shape(
        "file.txt".chars().collect(),
        vec![1, 8, 1],
    )
    .expect("trailing singleton");
    assert_eq!(
        resolve_path(&Value::CharArray(trailing)).expect("row"),
        PathBuf::from("file.txt")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn relative_import_reads_from_installed_memory_provider() {
    let _lock = fs::provider_override_lock();
    let provider = std::sync::Arc::new(fs::MemoryFsProvider::with_current_dir("/import"));
    provider
        .write_project_path("/import/data.txt", b"1 2\n3 4\n")
        .expect("fixture");
    let _owner = fs::replace_provider(provider);
    let out = block_on(importdata_builtin(
        None,
        Value::from("data.txt"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("import from captured owner");
    assert_eq!(tensor_data(&out), (vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn excessive_outputs_reject_before_missing_file_read() {
    let _requested = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let err = block_on(importdata_builtin(
        None,
        Value::from("missing.txt"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("output bound");
    assert_eq!(err.identifier(), Some("RunMat:importdata:InvalidArgument"));
    assert!(err.message().contains("at most three outputs"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn importdata_reads_plain_numeric_matrix() {
    let path = write_fixture("txt", "1 2 3\n4 5 6\n");
    let out = block_on(importdata_builtin(
        None,
        Value::from(path.to_string_lossy().into_owned()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("importdata");
    let (data, shape) = tensor_data(&out);
    assert_eq!(shape, &[2, 3]);
    assert_eq!(data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn importdata_headerlines_zero_numeric_input_returns_tensor() {
    let path = write_fixture("txt", "1 2\n3 4\n");
    let out = block_on(importdata_builtin(
        None,
        Value::from(path.to_string_lossy().into_owned()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from(" "),
            Value::Num(0.0),
        ]),
    ))
    .expect("importdata");
    let (data, shape) = tensor_data(&out);
    assert_eq!(shape, &[2, 2]);
    assert_eq!(data, &[1.0, 3.0, 2.0, 4.0]);
    let _ = fs::remove_file(path);
}

#[test]
fn importdata_headerlines_reads_typed_integer_storage_exactly() {
    let header_lines = Tensor::new_integer(runmat_value::IntegerStorage::U16(vec![2]), vec![1, 1])
        .expect("header lines");
    assert_eq!(
        parse_header_lines(&Value::Tensor(header_lines)).expect("header lines"),
        2
    );

    let negative = Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![-1]), vec![1, 1])
        .expect("negative header lines");
    assert!(parse_header_lines(&Value::Tensor(negative)).is_err());

    assert_eq!(
        parse_header_lines(&Value::Int(runmat_value::IntValue::U64(u64::MAX))).ok(),
        usize::try_from(u64::MAX).ok()
    );
    assert!(parse_header_lines(&Value::Num(1.0e300)).is_err());
}

#[test]
fn importdata_headerlines_typed_integer_tensors_ignore_poisoned_f64_mirrors() {
    let classes = [
        runmat_value::IntegerStorage::I8(vec![2]),
        runmat_value::IntegerStorage::I16(vec![2]),
        runmat_value::IntegerStorage::I32(vec![2]),
        runmat_value::IntegerStorage::I64(vec![2]),
        runmat_value::IntegerStorage::U8(vec![2]),
        runmat_value::IntegerStorage::U16(vec![2]),
        runmat_value::IntegerStorage::U32(vec![2]),
        runmat_value::IntegerStorage::U64(vec![2]),
    ];

    for storage in classes {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("header lines");
        assert_eq!(parse_header_lines(&Value::Tensor(tensor)).unwrap(), 2);
    }
}

#[test]
fn importdata_returns_documented_delimiter_and_header_count_outputs() {
    let path = write_fixture("csv", "name,value\na,1\nb,2\n");
    let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let result = block_on(importdata_builtin(
        None,
        Value::from(path.to_string_lossy().into_owned()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("importdata outputs");
    let Value::OutputList(outputs) = result else {
        panic!("expected three outputs");
    };
    assert_eq!(outputs.len(), 3);
    assert_eq!(
        outputs[1],
        Value::CharArray(runmat_value::CharArray::new_row(","))
    );
    assert_eq!(outputs[2], Value::Num(1.0));
    let _ = fs::remove_file(path);
}

#[test]
fn importdata_rejects_resident_controls_before_provider_or_file_access() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered allocation");
    let original = allocation.handle.wire_descriptor();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let resident = Value::GpuTensor(allocation.handle.clone());
    let error = block_on(importdata_builtin(
        None,
        resident,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("resident filename must reject");
    assert_eq!(
        error.identifier(),
        Some("RunMat:importdata:InvalidArgument")
    );
    assert_eq!(allocation.handle.wire_descriptor(), original);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn importdata_detects_csv_header_and_colheaders() {
    let path = write_fixture("csv", "time,value\n0,1.5\n1,2.5\n");
    let out = block_on(importdata_builtin(
        None,
        Value::from(path.to_string_lossy().into_owned()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("importdata");
    let data = struct_field(&out, "data");
    let (values, shape) = tensor_data(data);
    assert_eq!(shape, &[2, 2]);
    assert_eq!(values, &[0.0, 1.0, 1.5, 2.5]);
    assert_eq!(cell_text(struct_field(&out, "colheaders"), 0, 0), "time");
    assert_eq!(cell_text(struct_field(&out, "colheaders"), 0, 1), "value");
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn importdata_honors_explicit_delimiter_and_header_lines() {
    let path = write_fixture("dat", "# instrument log\nA|B\n10|20\n30|40\n");
    let out = block_on(importdata_builtin(
        None,
        Value::from(path.to_string_lossy().into_owned()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("|"),
            Value::Num(2.0),
        ]),
    ))
    .expect("importdata");
    let data = struct_field(&out, "data");
    let (values, shape) = tensor_data(data);
    assert_eq!(shape, &[2, 2]);
    assert_eq!(values, &[10.0, 30.0, 20.0, 40.0]);
    assert_eq!(
        cell_text(struct_field(&out, "textdata"), 0, 0),
        "# instrument log"
    );
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn importdata_preserves_rowheaders() {
    let path = write_fixture("txt", "label x y\nr1 1 2\nr2 3 4\n");
    let out = block_on(importdata_builtin(
        None,
        Value::from(path.to_string_lossy().into_owned()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("importdata");
    assert_eq!(cell_text(struct_field(&out, "rowheaders"), 0, 0), "r1");
    assert_eq!(cell_text(struct_field(&out, "rowheaders"), 1, 0), "r2");
    assert_eq!(cell_text(struct_field(&out, "colheaders"), 0, 0), "x");
    assert_eq!(cell_text(struct_field(&out, "colheaders"), 0, 1), "y");
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn importdata_reports_mixed_unsupported_data() {
    let path = write_fixture("txt", "1 2\n3 nope\n");
    let err = block_on(importdata_builtin(
        None,
        Value::from(path.to_string_lossy().into_owned()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("parse error");
    assert!(err.message().contains("nonnumeric token"));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn importdata_rejects_rows_with_extra_numeric_columns() {
    let path = write_fixture("txt", "1 2\n3 4 5\n");
    let err = block_on(importdata_builtin(
        None,
        Value::from(path.to_string_lossy().into_owned()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("width mismatch");
    assert!(err.message().contains("expected 2"));
    let _ = fs::remove_file(path);
}
