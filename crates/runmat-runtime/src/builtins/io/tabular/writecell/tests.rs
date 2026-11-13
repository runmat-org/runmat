use super::controls::parse_sheet;
use super::*;

async fn writecell_builtin(data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    super::writecell_builtin(
        None,
        data,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}
use calamine::{open_workbook_auto, Data, Reader};
use futures::executor::block_on;
use runmat_time::unix_timestamp_ms;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Barrier;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
fn selected_path(path: &Path) -> TabularFsPath {
    TabularFsOwner::capture(Some(path))
        .expect("captured provider")
        .path(path.to_path_buf())
        .expect("selected path")
}

use runmat_value::{CharArray, IntValue, IntegerStorage, LogicalArray, Tensor};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn temp_path(ext: &str) -> PathBuf {
    let millis = unix_timestamp_ms();
    let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "runmat_writecell_{}_{}_{}.{}",
        std::process::id(),
        millis,
        unique,
        ext
    ));
    path
}

fn cell(values: Vec<Value>, rows: usize, cols: usize) -> Value {
    Value::Cell(CellArray::new(values, rows, cols).expect("cell array"))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn overwrite_streams_long_quoted_cells_through_selected_memory_owner() {
    let _lock = runmat_filesystem::provider_override_lock();
    let root = "/tabular";
    let provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(root));
    let _guard = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir(root).expect("memory working directory");
    let path = PathBuf::from(root).join("quoted.csv");
    let text = format!("{}\"tail", "a".repeat(8193));
    let value = cell(vec![Value::from(text.clone())], 1, 1);
    block_on(writecell_builtin(
        value,
        vec![Value::from(path.to_string_lossy().as_ref())],
    ))
    .expect("staged overwrite");
    let expected = format!("\"{}\"\n", text.replace('"', "\"\""));
    assert_eq!(
        provider.read_project_path(&path).expect("selected output"),
        expected.as_bytes()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writecell_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = WRITECELL_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"writecell(C, filename)"));
    assert!(labels.contains(&"writecell(C, filename, name, optionValue)"));
    assert!(labels.contains(&"writecell(C, filename, nameValuePairs...)"));
}

#[test]
fn scalar_text_geometry_rejects_paged_controls_and_accepts_singleton_pages() {
    let paged = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("ab".chars().collect(), vec![1, 1, 2])
            .expect("paged characters"),
    );
    assert!(resolve_path(&paged).is_err());
    assert!(string_scalar_from_value(&paged, "Delimiter").is_err());
    assert!(table::cell_value_from_value(paged).is_err());

    let singleton = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("ab".chars().collect(), vec![1, 2, 1])
            .expect("singleton character page"),
    );
    assert_eq!(
        string_scalar_from_value(&singleton, "Delimiter").unwrap(),
        "ab"
    );
    assert!(
        matches!(table::cell_value_from_value(singleton), Ok(CellValue::Text(text)) if text == "ab")
    );
}

#[test]
fn malformed_host_pair_precedes_earlier_resident_option_gather() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident option");
    let data = cell(vec![Value::Num(1.0)], 1, 1);
    let error = block_on(writecell_builtin(
        data,
        vec![
            Value::from("unused.csv"),
            Value::GpuTensor(allocation.handle.clone()),
            Value::from("ignored"),
            Value::from("QuoteStrings"),
            Value::from("not-a-boolean"),
        ],
    ))
    .expect_err("known malformed host pair must precede resident gather");
    assert!(error.message().contains("QuoteStrings"));
    assert_eq!(allocation.probe.retirements(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writecell_writes_heterogeneous_csv() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("csv");
    let filename = path.to_string_lossy().into_owned();
    let values = cell(
        vec![
            Value::Num(1.5),
            Value::from("alpha"),
            Value::Bool(true),
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor")),
        ],
        2,
        2,
    );

    block_on(writecell_builtin(values, vec![Value::from(filename)])).expect("writecell");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, "1.5,\"alpha\"\n1,\n");
    let _ = fs::remove_file(path);
}

#[path = "tests/append.rs"]
mod append;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn writecell_overwrite_uses_same_path_write_lock() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("txt");
    fs::write(&path, "existing\n").expect("seed");
    let filename = path.to_string_lossy().into_owned();
    let lock = block_on(write_owner::write_lock_for_path(&selected_path(&path)))
        .expect("selected write lock");
    let guard = block_on(lock.lock());
    let (tx, rx) = mpsc::channel();

    let handle = thread::spawn(move || {
        let values = cell(vec![Value::from("replacement")], 1, 1);
        block_on(writecell_builtin(values, vec![Value::from(filename)])).expect("overwrite write");
        tx.send(()).expect("send completion");
    });

    thread::sleep(Duration::from_millis(50));
    assert!(rx.try_recv().is_err());
    drop(guard);
    handle.join().expect("writer thread");
    rx.recv_timeout(Duration::from_secs(1))
        .expect("overwrite completion");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, "\"replacement\"\n");
    let _ = fs::remove_file(path);
}

#[cfg(unix)]
#[test]
fn writecell_canonical_aliases_share_write_lock() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("txt");
    fs::write(&path, "").expect("seed");
    let mut link = path.clone();
    link.set_extension("link.txt");
    std::os::unix::fs::symlink(&path, &link).expect("symlink");

    let direct = block_on(write_owner::write_lock_for_path(&selected_path(&path)))
        .expect("direct write lock");
    let alias = block_on(write_owner::write_lock_for_path(&selected_path(&link)))
        .expect("alias write lock");
    assert!(Arc::ptr_eq(&direct, &alias));

    let _ = fs::remove_file(link);
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writecell_accepts_scalar_char_tensor_and_logical_cells() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("csv");
    let filename = path.to_string_lossy().into_owned();
    let typed = Tensor::new_integer(IntegerStorage::U16(vec![2026]), vec![1, 1])
        .expect("typed scalar tensor");
    let values = cell(
        vec![
            Value::CharArray(CharArray::new_row("name")),
            Value::Tensor(typed),
            Value::LogicalArray(LogicalArray::new(vec![0], vec![1, 1]).expect("logical")),
        ],
        1,
        3,
    );

    block_on(writecell_builtin(values, vec![Value::from(filename)])).expect("writecell");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, "\"name\",2026,0\n");
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writecell_preserves_integer_cell_text_exactly() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("csv");
    let filename = path.to_string_lossy().into_owned();
    let wide = (1_u64 << 53) + 1;
    let typed =
        Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("wide tensor");
    let values = cell(
        vec![Value::Int(IntValue::U64(u64::MAX)), Value::Tensor(typed)],
        1,
        2,
    );

    block_on(writecell_builtin(values, vec![Value::from(filename)])).expect("writecell");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, "18446744073709551615,9007199254740993\n");
    let _ = fs::remove_file(path);
}

#[test]
fn writecell_sheet_parser_rejects_unrepresentable_double_boundary() {
    assert!(parse_sheet(&Value::Num(usize::MAX as f64)).is_err());
    assert!(parse_sheet(&Value::Num((usize::MAX as f64) + 1.0)).is_err());

    let typed = Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1])
        .expect("typed sheet");
    let parsed = parse_sheet(&Value::Tensor(typed));
    match usize::try_from((1_u64 << 53) + 1) {
        Ok(expected) => {
            assert!(matches!(parsed, Ok(SheetSelector::Index(actual)) if actual == expected))
        }
        Err(_) => assert!(parsed.is_err()),
    }
}

#[test]
fn writecell_explicit_gpu_input_is_gated_before_filesystem_access() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let original = allocation.handle.wire_descriptor();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let handle = allocation
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("explicit immutable view");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let data = cell(vec![Value::GpuTensor(handle)], 1, 1);

    let error = block_on(writecell_builtin(data, vec![Value::from("unused.csv")]))
        .expect_err("strict mode rejects explicit GPU input before gather or file access");
    assert_eq!(
        error.identifier(),
        WRITECELL_EXPLICIT_GPU_EXTENSION.error_identifier
    );
    assert_eq!(allocation.handle.wire_descriptor(), original);
    assert_eq!(
        allocation.probe.retirements(),
        0,
        "caller alias survives nested preflight"
    );
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn writecell_bytes_output_is_gated_before_filesystem_access() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let data = cell(vec![Value::Num(1.0)], 1, 1);

    let error = block_on(writecell_builtin(
        data,
        vec![Value::from("definitely/missing/out.csv")],
    ))
    .expect_err("strict mode rejects the bytes-written output before file access");
    assert_eq!(
        error.identifier(),
        WRITECELL_BYTES_OUTPUT_EXTENSION.error_identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writecell_rejects_nested_cells_and_nonscalar_arrays() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("csv");
    let filename = path.to_string_lossy().into_owned();
    let nested = cell(vec![cell(vec![Value::Num(1.0)], 1, 1)], 1, 1);
    let err = block_on(writecell_builtin(
        nested,
        vec![Value::from(filename.clone())],
    ))
    .expect_err("nested cell error");
    assert!(err.message().contains("nested cell arrays"));

    let nonscalar = cell(
        vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor"),
        )],
        1,
        1,
    );
    let err = block_on(writecell_builtin(nonscalar, vec![Value::from(filename)]))
        .expect_err("nonscalar error");
    assert!(err.message().contains("unsupported cell value"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writecell_writes_xlsx_with_sheet_and_range() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("xlsx");
    let filename = path.to_string_lossy().into_owned();
    let values = cell(
        vec![Value::from("Voltage"), Value::Num(1.5), Value::Bool(true)],
        1,
        3,
    );

    block_on(writecell_builtin(
        values,
        vec![
            Value::from(filename),
            Value::from("Sheet"),
            Value::from("Measurements"),
            Value::from("Range"),
            Value::from("B2"),
        ],
    ))
    .expect("writecell xlsx");

    let mut workbook = open_workbook_auto(&path).expect("open workbook");
    assert_eq!(workbook.sheet_names()[0], "Measurements");
    let range = workbook
        .worksheet_range("Measurements")
        .expect("worksheet range");
    assert_eq!(
        range.get((0, 0)),
        Some(&Data::String("Voltage".to_string()))
    );
    assert_eq!(range.get((0, 1)), Some(&Data::Float(1.5)));
    assert_eq!(range.get((0, 2)), Some(&Data::Bool(true)));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writecell_rejects_unsupported_spreadsheet_extension() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("xls");
    let filename = path.to_string_lossy().into_owned();
    let values = cell(vec![Value::from("A"), Value::Num(1.0)], 1, 2);
    let err = block_on(writecell_builtin(values, vec![Value::from(filename)]))
        .expect_err("unsupported extension");
    assert!(err
        .message()
        .contains("unsupported spreadsheet file extension"));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn text_numeric_fields_keep_storage_precision_without_changing_xlsx_policy() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("csv");
    let single = 1.2345679_f32;
    let value = cell(
        vec![
            Value::Num(1.693061224489796),
            Value::Tensor(Tensor::from_f32(vec![single], vec![1, 1]).unwrap()),
            Value::Int(IntValue::U64(u64::MAX)),
            Value::Bool(true),
            Value::Num(-0.0),
            Value::Tensor(Tensor::from_f32(vec![-0.0], vec![1, 1]).unwrap()),
        ],
        1,
        6,
    );
    block_on(writecell_builtin(
        value,
        vec![Value::from(path.to_string_lossy().as_ref())],
    ))
    .expect("write class-aware text");
    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        "1.6930612244898,1.234568,18446744073709551615,1,0,0\n"
    );
    let _ = fs::remove_file(path);

    let single =
        scalar_tensor_cell_value(&Tensor::from_f32(vec![single], vec![1, 1]).unwrap()).unwrap();
    let mut xml = String::new();
    append_sheet_cell_xml(&mut xml, "A1", &single);
    assert!(
        xml.contains("<v>1.234567880630493</v>"),
        "XLSX retains its existing numeric encoding: {xml}"
    );
}
