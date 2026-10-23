use super::*;
use calamine::{open_workbook_auto, open_workbook_auto_from_rs, Data, Reader};
use futures::executor::block_on;
use runmat_time::unix_timestamp_ms;
use runmat_value::{IntegerStorage, SparseTensor};
use std::fs;
use std::io::Cursor;
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "tests/admission.rs"]
mod admission;
#[path = "tests/publication.rs"]
mod publication;

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

async fn xlswrite_call(filename: Value, data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    xlswrite_builtin(
        None,
        filename,
        data,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

async fn table_from_value(value: Value) -> BuiltinResult<XlsTable> {
    let path = temp_path("xlsx");
    let selected = TabularFsOwner::capture(Some(&path))
        .expect("captured test owner")
        .path(path)
        .expect("selected test path");
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    XlsTable::from_value_cooperative(value, None, &mut cooperation, &selected).await
}

#[test]
fn cancellation_is_not_converted_to_spreadsheet_status() {
    let cancelled = build_runtime_error("xlswrite: cancelled")
        .with_builtin(BUILTIN_NAME)
        .with_source(crate::execution::ExecutionServiceError::Cancelled)
        .build();
    assert!(is_cancelled(&cancelled));
    assert!(!is_cancelled(&xlswrite_error_with(
        &XLSWRITE_ERROR_IO,
        "ordinary spreadsheet failure"
    )));
}

#[test]
fn scalar_text_geometry_rejects_paged_filename_before_file_access() {
    let paged = CharArray::from_unicode_scalars_with_shape("ab".chars().collect(), vec![1, 1, 2])
        .expect("paged character array");
    assert!(resolve_path(&Value::CharArray(paged)).is_err());
    let trailing_singleton =
        CharArray::from_unicode_scalars_with_shape("ab".chars().collect(), vec![1, 2, 1])
            .expect("trailing singleton character row");
    assert_eq!(
        value_to_string_scalar(&Value::CharArray(trailing_singleton)).expect("scalar text"),
        "ab"
    );
}

fn temp_path(ext: &str) -> PathBuf {
    let millis = unix_timestamp_ms();
    let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "runmat_xlswrite_{}_{}_{}.{}",
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

fn output_list(value: &Value) -> &[Value] {
    let Value::OutputList(outputs) = value else {
        panic!("expected output list, got {value:?}");
    };
    outputs
}

#[test]
fn xlswrite_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = XLSWRITE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"status = xlswrite(filename, A)"));
    assert!(labels.contains(&"status = xlswrite(filename, A, sheetOrRange)"));
    assert!(labels.contains(&"status = xlswrite(filename, A, sheet, range)"));
    assert!(labels.contains(&"status = xlswrite(filename, A, sheet, range, '-basic')"));
    assert!(labels.contains(&"[status,message] = xlswrite(filename, A)"));
    assert!(labels.contains(&"[status,message] = xlswrite(filename, A, sheetOrRange)"));
    assert!(labels.contains(&"[status,message] = xlswrite(filename, A, sheet, range, '-basic')"));
}

#[test]
fn xlswrite_numeric_range_reads_typed_integer_storage_exactly() {
    let range = Tensor::new_integer(IntegerStorage::U16(vec![2, 3]), vec![1, 2]).expect("range");

    let parsed = parse_range_start(&Value::Tensor(range)).expect("range");

    assert_eq!(parsed.row, 1);
    assert_eq!(parsed.col, 2);

    let boundary = Value::Tensor(
        Tensor::new(vec![usize::MAX as f64, 1.0], vec![1, 2]).expect("boundary range"),
    );
    assert!(parse_range_start(&boundary).is_err());
}

#[test]
fn xlswrite_gates_numeric_range_before_file_access() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let range =
        Tensor::new_integer(IntegerStorage::U16(vec![1, 1]), vec![1, 2]).expect("numeric range");
    let error = futures::executor::block_on(xlswrite_call(
        Value::from("definitely/missing/out.xlsx"),
        Value::Num(1.0),
        vec![Value::from("Sheet1"), Value::Tensor(range)],
    ))
    .expect_err("strict mode rejects numeric ranges");
    assert_eq!(
        error.identifier(),
        XLSWRITE_NUMERIC_RANGE_EXTENSION.error_identifier
    );
    drop(strict);
}

#[test]
fn xlswrite_writes_numeric_matrix_with_sheet_and_range() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("xlsx");
    let filename = path.to_string_lossy().into_owned();
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();

    let ok = block_on(xlswrite_call(
        Value::from(filename),
        Value::Tensor(tensor),
        vec![Value::from("Data"), Value::from("B2")],
    ))
    .expect("xlswrite");
    assert_eq!(ok, Value::Bool(true));

    let mut workbook = open_workbook_auto(&path).expect("open workbook");
    assert_eq!(workbook.sheet_names()[0], "Data");
    let range = workbook.worksheet_range("Data").expect("worksheet");
    assert_eq!(range.get((0, 0)), Some(&Data::Float(1.0)));
    assert_eq!(range.get((0, 1)), Some(&Data::Float(2.0)));
    assert_eq!(range.get((0, 2)), Some(&Data::Float(3.0)));
    assert_eq!(range.get((1, 0)), Some(&Data::Float(4.0)));
    assert_eq!(range.get((1, 1)), Some(&Data::Float(5.0)));
    assert_eq!(range.get((1, 2)), Some(&Data::Float(6.0)));
    let _ = fs::remove_file(path);
}

#[test]
fn xlswrite_accepts_cell_strings_logicals_and_ints() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("xls");
    let filename = path.to_string_lossy().into_owned();
    let typed = Tensor::new_integer(IntegerStorage::U16(vec![2026]), vec![1, 1])
        .expect("typed scalar tensor");
    let values = cell(
        vec![
            Value::from("name"),
            Value::Bool(true),
            Value::Tensor(typed),
            Value::from("tail"),
        ],
        2,
        2,
    );

    let _status = block_on(xlswrite_call(
        Value::from(filename),
        values,
        vec![Value::from("A1")],
    ))
    .expect("xlswrite");

    let bytes = fs::read(&path).expect("read workbook bytes");
    let mut workbook = open_workbook_auto_from_rs(Cursor::new(bytes)).expect("open workbook");
    let range = workbook.worksheet_range("Sheet1").expect("worksheet");
    assert_eq!(range.get((0, 0)), Some(&Data::String("name".to_string())));
    assert_eq!(range.get((0, 1)), Some(&Data::Bool(true)));
    assert_eq!(range.get((1, 0)), Some(&Data::Float(2026.0)));
    assert_eq!(range.get((1, 1)), Some(&Data::String("tail".to_string())));
    let _ = fs::remove_file(path);
}

#[test]
fn xlswrite_preserves_integer_cell_xml_exactly() {
    let wide = (1_u64 << 53) + 1;
    let typed =
        Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("wide tensor");

    let scalar =
        block_on(table_from_value(Value::Int(IntValue::U64(u64::MAX)))).expect("scalar table");
    let scalar_xml =
        writecell::build_sheet_xml(&scalar.into_cell_table().unwrap(), RangeStart::default());
    assert!(scalar_xml.contains("<v>18446744073709551615</v>"));

    let tensor = block_on(table_from_value(Value::Tensor(typed))).expect("tensor table");
    let tensor_xml =
        writecell::build_sheet_xml(&tensor.into_cell_table().unwrap(), RangeStart::default());
    assert!(tensor_xml.contains("<v>9007199254740993</v>"));
}

#[test]
fn xlswrite_accepts_top_level_sparse_matrix() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("xlsx");
    let filename = path.to_string_lossy().into_owned();
    let sparse = SparseTensor::new(2, 2, vec![0, 1, 2], vec![1, 0], vec![9.0, 8.0]).unwrap();

    block_on(xlswrite_call(
        Value::from(filename),
        Value::SparseTensor(sparse),
        Vec::new(),
    ))
    .expect("xlswrite sparse");

    let mut workbook = open_workbook_auto(&path).expect("open workbook");
    let range = workbook.worksheet_range("Sheet1").expect("worksheet");
    assert_eq!(range.get((0, 0)), Some(&Data::Float(0.0)));
    assert_eq!(range.get((0, 1)), Some(&Data::Float(8.0)));
    assert_eq!(range.get((1, 0)), Some(&Data::Float(9.0)));
    assert_eq!(range.get((1, 1)), Some(&Data::Float(0.0)));
    let _ = fs::remove_file(path);
}

#[test]
fn xlswrite_preserves_sparse_integer_cells_exactly() {
    let sparse = SparseTensor::new_integer(
        2,
        2,
        vec![0, 1, 2],
        vec![1, 0],
        IntegerStorage::U64(vec![u64::MAX, (1_u64 << 53) + 1]),
    )
    .expect("integer sparse");
    let table = block_on(table_from_value(Value::SparseTensor(sparse))).expect("table");
    let xml = writecell::build_sheet_xml(&table.into_cell_table().unwrap(), RangeStart::default());

    assert!(xml.contains("<v>18446744073709551615</v>"));
    assert!(xml.contains("<v>9007199254740993</v>"));
}

#[test]
fn xlswrite_preserves_existing_cells_outside_overlay() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("xlsx");
    let filename = path.to_string_lossy().into_owned();
    let initial = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    block_on(xlswrite_call(
        Value::from(filename.clone()),
        Value::Tensor(initial),
        vec![Value::from("Sheet1"), Value::from("A1")],
    ))
    .expect("initial xlswrite");

    block_on(xlswrite_call(
        Value::from(filename),
        Value::Num(9.0),
        vec![Value::from("Sheet1"), Value::from("B2")],
    ))
    .expect("overlay xlswrite");

    let mut workbook = open_workbook_auto(&path).expect("open workbook");
    let range = workbook.worksheet_range("Sheet1").expect("worksheet");
    assert_eq!(range.get_value((0, 0)), Some(&Data::Float(1.0)));
    assert_eq!(range.get_value((0, 1)), Some(&Data::Float(2.0)));
    assert_eq!(range.get_value((1, 0)), Some(&Data::Float(3.0)));
    assert_eq!(range.get_value((1, 1)), Some(&Data::Float(9.0)));
    let _ = fs::remove_file(path);
}

#[test]
fn xlswrite_preserves_other_sheets_and_uses_numeric_sheet_ordinal() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("xlsx");
    let filename = path.to_string_lossy().into_owned();
    block_on(xlswrite_call(
        Value::from(filename.clone()),
        Value::Num(1.0),
        vec![Value::from("First"), Value::from("A1")],
    ))
    .expect("write first sheet");
    block_on(xlswrite_call(
        Value::from(filename),
        Value::Num(2.0),
        vec![Value::Num(2.0), Value::from("A1")],
    ))
    .expect("write second sheet");

    let mut workbook = open_workbook_auto(&path).expect("open workbook");
    assert_eq!(
        workbook.sheet_names(),
        &["First".to_string(), "Sheet2".to_string()]
    );
    assert_eq!(
        workbook
            .worksheet_range("First")
            .expect("first")
            .get_value((0, 0)),
        Some(&Data::Float(1.0))
    );
    assert_eq!(
        workbook
            .worksheet_range_at(1)
            .expect("second exists")
            .expect("second")
            .get_value((0, 0)),
        Some(&Data::Float(2.0))
    );
    let _ = fs::remove_file(path);
}

#[test]
fn xlswrite_numeric_sheet_placeholders_avoid_name_collisions() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("xlsx");
    let filename = path.to_string_lossy().into_owned();
    block_on(xlswrite_call(
        Value::from(filename.clone()),
        Value::Num(1.0),
        vec![Value::from("First"), Value::from("A1")],
    ))
    .expect("write first sheet");
    block_on(xlswrite_call(
        Value::from(filename.clone()),
        Value::Num(3.0),
        vec![Value::from("Sheet3"), Value::from("A1")],
    ))
    .expect("write named sheet3");
    block_on(xlswrite_call(
        Value::from(filename),
        Value::Num(2.0),
        vec![Value::Num(3.0), Value::from("B2")],
    ))
    .expect("write ordinal third sheet");

    let workbook = open_workbook_auto(&path).expect("open workbook");
    let names = workbook.sheet_names();
    assert_eq!(names.len(), 3);
    let unique: std::collections::HashSet<&String> = names.iter().collect();
    assert_eq!(unique.len(), names.len());
    assert_eq!(names[0], "First");
    assert_eq!(names[1], "Sheet3");
    assert_ne!(names[2], "Sheet3");
    let _ = fs::remove_file(path);
}

#[test]
fn xlswrite_accepts_scalar_tensor_sheet_selector_and_absolute_range() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("xlsx");
    let filename = path.to_string_lossy().into_owned();
    let sheet = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
    block_on(xlswrite_call(
        Value::from(filename),
        Value::Num(42.0),
        vec![Value::Tensor(sheet), Value::from("$B$2:$B$2")],
    ))
    .expect("xlswrite");

    let mut workbook = open_workbook_auto(&path).expect("open workbook");
    let sheet2 = workbook
        .worksheet_range_at(1)
        .expect("second sheet exists")
        .expect("second sheet");
    assert_eq!(sheet2.get_value((1, 1)), Some(&Data::Float(42.0)));
    let _ = fs::remove_file(path);
}

#[test]
fn xlswrite_rejects_malformed_range_end() {
    let err = block_on(xlswrite_call(
        Value::from("out.xlsx"),
        Value::Num(1.0),
        vec![Value::from("Sheet1"), Value::from("B2:not-a-cell")],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:xlswrite:Range"));
}

#[test]
fn xlswrite_rejects_sparse_inputs_that_would_densify_too_large() {
    let sparse = SparseTensor::zeros(MAX_XLSWRITE_CELLS + 1, 1);
    let err = block_on(xlswrite_call(
        Value::from("out.xlsx"),
        Value::SparseTensor(sparse),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:xlswrite:Data"));
}

#[test]
fn xlswrite_rejects_unmarked_existing_workbook_instead_of_flattening_it() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("xlsx");
    let table = CellTable::from_cells(1, 1, vec![CellValue::Number(1.0)]).expect("table");
    let bytes = writecell::build_xlsx_workbook(&table, "External", RangeStart::default())
        .expect("workbook bytes");
    fs::write(&path, bytes).expect("write workbook");

    let err = block_on(xlswrite_call(
        Value::from(path.to_string_lossy().into_owned()),
        Value::Num(2.0),
        vec![Value::from("External"), Value::from("A1")],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:xlswrite:Io"));
    assert!(err.message().contains("not created by RunMat xlswrite"));
    let _ = fs::remove_file(path);
}

#[test]
fn xlswrite_rejects_excessive_numeric_sheet_index() {
    let err = block_on(xlswrite_call(
        Value::from("out.xlsx"),
        Value::Num(1.0),
        vec![
            Value::Num((MAX_XLSWRITE_SHEETS + 1) as f64),
            Value::from("A1"),
        ],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:xlswrite:Sheet"));
}

#[test]
fn xlswrite_sheet_selector_preserves_typed_integer_tensor_bounds() {
    let tensor =
        Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("typed sheet tensor");
    assert!(matches!(
        parse_sheet_selector(&Value::Tensor(tensor)),
        Ok(SheetSelector::Index(6))
    ));

    let too_large = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
        .expect("typed sheet tensor");
    assert!(parse_sheet_selector(&Value::Tensor(too_large)).is_err());
    let negative =
        Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).expect("negative");
    assert!(parse_sheet_selector(&Value::Tensor(negative)).is_err());

    assert!(parse_sheet_selector(&Value::Num(usize::MAX as f64)).is_err());
    assert!(parse_sheet_selector(&Value::Num((usize::MAX as f64) + 1.0)).is_err());
}

#[test]
fn xlswrite_rejects_invalid_sheet_names_instead_of_rewriting_them() {
    let err = block_on(xlswrite_call(
        Value::from("out.xlsx"),
        Value::Num(1.0),
        vec![Value::from("A/B"), Value::from("A1")],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:xlswrite:Sheet"));
}

#[test]
fn xlswrite_returns_status_and_message_outputs() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("xlsx");
    let filename = path.to_string_lossy().into_owned();
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = block_on(xlswrite_call(
        Value::from(filename),
        Value::Complex(1.0, 2.0),
        Vec::new(),
    ))
    .expect("xlswrite captures failure");
    let outputs = output_list(&result);
    assert_eq!(outputs[0], Value::Bool(false));
    let Value::Struct(message) = &outputs[1] else {
        panic!("expected message struct");
    };
    assert!(matches!(
        message.fields.get("identifier"),
        Some(Value::String(id)) if id == "RunMat:xlswrite:Data"
    ));
    let _ = fs::remove_file(path);
}

#[test]
fn xlswrite_rejects_too_many_outputs() {
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let err = block_on(xlswrite_call(
        Value::from("out.xlsx"),
        Value::Num(1.0),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:xlswrite:OutputCount"));
}
