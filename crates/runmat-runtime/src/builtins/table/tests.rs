use super::*;
async fn table2array_builtin(value: Value) -> BuiltinResult<Value> {
    super::table2array_builtin(None, value).await
}
async fn table2cell_builtin(value: Value) -> BuiltinResult<Value> {
    super::table2cell_builtin(None, value).await
}
fn grpstats_impl(value: Value, group: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    super::object::grpstats_impl(
        value,
        group,
        crate::call::FunctionArgumentView::from_slice(&rest),
    )
}
fn groupsummary_impl(
    value: Value,
    group: Value,
    method: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    super::object::groupsummary_impl(
        value,
        group,
        method,
        crate::call::FunctionArgumentView::from_slice(&rest),
    )
}

async fn struct2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    super::struct2table_builtin(None, value, rest).await
}

async fn table2struct_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    super::table2struct_builtin(None, value, rest).await
}

use crate::{RuntimeError, OBJECT_INDEX_PAREN};
#[cfg(not(target_arch = "wasm32"))]
use async_trait::async_trait;
use futures::executor::block_on;
use runmat_builtins::{
    ARRAY2TABLE_EXTENSIONS, ARRAY2TABLE_GPU_INPUT_EXTENSION, ARRAY2TABLE_INTEGER_CAPABILITIES,
    ARRAY2TIMETABLE_EXTENSIONS, ARRAY2TIMETABLE_INTEGER_CAPABILITIES,
};
#[cfg(not(target_arch = "wasm32"))]
use runmat_filesystem::{
    DirEntry, FileHandle, FsMetadata, FsProvider, NativeFsProvider, OpenFlags, SandboxFsProvider,
};
use runmat_time::unix_timestamp_ms;
use runmat_value::IntValue;
use runmat_value::IntegerStorage;
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::io;
use std::io::Write;

mod indexing_methods;
#[cfg(not(target_arch = "wasm32"))]
mod timetable_detection;

fn run_parquetread(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    block_on(parquetread_builtin(None, path, rest))
}

async fn readtable_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    path: Value,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    let arguments = crate::call::BuiltinArguments::from_borrowed(&arguments, runtime.as_ref())?;
    super::readtable_builtin(runtime, path, arguments).await
}

async fn readcell_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    path: Value,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    let arguments = crate::call::BuiltinArguments::from_borrowed(&arguments, runtime.as_ref())?;
    super::readcell_builtin(runtime, path, arguments).await
}

async fn detect_import_options_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    path: Value,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    let arguments = crate::call::BuiltinArguments::from_borrowed(&arguments, runtime.as_ref())?;
    super::detect_import_options_builtin(runtime, path, arguments).await
}

async fn spreadsheet_import_options_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    let arguments = crate::call::BuiltinArguments::from_borrowed(&arguments, runtime.as_ref())?;
    super::spreadsheet_import_options_builtin(runtime, arguments).await
}

async fn timetable2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    super::timetable2table_builtin(None, value, rest).await
}

async fn timetable_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    super::timetable_builtin(None, args).await
}

async fn array2timetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    super::array2timetable_builtin(None, value, rest).await
}

async fn table2timetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    super::table2timetable_builtin(None, value, rest).await
}

async fn array2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    super::array2table_builtin(None, value, rest).await
}

async fn readtimetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    super::readtimetable_builtin(None, value, rest).await
}

async fn timerange_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    super::timerange_builtin(None, args).await
}

async fn vartype_builtin(value: Value) -> BuiltinResult<Value> {
    super::vartype_builtin(None, value).await
}

async fn rowfilter_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    super::rowfilter_builtin(None, args).await
}

async fn array_datastore_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    super::array_datastore_builtin(None, args).await
}

async fn file_datastore_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    super::file_datastore_builtin(None, args).await
}

async fn parquet_datastore_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    super::parquet_datastore_builtin(None, args).await
}

async fn uitable_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    super::uitable_builtin(None, args).await
}

#[cfg(feature = "wgpu")]
fn register_wgpu_provider_available() -> bool {
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_ok()
        && runmat_accelerate_api::provider().is_some()
}

#[cfg(not(target_arch = "wasm32"))]
struct PrefixSandboxProvider {
    prefix: &'static str,
    sandbox: SandboxFsProvider,
    native: NativeFsProvider,
}

#[cfg(not(target_arch = "wasm32"))]
impl PrefixSandboxProvider {
    fn is_virtual(&self, path: &Path) -> bool {
        path.to_string_lossy().starts_with(self.prefix)
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait(?Send)]
impl FsProvider for PrefixSandboxProvider {
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        if self.is_virtual(path) {
            self.sandbox.open(path, flags)
        } else {
            self.native.open(path, flags)
        }
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        if self.is_virtual(path) {
            self.sandbox.read(path).await
        } else {
            self.native.read(path).await
        }
    }

    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        if self.is_virtual(path) {
            self.sandbox.write(path, data).await
        } else {
            self.native.write(path, data).await
        }
    }

    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        if self.is_virtual(path) {
            self.sandbox.remove_file(path).await
        } else {
            self.native.remove_file(path).await
        }
    }

    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        if self.is_virtual(path) {
            self.sandbox.metadata(path).await
        } else {
            self.native.metadata(path).await
        }
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        if self.is_virtual(path) {
            self.sandbox.symlink_metadata(path).await
        } else {
            self.native.symlink_metadata(path).await
        }
    }

    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        if self.is_virtual(path) {
            self.sandbox.read_dir(path).await
        } else {
            self.native.read_dir(path).await
        }
    }

    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        if self.is_virtual(path) {
            self.sandbox.canonicalize(path).await
        } else {
            self.native.canonicalize(path).await
        }
    }

    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        if self.is_virtual(path) {
            self.sandbox.create_dir(path).await
        } else {
            self.native.create_dir(path).await
        }
    }

    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        if self.is_virtual(path) {
            self.sandbox.create_dir_all(path).await
        } else {
            self.native.create_dir_all(path).await
        }
    }

    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        if self.is_virtual(path) {
            self.sandbox.remove_dir(path).await
        } else {
            self.native.remove_dir(path).await
        }
    }

    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        if self.is_virtual(path) {
            self.sandbox.remove_dir_all(path).await
        } else {
            self.native.remove_dir_all(path).await
        }
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        match (self.is_virtual(from), self.is_virtual(to)) {
            (true, true) => self.sandbox.rename(from, to).await,
            (false, false) => self.native.rename(from, to).await,
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "cross-provider rename is unsupported in test provider",
            )),
        }
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        if self.is_virtual(path) {
            self.sandbox.set_readonly(path, readonly).await
        } else {
            self.native.set_readonly(path, readonly).await
        }
    }
}

fn unique_path(prefix: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "runmat_{prefix}_{}_{}",
        std::process::id(),
        unix_timestamp_ms()
    ));
    path
}

fn read_table(path: &Path, args: Vec<Value>) -> Value {
    block_on(readtable_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        args,
    ))
    .expect("readtable")
}

fn read_table_err(path: &Path, args: Vec<Value>) -> RuntimeError {
    block_on(readtable_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        args,
    ))
    .expect_err("expected readtable failure")
}

fn spreadsheet_options(args: Vec<Value>) -> StructValue {
    let value =
        block_on(spreadsheet_import_options_builtin(None, args)).expect("spreadsheetImportOptions");
    crate::builtins::io::tabular::import_options::ImportOptions::decode(None, &value)
        .expect("decode spreadsheet options")
        .expect("spreadsheet options object")
        .base()
        .clone()
}

fn detect_options(path: &Path, args: Vec<Value>) -> StructValue {
    let value = block_on(detect_import_options_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        args,
    ))
    .expect("detectImportOptions");
    crate::builtins::io::tabular::import_options::ImportOptions::decode(None, &value)
        .expect("decode detected options")
        .expect("detected options object")
        .base()
        .clone()
}

#[test]
fn variable_import_options_modify_real_table_columns_and_preserve_object_identity() {
    let path = unique_path("setvaropts_table_columns");
    fs::write(
        &path,
        "Force_N,Amount,When\nN/A,1.234,31.12.2024\n2,2.500,01.01.2025\n",
    )
    .expect("write import fixture");
    let filename = Value::from(path.to_string_lossy().to_string());
    let detected = block_on(detect_import_options_builtin(
        None,
        filename.clone(),
        Vec::new(),
    ))
    .expect("detect import options");
    let numeric = block_on(setvartype_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            detected,
            Value::from("Force_N"),
            Value::from("double"),
        ]),
    ))
    .expect("set numeric variable type");
    let typed = block_on(setvartype_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            numeric,
            Value::from("When"),
            Value::from("datetime"),
        ]),
    ))
    .expect("set datetime variable type");
    let missing = block_on(setvaropts_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            typed,
            Value::from("Force_N"),
            Value::from("TreatAsMissing"),
            Value::from("N/A"),
            Value::from("FillValue"),
            Value::Num(-1.0),
        ]),
    ))
    .expect("set missing rule");
    let separated = block_on(setvaropts_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            missing,
            Value::from("Amount"),
            Value::from("DecimalSeparator"),
            Value::from(","),
            Value::from("ThousandsSeparator"),
            Value::from("."),
        ]),
    ))
    .expect("set separators");
    let formatted = block_on(setvaropts_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            separated,
            Value::from("When"),
            Value::from("InputFormat"),
            Value::from("dd.MM.yyyy"),
        ]),
    ))
    .expect("set datetime input format");
    let variable = block_on(getvaropts_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            formatted.clone(),
            Value::from("Amount"),
        ]),
    ))
    .expect("get variable options");
    assert!(matches!(variable, Value::Object(_)));
    let table = object(
        block_on(readtable_builtin(
            None,
            filename.clone(),
            vec![formatted.clone()],
        ))
        .expect("readtable with variable options"),
    );
    let Value::Tensor(force) = table_member_get(&table, &Value::from("Force_N")).unwrap() else {
        panic!("expected numeric Force_N variable")
    };
    assert_eq!(force.materialize_f64(), vec![-1.0, 2.0]);
    let Value::Tensor(amount) = table_member_get(&table, &Value::from("Amount")).unwrap() else {
        panic!("expected numeric Amount variable")
    };
    assert_eq!(amount.materialize_f64(), vec![1234.0, 2500.0]);
    let Value::Object(when) = table_member_get(&table, &Value::from("When")).unwrap() else {
        panic!("expected parsed datetime variable")
    };
    assert!(when.is_class(runmat_types::standard::DATETIME));
    let ranged = object(
        block_on(readtable_builtin(
            None,
            filename,
            vec![formatted, Value::from("Range"), Value::from("B:C")],
        ))
        .expect("ranged table retains source-column options"),
    );
    assert_eq!(
        table_variable_names_from_object(&ranged).unwrap(),
        vec!["Amount", "When"]
    );
    let Value::Tensor(ranged_amount) = table_member_get(&ranged, &Value::from("Amount")).unwrap()
    else {
        panic!("expected ranged Amount variable")
    };
    assert_eq!(ranged_amount.materialize_f64(), vec![1234.0, 2500.0]);
    let range_detected = block_on(detect_import_options_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        vec![Value::from("Range"), Value::from("B:C")],
    ))
    .expect("detect projected variables");
    let range_set = block_on(setvaropts_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            range_detected,
            Value::from("Amount"),
            Value::from("DecimalSeparator"),
            Value::from(","),
            Value::from("ThousandsSeparator"),
            Value::from("."),
        ]),
    ))
    .expect("configure detected ranged variable");
    let from_range_detected = object(
        block_on(readtable_builtin(
            None,
            Value::from(path.to_string_lossy().to_string()),
            vec![range_set],
        ))
        .expect("read detected ranged options"),
    );
    let Value::Tensor(range_detected_amount) =
        table_member_get(&from_range_detected, &Value::from("Amount")).unwrap()
    else {
        panic!("expected detected ranged Amount variable")
    };
    assert_eq!(
        range_detected_amount.materialize_f64(),
        vec![1234.0, 2500.0]
    );
    let _ = fs::remove_file(path);
}

fn char_row(array: &CharArray, row: usize) -> String {
    let start = row * array.cols;
    runmat_value::decode_utf16_code_units(&array.data()[start..start + array.cols])
        .expect("valid UTF-16 fixture")
}

fn object(value: Value) -> ObjectInstance {
    match value {
        Value::Object(object) => object,
        other => panic!("expected table object, got {other:?}"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn write_sample_parquet(path: &Path) {
    write_sample_parquet_with_rows(path, true);
}

#[cfg(not(target_arch = "wasm32"))]
fn write_sample_parquet_with_rows(path: &Path, include_rows: bool) {
    use arrow_array::{
        BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray as ArrowStringArray,
    };
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("Name", DataType::Utf8, true),
        Field::new("Score", DataType::Float64, true),
        Field::new("Passed", DataType::Boolean, true),
        Field::new("Group", DataType::Int32, true),
    ]));
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, schema.clone(), Some(properties))
        .expect("create parquet writer");
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ArrowStringArray::from(vec![
                Some("item-a"),
                Some("item-b"),
                Some("item-c"),
            ])),
            Arc::new(Float64Array::from(vec![Some(10.0), Some(12.5), None])),
            Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])),
            Arc::new(Int32Array::from(vec![Some(1), Some(1), Some(2)])),
        ],
    )
    .expect("create parquet batch");
    if include_rows {
        writer.write(&batch).expect("write parquet batch");
    }
    writer.close().expect("close parquet writer");
    fs::write(path, bytes).expect("write parquet sample");
}

#[cfg(not(target_arch = "wasm32"))]
fn write_integer_parquet(path: &Path, include_rows: bool) {
    use arrow_array::{
        Int16Array, Int32Array, Int64Array, Int8Array, RecordBatch, UInt16Array, UInt32Array,
        UInt64Array, UInt8Array,
    };
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("I8", DataType::Int8, true),
        Field::new("I16", DataType::Int16, true),
        Field::new("I32", DataType::Int32, true),
        Field::new("I64", DataType::Int64, true),
        Field::new("U8", DataType::UInt8, true),
        Field::new("U16", DataType::UInt16, true),
        Field::new("U32", DataType::UInt32, true),
        Field::new("U64", DataType::UInt64, true),
    ]));
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, schema.clone(), Some(properties))
        .expect("create parquet writer");
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int8Array::from(vec![Some(i8::MIN), None])),
            Arc::new(Int16Array::from(vec![Some(i16::MIN), None])),
            Arc::new(Int32Array::from(vec![Some(i32::MIN), None])),
            Arc::new(Int64Array::from(vec![Some(i64::MIN), None])),
            Arc::new(UInt8Array::from(vec![Some(u8::MAX), None])),
            Arc::new(UInt16Array::from(vec![Some(u16::MAX), None])),
            Arc::new(UInt32Array::from(vec![Some(u32::MAX), None])),
            Arc::new(UInt64Array::from(vec![Some(u64::MAX), None])),
        ],
    )
    .expect("create parquet batch");
    if include_rows {
        writer.write(&batch).expect("write parquet batch");
    }
    writer.close().expect("close parquet writer");
    fs::write(path, bytes).expect("write parquet integer sample");
}

#[test]
fn readtable_imports_headered_numeric_and_text_columns() {
    let path = unique_path("readtable_basic");
    fs::write(&path, "Name,Score\nitem-a,10\nitem-b,12\n").expect("write sample");
    let table = object(read_table(&path, Vec::new()));
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["Name".to_string(), "Score".to_string()]
    );
    match table_member_get(&table, &Value::from("Score")).unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2, 1]);
            assert_eq!(tensor.materialize_f64(), vec![10.0, 12.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    match table_member_get(&table, &Value::from("Name")).unwrap() {
        Value::StringArray(array) => {
            assert_eq!(
                array.data(),
                vec!["item-a".to_string(), "item-b".to_string()]
            );
        }
        other => panic!("expected string array, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn parquetread_imports_common_column_types() {
    let path = unique_path("parquetread_basic").with_extension("parquet");
    write_sample_parquet(&path);

    let table = object(
        run_parquetread(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect("parquetread"),
    );
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec![
            "Name".to_string(),
            "Score".to_string(),
            "Passed".to_string(),
            "Group".to_string()
        ]
    );
    match table_member_get(&table, &Value::from("Name")).unwrap() {
        Value::StringArray(array) => {
            assert_eq!(array.shape, vec![3, 1]);
            assert_eq!(
                array.data(),
                vec![
                    "item-a".to_string(),
                    "item-b".to_string(),
                    "item-c".to_string()
                ]
            );
        }
        other => panic!("expected string column, got {other:?}"),
    }
    match table_member_get(&table, &Value::from("Score")).unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[3, 1]);
            assert_eq!(tensor.materialize_f64()[0], 10.0);
            assert_eq!(tensor.materialize_f64()[1], 12.5);
            assert!(tensor.materialize_f64()[2].is_nan());
        }
        other => panic!("expected tensor column, got {other:?}"),
    }
    match table_member_get(&table, &Value::from("Passed")).unwrap() {
        Value::LogicalArray(array) => assert_eq!(array.data, vec![1, 0, 0]),
        other => panic!("expected logical column, got {other:?}"),
    }
    match table_member_get(&table, &Value::from("Group")).unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[3, 1]);
            assert_eq!(
                tensor.integer_storage(),
                Some(&IntegerStorage::I32(vec![1, 1, 2]))
            );
        }
        other => panic!("expected integer tensor column, got {other:?}"),
    }

    let _ = fs::remove_file(&path);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn parquetread_preserves_all_integer_physical_column_classes() {
    let directory = tempfile::tempdir().expect("owned integer Parquet fixture");
    let path = directory.path().join("integers.parquet");
    for (include_rows, empty_selection) in [(true, false), (true, true), (false, false)] {
        write_integer_parquet(&path, include_rows);
        let options = if empty_selection {
            vec![
                Value::from("RowGroups"),
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 1]).unwrap()),
            ]
        } else {
            Vec::new()
        };
        let table = object(
            run_parquetread(Value::from(path.to_string_lossy().to_string()), options)
                .expect("parquetread"),
        );
        let cases = [
            ("I8", IntegerStorage::I8(vec![i8::MIN, 0])),
            ("I16", IntegerStorage::I16(vec![i16::MIN, 0])),
            ("I32", IntegerStorage::I32(vec![i32::MIN, 0])),
            ("I64", IntegerStorage::I64(vec![i64::MIN, 0])),
            ("U8", IntegerStorage::U8(vec![u8::MAX, 0])),
            ("U16", IntegerStorage::U16(vec![u16::MAX, 0])),
            ("U32", IntegerStorage::U32(vec![u32::MAX, 0])),
            ("U64", IntegerStorage::U64(vec![u64::MAX, 0])),
        ];
        for (name, expected) in cases {
            let Value::Tensor(column) = table_member_get(&table, &Value::from(name)).unwrap()
            else {
                panic!("expected tensor column for {name}");
            };
            if include_rows && !empty_selection {
                assert_eq!(column.shape(), &[2, 1], "{name}");
                assert_eq!(column.integer_storage(), Some(&expected), "{name}");
            } else {
                assert_eq!(column.shape(), &[0, 1], "{name}");
                assert_eq!(column.numeric_dtype(), expected.numeric_dtype(), "{name}");
                assert_eq!(column.integer_storage().unwrap().len(), 0, "{name}");
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn parquetread_supports_selected_variables_and_row_groups() {
    let directory = tempfile::tempdir().expect("owned mixed Parquet fixture");
    let path = directory.path().join("mixed.parquet");
    for (include_rows, empty_selection) in [(true, false), (true, true), (false, true)] {
        write_sample_parquet_with_rows(&path, include_rows);
        let names = ["Group", "Passed", "Score", "Name"];
        let selected = StringArray::new(
            names.iter().map(|name| (*name).into()).collect(),
            vec![1, 4],
        )
        .expect("selected variables");
        let table = object(
            run_parquetread(
                Value::from(path.to_string_lossy().to_string()),
                vec![
                    Value::from("SelectedVariableNames"),
                    Value::StringArray(selected),
                    Value::from("RowGroups"),
                    if empty_selection {
                        Value::Tensor(Tensor::new(Vec::new(), vec![0, 1]).unwrap())
                    } else {
                        Value::Num(2.0)
                    },
                ],
            )
            .expect("parquetread selected"),
        );
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            names
                .iter()
                .map(|name| (*name).to_string())
                .collect::<Vec<_>>()
        );
        match table_member_get(&table, &Value::from("Group")).unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[usize::from(!empty_selection), 1]);
                assert_eq!(
                    tensor.integer_storage(),
                    Some(&IntegerStorage::I32(if empty_selection {
                        Vec::new()
                    } else {
                        vec![2]
                    }))
                );
            }
            other => panic!("expected tensor column, got {other:?}"),
        }
        let score = table_member_get(&table, &Value::from("Score")).unwrap();
        assert!(
            matches!(score, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64
            && tensor.shape() == [usize::from(!empty_selection), 1]
            && (empty_selection || tensor.materialize_f64()[0].is_nan()))
        );
        let name = table_member_get(&table, &Value::from("Name")).unwrap();
        assert!(
            matches!(name, Value::StringArray(array) if array.shape == [usize::from(!empty_selection), 1]
            && array.data() == if empty_selection { Vec::new() } else { vec!["item-c".to_string()] })
        );
        let passed = table_member_get(&table, &Value::from("Passed")).unwrap();
        assert!(
            matches!(passed, Value::LogicalArray(array) if array.shape == [usize::from(!empty_selection), 1]
            && array.data == if empty_selection { Vec::new() } else { vec![0] })
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn parquetinfo_reports_schema_and_row_groups() {
    let path = unique_path("parquetinfo_basic").with_extension("parquet");
    write_sample_parquet(&path);

    let info = match block_on(parquetinfo_builtin(Value::from(
        path.to_string_lossy().to_string(),
    )))
    .expect("parquetinfo")
    {
        Value::Struct(info) => info,
        other => panic!("expected struct, got {other:?}"),
    };
    assert_eq!(info.fields.get("NumRows"), Some(&Value::Num(3.0)));
    assert_eq!(info.fields.get("NumVariables"), Some(&Value::Num(4.0)));
    match info.fields.get("VariableNames").unwrap() {
        Value::StringArray(array) => assert_eq!(
            array.data(),
            vec![
                "Name".to_string(),
                "Score".to_string(),
                "Passed".to_string(),
                "Group".to_string()
            ]
        ),
        other => panic!("expected variable names, got {other:?}"),
    }
    match info.fields.get("RowGroups").unwrap() {
        Value::Struct(row_groups) => {
            assert!(row_groups.fields.contains_key("RowGroup1"));
            assert!(row_groups.fields.contains_key("RowGroup2"));
        }
        other => panic!("expected row group struct, got {other:?}"),
    }

    let _ = fs::remove_file(&path);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn canonical_parquet_example_bytes_have_authentic_schema_and_selected_values() {
    use runmat_builtins::{BuiltinExampleFixture, BuiltinFilesystemEntry, BuiltinFixtureContent};
    let BuiltinExampleFixture::Filesystem(fixture) = runmat_builtins::PARQUETREAD_CATALOG_ENTRY
        .documentation
        .examples[0]
        .fixture
    else {
        panic!("typed filesystem fixture");
    };
    let BuiltinFilesystemEntry::File {
        content: BuiltinFixtureContent::Bytes(bytes),
        ..
    } = fixture.entries[0]
    else {
        panic!("actual ArrowWriter bytes");
    };
    let directory = tempfile::tempdir().expect("owned fixture directory");
    let path = directory.path().join("measurements.parquet");
    fs::write(&path, bytes).expect("owned example fixture");
    let run = || {
        let text = path.to_string_lossy().to_string();
        let path_value = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape(
                text.chars().collect(),
                vec![1, 1, text.chars().count()],
            )
            .unwrap(),
        );
        let Value::Struct(info) = block_on(parquetinfo_builtin(path_value.clone())).unwrap() else {
            panic!("metadata struct");
        };
        for (name, count) in [
            ("NumRows", 4.0),
            ("NumRowGroups", 2.0),
            ("NumVariables", 2.0),
        ] {
            assert!(matches!(info.fields.get(name), Some(Value::Num(actual)) if *actual == count));
        }
        let table = object(
            run_parquetread(
                path_value,
                vec![
                    Value::from("SelectedVariableNames"),
                    Value::StringArray(
                        runmat_value::StringArray::new(
                            vec!["time".into(), "value".into()],
                            vec![1, 2],
                        )
                        .unwrap(),
                    ),
                    Value::from("RowGroups"),
                    Value::Num(2.0),
                ],
            )
            .unwrap(),
        );
        for (name, expected) in [("time", vec![3.0, 4.0]), ("value", vec![30.0, 40.0])] {
            let value = table_member_get(&table, &Value::from(name)).unwrap();
            assert!(matches!(value, Value::Tensor(tensor) if tensor.materialize_f64() == expected));
        }
    };
    run();
    fs::remove_file(path).expect("remove owned example fixture");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn parquetread_rejects_rowfilter_until_predicates_are_available() {
    let path = unique_path("parquetread_rowfilter").with_extension("parquet");
    write_sample_parquet(&path);

    let err = run_parquetread(
        Value::from(path.to_string_lossy().to_string()),
        vec![Value::from("RowFilter"), Value::Bool(true)],
    )
    .expect_err("expected rowfilter failure");
    assert!(err.to_string().contains("RowFilter is not supported"));

    let _ = fs::remove_file(&path);
}

#[test]
fn readtable_auto_does_not_consume_headerless_numeric_rows() {
    let path = unique_path("readtable_headerless_numeric");
    fs::write(&path, "1,2\n3,4\n").expect("write sample");
    let table = object(read_table(&path, Vec::new()));
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["Var1".to_string(), "Var2".to_string()]
    );
    match table_member_get(&table, &Value::from("Var1")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0, 3.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
    match table_member_get(&table, &Value::from("Var2")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0, 4.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn readtable_rejects_unknown_and_invalid_options() {
    let path = unique_path("readtable_invalid_options");
    fs::write(&path, "A\n1\n").expect("write sample");
    let err = read_table_err(
        &path,
        vec![Value::from("DefinitelyNotAnOption"), Value::from(1.0)],
    );
    assert!(err.message().contains("unsupported option"));
    let err = read_table_err(
        &path,
        vec![Value::from("VariableNamingRule"), Value::from("mangle")],
    );
    assert!(err.message().contains("unsupported VariableNamingRule"));
    let _ = fs::remove_file(&path);
}

#[test]
fn readtable_handles_quoted_delimiters_and_newlines() {
    let path = unique_path("readtable_quoted_newlines");
    fs::write(
        &path,
        "Name,Note\nAda,\"hello, world\"\nGrace,\"line one\nline two\"\n",
    )
    .expect("write sample");
    let table = object(read_table(&path, Vec::new()));
    match table_member_get(&table, &Value::from("Note")).unwrap() {
        Value::StringArray(array) => assert_eq!(
            array.data(),
            vec!["hello, world".to_string(), "line one\nline two".to_string()]
        ),
        other => panic!("expected string array, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn readtable_supports_explicit_names_and_missing_tokens() {
    let path = unique_path("readtable_options");
    fs::write(&path, "1,NA\n2,4\n").expect("write sample");
    let names =
        StringArray::new(vec!["A".to_string(), "B".to_string()], vec![1, 2]).expect("names");
    let table = object(read_table(
        &path,
        vec![
            Value::from("ReadVariableNames"),
            Value::Bool(false),
            Value::from("VariableNames"),
            Value::StringArray(names),
            Value::from("TreatAsMissing"),
            Value::from("NA"),
        ],
    ));
    match table_member_get(&table, &Value::from("B")).unwrap() {
        Value::Tensor(tensor) => {
            assert!(tensor.materialize_f64()[0].is_nan());
            assert_eq!(tensor.materialize_f64()[1], 4.0);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn readtable_preserves_variable_names_when_requested() {
    let path = unique_path("readtable_preserve_names");
    fs::write(&path, " daily revenue ,total orders\n100,10\n").expect("write sample");
    let table = object(read_table(
        &path,
        vec![Value::from("VariableNamingRule"), Value::from("preserve")],
    ));
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec![" daily revenue ".to_string(), "total orders".to_string()]
    );
    match table_member_get(&table, &Value::from(" daily revenue ")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![100.0]),
        other => panic!("expected numeric column, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

fn write_zip_file(zip: &mut zip::ZipWriter<std::fs::File>, name: &str, contents: &str) {
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file(name, options).expect("start xlsx part");
    zip.write_all(contents.as_bytes()).expect("write xlsx part");
}

fn write_minimal_xlsx(path: &Path) {
    let file = std::fs::File::create(path).expect("create xlsx");
    let mut zip = zip::ZipWriter::new(file);
    write_zip_file(
        &mut zip,
        "[Content_Types].xml",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"#,
    );
    write_zip_file(
        &mut zip,
        "_rels/.rels",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#,
    );
    write_zip_file(
        &mut zip,
        "xl/workbook.xml",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
<sheet name="Data" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"#,
    );
    write_zip_file(
        &mut zip,
        "xl/_rels/workbook.xml.rels",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"#,
    );
    write_zip_file(
        &mut zip,
        "xl/styles.xml",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellXfs>
</styleSheet>"#,
    );
    write_zip_file(
        &mut zip,
        "xl/worksheets/sheet1.xml",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
<row r="1">
  <c r="A1" t="inlineStr"><is><t>Date</t></is></c>
  <c r="B1" t="inlineStr"><is><t>Orders</t></is></c>
  <c r="C1" t="inlineStr"><is><t>Revenue</t></is></c>
</row>
<row r="2">
  <c r="A2" t="inlineStr"><is><t>2026-06-01</t></is></c>
  <c r="B2"><v>10</v></c>
  <c r="C2"><v>200</v></c>
</row>
<row r="3">
  <c r="A3" t="inlineStr"><is><t>2026-06-02</t></is></c>
  <c r="B3"><v>4</v></c>
  <c r="C3"><v>90</v></c>
</row>
  </sheetData>
</worksheet>"#,
    );
    zip.finish().expect("finish xlsx");
}

#[test]
fn readtable_imports_xlsx_sheet_and_range() {
    let path = unique_path("readtable_spreadsheet");
    let path = path.with_extension("xlsx");
    write_minimal_xlsx(&path);
    let table = object(read_table(
        &path,
        vec![
            Value::from("Sheet"),
            Value::from("Data"),
            Value::from("Range"),
            Value::from("A1:C3"),
        ],
    ));
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec![
            "Date".to_string(),
            "Orders".to_string(),
            "Revenue".to_string()
        ]
    );
    match table_member_get(&table, &Value::from("Revenue")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![200.0, 90.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn spreadsheet_import_options_registers_public_descriptor() {
    let entry = &runmat_builtins::SPREADSHEET_IMPORT_OPTIONS_CATALOG_ENTRY;
    let bindings = crate::builtin::runtime_builtin_bindings_by_name(entry.identity.name);
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].identity,
        entry.binding_identity(&entry.bindings[0])
    );
    assert!(runmat_builtins::builtin_function_by_name(entry.identity.name).is_none());
    let labels = SPREADSHEET_IMPORT_OPTIONS_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert!(labels.contains(&"opts = spreadsheetImportOptions()"));
    assert!(labels.contains(&"opts = spreadsheetImportOptions(nameValuePairs...)"));
    assert_eq!(SPREADSHEET_IMPORT_OPTIONS_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(SPREADSHEET_IMPORT_OPTIONS_EXTENSIONS.len(), 1);
}

#[test]
fn table_import_integer_control_gates_match_documented_forms() {
    let missing = Value::from("definitely-missing-table-import.csv");
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);

    let readtable_error = block_on(readtable_builtin(
        None,
        missing.clone(),
        vec![
            Value::from("NumHeaderLines"),
            Value::Int(runmat_value::IntValue::U8(1)),
        ],
    ))
    .expect_err("typed readtable NumHeaderLines must be gated before file access");
    assert_eq!(
        readtable_error.identifier(),
        READTABLE_TYPED_INTEGER_CONTROL_EXTENSION.error_identifier
    );

    let readcell_error = block_on(readcell_builtin(
        None,
        missing.clone(),
        vec![
            Value::from("Range"),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U16(vec![1, 1]), vec![1, 2]).unwrap(),
            ),
        ],
    ))
    .expect_err("typed readcell Range must be gated before file access");
    assert_eq!(
        readcell_error.identifier(),
        READCELL_TYPED_INTEGER_CONTROL_EXTENSION.error_identifier
    );

    let readtimetable_error = block_on(readtimetable_builtin(
        missing,
        vec![
            Value::from("Sheet"),
            Value::Int(runmat_value::IntValue::U8(1)),
        ],
    ))
    .expect_err("typed readtimetable Sheet must be gated before file access");
    assert_eq!(
        readtimetable_error.identifier(),
        READTIMETABLE_TYPED_INTEGER_CONTROL_EXTENSION.error_identifier
    );

    let spreadsheet_error = block_on(spreadsheet_import_options_builtin(
        None,
        vec![
            Value::from("DataRange"),
            Value::Int(runmat_value::IntValue::U8(2)),
        ],
    ))
    .expect_err("typed spreadsheet location must be gated before option construction");
    assert_eq!(
        spreadsheet_error.identifier(),
        SPREADSHEET_OPTIONS_TYPED_INTEGER_CONTROL_EXTENSION.error_identifier
    );
}

#[test]
fn documented_table_import_integer_controls_remain_enabled_in_matlab_mode() {
    let path = unique_path("table_import_documented_integer_controls");
    fs::write(&path, "ignored\nA\n9007199254740993\n").expect("write sample");
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);

    let table = object(
        block_on(readtable_builtin(
            None,
            Value::from(path.to_string_lossy().to_string()),
            vec![
                Value::from("VariableNamesLine"),
                Value::Int(runmat_value::IntValue::U8(2)),
                Value::from("VariableTypes"),
                Value::StringArray(StringArray::new(vec!["uint64".into()], vec![1, 1]).unwrap()),
            ],
        ))
        .expect("documented readtable integer line"),
    );
    let Value::Tensor(column) = table_member_get(&table, &Value::from("A")).unwrap() else {
        panic!("expected integer table variable");
    };
    assert_eq!(
        column.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993]))
    );

    let cells = block_on(readcell_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        vec![
            Value::from("NumHeaderLines"),
            Value::Int(runmat_value::IntValue::U8(1)),
        ],
    ))
    .expect("documented readcell integer header count");
    let Value::Cell(cells) = cells else {
        panic!("expected cell output");
    };
    assert_eq!(
        cells.get(0, 0).expect("first imported cell"),
        Value::from("A")
    );

    let options = block_on(spreadsheet_import_options_builtin(
        None,
        vec![
            Value::from("NumVariables"),
            Value::Int(runmat_value::IntValue::U8(2)),
        ],
    ))
    .expect("documented integer NumVariables");
    let options =
        crate::builtins::io::tabular::import_options::ImportOptions::decode(None, &options)
            .expect("decode constructed options")
            .expect("import options object");
    assert_eq!(
        options.base().fields.get("NumVariables"),
        Some(&Value::Num(2.0))
    );
    let _ = fs::remove_file(path);
}

#[test]
fn detect_import_options_registers_public_descriptor() {
    let entry = &runmat_builtins::DETECT_IMPORT_OPTIONS_CATALOG_ENTRY;
    let bindings = crate::builtin::runtime_builtin_bindings_by_name(entry.identity.name);
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].identity,
        entry.binding_identity(&entry.bindings[0])
    );
    assert!(runmat_builtins::builtin_function_by_name(entry.identity.name).is_none());
    let labels = DETECT_IMPORT_OPTIONS_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert!(labels.contains(&"opts = detectImportOptions(filename)"));
    assert!(labels.contains(&"opts = detectImportOptions(filename, nameValuePairs...)"));
    assert_eq!(DETECT_IMPORT_OPTIONS_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(DETECT_IMPORT_OPTIONS_EXTENSIONS.len(), 1);
}

#[test]
fn detect_import_options_integer_num_header_lines_is_gated_before_file_access() {
    let missing = Value::from("definitely-missing.csv");
    let args = vec![
        Value::from("NumHeaderLines"),
        Value::Int(runmat_value::IntValue::U8(1)),
    ];
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(detect_import_options_builtin(None, missing, args))
        .expect_err("typed NumHeaderLines must be gated before opening the file");
    assert_eq!(
        error.identifier(),
        DETECT_IMPORT_OPTIONS_INTEGER_NUM_HEADER_LINES_EXTENSION.error_identifier
    );
}

#[test]
fn detect_import_options_rejects_nested_resident_input_before_provider_access() {
    let allocation = unregistered_table_allocation(vec![1, 1]);
    let resident = Value::GpuTensor(allocation.handle.clone());
    let mut options = runmat_value::StructValue::new();
    crate::builtins::common::test_support::replace_struct_field_unbounded(
        &mut options,
        "Range",
        resident,
    );
    let error = block_on(detect_import_options_builtin(
        None,
        Value::from("definitely-missing.csv"),
        vec![Value::Struct(options)],
    ))
    .expect_err("nested resident option must reject before gather");
    assert!(error.message().contains("resident arguments"));
    assert!(!error.message().to_ascii_lowercase().contains("provider"));
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

fn unregistered_table_allocation(
    shape: Vec<usize>,
) -> runmat_accelerate_api::test_support::TestAllocation {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            shape,
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    allocation
}

#[test]
fn detect_import_options_infers_text_delimiter_names_and_types() {
    let path = unique_path("detect_import_options_text");
    fs::write(
        &path,
        "Name;Score;Flag;When\nAda;10;true;2026-06-01\nGrace;12;false;2026-06-02\n",
    )
    .expect("write sample");
    let options = detect_options(&path, Vec::new());
    assert_eq!(options.fields.get("FileType"), Some(&Value::from("text")));
    assert_eq!(options.fields.get("Delimiter"), Some(&Value::from(";")));
    assert_eq!(options.fields.get("NumHeaderLines"), Some(&Value::Num(1.0)));
    assert_eq!(
        options.fields.get("ReadVariableNames"),
        Some(&Value::Bool(false))
    );
    match options.fields.get("VariableNames").unwrap() {
        Value::StringArray(array) => assert_eq!(
            array.data(),
            vec![
                "Name".to_string(),
                "Score".to_string(),
                "Flag".to_string(),
                "When".to_string()
            ]
        ),
        other => panic!("expected string array, got {other:?}"),
    }
    match options.fields.get("VariableTypes").unwrap() {
        Value::StringArray(array) => assert_eq!(
            array.data(),
            vec![
                "string".to_string(),
                "double".to_string(),
                "logical".to_string(),
                "datetime".to_string()
            ]
        ),
        other => panic!("expected string array, got {other:?}"),
    }
    let table = object(read_table(&path, vec![Value::Struct(options)]));
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec![
            "Name".to_string(),
            "Score".to_string(),
            "Flag".to_string(),
            "When".to_string()
        ]
    );
    match table_member_get(&table, &Value::from("Score")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![10.0, 12.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn detect_import_options_struct_can_drive_readmatrix() {
    let path = unique_path("detect_import_options_readmatrix");
    fs::write(&path, "A,B\n1,2\n3,4\n").expect("write sample");
    let options = detect_options(&path, Vec::new());
    let matrix = block_on(
        crate::builtins::io::tabular::readmatrix::readmatrix_builtin(
            None,
            Value::from(path.to_string_lossy().to_string()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(options)]),
        ),
    )
    .expect("readmatrix");
    match matrix {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2, 2]);
            assert_eq!(tensor.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn detect_import_options_strips_bom_from_detected_names() {
    let path = unique_path("detect_import_options_bom");
    fs::write(&path, "\u{FEFF}A,B\n1,2\n3,4\n").expect("write sample");
    let options = detect_options(&path, Vec::new());
    match options.fields.get("VariableNames").unwrap() {
        Value::StringArray(array) => {
            assert_eq!(array.data(), vec!["A".to_string(), "B".to_string()])
        }
        other => panic!("expected string array, got {other:?}"),
    }
    let table = object(read_table(&path, vec![Value::Struct(options)]));
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["A".to_string(), "B".to_string()]
    );
    match table_member_get(&table, &Value::from("A")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0, 3.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn detect_import_options_preserves_partial_ranges_for_replay() {
    let path = unique_path("detect_import_options_partial_range");
    fs::write(&path, "ID,A,B,C\nr1,1,2,3\nr2,4,5,6\nr3,7,8,9\n").expect("write sample");

    let column_options = detect_options(&path, vec![Value::from("Range"), Value::from("C:D")]);
    assert_eq!(
        column_options.fields.get("Range"),
        Some(&Value::from("C2:D"))
    );
    let table = object(read_table(&path, vec![Value::Struct(column_options)]));
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["B".to_string(), "C".to_string()]
    );
    match table_member_get(&table, &Value::from("B")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0, 5.0, 8.0]),
        other => panic!("expected tensor, got {other:?}"),
    }

    fs::write(&path, "11,12\n21,22\n31,32\n41,42\n").expect("write numeric sample");
    let row_options = detect_options(&path, vec![Value::from("Range"), Value::from("2:3")]);
    assert_eq!(row_options.fields.get("Range"), Some(&Value::from("2:3")));
    let table = object(read_table(&path, vec![Value::Struct(row_options)]));
    match table_member_get(&table, &Value::from("Var2")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![22.0, 32.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn detect_import_options_read_row_names_replays_through_readtable() {
    let path = unique_path("detect_import_options_row_names");
    fs::write(&path, "Row,Name,Score\nr1,item-a,10\nr2,item-b,12\n").expect("write sample");
    let options = detect_options(&path, vec![Value::from("ReadRowNames"), Value::Bool(true)]);
    assert_eq!(options.fields.get("NumVariables"), Some(&Value::Num(2.0)));
    match options.fields.get("VariableNames").unwrap() {
        Value::StringArray(array) => assert_eq!(
            array.data(),
            vec!["Row".to_string(), "Name".to_string(), "Score".to_string()]
        ),
        other => panic!("expected string array, got {other:?}"),
    }
    let table = object(read_table(&path, vec![Value::Struct(options)]));
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["Name".to_string(), "Score".to_string()]
    );
    let props = table_public_properties(&table).unwrap();
    match props.fields.get(ROW_NAMES).unwrap() {
        Value::StringArray(array) => {
            assert_eq!(array.data(), vec!["r1".to_string(), "r2".to_string()])
        }
        other => panic!("expected row names, got {other:?}"),
    }
    match table_member_get(&table, &Value::from("Score")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![10.0, 12.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn detect_import_options_encoding_replays_through_readmatrix() {
    let path = unique_path("detect_import_options_encoding_readmatrix");
    fs::write(&path, b"Caf\xe9,Score\n1,2\n3,4\n").expect("write sample");
    let options = detect_options(
        &path,
        vec![Value::from("Encoding"), Value::from("windows-1252")],
    );
    let matrix = block_on(
        crate::builtins::io::tabular::readmatrix::readmatrix_builtin(
            None,
            Value::from(path.to_string_lossy().to_string()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(options)]),
        ),
    )
    .expect("readmatrix");
    match matrix {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2, 2]);
            assert_eq!(tensor.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn detect_import_options_replays_through_filesystem_provider() {
    let root = unique_path("detect_import_options_provider_root");
    {
        let _provider_lock = runmat_filesystem::provider_override_lock();
        let provider = PrefixSandboxProvider {
            prefix: "/provider",
            sandbox: SandboxFsProvider::new(root.clone()).expect("sandbox provider"),
            native: NativeFsProvider,
        };
        let _provider_guard = runmat_filesystem::replace_provider(std::sync::Arc::new(provider));
        block_on(runmat_filesystem::write_async(
            "/provider.csv",
            b"Name,Score\nAda,10\nGrace,12\n",
        ))
        .expect("write provider sample");

        let virtual_path = Path::new("/provider.csv");
        let options = detect_options(virtual_path, Vec::new());
        let table = object(read_table(
            virtual_path,
            vec![Value::Struct(options.clone())],
        ));
        assert_eq!(
            table_variable_names_from_object(&table).unwrap(),
            vec!["Name".to_string(), "Score".to_string()]
        );
        match table_member_get(&table, &Value::from("Score")).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![10.0, 12.0]),
            other => panic!("expected tensor, got {other:?}"),
        }

        block_on(runmat_filesystem::write_async(
            "/provider_numeric.csv",
            b"A,B\n1,2\n3,4\n",
        ))
        .expect("write provider numeric sample");
        let matrix_options = detect_options(Path::new("/provider_numeric.csv"), Vec::new());
        let matrix = block_on(
            crate::builtins::io::tabular::readmatrix::readmatrix_builtin(
                None,
                Value::from("/provider_numeric.csv"),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(
                    matrix_options,
                )]),
            ),
        )
        .expect("readmatrix");
        match matrix {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[2, 2]);
                assert_eq!(tensor.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn detect_import_options_honors_overrides_and_range() {
    let path = unique_path("detect_import_options_overrides");
    fs::write(&path, "ignore me\nRaw A|Raw B\n5|yes\n6|no\n").expect("write sample");
    let options = detect_options(
        &path,
        vec![
            Value::from("Delimiter"),
            Value::from("|"),
            Value::from("NumHeaderLines"),
            Value::Num(1.0),
            Value::from("VariableNamingRule"),
            Value::from("preserve"),
            Value::from("TextType"),
            Value::from("char"),
        ],
    );
    assert_eq!(options.fields.get("Delimiter"), Some(&Value::from("|")));
    assert_eq!(options.fields.get("NumHeaderLines"), Some(&Value::Num(2.0)));
    assert_eq!(
        options.fields.get("VariableNamingRule"),
        Some(&Value::from("preserve"))
    );
    match options.fields.get("VariableNames").unwrap() {
        Value::StringArray(array) => {
            assert_eq!(array.data(), vec!["Raw A".to_string(), "Raw B".to_string()])
        }
        other => panic!("expected string array, got {other:?}"),
    }
    match options.fields.get("VariableTypes").unwrap() {
        Value::StringArray(array) => {
            assert_eq!(
                array.data(),
                vec!["double".to_string(), "logical".to_string()]
            )
        }
        other => panic!("expected string array, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn spreadsheet_import_options_builds_editable_options_struct() {
    let options = spreadsheet_options(vec![
        Value::from("NumVariables"),
        Value::Num(2.0),
        Value::from("VariableTypes"),
        Value::StringArray(
            StringArray::new(vec!["double".into(), "string".into()], vec![1, 2]).unwrap(),
        ),
        Value::from("DataRange"),
        Value::from("A2:B5"),
    ]);
    assert_eq!(
        options.fields.get("FileType"),
        Some(&Value::from("spreadsheet"))
    );
    assert_eq!(options.fields.get("NumVariables"), Some(&Value::Num(2.0)));
    assert_eq!(options.fields.get("DataRange"), Some(&Value::from("A2:B5")));
    match options.fields.get("VariableNames").unwrap() {
        Value::StringArray(array) => {
            assert_eq!(array.data(), vec!["Var1".to_string(), "Var2".to_string()]);
            assert_eq!(array.shape, vec![1, 2]);
        }
        other => panic!("expected string array, got {other:?}"),
    }
    match options.fields.get("VariableTypes").unwrap() {
        Value::StringArray(array) => {
            assert_eq!(
                array.data(),
                vec!["double".to_string(), "string".to_string()]
            );
            assert_eq!(array.shape, vec![1, 2]);
        }
        other => panic!("expected string array, got {other:?}"),
    }
}

#[test]
fn readtable_consumes_spreadsheet_import_options_struct() {
    let path = unique_path("readtable_spreadsheet_options");
    let path = path.with_extension("xlsx");
    write_minimal_xlsx(&path);
    let mut options = spreadsheet_options(vec![Value::from("NumVariables"), Value::Num(1.0)]);
    crate::builtins::common::test_support::replace_struct_field_unbounded(
        &mut options,
        "Sheet",
        Value::from("Data"),
    );
    crate::builtins::common::test_support::replace_struct_field_unbounded(
        &mut options,
        "DataRange",
        Value::from("C2:C3"),
    );
    crate::builtins::common::test_support::replace_struct_field_unbounded(
        &mut options,
        "VariableNames",
        Value::StringArray(StringArray::new(vec!["Amount".into()], vec![1, 1]).unwrap()),
    );
    crate::builtins::common::test_support::replace_struct_field_unbounded(
        &mut options,
        "VariableTypes",
        Value::StringArray(StringArray::new(vec!["double".into()], vec![1, 1]).unwrap()),
    );
    let table = object(read_table(&path, vec![Value::Struct(options)]));
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["Amount".to_string()]
    );
    match table_member_get(&table, &Value::from("Amount")).unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2, 1]);
            assert_eq!(tensor.materialize_f64(), vec![200.0, 90.0]);
            assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn readtable_default_spreadsheet_options_still_infers_headers() {
    let path = unique_path("readtable_default_spreadsheet_options");
    let path = path.with_extension("xlsx");
    write_minimal_xlsx(&path);
    let options = spreadsheet_options(Vec::new());
    let table = object(read_table(&path, vec![Value::Struct(options)]));
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec![
            "Date".to_string(),
            "Orders".to_string(),
            "Revenue".to_string()
        ]
    );
    let _ = fs::remove_file(&path);
}

#[test]
fn readtable_variable_types_coerce_imported_columns() {
    let path = unique_path("readtable_variable_types");
    fs::write(
        &path,
        "Value,Flag,When,Elapsed,Kind\n1.5,true,2026-06-01,01:30:00,A\n2.25,false,2026-06-02,02:00:00,B\n",
    )
    .expect("write sample");
    let types = StringArray::new(
        vec![
            "single".to_string(),
            "logical".to_string(),
            "datetime".to_string(),
            "duration".to_string(),
            "categorical".to_string(),
        ],
        vec![1, 5],
    )
    .unwrap();
    let table = object(read_table(
        &path,
        vec![Value::from("VariableTypes"), Value::StringArray(types)],
    ));
    match table_member_get(&table, &Value::from("Value")).unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
            assert_eq!(tensor.materialize_f64(), vec![1.5, 2.25]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    match table_member_get(&table, &Value::from("Flag")).unwrap() {
        Value::LogicalArray(array) => assert_eq!(array.data, vec![1, 0]),
        other => panic!("expected logical array, got {other:?}"),
    }
    match table_member_get(&table, &Value::from("When")).unwrap() {
        Value::Object(object) => assert!(object.is_class(runmat_types::standard::DATETIME)),
        other => panic!("expected datetime object, got {other:?}"),
    }
    match table_member_get(&table, &Value::from("Elapsed")).unwrap() {
        Value::Object(object) => assert!(object.is_class(runmat_types::standard::DURATION)),
        other => panic!("expected duration object, got {other:?}"),
    }
    match table_member_get(&table, &Value::from("Kind")).unwrap() {
        Value::Object(object) => assert!(object.is_class(CATEGORICAL_CLASS)),
        other => panic!("expected categorical object, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn readtable_preserves_explicit_import_variable_names_when_requested() {
    let path = unique_path("readtable_preserve_explicit_names");
    fs::write(&path, "100,10\n125,12\n").expect("write sample");
    let names = StringArray::new(
        vec!["daily revenue".to_string(), "total orders".to_string()],
        vec![1, 2],
    )
    .unwrap();
    let table = object(read_table(
        &path,
        vec![
            Value::from("ReadVariableNames"),
            Value::Bool(false),
            Value::from("VariableNames"),
            Value::StringArray(names),
            Value::from("VariableNamingRule"),
            Value::from("preserve"),
        ],
    ));
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["daily revenue".to_string(), "total orders".to_string()]
    );
    let _ = fs::remove_file(&path);
}

#[test]
fn readtable_text_type_char_imports_text_columns_as_char_matrix() {
    let path = unique_path("readtable_text_type_char");
    fs::write(&path, "Name\nitem-a\nitem-b\n").expect("write sample");
    let table = object(read_table(
        &path,
        vec![Value::from("TextType"), Value::from("char")],
    ));
    match table_member_get(&table, &Value::from("Name")).unwrap() {
        Value::CharArray(array) => {
            assert_eq!(array.rows, 2);
            assert_eq!(array.cols, 6);
            assert_eq!(char_row(&array, 0), "item-a");
            assert_eq!(char_row(&array, 1), "item-b");
        }
        other => panic!("expected char array, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn readtable_variable_types_cellstr_imports_cell_column() {
    let path = unique_path("readtable_variable_types_cellstr");
    fs::write(&path, "Name\nitem-a\nitem-b\n").expect("write sample");
    let types = StringArray::new(vec!["cellstr".to_string()], vec![1, 1]).unwrap();
    let table = object(read_table(
        &path,
        vec![Value::from("VariableTypes"), Value::StringArray(types)],
    ));
    match table_member_get(&table, &Value::from("Name")).unwrap() {
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 2);
            assert_eq!(cell.cols, 1);
            assert_eq!(
                cell.get(0, 0).unwrap(),
                Value::CharArray(CharArray::new_row("item-a"))
            );
            assert_eq!(
                cell.get(1, 0).unwrap(),
                Value::CharArray(CharArray::new_row("item-b"))
            );
        }
        other => panic!("expected cell array, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn readtable_variable_types_preserve_all_exact_integer_classes() {
    let cases = [
        ("int8", "127\n-128\n", IntegerStorage::I8(vec![127, -128])),
        (
            "int16",
            "32767\n-32768\n",
            IntegerStorage::I16(vec![32767, -32768]),
        ),
        (
            "int32",
            "2147483647\n-2147483648\n",
            IntegerStorage::I32(vec![i32::MAX, i32::MIN]),
        ),
        (
            "int64",
            "9223372036854775807\n-9223372036854775808\n",
            IntegerStorage::I64(vec![i64::MAX, i64::MIN]),
        ),
        ("uint8", "255\n-1\n", IntegerStorage::U8(vec![255, 0])),
        (
            "uint16",
            "65535\n-1\n",
            IntegerStorage::U16(vec![u16::MAX, 0]),
        ),
        (
            "uint32",
            "4294967295\n-1\n",
            IntegerStorage::U32(vec![u32::MAX, 0]),
        ),
        (
            "uint64",
            "18446744073709551615\n-1\n",
            IntegerStorage::U64(vec![u64::MAX, 0]),
        ),
    ];
    for (class, values, expected) in cases {
        let path = unique_path(&format!("readtable_{class}"));
        fs::write(&path, format!("A\n{values}")).expect("write sample");
        let types = StringArray::new(vec![class.to_string()], vec![1, 1]).unwrap();
        let table = object(read_table(
            &path,
            vec![Value::from("VariableTypes"), Value::StringArray(types)],
        ));
        let Value::Tensor(column) = table_member_get(&table, &Value::from("A")).unwrap() else {
            panic!("expected integer tensor column");
        };
        assert_eq!(column.integer_storage(), Some(&expected), "{class}");
        let _ = fs::remove_file(&path);
    }
}

#[test]
fn table_properties_variable_names_rename_columns() {
    let a = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let b = Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap());
    let mut table = object(table_from_columns(vec!["A".into(), "B".into()], vec![a, b]).unwrap());
    let mut props = table_public_properties(&table).unwrap();
    crate::builtins::common::test_support::replace_struct_field_unbounded(
        &mut props,
        VARIABLE_NAMES,
        Value::StringArray(StringArray::new(vec!["X".into(), "Y".into()], vec![1, 2]).unwrap()),
    );
    table_member_set(None, &mut table, PROPERTIES_MEMBER, Value::Struct(props)).unwrap();
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["X".to_string(), "Y".to_string()]
    );

    let first = Value::Tensor(Tensor::new(vec![7.0], vec![1, 1]).unwrap());
    let second = Value::Tensor(Tensor::new(vec![9.0], vec![1, 1]).unwrap());
    let mut nonlexical =
        object(table_from_columns(vec!["Z".into(), "A".into()], vec![first, second]).unwrap());
    assert_eq!(
        table_variable_names_from_object(&nonlexical).unwrap(),
        vec!["Z".to_string(), "A".to_string()]
    );
    let mut properties = table_public_properties(&nonlexical).unwrap();
    crate::builtins::common::test_support::replace_struct_field_unbounded(
        &mut properties,
        VARIABLE_NAMES,
        Value::StringArray(
            StringArray::new(vec!["First".into(), "Second".into()], vec![1, 2]).unwrap(),
        ),
    );
    table_member_set(
        None,
        &mut nonlexical,
        PROPERTIES_MEMBER,
        Value::Struct(properties),
    )
    .unwrap();
    let Value::Tensor(renamed_first) =
        table_member_get(&nonlexical, &Value::from("First")).unwrap()
    else {
        panic!("first declared column was not retained across rename")
    };
    assert_eq!(renamed_first.materialize_f64(), vec![7.0]);
    assert_eq!(
        table_variable_names_from_object(&nonlexical).unwrap(),
        vec!["First".to_string(), "Second".to_string()]
    );
}

#[test]
fn table_paren_selects_rows_and_named_variables() {
    let a = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
    let b = Value::Tensor(Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap());
    let table = object(table_from_columns(vec!["A".into(), "B".into()], vec![a, b]).unwrap());
    let selector = CellArray::new(
        vec![
            Value::Tensor(Tensor::new(vec![3.0, 1.0], vec![1, 2]).unwrap()),
            Value::Cell(CellArray::new(vec![Value::from("B")], 1, 1).unwrap()),
        ],
        1,
        2,
    )
    .unwrap();
    let subset = object(table_paren_get(&table, &Value::Cell(selector)).unwrap());
    assert_eq!(
        table_variable_names_from_object(&subset).unwrap(),
        vec!["B".to_string()]
    );
    match table_member_get(&subset, &Value::from("B")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![6.0, 4.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn table_paren_selects_rows_by_row_names() {
    let a = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
    let table = object(
        table_from_columns_with_properties(
            vec!["A".into()],
            vec![a],
            Some(vec!["r1".into(), "r2".into(), "r3".into()]),
        )
        .unwrap(),
    );
    let selector = CellArray::new(
        vec![
            Value::Cell(CellArray::new(vec![Value::from("r3"), Value::from("r1")], 1, 2).unwrap()),
            Value::from(":"),
        ],
        1,
        2,
    )
    .unwrap();

    let subset = object(table_paren_get(&table, &Value::Cell(selector)).unwrap());
    match table_member_get(&subset, &Value::from("A")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0, 1.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
    let props = table_public_properties(&subset).unwrap();
    match props.fields.get(ROW_NAMES).unwrap() {
        Value::StringArray(array) => assert_eq!(array.data(), vec!["r3", "r1"]),
        other => panic!("expected selected row names, got {other:?}"),
    }
}

#[test]
fn sortrows_preserves_row_names() {
    let values = Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![2, 1]).unwrap());
    let table = table_from_columns_with_properties(
        vec!["X".into()],
        vec![values],
        Some(vec!["second".into(), "first".into()]),
    )
    .unwrap();
    let (sorted, _) = sortrows_table(table, &[Value::from("X")]).unwrap();
    let sorted = object(sorted);
    let props = table_public_properties(&sorted).unwrap();
    match props.fields.get(ROW_NAMES).unwrap() {
        Value::StringArray(array) => {
            assert_eq!(
                array.data(),
                vec!["first".to_string(), "second".to_string()]
            );
        }
        other => panic!("expected row names, got {other:?}"),
    }
}

#[test]
fn groupsummary_mean_counts_groups() {
    let group = Value::StringArray(
        StringArray::new(vec!["a".into(), "b".into(), "a".into()], vec![3, 1]).unwrap(),
    );
    let value = Value::Tensor(Tensor::new(vec![2.0, 5.0, 4.0], vec![3, 1]).unwrap());
    let table = table_from_columns(vec!["G".into(), "X".into()], vec![group, value]).unwrap();
    let summary = groupsummary_impl(
        table,
        Value::from("G"),
        Value::from("mean"),
        vec![Value::from("X")],
    )
    .unwrap();
    let summary = object(summary);
    assert_eq!(
        table_variable_names_from_object(&summary).unwrap(),
        vec![
            "G".to_string(),
            "GroupCount".to_string(),
            "mean_X".to_string()
        ]
    );
    match table_member_get(&summary, &Value::from("mean_X")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0, 5.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn groupsummary_orders_numeric_groups_numerically() {
    let group = Value::Tensor(Tensor::new(vec![10.0, 2.0, 10.0], vec![3, 1]).unwrap());
    let value = Value::Tensor(Tensor::new(vec![1.0, 5.0, 3.0], vec![3, 1]).unwrap());
    let table = table_from_columns(vec!["G".into(), "X".into()], vec![group, value]).unwrap();
    let summary =
        object(groupsummary_impl(table, Value::from("G"), Value::from("sum"), vec![]).unwrap());
    match table_member_get(&summary, &Value::from("G")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0, 10.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
    match table_member_get(&summary, &Value::from("sum_X")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![5.0, 4.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn groupsummary_preserves_exact_integer_groups_and_extrema() {
    let large = 9_007_199_254_740_992_u64;
    let group = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![large + 1, large, large + 1]),
            vec![3, 1],
        )
        .unwrap(),
    );
    let values = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, large + 1, large + 2]),
            vec![3, 1],
        )
        .unwrap(),
    );
    let table = table_from_columns(vec!["G".into(), "X".into()], vec![group, values]).unwrap();
    let summary = object(
        groupsummary_impl(
            table,
            Value::from("G"),
            Value::from("min"),
            vec![Value::from("X")],
        )
        .unwrap(),
    );
    let Value::Tensor(groups) = table_member_get(&summary, &Value::from("G")).unwrap() else {
        panic!("expected integer grouping column");
    };
    assert_eq!(
        groups.integer_storage(),
        Some(&IntegerStorage::U64(vec![large, large + 1]))
    );
    let Value::Tensor(minima) = table_member_get(&summary, &Value::from("min_X")).unwrap() else {
        panic!("expected integer minima");
    };
    assert_eq!(
        minima.integer_storage(),
        Some(&IntegerStorage::U64(vec![large + 1, large + 2]))
    );
}

#[test]
fn groupsummary_rejects_lossy_integer_floating_summaries() {
    let large = 9_007_199_254_740_993_u64;
    let group = Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap());
    let values = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![large, large + 2]), vec![2, 1]).unwrap(),
    );
    let table = table_from_columns(vec!["G".into(), "X".into()], vec![group, values]).unwrap();
    let error = groupsummary_impl(
        table,
        Value::from("G"),
        Value::from("mean"),
        vec![Value::from("X")],
    )
    .expect_err("lossy integer mean must reject");
    assert!(error.message.contains("exactly representable as double"));
}

#[test]
fn groupsummary_missing_controls_default_true_and_require_zero_or_one() {
    let group = Value::Tensor(Tensor::new(vec![f64::NAN, 1.0, f64::NAN], vec![3, 1]).unwrap());
    let values = Value::Tensor(Tensor::new(vec![2.0, 4.0, 6.0], vec![3, 1]).unwrap());
    let table = table_from_columns(
        vec!["G".into(), "X".into()],
        vec![group.clone(), values.clone()],
    )
    .unwrap();
    let summary = object(
        groupsummary_impl(
            table,
            Value::from("G"),
            Value::from("sum"),
            vec![Value::from("X")],
        )
        .unwrap(),
    );
    let Value::Tensor(counts) = table_member_get(&summary, &Value::from("GroupCount")).unwrap()
    else {
        panic!("expected counts");
    };
    assert_eq!(counts.materialize_f64(), vec![1.0, 2.0]);

    let table = table_from_columns(vec!["G".into(), "X".into()], vec![group, values]).unwrap();
    let error = groupsummary_impl(
        table,
        Value::from("G"),
        Value::from("sum"),
        vec![
            Value::from("X"),
            Value::from("IncludeMissingGroups"),
            Value::Int(runmat_value::IntValue::U8(2)),
        ],
    )
    .expect_err("nonbinary control must reject");
    assert!(error.message.contains("0 or 1"));
}

#[test]
fn groupsummary_canonicalizes_nan_and_typed_missing_but_preserves_empty_strings() {
    let nan_a = f64::from_bits(0x7ff8_0000_0000_0001);
    let nan_b = f64::from_bits(0x7ff8_0000_0000_0002);
    let numeric_group = Value::Tensor(Tensor::new(vec![nan_a, 1.0, nan_b], vec![3, 1]).unwrap());
    let values = Value::Tensor(Tensor::new(vec![2.0, 4.0, 6.0], vec![3, 1]).unwrap());
    let table = table_from_columns(
        vec!["G".into(), "X".into()],
        vec![numeric_group, values.clone()],
    )
    .unwrap();
    let summary = object(
        groupsummary_impl(
            table,
            Value::from("G"),
            Value::from("sum"),
            vec![Value::from("X")],
        )
        .unwrap(),
    );
    let Value::Tensor(counts) = table_member_get(&summary, &Value::from("GroupCount")).unwrap()
    else {
        panic!("expected counts");
    };
    assert_eq!(counts.materialize_f64(), vec![1.0, 2.0]);

    let text_group = Value::StringArray(
        StringArray::from_elements_unbounded(
            vec![
                "".into(),
                "a".into(),
                runmat_value::StringElement::Missing,
                "".into(),
                runmat_value::StringElement::Missing,
            ],
            vec![5, 1],
        )
        .unwrap(),
    );
    let values = Value::Tensor(Tensor::new(vec![2.0, 4.0, 6.0, 8.0, 10.0], vec![5, 1]).unwrap());
    let table = table_from_columns(vec!["G".into(), "X".into()], vec![text_group, values]).unwrap();
    let summary = object(
        groupsummary_impl(
            table,
            Value::from("G"),
            Value::from("sum"),
            vec![Value::from("X")],
        )
        .unwrap(),
    );
    let Value::Tensor(counts) = table_member_get(&summary, &Value::from("GroupCount")).unwrap()
    else {
        panic!("expected counts");
    };
    assert_eq!(counts.materialize_f64(), vec![2.0, 1.0, 2.0]);
    let Value::StringArray(groups) = table_member_get(&summary, &Value::from("G")).unwrap() else {
        panic!("expected string groups");
    };
    assert_eq!(
        groups.data(),
        &["".into(), "a".into(), runmat_value::StringElement::Missing]
    );
}

#[test]
fn groupsummary_places_undefined_categorical_and_nan_duration_groups_last() {
    let categorical = categorical_from_args(crate::call::FunctionArgumentView::from_slice(&[
        Value::StringArray(
            StringArray::new(vec!["b".into(), "missing".into(), "a".into()], vec![3, 1]).unwrap(),
        ),
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
    ]))
    .unwrap();
    let values = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
    let table = table_from_columns(
        vec!["G".into(), "X".into()],
        vec![categorical, values.clone()],
    )
    .unwrap();
    let summary = object(
        groupsummary_impl(
            table,
            Value::from("G"),
            Value::from("sum"),
            vec![Value::from("X")],
        )
        .unwrap(),
    );
    let categorical_groups = table_member_get(&summary, &Value::from("G")).unwrap();
    assert_eq!(cell_key_string(&categorical_groups, 0).unwrap(), "a");
    assert_eq!(cell_key_string(&categorical_groups, 1).unwrap(), "b");
    assert_eq!(
        cell_key_string(&categorical_groups, 2).unwrap(),
        "<undefined>"
    );

    let duration = crate::builtins::duration::duration_object_from_days_tensor(
        None,
        Tensor::new(vec![2.0, f64::NAN, 1.0], vec![3, 1]).unwrap(),
        "dd:hh:mm:ss",
    )
    .unwrap();
    let table = table_from_columns(vec!["G".into(), "X".into()], vec![duration, values]).unwrap();
    let summary = object(
        groupsummary_impl(
            table,
            Value::from("G"),
            Value::from("sum"),
            vec![Value::from("X")],
        )
        .unwrap(),
    );
    let duration_groups = table_member_get(&summary, &Value::from("G")).unwrap();
    assert_eq!(cell_key_string(&duration_groups, 0).unwrap(), "1");
    assert_eq!(cell_key_string(&duration_groups, 1).unwrap(), "2");
    assert_eq!(cell_key_string(&duration_groups, 2).unwrap(), "NaN");
}

#[test]
fn grpstats_matrix_returns_multiple_statistics_and_group_names() {
    let x = Value::Tensor(
        Tensor::new(vec![1.0, 3.0, 2.0, 4.0, 10.0, 30.0, 20.0, 40.0], vec![4, 2]).unwrap(),
    );
    let group = Value::Tensor(Tensor::new(vec![2.0, 1.0, 2.0, 1.0], vec![4, 1]).unwrap());
    let whichstats = Value::StringArray(
        StringArray::new(
            vec!["mean".into(), "std".into(), "gname".into()],
            vec![1, 3],
        )
        .unwrap(),
    );
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let result = grpstats_impl(x, group, vec![whichstats]).unwrap();
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 3);
    match &outputs[0] {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2, 2]);
            assert_eq!(tensor.materialize_f64(), vec![3.5, 1.5, 35.0, 15.0]);
        }
        other => panic!("expected mean tensor, got {other:?}"),
    }
    match &outputs[1] {
        Value::Tensor(tensor) => {
            let root_half = 0.5_f64.sqrt();
            assert_eq!(tensor.shape(), &[2, 2]);
            let expected = [root_half, root_half, 10.0 * root_half, 10.0 * root_half];
            for (value, expected) in tensor.materialize_f64().iter().zip(expected) {
                assert!((*value - expected).abs() < 1.0e-12);
            }
        }
        other => panic!("expected std tensor, got {other:?}"),
    }
    match &outputs[2] {
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 2);
            assert_eq!(cell.cols, 1);
            assert_eq!(cell.data(), vec![Value::from("1"), Value::from("2")]);
        }
        other => panic!("expected group-name cell, got {other:?}"),
    }
}

#[test]
fn grpstats_matrix_handles_empty_group_missing_groups_and_output_count_contract() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![4, 1]).unwrap());
    let empty_group = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
    let result = grpstats_impl(x.clone(), empty_group, Vec::new()).unwrap();
    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[1, 1]);
            assert_eq!(tensor.materialize_f64(), vec![2.5]);
        }
        other => panic!("expected all-group mean tensor, got {other:?}"),
    }

    let group = Value::Tensor(Tensor::new(vec![f64::NAN, 2.0, 1.0, 2.0], vec![4, 1]).unwrap());
    let names = grpstats_impl(x.clone(), group.clone(), vec![Value::from("gname")]).unwrap();
    match names {
        Value::Cell(cell) => assert_eq!(cell.data(), vec![Value::from("1"), Value::from("2")]),
        other => panic!("expected group names, got {other:?}"),
    }
    let means = grpstats_impl(x.clone(), group, vec![Value::from("mean")]).unwrap();
    match means {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0, 3.5]),
        other => panic!("expected means, got {other:?}"),
    }

    let _guard = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let err = grpstats_impl(
        x,
        Value::Tensor(Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]).unwrap()),
        vec![Value::StringArray(
            StringArray::new(vec!["mean".into(), "std".into()], vec![1, 2]).unwrap(),
        )],
    )
    .expect_err("expected output-count mismatch");
    assert!(err.message.contains("number of outputs"));
}

#[test]
fn grpstats_matrix_preserves_text_group_first_seen_order_and_builtin_handles() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![4, 1]).unwrap());
    let group = Value::StringArray(
        StringArray::new(
            vec!["b".into(), "a".into(), "".into(), "b".into()],
            vec![4, 1],
        )
        .unwrap(),
    );
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = grpstats_impl(
        x,
        group,
        vec![Value::Cell(
            CellArray::new(
                vec![
                    Value::FunctionHandle("mean".into()),
                    Value::FunctionHandle("gname".into()),
                ],
                1,
                2,
            )
            .unwrap(),
        )],
    )
    .unwrap();
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    match &outputs[0] {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.5, 3.0]),
        other => panic!("expected mean tensor, got {other:?}"),
    }
    match &outputs[1] {
        Value::Cell(cell) => assert_eq!(cell.data(), vec![Value::from("b"), Value::from("a")]),
        other => panic!("expected group names, got {other:?}"),
    }
}

#[test]
fn grpstats_table_supports_datavars_varnames_and_multiple_stats() {
    let group = Value::StringArray(
        StringArray::new(
            vec!["b".into(), "a".into(), "b".into(), "a".into()],
            vec![4, 1],
        )
        .unwrap(),
    );
    let x = Value::Tensor(Tensor::new(vec![2.0, 10.0, 4.0, 14.0], vec![4, 1]).unwrap());
    let y = Value::Tensor(Tensor::new(vec![1.0, 3.0, 5.0, 7.0], vec![4, 1]).unwrap());
    let table =
        table_from_columns(vec!["G".into(), "X".into(), "Y".into()], vec![group, x, y]).unwrap();
    let stats = Value::StringArray(
        StringArray::new(vec!["mean".into(), "max".into()], vec![1, 2]).unwrap(),
    );
    let names = Value::StringArray(
        StringArray::new(
            vec![
                "Group".into(),
                "N".into(),
                "AverageX".into(),
                "PeakX".into(),
                "AverageY".into(),
                "PeakY".into(),
            ],
            vec![1, 6],
        )
        .unwrap(),
    );
    let summary = grpstats_impl(
        table,
        Value::from("G"),
        vec![
            stats,
            Value::from("DataVars"),
            Value::StringArray(StringArray::new(vec!["X".into(), "Y".into()], vec![1, 2]).unwrap()),
            Value::from("VarNames"),
            names,
        ],
    )
    .unwrap();
    let summary = object(summary);
    assert_eq!(
        table_variable_names_from_object(&summary).unwrap(),
        vec![
            "Group".to_string(),
            "N".to_string(),
            "AverageX".to_string(),
            "PeakX".to_string(),
            "AverageY".to_string(),
            "PeakY".to_string()
        ]
    );
    match table_member_get(&summary, &Value::from("Group")).unwrap() {
        Value::StringArray(array) => {
            assert_eq!(array.data(), vec!["b".to_string(), "a".to_string()])
        }
        other => panic!("expected group strings, got {other:?}"),
    }
    match table_member_get(&summary, &Value::from("N")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0, 2.0]),
        other => panic!("expected count tensor, got {other:?}"),
    }
    match table_member_get(&summary, &Value::from("AverageX")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0, 12.0]),
        other => panic!("expected average tensor, got {other:?}"),
    }
    match table_member_get(&summary, &Value::from("PeakX")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![4.0, 14.0]),
        other => panic!("expected max tensor, got {other:?}"),
    }
    match table_member_get(&summary, &Value::from("AverageY")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0, 5.0]),
        other => panic!("expected Y average tensor, got {other:?}"),
    }
    match table_member_get(&summary, &Value::from("PeakY")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![5.0, 7.0]),
        other => panic!("expected Y max tensor, got {other:?}"),
    }
}

#[test]
fn grpstats_table_supports_empty_group_and_interval_stats() {
    let x = Value::Tensor(Tensor::new(vec![2.0, 4.0, 6.0], vec![3, 1]).unwrap());
    let table = table_from_columns(vec!["X".into()], vec![x]).unwrap();
    let summary = object(
        grpstats_impl(
            table,
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            vec![Value::from("meanci")],
        )
        .unwrap(),
    );
    assert_eq!(
        table_variable_names_from_object(&summary).unwrap(),
        vec!["GroupCount".to_string(), "meanci_X".to_string()]
    );
    match table_member_get(&summary, &Value::from("GroupCount")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0]),
        other => panic!("expected count tensor, got {other:?}"),
    }
    match table_member_get(&summary, &Value::from("meanci_X")).unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[1, 2]);
            assert!(tensor.materialize_f64()[0] < 4.0);
            assert!(tensor.materialize_f64()[1] > 4.0);
        }
        other => panic!("expected interval tensor, got {other:?}"),
    }
}

#[test]
fn grpstats_table_preserves_wide_integer_extrema_and_checks_floating_boundary() {
    let base = (1_u64 << 53) + 1;
    let group = Value::Tensor(Tensor::new(vec![1.0, 1.0, 2.0], vec![3, 1]).unwrap());
    let values = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![base + 1, base, u64::MAX]),
            vec![3, 1],
        )
        .unwrap(),
    );
    let table = table_from_columns(
        vec!["G".into(), "X".into()],
        vec![group.clone(), values.clone()],
    )
    .unwrap();
    let stats =
        Value::StringArray(StringArray::new(vec!["min".into(), "max".into()], vec![1, 2]).unwrap());
    let summary = object(grpstats_impl(table, Value::from("G"), vec![stats]).unwrap());
    let Value::Tensor(minima) = table_member_get(&summary, &Value::from("min_X")).unwrap() else {
        panic!("expected exact minima");
    };
    assert_eq!(
        minima.integer_storage(),
        Some(&IntegerStorage::U64(vec![base, u64::MAX]))
    );
    let Value::Tensor(maxima) = table_member_get(&summary, &Value::from("max_X")).unwrap() else {
        panic!("expected exact maxima");
    };
    assert_eq!(
        maxima.integer_storage(),
        Some(&IntegerStorage::U64(vec![base + 1, u64::MAX]))
    );

    let table = table_from_columns(vec!["G".into(), "X".into()], vec![group, values]).unwrap();
    let error = grpstats_impl(table, Value::from("G"), vec![Value::from("mean")])
        .expect_err("lossy integer table mean must reject");
    assert!(error.message.contains("exactly representable as double"));
}

#[test]
fn grpstats_table_ignores_lossy_integer_values_in_rows_excluded_by_missing_groups() {
    let group = Value::Tensor(Tensor::new(vec![1.0, f64::NAN], vec![2, 1]).unwrap());
    let values = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![4, (1_u64 << 53) + 1]), vec![2, 1]).unwrap(),
    );
    let table = table_from_columns(vec!["G".into(), "X".into()], vec![group, values]).unwrap();
    let summary =
        object(grpstats_impl(table, Value::from("G"), vec![Value::from("mean")]).unwrap());
    let Value::Tensor(means) = table_member_get(&summary, &Value::from("mean_X")).unwrap() else {
        panic!("expected mean tensor");
    };
    assert_eq!(means.materialize_f64(), vec![4.0]);
}

#[test]
fn table_conversion_builtins_round_trip_arrays_cells_and_structs() {
    let matrix = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let table = block_on(array2table_builtin(
        matrix,
        vec![
            Value::from("VariableNames"),
            Value::Cell(CellArray::new(vec![Value::from("A"), Value::from("B")], 1, 2).unwrap()),
        ],
    ))
    .unwrap();
    assert!(matches!(
        block_on(istable_builtin(table.clone())).unwrap(),
        Value::Bool(true)
    ));
    let array = block_on(table2array_builtin(table.clone())).unwrap();
    match array {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
    let cells = block_on(table2cell_builtin(table.clone())).unwrap();
    match cells {
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 2);
            assert_eq!(cell.cols, 2);
            assert_eq!(cell.get(0, 0).unwrap(), Value::Num(1.0));
            assert_eq!(cell.get(0, 1).unwrap(), Value::Num(3.0));
            assert_eq!(cell.get(1, 0).unwrap(), Value::Num(2.0));
            assert_eq!(cell.get(1, 1).unwrap(), Value::Num(4.0));
        }
        other => panic!("expected cell, got {other:?}"),
    }
    let st = block_on(table2struct_builtin(table.clone(), Vec::new())).unwrap();
    let round_trip = block_on(struct2table_builtin(st, Vec::new())).unwrap();
    assert_eq!(table_width(&object(round_trip)).unwrap(), 2);
}

#[test]
fn array2table_preserves_every_integer_storage_class_exactly() {
    let storages = [
        IntegerStorage::I8(vec![i8::MIN, -1, 0, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, -1, 0, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, -1, 0, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, -1, 0, i64::MAX]),
        IntegerStorage::U8(vec![0, 1, u8::MAX - 1, u8::MAX]),
        IntegerStorage::U16(vec![0, 1, u16::MAX - 1, u16::MAX]),
        IntegerStorage::U32(vec![0, 1, u32::MAX - 1, u32::MAX]),
        IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX - 1, u64::MAX]),
    ];
    for storage in storages {
        let table = block_on(array2table_builtin(
            Value::Tensor(Tensor::new_integer(storage.clone(), vec![2, 2]).unwrap()),
            vec![
                Value::from("VariableNames"),
                Value::StringArray(
                    StringArray::new(vec!["A".into(), "B".into()], vec![1, 2]).unwrap(),
                ),
            ],
        ))
        .unwrap();
        let object = object(table.clone());
        let variables = table_variables(&object).unwrap();
        for name in ["A", "B"] {
            let Value::Tensor(column) = variables.fields.get(name).unwrap() else {
                panic!("expected typed integer column");
            };
            assert_eq!(column.shape(), &[2, 1]);
            assert_eq!(
                column.integer_storage().unwrap().numeric_dtype(),
                storage.numeric_dtype(),
                "{name}"
            );
        }
        let Value::Tensor(round_trip) = block_on(table2array_builtin(table)).unwrap() else {
            panic!("expected typed integer array");
        };
        assert_eq!(round_trip.shape(), &[2, 2]);
        assert_eq!(round_trip.integer_storage(), Some(&storage));
    }
}

#[test]
fn array2table_supports_current_names_and_properties_rules() {
    let table = block_on(array2table_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
        vec![
            Value::from("VariableNames"),
            Value::StringArray(
                StringArray::new(vec![" 1 café ".into(), "A".into()], vec![1, 2]).unwrap(),
            ),
            Value::from("RowNames"),
            Value::StringArray(
                StringArray::new(vec![" first ".into(), "second".into()], vec![2, 1]).unwrap(),
            ),
            Value::from("DimensionNames"),
            Value::StringArray(
                StringArray::new(vec!["Samples".into(), "Signals".into()], vec![1, 2]).unwrap(),
            ),
        ],
    ))
    .unwrap();
    let table_object_value = object(table);
    assert_eq!(
        table_variable_names_from_object(&table_object_value).unwrap(),
        vec![" 1 café ".to_string(), "A".to_string()]
    );
    let properties = table_public_properties(&table_object_value).unwrap();
    assert_eq!(
        properties.fields.get(ROW_NAMES),
        Some(&Value::StringArray(
            StringArray::new(vec!["first".into(), "second".into()], vec![2, 1]).unwrap()
        ))
    );
    assert_eq!(
        properties.fields.get(DIMENSION_NAMES),
        Some(&Value::StringArray(
            StringArray::new(vec!["Samples".into(), "Signals".into()], vec![1, 2]).unwrap()
        ))
    );
    let case_distinct = block_on(array2table_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        vec![
            Value::from("VariableNames"),
            Value::StringArray(StringArray::new(vec!["A".into(), "a".into()], vec![1, 2]).unwrap()),
        ],
    ))
    .unwrap();
    assert_eq!(
        table_variable_names_from_object(&object(case_distinct)).unwrap(),
        vec!["A".to_string(), "a".to_string()]
    );
    let whitespace = object(
        block_on(array2table_builtin(
            Value::Tensor(Tensor::new(vec![7.0], vec![1, 1]).unwrap()),
            vec![
                Value::from("VariableNames"),
                Value::from(" "),
                Value::from("DimensionNames"),
                Value::StringArray(
                    StringArray::new(vec!["  ".into(), "Variables".into()], vec![1, 2]).unwrap(),
                ),
            ],
        ))
        .unwrap(),
    );
    assert_eq!(
        table_variable_names_from_object(&whitespace).unwrap(),
        vec![" ".to_string()]
    );
    assert_eq!(
        table_public_properties(&whitespace)
            .unwrap()
            .fields
            .get(DIMENSION_NAMES),
        Some(&Value::StringArray(
            StringArray::new(vec!["  ".into(), "Variables".into()], vec![1, 2]).unwrap()
        ))
    );
    match table_member_get(&whitespace, &Value::from(" ")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![7.0]),
        other => panic!("expected numeric column, got {other:?}"),
    }

    for options in [
        vec![
            Value::from("VariableNames"),
            Value::StringArray(StringArray::new(vec!["A".into(), "A".into()], vec![1, 2]).unwrap()),
        ],
        vec![
            Value::from("VariableNames"),
            Value::StringArray(
                StringArray::new(vec!["Rows".into(), "B".into()], vec![1, 2]).unwrap(),
            ),
        ],
        vec![
            Value::from("RowNames"),
            Value::StringArray(
                StringArray::new(vec!["same".into(), " same ".into()], vec![2, 1]).unwrap(),
            ),
        ],
        vec![
            Value::from("DimensionNames"),
            Value::StringArray(StringArray::new(vec!["A".into(), "A".into()], vec![1, 2]).unwrap()),
        ],
    ] {
        let error = block_on(array2table_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
            options,
        ))
        .expect_err("invalid names must reject");
        assert!(error.message.contains("array2table"));
    }
}

#[test]
fn array2table_resident_integer_input_is_mode_gated_and_gathers_exactly() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let storages = [
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];
        for storage in storages {
            let tensor = Tensor::new_integer(storage.clone(), vec![2, 1]).unwrap();
            let handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block_on(array2table_builtin(
                    Value::GpuTensor(handle.clone()),
                    Vec::new(),
                ))
                .expect_err("MATLAB mode must reject interactive GPU input");
                assert_eq!(
                    error.identifier(),
                    ARRAY2TABLE_GPU_INPUT_EXTENSION.error_identifier
                );
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                let table =
                    block_on(array2table_builtin(Value::GpuTensor(handle), Vec::new())).unwrap();
                let Value::Tensor(round_trip) = block_on(table2array_builtin(table)).unwrap()
                else {
                    panic!("expected typed integer array");
                };
                assert_eq!(round_trip.integer_storage(), Some(&storage));
            }
        }
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn array2table_wgpu_gathers_every_resident_integer_class_exactly() {
    let _guard = crate::builtins::common::test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = runmat_accelerate_api::provider().expect("WGPU provider");
    for storage in [
        IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![0, u8::MAX]),
        IntegerStorage::U16(vec![0, u16::MAX]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
    ] {
        let tensor = Tensor::new_integer(storage.clone(), vec![2, 1]).unwrap();
        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        let table = block_on(array2table_builtin(Value::GpuTensor(handle), Vec::new())).unwrap();
        let Value::Tensor(round_trip) = block_on(table2array_builtin(table)).unwrap() else {
            panic!("expected typed integer array");
        };
        assert_eq!(round_trip.integer_storage(), Some(&storage));
    }
}

#[test]
fn array2table_metadata_classifies_integer_and_gpu_forms() {
    assert_eq!(ARRAY2TABLE_INTEGER_CAPABILITIES.len(), 1);
    assert_eq!(
        ARRAY2TABLE_INTEGER_CAPABILITIES[0].output_class,
        runmat_builtins::BuiltinIntegerOutputClassRule::PreserveInput
    );
    assert_eq!(
        ARRAY2TABLE_INTEGER_CAPABILITIES[0].backend,
        runmat_builtins::BuiltinIntegerBackendRule::GatherFallback
    );
    assert_eq!(ARRAY2TABLE_EXTENSIONS, [ARRAY2TABLE_GPU_INPUT_EXTENSION]);
}

fn array2timetable_duration_row_times(days: Vec<f64>) -> Value {
    crate::builtins::duration::duration_object_from_days_tensor(
        None,
        Tensor::new(days.clone(), vec![days.len(), 1]).unwrap(),
        "hh:mm:ss".to_string(),
    )
    .unwrap()
}

fn assert_array2timetable_duration_row_times(value: &Value, expected_days: &[f64]) {
    let Value::Object(object) = value else {
        panic!("expected timetable object");
    };
    let row_times = timetable_row_times(object)
        .unwrap()
        .expect("array2timetable row times");
    let tensor =
        crate::builtins::duration::duration_tensor_from_duration_value(&row_times).unwrap();
    let actual = tensor.materialize_f64();
    assert_eq!(actual.len(), expected_days.len());
    for (actual, expected) in actual.iter().zip(expected_days) {
        assert!(
            (actual - expected).abs() <= f64::EPSILON,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn array2timetable_preserves_every_integer_storage_class_exactly() {
    let row_times = array2timetable_duration_row_times(vec![0.0, 1.0]);
    for storage in [
        IntegerStorage::I8(vec![i8::MIN, -1, 0, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, -1, 0, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, -1, 0, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, -1, 0, i64::MAX]),
        IntegerStorage::U8(vec![0, 1, u8::MAX - 1, u8::MAX]),
        IntegerStorage::U16(vec![0, 1, u16::MAX - 1, u16::MAX]),
        IntegerStorage::U32(vec![0, 1, u32::MAX - 1, u32::MAX]),
        IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX - 1, u64::MAX]),
    ] {
        let timetable = block_on(array2timetable_builtin(
            Value::Tensor(Tensor::new_integer(storage.clone(), vec![2, 2]).unwrap()),
            vec![
                Value::from("RowTimes"),
                row_times.clone(),
                Value::from("VariableNames"),
                Value::StringArray(
                    StringArray::new(vec!["A".into(), "B".into()], vec![1, 2]).unwrap(),
                ),
            ],
        ))
        .unwrap();
        assert_array2timetable_duration_row_times(&timetable, &[0.0, 1.0]);
        let Value::Tensor(round_trip) = block_on(table2array_builtin(
            block_on(timetable2table_builtin(
                timetable,
                vec![Value::from("ConvertRowTimes"), Value::Bool(false)],
            ))
            .unwrap(),
        ))
        .unwrap() else {
            panic!("expected typed integer array");
        };
        assert_eq!(round_trip.integer_storage(), Some(&storage));
    }
}

#[test]
fn array2timetable_accepts_every_integer_sample_rate_class() {
    let input = Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap());
    for storage in [
        IntegerStorage::I8(vec![2]),
        IntegerStorage::I16(vec![2]),
        IntegerStorage::I32(vec![2]),
        IntegerStorage::I64(vec![2]),
        IntegerStorage::U8(vec![2]),
        IntegerStorage::U16(vec![2]),
        IntegerStorage::U32(vec![2]),
        IntegerStorage::U64(vec![2]),
    ] {
        let timetable = block_on(array2timetable_builtin(
            input.clone(),
            vec![
                Value::from("SampleRate"),
                Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap()),
            ],
        ))
        .unwrap();
        assert_array2timetable_duration_row_times(&timetable, &[0.0, 0.5 / 86_400.0]);
    }
}

#[test]
fn array2timetable_supports_current_timing_and_name_rules() {
    let input = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let timetable = block_on(array2timetable_builtin(
        input.clone(),
        vec![
            Value::from("TimeStep"),
            array2timetable_duration_row_times(vec![0.25]),
            Value::from("StartTime"),
            array2timetable_duration_row_times(vec![1.0]),
            Value::from("VariableNames"),
            Value::StringArray(
                StringArray::new(vec![" 1 café ".into(), "A".into()], vec![1, 2]).unwrap(),
            ),
            Value::from("DimensionNames"),
            Value::StringArray(
                StringArray::new(vec!["Observations".into(), "Signals".into()], vec![1, 2])
                    .unwrap(),
            ),
        ],
    ))
    .unwrap();
    assert_array2timetable_duration_row_times(&timetable, &[1.0, 1.25]);
    let object = object(timetable);
    assert_eq!(
        table_variable_names_from_object(&object).unwrap(),
        vec![" 1 café ".to_string(), "A".to_string()]
    );
    assert_eq!(
        table_public_properties(&object)
            .unwrap()
            .fields
            .get(DIMENSION_NAMES),
        Some(&Value::StringArray(
            StringArray::new(vec!["Observations".into(), "Signals".into()], vec![1, 2]).unwrap()
        ))
    );

    for options in [
        Vec::new(),
        vec![
            Value::from("RowNames"),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap()),
        ],
        vec![
            Value::from("RowTimes"),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        ],
        vec![
            Value::from("RowTimes"),
            array2timetable_duration_row_times(vec![0.0, 1.0]),
            Value::from("SampleRate"),
            Value::Num(1.0),
        ],
        vec![
            Value::from("RowTimes"),
            array2timetable_duration_row_times(vec![0.0, 1.0]),
            Value::from("StartTime"),
            array2timetable_duration_row_times(vec![1.0]),
        ],
        vec![
            Value::from("SampleRate"),
            Value::Num(1.0),
            Value::from("DimensionNames"),
            Value::StringArray(StringArray::new(vec!["A".into(), "A".into()], vec![1, 2]).unwrap()),
        ],
    ] {
        let error = block_on(array2timetable_builtin(input.clone(), options))
            .expect_err("invalid array2timetable form must reject");
        assert!(
            error.message.contains("array2timetable"),
            "{}",
            error.message
        );
    }
}

#[test]
fn array2timetable_calendar_step_uses_datetime_calendar_arithmetic() {
    let start = crate::builtins::datetime::datetime_object_from_serial_tensor(
        None,
        Tensor::new(vec![739_282.0], vec![1, 1]).unwrap(),
        "yyyy-MM-dd".to_string(),
    )
    .unwrap();
    let mut step = ObjectInstance::new("calendarDuration".to_string());
    step.replace_property_unbounded(
        "__months".to_string(),
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
    );
    step.replace_property_unbounded(
        "__days".to_string(),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
    );
    let timetable = block_on(array2timetable_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
        vec![
            Value::from("TimeStep"),
            Value::Object(step),
            Value::from("StartTime"),
            start,
        ],
    ))
    .unwrap();
    let Value::Object(object) = timetable else {
        panic!("expected timetable object");
    };
    let row_times = timetable_row_times(&object).unwrap().unwrap();
    let serials = crate::builtins::datetime::serials_from_datetime_value(&row_times).unwrap();
    assert_eq!(
        serials.materialize_f64(),
        vec![739_282.0, 739_311.0, 739_342.0]
    );
}

#[test]
fn array2timetable_resident_integer_input_is_mode_gated_and_gathers_exactly() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let row_times = array2timetable_duration_row_times(vec![0.0, 1.0]);
        for storage in [
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ] {
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new_integer(storage.clone(), vec![2, 1]).unwrap(),
            )
            .unwrap();
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block_on(array2timetable_builtin(
                    Value::GpuTensor(handle.clone()),
                    vec![Value::from("RowTimes"), row_times.clone()],
                ))
                .expect_err("MATLAB mode must reject interactive GPU input");
                assert_eq!(
                    error.identifier(),
                    ARRAY2TIMETABLE_GPU_INPUT_EXTENSION.error_identifier
                );
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                let timetable = block_on(array2timetable_builtin(
                    Value::GpuTensor(handle),
                    vec![Value::from("RowTimes"), row_times.clone()],
                ))
                .unwrap();
                let Value::Tensor(round_trip) = block_on(table2array_builtin(
                    block_on(timetable2table_builtin(
                        timetable,
                        vec![Value::from("ConvertRowTimes"), Value::Bool(false)],
                    ))
                    .unwrap(),
                ))
                .unwrap() else {
                    panic!("expected typed integer array");
                };
                assert_eq!(round_trip.integer_storage(), Some(&storage));
            }
        }
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn array2timetable_wgpu_gathers_every_resident_integer_class_exactly() {
    let _guard = crate::builtins::common::test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = runmat_accelerate_api::provider().expect("WGPU provider");
    let row_times = array2timetable_duration_row_times(vec![0.0, 1.0]);
    for storage in [
        IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![0, u8::MAX]),
        IntegerStorage::U16(vec![0, u16::MAX]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
    ] {
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(
            provider,
            &Tensor::new_integer(storage.clone(), vec![2, 1]).unwrap(),
        )
        .unwrap();
        let timetable = block_on(array2timetable_builtin(
            Value::GpuTensor(handle),
            vec![Value::from("RowTimes"), row_times.clone()],
        ))
        .unwrap();
        let Value::Tensor(round_trip) = block_on(table2array_builtin(
            block_on(timetable2table_builtin(
                timetable,
                vec![Value::from("ConvertRowTimes"), Value::Bool(false)],
            ))
            .unwrap(),
        ))
        .unwrap() else {
            panic!("expected typed integer array");
        };
        assert_eq!(round_trip.integer_storage(), Some(&storage));
    }
}

#[test]
fn array2timetable_metadata_classifies_integer_and_gpu_forms() {
    assert_eq!(ARRAY2TIMETABLE_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(
        ARRAY2TIMETABLE_INTEGER_CAPABILITIES[0].output_class,
        runmat_builtins::BuiltinIntegerOutputClassRule::PreserveInput
    );
    assert_eq!(
        ARRAY2TIMETABLE_INTEGER_CAPABILITIES[1].overload,
        runmat_builtins::BuiltinIntegerOverloadKind::StructuralParameter
    );
    assert_eq!(
        ARRAY2TIMETABLE_EXTENSIONS,
        [ARRAY2TIMETABLE_GPU_INPUT_EXTENSION]
    );
}

#[test]
fn explicit_variable_names_reject_duplicates_and_accept_current_identifiers() {
    let duplicate = block_on(table_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap()),
            Value::from("VariableNames"),
            Value::Cell(CellArray::new(vec![Value::from("A"), Value::from("A")], 1, 2).unwrap()),
        ]),
    ));
    assert!(duplicate
        .unwrap_err()
        .message
        .contains("duplicate variable name"));

    let current = block_on(array2table_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        vec![
            Value::from("VariableNames"),
            Value::Cell(
                CellArray::new(vec![Value::from("1 bad"), Value::from("B")], 1, 2).unwrap(),
            ),
        ],
    ))
    .unwrap();
    assert_eq!(
        table_variable_names_from_object(&object(current)).unwrap(),
        vec!["1 bad".to_string(), "B".to_string()]
    );
    let whitespace = object(
        block_on(table_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(Tensor::new(vec![9.0], vec![1, 1]).unwrap()),
                Value::from("VariableNames"),
                Value::from(" "),
            ]),
        ))
        .unwrap(),
    );
    match table_member_get(&whitespace, &Value::from(" ")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![9.0]),
        other => panic!("expected numeric column, got {other:?}"),
    }
}

#[test]
fn timetable_conversion_predicates_and_head_work() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let values = Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap());
    let times = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
    let timetable = block_on(timetable_builtin(vec![
        times,
        values,
        Value::from("VariableNames"),
        Value::Cell(CellArray::new(vec![Value::from("X")], 1, 1).unwrap()),
    ]))
    .unwrap();
    assert!(matches!(
        block_on(istimetable_builtin(timetable.clone())).unwrap(),
        Value::Bool(true)
    ));
    let first_two = block_on(head_builtin(
        None,
        timetable.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
    ))
    .unwrap();
    let first_two_object = object(first_two);
    assert!(first_two_object.class_name().is(TIMETABLE_CLASS));
    assert_eq!(table_height(&first_two_object).unwrap(), 2);
    match timetable_row_times(&first_two_object).unwrap().unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0]),
        other => panic!("expected selected row times, got {other:?}"),
    }
    let table = block_on(timetable2table_builtin(
        timetable,
        vec![Value::from("ConvertRowTimes"), Value::Bool(true)],
    ))
    .unwrap();
    assert_eq!(
        table_variable_names_from_object(&object(table.clone())).unwrap(),
        vec!["Time".to_string(), "X".to_string()]
    );
    let converted = block_on(table2timetable_builtin(table, Vec::new())).unwrap();
    let converted_object = object(converted);
    assert!(converted_object.class_name().is(TIMETABLE_CLASS));
    assert_eq!(
        table_variable_names_from_object(&converted_object).unwrap(),
        vec!["Time".to_string(), "X".to_string()]
    );
    match timetable_row_times(&converted_object).unwrap().unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0]),
        other => panic!("expected timetable Time member, got {other:?}"),
    }
    match table_member_get(&converted_object, &Value::from("Time")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0]),
        other => panic!("expected retained Time variable, got {other:?}"),
    }
}

#[test]
fn head_preserves_exact_integer_rows_and_requires_positive_count() {
    assert_eq!(runmat_builtins::HEAD_INTEGER_CAPABILITIES.len(), 2);
    let source = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, 5, 7, 9]),
            vec![2, 2],
        )
        .unwrap(),
    );
    let selected = block_on(head_builtin(
        None,
        source,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
            runmat_value::IntValue::U8(1),
        )]),
    ))
    .unwrap();
    let Value::Tensor(selected) = selected else {
        panic!("expected exact integer tensor")
    };
    assert_eq!(selected.shape(), &[1, 2]);
    assert_eq!(
        selected.into_numeric_storage().unwrap(),
        runmat_value::NumericStorage::U64(vec![9_007_199_254_740_993, 7])
    );

    for controls in [
        vec![Value::Num(0.0)],
        vec![Value::Num(1.0), Value::Num(2.0)],
    ] {
        assert!(block_on(head_builtin(
            None,
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(controls)
        ))
        .is_err());
    }
}

#[test]
fn head_host_preserves_every_integer_class() {
    for storage in [
        IntegerStorage::I8(vec![-8, 8, 1, 2]),
        IntegerStorage::I16(vec![-16, 16, 1, 2]),
        IntegerStorage::I32(vec![-32, 32, 1, 2]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX, 1, 2]),
        IntegerStorage::U8(vec![0, u8::MAX, 1, 2]),
        IntegerStorage::U16(vec![0, u16::MAX, 1, 2]),
        IntegerStorage::U32(vec![0, u32::MAX, 1, 2]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX, 1, 2]),
    ] {
        let source = Tensor::new_integer(storage.clone(), vec![2, 2]).unwrap();
        let Value::Tensor(selected) = block_on(head_builtin(
            None,
            Value::Tensor(source),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .unwrap() else {
            panic!("expected exact integer tensor")
        };
        assert_eq!(selected.shape(), &[1, 2]);
        let expected = storage
            .from_exact_values_like(vec![
                storage.value_at(0).unwrap(),
                storage.value_at(2).unwrap(),
            ])
            .unwrap();
        assert_eq!(selected.integer_storage(), Some(&expected));
    }
}

#[test]
fn head_preserves_nd_real_complex_and_logical_pages() {
    let real_f64 = Tensor::new((1..=12).map(f64::from).collect(), vec![2, 2, 3]).unwrap();
    let Value::Tensor(real_f64) = block_on(head_builtin(
        None,
        Value::Tensor(real_f64),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .unwrap() else {
        panic!("expected double tensor")
    };
    assert_eq!(real_f64.shape(), &[1, 2, 3]);
    assert_eq!(
        real_f64.materialize_f64(),
        vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    );

    let real = Tensor::from_f32((1..=12).map(|value| value as f32).collect(), vec![2, 2, 3])
        .expect("single pages");
    let Value::Tensor(real) = block_on(head_builtin(
        None,
        Value::Tensor(real),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .unwrap() else {
        panic!("expected single tensor");
    };
    assert_eq!(real.shape(), &[1, 2, 3]);
    assert_eq!(real.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(real.materialize_f64(), vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0]);

    let complex = ComplexTensor::from_f32(
        (1..=12)
            .map(|value| (value as f32, -(value as f32)))
            .collect(),
        vec![2, 2, 3],
    )
    .expect("complex pages");
    let Value::ComplexTensor(complex) = block_on(head_builtin(
        None,
        Value::ComplexTensor(complex),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .unwrap() else {
        panic!("expected complex tensor");
    };
    assert_eq!(complex.shape(), &[1, 2, 3]);
    assert_eq!(complex.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(
        complex.materialize_f64(),
        vec![
            (1.0, -1.0),
            (3.0, -3.0),
            (5.0, -5.0),
            (7.0, -7.0),
            (9.0, -9.0),
            (11.0, -11.0)
        ]
    );

    let complex_f64 = ComplexTensor::new(
        (1..=12)
            .map(|value| (f64::from(value), -f64::from(value)))
            .collect(),
        vec![2, 2, 3],
    )
    .unwrap();
    let Value::ComplexTensor(complex_f64) = block_on(head_builtin(
        None,
        Value::ComplexTensor(complex_f64),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .unwrap() else {
        panic!("expected double complex tensor")
    };
    assert_eq!(complex_f64.shape(), &[1, 2, 3]);

    let integer_complex = ComplexTensor::new_integer(
        runmat_value::IntegerComplexStorage::new(
            IntegerStorage::I64((1..=12).collect()),
            IntegerStorage::I64((-12..=-1).collect()),
        )
        .unwrap(),
        vec![2, 2, 3],
    )
    .unwrap();
    let Value::ComplexTensor(integer_complex) = block_on(head_builtin(
        None,
        Value::ComplexTensor(integer_complex),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .unwrap() else {
        panic!("expected integer complex tensor")
    };
    assert_eq!(integer_complex.shape(), &[1, 2, 3]);
    assert!(integer_complex.integer_storage().is_some());

    let logical = LogicalArray::new(
        (0..12).map(|index| u8::from(index % 2 == 0)).collect(),
        vec![2, 2, 3],
    )
    .expect("logical pages");
    let Value::LogicalArray(logical) = block_on(head_builtin(
        None,
        Value::LogicalArray(logical),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .unwrap() else {
        panic!("expected logical array");
    };
    assert_eq!(logical.shape, vec![1, 2, 3]);
    assert_eq!(logical.data, vec![1, 1, 1, 1, 1, 1]);
}

#[test]
fn head_strict_gpu_row_count_extension_rejects_before_provider_access() {
    let allocation = unregistered_table_allocation(vec![2, 2]);
    let resident = Value::GpuTensor(allocation.handle.clone());
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let count_error = block_on(head_builtin(
        None,
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![resident]),
    ))
    .expect_err("resident row count must be gated");
    assert_eq!(
        count_error.identifier(),
        runmat_builtins::HEAD_EXTENSIONS[0].error_identifier
    );
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn head_preserves_resident_integer_class_owner_and_source() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, 5, 7, 9]),
            vec![2, 2],
        )
        .unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &source)
            .expect("upload exact integer source");
        let result = block_on(head_builtin(
            None,
            crate::builtins::common::gpu_helpers::resident_gpu_value(handle.clone())
                .expect("resident view"),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .expect("resident head");
        let Value::GpuTensor(result) = result else {
            panic!("expected resident result")
        };
        assert!(!result.shares_allocation_with(&handle));
        assert!(runmat_accelerate_api::provider_for_handle(&handle).is_some());
        assert!(runmat_accelerate_api::provider_for_handle(&result)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider)));
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&result),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        let downloaded = block_on(provider.download_integer(&result)).expect("download result");
        assert_eq!(downloaded.shape, vec![1, 2]);
        assert_eq!(
            downloaded.data,
            runmat_accelerate_api::HostIntegerDataOwned::U64(vec![9_007_199_254_740_993, 7])
        );
        for output in [result, handle] {
            let lease = output.allocation_lease().downgrade();
            output.release().unwrap();
            assert!(lease.acquire().is_none());
        }
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn head_wgpu_preserves_every_integer_class_exactly() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let _guard = crate::builtins::common::test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("WGPU provider");
    for storage in [
        IntegerStorage::I8(vec![i8::MIN, i8::MAX, 1, 2]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX, 1, 2]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX, 1, 2]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX, 1, 2]),
        IntegerStorage::U8(vec![0, u8::MAX, 1, 2]),
        IntegerStorage::U16(vec![0, u16::MAX, 1, 2]),
        IntegerStorage::U32(vec![0, u32::MAX, 1, 2]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX, 1, 2]),
    ] {
        let source = Tensor::new_integer(storage.clone(), vec![2, 2]).unwrap();
        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &source).unwrap();
        let output = block_on(head_builtin(
            None,
            crate::builtins::common::gpu_helpers::resident_gpu_value(handle.clone())
                .expect("resident view"),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .expect("resident integer head");
        let Value::GpuTensor(output) = output else {
            panic!("expected resident integer output")
        };
        let downloaded = block_on(provider.download_integer(&output)).expect("download head");
        let expected = storage
            .from_exact_values_like(vec![
                storage.value_at(0).unwrap(),
                storage.value_at(2).unwrap(),
            ])
            .unwrap();
        let actual = match downloaded.data {
            runmat_accelerate_api::HostIntegerDataOwned::I8(values) => IntegerStorage::I8(values),
            runmat_accelerate_api::HostIntegerDataOwned::I16(values) => IntegerStorage::I16(values),
            runmat_accelerate_api::HostIntegerDataOwned::I32(values) => IntegerStorage::I32(values),
            runmat_accelerate_api::HostIntegerDataOwned::I64(values) => IntegerStorage::I64(values),
            runmat_accelerate_api::HostIntegerDataOwned::U8(values) => IntegerStorage::U8(values),
            runmat_accelerate_api::HostIntegerDataOwned::U16(values) => IntegerStorage::U16(values),
            runmat_accelerate_api::HostIntegerDataOwned::U32(values) => IntegerStorage::U32(values),
            runmat_accelerate_api::HostIntegerDataOwned::U64(values) => IntegerStorage::U64(values),
        };
        assert_eq!(actual, expected);
        assert!(runmat_accelerate_api::provider_for_handle(&handle).is_some());
        for resident in [output, handle] {
            let lease = resident.allocation_lease().downgrade();
            resident.release().unwrap();
            assert!(lease.acquire().is_none());
        }
    }
}

#[test]
fn categorical_dictionary_and_selector_objects_materialize() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let categorical = block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::StringArray(
            StringArray::new(vec!["red".into(), "blue".into(), "red".into()], vec![3, 1]).unwrap(),
        )]),
    ))
    .unwrap();
    assert!(matches!(
        block_on(iscategorical_builtin(categorical.clone())).unwrap(),
        Value::Bool(true)
    ));
    let Value::Object(cat) = categorical else {
        panic!("expected categorical object");
    };
    match cat.properties().get("Categories").unwrap() {
        Value::StringArray(array) => assert_eq!(array.data(), vec!["blue", "red"]),
        other => panic!("expected categories, got {other:?}"),
    }

    let dictionary = block_on(dictionary_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        ]),
    ))
    .unwrap();
    assert!(matches!(dictionary, Value::Object(obj) if obj.class_name().is(DICTIONARY_CLASS)));
    assert!(matches!(
        block_on(timerange_builtin(vec![Value::Num(1.0), Value::Num(3.0)])).unwrap(),
        Value::Object(obj) if obj.class_name().is(TIMERANGE_CLASS)
    ));
    assert!(matches!(
        block_on(vartype_builtin(Value::from("numeric"))).unwrap(),
        Value::Object(obj) if obj.class_name().is(VARTYPE_CLASS)
    ));
}

#[test]
fn dictionary_declares_integer_capabilities_and_gates_resident_input_before_gather() {
    assert_eq!(DICTIONARY_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(DICTIONARY_EXTENSIONS.len(), 1);
    let labels = DICTIONARY_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert!(labels.contains(&"d = dictionary()"));
    assert!(labels.contains(&"d = dictionary(keys, values)"));
    assert!(labels.contains(&"d = dictionary(k1, v1, ..., kN, vN)"));
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .unwrap();
        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(dictionary_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(handle),
                Value::Int(runmat_value::IntValue::U8(1)),
            ]),
        ))
        .expect_err("resident dictionary input must be gated before gather");
        assert_eq!(
            error.identifier(),
            DICTIONARY_GPU_INPUT_EXTENSION.error_identifier
        );
    });

    let allocation = unregistered_table_allocation(vec![1, 1]);
    let resident = Value::GpuTensor(allocation.handle.clone());
    let nested = crate::make_cell(vec![resident], 1, 1).unwrap();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(dictionary_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            nested,
            Value::Int(runmat_value::IntValue::U8(1)),
        ]),
    ))
    .expect_err("nested resident dictionary input must be gated before gather");
    assert_eq!(
        error.identifier(),
        DICTIONARY_GPU_INPUT_EXTENSION.error_identifier
    );
    assert!(!error.message().to_ascii_lowercase().contains("provider"));
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn array_datastore_preserves_every_integer_storage_class_and_options() {
    for storage in [
        IntegerStorage::I8(vec![i8::MIN, -1, 0, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, -1, 0, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, -1, 0, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, -1, 0, i64::MAX]),
        IntegerStorage::U8(vec![0, 1, u8::MAX - 1, u8::MAX]),
        IntegerStorage::U16(vec![0, 1, u16::MAX - 1, u16::MAX]),
        IntegerStorage::U32(vec![0, 1, u32::MAX - 1, u32::MAX]),
        IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX - 1, u64::MAX]),
    ] {
        let datastore = block_on(array_datastore_builtin(vec![
            Value::Tensor(Tensor::new_integer(storage.clone(), vec![2, 2]).unwrap()),
            Value::from("ReadSize"),
            Value::Num(2.0),
            Value::from("IterationDimension"),
            Value::Num(1.0),
            Value::from("OutputType"),
            Value::from("same"),
        ]))
        .unwrap();
        let Value::Object(datastore) = datastore else {
            panic!("expected ArrayDatastore object");
        };
        assert!(datastore.class_name().is(ARRAY_DATASTORE_CLASS));
        let Some(Value::Tensor(data)) = datastore.properties().get("Data") else {
            panic!("expected typed datastore data");
        };
        assert_eq!(data.integer_storage(), Some(&storage));
        assert_eq!(
            datastore.properties().get("ReadSize"),
            Some(&Value::Num(2.0))
        );
        assert_eq!(
            datastore.properties().get("IterationDimension"),
            Some(&Value::Num(1.0))
        );
        assert_eq!(
            datastore.properties().get("OutputType"),
            Some(&Value::from("same"))
        );
    }
}

#[test]
fn array_datastore_rejects_typed_integer_controls_and_invalid_forms() {
    let data = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    for storage in [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ] {
        for property in ["ReadSize", "IterationDimension"] {
            let error = block_on(array_datastore_builtin(vec![
                data.clone(),
                Value::from(property),
                Value::Tensor(Tensor::new_integer(storage.clone(), vec![1, 1]).unwrap()),
            ]))
            .expect_err("typed integer property must reject");
            assert!(
                error.message.contains("positive double integer"),
                "{}",
                error.message
            );
        }
    }
    for args in [
        Vec::new(),
        vec![data.clone(), Value::from("ReadSize")],
        vec![data.clone(), Value::from("ReadSize"), Value::Num(0.0)],
        vec![data.clone(), Value::from("ReadSize"), Value::Num(1.5)],
        vec![
            data.clone(),
            Value::from("OutputType"),
            Value::from("table"),
        ],
        vec![
            data.clone(),
            Value::from("IterationDimension"),
            Value::Num(2.0),
            Value::from("OutputType"),
            Value::from("same"),
        ],
        vec![data.clone(), Value::from("Unknown"), Value::Num(1.0)],
    ] {
        let error = block_on(array_datastore_builtin(args))
            .expect_err("invalid arrayDatastore form must reject");
        assert!(
            error.message.contains("arrayDatastore"),
            "{}",
            error.message
        );
    }
}

#[test]
fn array_datastore_resident_integer_input_is_mode_gated_and_gathers_exactly() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        for storage in [
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ] {
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new_integer(storage.clone(), vec![2, 1]).unwrap(),
            )
            .unwrap();
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block_on(array_datastore_builtin(vec![Value::GpuTensor(
                    handle.clone(),
                )]))
                .expect_err("MATLAB mode must reject interactive GPU input");
                assert_eq!(
                    error.identifier(),
                    ARRAY_DATASTORE_GPU_INPUT_EXTENSION.error_identifier
                );
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                let datastore =
                    block_on(array_datastore_builtin(vec![Value::GpuTensor(handle)])).unwrap();
                let Value::Object(datastore) = datastore else {
                    panic!("expected ArrayDatastore object");
                };
                let Some(Value::Tensor(data)) = datastore.properties().get("Data") else {
                    panic!("expected gathered typed data");
                };
                assert_eq!(data.integer_storage(), Some(&storage));
            }
        }
    });
}

#[test]
fn parquet_datastore_rejects_typed_integer_arguments_before_gather() {
    assert_eq!(PARQUET_DATASTORE_INTEGER_CAPABILITIES.len(), 1);
    for value in [
        Value::Int(runmat_value::IntValue::I8(1)),
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap(),
        ),
    ] {
        let error = block_on(parquet_datastore_builtin(vec![
            Value::from("sample.parquet"),
            Value::from("ReadSize"),
            value,
        ]))
        .expect_err("typed integer parquet constructor control must reject");
        assert!(error.message().contains("typed-integer"));
    }
}

#[test]
fn canonical_constructor_contracts_are_shared_with_runtime() {
    assert_eq!(TIMERANGE_DESCRIPTOR.signatures.len(), 1);
    assert_eq!(TIMERANGE_EXTENSIONS.len(), 2);
    assert_eq!(TIMERANGE_INTEGER_CAPABILITIES.len(), 1);
    assert_eq!(
        ROWFILTER_INTEGER_AUDIT.kind,
        runmat_builtins::BuiltinIntegerAuditKind::NotApplicable
    );
    assert_eq!(UITABLE_EXTENSIONS.len(), 1);
    assert_eq!(UITABLE_INTEGER_CAPABILITIES.len(), 1);
}

#[test]
fn rowfilter_rejects_surplus_payload_before_gather_and_keeps_default_forms() {
    let Value::Object(empty) = block_on(rowfilter_builtin(Vec::new())).unwrap() else {
        panic!("expected rowfilter descriptor");
    };
    assert_eq!(empty.properties().get("Predicate"), Some(&Value::from("")));
    let Value::Cell(variables) = empty.properties().get("Variables").unwrap() else {
        panic!("expected empty Variables");
    };
    assert!(variables.data().is_empty());
    let Value::Object(source) = block_on(rowfilter_builtin(vec![Value::from("A")])).unwrap() else {
        panic!("expected source descriptor");
    };
    assert_eq!(
        source.properties().get("Variables"),
        Some(&Value::from("A"))
    );
    let allocation = unregistered_table_allocation(vec![1, 1]);
    let error = block_on(rowfilter_builtin(vec![
        Value::from("A"),
        Value::from("@gt0"),
        Value::GpuTensor(allocation.handle.clone()),
    ]))
    .expect_err("surplus argument must not be discarded or gathered");
    assert!(error.message().contains("at most a source"));
    assert!(!error.message().contains("provider"));
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn parquet_datastore_rejects_sparse_complex_and_nested_integer_storage() {
    let sparse = Value::SparseTensor(
        runmat_value::SparseTensor::new_integer(
            2,
            1,
            vec![0, 1],
            vec![1],
            IntegerStorage::I64(vec![i64::MIN]),
        )
        .unwrap(),
    );
    let complex = Value::ComplexTensor(
        ComplexTensor::new_integer(
            runmat_value::IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX]),
                IntegerStorage::U64(vec![1]),
            )
            .unwrap(),
            vec![1, 1],
        )
        .unwrap(),
    );
    for value in [sparse, complex] {
        let nested = Value::Cell(CellArray::new(vec![value.clone()], 1, 1).unwrap());
        for argument in [value, nested] {
            let error = block_on(parquet_datastore_builtin(vec![argument]))
                .expect_err("native integer locations reject before descriptor construction");
            assert!(error.message().contains("typed-integer"));
        }
    }
}

#[test]
fn parquet_datastore_records_supported_double_controls_and_rejects_unknown_options() {
    let datastore = block_on(parquet_datastore_builtin(vec![
        Value::from("sample.parquet"),
        Value::from("ReadSize"),
        Value::Num(32.0),
        Value::from("BlockSize"),
        Value::Num(1_000_000.0),
    ]))
    .expect("parquet datastore");
    let Value::Object(datastore) = datastore else {
        panic!("expected parquet datastore object");
    };
    assert_eq!(
        datastore.properties().get("ReadSize"),
        Some(&Value::Num(32.0))
    );
    assert_eq!(
        datastore.properties().get("BlockSize"),
        Some(&Value::Num(1_000_000.0))
    );
    for property in ["Files", "ReadSize", "BlockSize"] {
        assert!(
            crate::class_registry::lookup_property(
                &PARQUET_DATASTORE_CLASS.owned(),
                &property.into(),
            )
            .is_some(),
            "constructor property must be registered: {property}"
        );
    }

    let error = block_on(parquet_datastore_builtin(vec![
        Value::from("sample.parquet"),
        Value::from("Unknown"),
        Value::Num(1.0),
    ]))
    .expect_err("unknown option must not be ignored");
    assert!(error.message().contains("unsupported option"));
}

#[test]
#[cfg(feature = "wgpu")]
fn array_datastore_wgpu_gathers_every_resident_integer_class_exactly() {
    let _guard = crate::builtins::common::test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = runmat_accelerate_api::provider().expect("WGPU provider");
    for storage in [
        IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![0, u8::MAX]),
        IntegerStorage::U16(vec![0, u16::MAX]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
    ] {
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(
            provider,
            &Tensor::new_integer(storage.clone(), vec![2, 1]).unwrap(),
        )
        .unwrap();
        let datastore = block_on(array_datastore_builtin(vec![Value::GpuTensor(handle)])).unwrap();
        let Value::Object(datastore) = datastore else {
            panic!("expected ArrayDatastore object");
        };
        let Some(Value::Tensor(data)) = datastore.properties().get("Data") else {
            panic!("expected gathered typed data");
        };
        assert_eq!(data.integer_storage(), Some(&storage));
    }
}

#[test]
fn array_datastore_metadata_classifies_integer_and_gpu_forms() {
    assert_eq!(ARRAY_DATASTORE_INTEGER_CAPABILITIES.len(), 3);
    assert_eq!(
        ARRAY_DATASTORE_INTEGER_CAPABILITIES[0].output_class,
        runmat_builtins::BuiltinIntegerOutputClassRule::PreserveInput
    );
    assert!(ARRAY_DATASTORE_INTEGER_CAPABILITIES[1..]
        .iter()
        .flat_map(|capability| capability.inputs)
        .all(|input| input.availability
            == runmat_builtins::BuiltinIntegerInputAvailability::Rejected));
    assert_eq!(
        ARRAY_DATASTORE_EXTENSIONS,
        [ARRAY_DATASTORE_GPU_INPUT_EXTENSION]
    );
}

#[test]
fn file_datastore_preserves_constructor_metadata() {
    let files = Value::StringArray(
        StringArray::new(vec!["a.txt".into(), "b.log".into()], vec![1, 2]).unwrap(),
    );
    let extensions = Value::StringArray(
        StringArray::new(vec![".txt".into(), ".log".into()], vec![1, 2]).unwrap(),
    );
    let datastore = block_on(file_datastore_builtin(vec![
        files.clone(),
        Value::from("ReadFcn"),
        Value::FunctionHandle("readlines".into()),
        Value::from("FileExtensions"),
        extensions.clone(),
        Value::from("IncludeSubfolders"),
        Value::Bool(true),
        Value::from("ReadMode"),
        Value::from("partialfile"),
    ]))
    .unwrap();
    let Value::Object(datastore) = datastore else {
        panic!("expected fileDatastore object");
    };
    assert!(datastore.class_name().is(FILE_DATASTORE_CLASS));
    assert_eq!(datastore.properties().get("Files"), Some(&files));
    assert_eq!(
        datastore.properties().get("ReadFcn"),
        Some(&Value::FunctionHandle("readlines".into()))
    );
    assert_eq!(
        datastore.properties().get("FileExtensions"),
        Some(&extensions)
    );
    assert_eq!(
        datastore.properties().get("IncludeSubfolders"),
        Some(&Value::Bool(true))
    );
    assert_eq!(
        datastore.properties().get("ReadMode"),
        Some(&Value::from("partialfile"))
    );
}

#[test]
fn file_datastore_validates_option_pairs_and_modes() {
    let err = block_on(file_datastore_builtin(Vec::new())).expect_err("missing location");
    assert!(err.message().contains("location is required"));

    let err = block_on(file_datastore_builtin(vec![
        Value::from("*.txt"),
        Value::from("ReadFcn"),
    ]))
    .expect_err("unpaired option");
    assert!(err.message().contains("name-value options"));

    let err = block_on(file_datastore_builtin(vec![
        Value::from("*.txt"),
        Value::from("ReadMode"),
        Value::from("chunk"),
    ]))
    .expect_err("invalid ReadMode");
    assert!(err.message().contains("ReadMode"));
}

#[test]
fn file_datastore_requires_reader_but_never_resolves_or_invokes_it() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    for read_fcn in [None, Some(Value::from("readlines")), Some(Value::Num(7.0))] {
        let mut args = vec![Value::from("missing/*.txt")];
        if let Some(reader) = read_fcn {
            args.extend([Value::from("ReadFcn"), reader]);
        }
        let error = block_on(file_datastore_builtin(args)).unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::FILE_DATASTORE_METADATA_EXTENSION.error_identifier
        );
    }
    // An unresolved handle is valid stored metadata; resolving or invoking it
    // would fail. The missing path must not be opened or enumerated either.
    let reader = Value::function_handle("file_datastore_reader_must_not_execute".into());
    let value = block_on(file_datastore_builtin(vec![
        Value::from("missing/*.txt"),
        Value::from("ReadFcn"),
        reader.clone(),
    ]))
    .unwrap();
    let Value::Object(object) = value else {
        panic!("descriptor")
    };
    assert_eq!(object.properties().get("ReadFcn"), Some(&reader));
    assert_eq!(
        object.properties().get("Files"),
        Some(&Value::from("missing/*.txt"))
    );
}

#[test]
fn file_datastore_boolean_controls_and_runmat_payloads_are_explicit() {
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (flag, expected) in [
            (Value::Bool(true), true),
            (Value::Num(1.0), true),
            (
                Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
                true,
            ),
            (
                Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
                false,
            ),
        ] {
            let value = block_on(file_datastore_builtin(vec![
                Value::from("*.txt"),
                Value::from("ReadFcn"),
                Value::function_handle("unresolved_reader".into()),
                Value::from("IncludeSubfolders"),
                flag,
            ]))
            .unwrap();
            let Value::Object(object) = value else {
                panic!("descriptor")
            };
            assert_eq!(
                object.properties().get("IncludeSubfolders"),
                Some(&Value::Bool(expected))
            );
        }
        for flag in [
            Value::Num(2.0),
            Value::from("yes"),
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        ] {
            let error = block_on(file_datastore_builtin(vec![
                Value::from("*.txt"),
                Value::from("ReadFcn"),
                Value::function_handle("reader".into()),
                Value::from("IncludeSubfolders"),
                flag,
            ]))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                runmat_builtins::FILE_DATASTORE_BOOLEAN_EXTENSION.error_identifier
            );
        }
    }
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let payload = Value::Tensor(
        Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![2, 1],
        )
        .unwrap(),
    );
    let value = block_on(file_datastore_builtin(vec![
        payload.clone(),
        Value::from("ReadFcn"),
        payload.clone(),
        Value::from("IncludeSubfolders"),
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
    ]))
    .unwrap();
    let Value::Object(object) = value else {
        panic!("descriptor")
    };
    assert_eq!(object.properties().get("Files"), Some(&payload));
    assert_eq!(object.properties().get("ReadFcn"), Some(&payload));
    assert_eq!(
        object.properties().get("IncludeSubfolders"),
        Some(&Value::Bool(true))
    );
    let default = block_on(file_datastore_builtin(vec![Value::from("*.txt")])).unwrap();
    let Value::Object(default) = default else {
        panic!("descriptor")
    };
    assert_eq!(default.properties().get("ReadFcn"), Some(&Value::from("")));
    assert_eq!(
        default.properties().get("ReadMode"),
        Some(&Value::from("file"))
    );
    assert_eq!(
        default.properties().get("IncludeSubfolders"),
        Some(&Value::Bool(false))
    );
    let Some(Value::StringArray(extensions)) = default.properties().get("FileExtensions") else {
        panic!("default extensions")
    };
    assert_eq!(extensions.shape.as_slice(), &[0, 1]);
    for flag in [
        Value::Num(f64::NAN),
        Value::Tensor(
            Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![1]), vec![1, 1]).unwrap(),
        ),
    ] {
        assert!(block_on(file_datastore_builtin(vec![
            Value::from("*.txt"),
            Value::from("IncludeSubfolders"),
            flag
        ]))
        .is_err());
    }
}

#[test]
fn file_datastore_default_binding_enforces_fixed_outputs_and_validates_discarded_calls() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let args = [
        Value::from("*.txt"),
        Value::from("ReadFcn"),
        Value::function_handle("unresolved_reader".into()),
    ];
    assert!(block_on(crate::call_builtin_async_with_outputs(
        "fileDatastore",
        &args,
        1
    ))
    .is_ok());
    assert!(block_on(crate::call_builtin_async_with_outputs(
        "fileDatastore",
        &args,
        0
    ))
    .is_ok());
    assert!(block_on(crate::call_builtin_async_with_outputs(
        "fileDatastore",
        &args,
        2
    ))
    .is_err());
    assert!(block_on(crate::call_builtin_async_with_outputs(
        "fileDatastore",
        &[],
        0
    ))
    .is_err());
    for name in [
        "BlockSize",
        "PreviewFcn",
        "UniformRead",
        "AlternateFileSystemRoots",
        "CreateLocalCopy",
    ] {
        let invalid = [
            args[0].clone(),
            args[1].clone(),
            args[2].clone(),
            Value::from(name),
            Value::Bool(false),
        ];
        assert!(
            block_on(file_datastore_builtin(invalid.to_vec())).is_err(),
            "{name}"
        );
    }
}

#[test]
fn ordinal_constructor_sets_ordered_categorical_semantics() {
    let default_ordered = block_on(ordinal_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::StringArray(
            StringArray::new(
                vec!["medium".into(), "low".into(), "high".into()],
                vec![3, 1],
            )
            .unwrap(),
        )]),
    ))
    .unwrap();
    let Value::Object(default_cat) = default_ordered else {
        panic!("expected default ordinal categorical object");
    };
    match default_cat.properties().get("Categories").unwrap() {
        Value::StringArray(array) => assert_eq!(array.data(), vec!["high", "low", "medium"]),
        other => panic!("expected sorted categories, got {other:?}"),
    }

    let ordinal = block_on(ordinal_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::StringArray(
                StringArray::new(
                    vec!["medium".into(), "low".into(), "high".into()],
                    vec![3, 1],
                )
                .unwrap(),
            ),
            Value::StringArray(
                StringArray::new(
                    vec!["low".into(), "medium".into(), "high".into()],
                    vec![1, 3],
                )
                .unwrap(),
            ),
            Value::StringArray(
                StringArray::new(
                    vec!["low".into(), "medium".into(), "high".into()],
                    vec![1, 3],
                )
                .unwrap(),
            ),
        ]),
    ))
    .unwrap();
    assert!(matches!(
        block_on(iscategorical_builtin(ordinal.clone())).unwrap(),
        Value::Bool(true)
    ));
    assert!(matches!(
        block_on(isordinal_builtin(ordinal.clone())).unwrap(),
        Value::Bool(true)
    ));
    let max_value = block_on(crate::call_builtin_async(
        "max",
        std::slice::from_ref(&ordinal),
    ))
    .unwrap();
    let Value::Object(max_cat) = max_value.value() else {
        panic!("expected categorical max result");
    };
    match max_cat.properties().get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0]),
        other => panic!("expected categorical max code, got {other:?}"),
    }
    let min_value = block_on(crate::call_builtin_async(
        "min",
        std::slice::from_ref(&ordinal),
    ))
    .unwrap();
    let Value::Object(min_cat) = min_value.value() else {
        panic!("expected categorical min result");
    };
    match min_cat.properties().get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0]),
        other => panic!("expected categorical min code, got {other:?}"),
    }
    let all_max = block_on(crate::call_builtin_async(
        "max",
        &[
            ordinal.clone(),
            Value::Tensor(Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap()),
            Value::from("all"),
        ],
    ))
    .unwrap();
    let Value::Object(all_max_cat) = all_max.value() else {
        panic!("expected categorical all max result");
    };
    match all_max_cat.properties().get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0]),
        other => panic!("expected categorical all max code, got {other:?}"),
    }

    let Value::Object(cat) = ordinal else {
        panic!("expected ordinal categorical object");
    };
    assert_eq!(cat.properties().get("Ordinal"), Some(&Value::Bool(true)));
    match cat.properties().get("Categories").unwrap() {
        Value::StringArray(array) => {
            assert_eq!(array.data(), vec!["low", "medium", "high"]);
            assert_eq!(array.shape, vec![1, 3]);
        }
        other => panic!("expected categories, got {other:?}"),
    }
    match cat.properties().get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0, 1.0, 3.0]),
        other => panic!("expected categorical codes, got {other:?}"),
    }

    let renamed = block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![3.0, 1.0, 2.0], vec![3, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
            Value::StringArray(
                StringArray::new(
                    vec!["child".into(), "adult".into(), "senior".into()],
                    vec![1, 3],
                )
                .unwrap(),
            ),
            Value::from("Ordinal"),
            Value::Bool(true),
        ]),
    ))
    .unwrap();
    assert!(matches!(
        block_on(isordinal_builtin(renamed.clone())).unwrap(),
        Value::Bool(true)
    ));
    let Value::Object(renamed_cat) = renamed else {
        panic!("expected renamed categorical");
    };
    match renamed_cat.properties().get("Categories").unwrap() {
        Value::StringArray(array) => assert_eq!(array.data(), vec!["child", "adult", "senior"]),
        other => panic!("expected renamed categories, got {other:?}"),
    }
    match renamed_cat.properties().get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0, 1.0, 2.0]),
        other => panic!("expected renamed codes, got {other:?}"),
    }

    let nominal = block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::StringArray(
            StringArray::new(vec!["low".into(), "high".into()], vec![2, 1]).unwrap(),
        )]),
    ))
    .unwrap();
    assert!(matches!(
        block_on(isordinal_builtin(nominal)).unwrap(),
        Value::Bool(false)
    ));
    assert!(matches!(
        block_on(isordinal_builtin(Value::Num(1.0))).unwrap(),
        Value::Bool(false)
    ));
}

#[test]
fn ordinal_constructor_preserves_adjacent_wide_integer_levels() {
    let lower = (1_u64 << 53) + 1;
    let upper = lower + 1;
    let source = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![upper, lower, upper]), vec![3, 1]).unwrap(),
    );
    let labels = Value::StringArray(
        StringArray::new(vec!["lower".into(), "upper".into()], vec![1, 2]).unwrap(),
    );
    let levels = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![lower, upper]), vec![1, 2]).unwrap(),
    );
    let Value::Object(ordinal) = block_on(ordinal_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![source, labels, levels]),
    ))
    .expect("ordinal") else {
        panic!("ordinal object")
    };
    let categories = match ordinal.properties().get("Categories") {
        Some(Value::StringArray(array)) => array.data(),
        other => panic!("categories {other:?}"),
    };
    assert_eq!(categories, vec!["lower", "upper"]);
    assert_eq!(
        crate::builtins::table::categorical_labels(&Value::Object(ordinal)).unwrap(),
        vec!["upper", "lower", "upper"]
    );
}

#[test]
fn ordinal_constructor_accepts_every_integer_level_class() {
    for storage in [
        IntegerStorage::I8(vec![2, 1]),
        IntegerStorage::I16(vec![2, 1]),
        IntegerStorage::I32(vec![2, 1]),
        IntegerStorage::I64(vec![2, 1]),
        IntegerStorage::U8(vec![2, 1]),
        IntegerStorage::U16(vec![2, 1]),
        IntegerStorage::U32(vec![2, 1]),
        IntegerStorage::U64(vec![2, 1]),
    ] {
        let source = Value::Tensor(Tensor::new_integer(storage.clone(), vec![2, 1]).unwrap());
        let levels = Value::Tensor(Tensor::new_integer(storage, vec![1, 2]).unwrap());
        let labels = Value::StringArray(
            StringArray::new(vec!["second".into(), "first".into()], vec![1, 2]).unwrap(),
        );
        let Value::Object(ordinal) = block_on(ordinal_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![source, labels, levels]),
        ))
        .expect("ordinal") else {
            panic!("ordinal object")
        };
        assert_eq!(
            crate::builtins::table::categorical_labels(&Value::Object(ordinal)).unwrap(),
            vec!["second", "first"]
        );
    }
}

#[test]
fn ordinal_constructor_bins_adjacent_wide_integer_edges_exactly() {
    let lower = (1_u64 << 53) + 1;
    let middle = lower + 1;
    let upper = middle + 1;
    let source = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![lower, middle, upper]), vec![3, 1]).unwrap(),
    );
    let labels = Value::StringArray(
        StringArray::new(vec!["first".into(), "second".into()], vec![1, 2]).unwrap(),
    );
    let empty = Value::Tensor(Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap());
    let edges = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![lower, middle, upper]), vec![1, 3]).unwrap(),
    );
    let Value::Object(ordinal) = block_on(ordinal_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            source, labels, empty, edges,
        ]),
    ))
    .expect("ordinal") else {
        panic!("ordinal object")
    };
    assert_eq!(
        crate::builtins::table::categorical_labels(&Value::Object(ordinal)).unwrap(),
        vec!["first", "second", "second"]
    );
}

#[test]
fn categorical_compatibility_edges_match_matlab_surface() {
    let option_word_category = block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Ordinal"),
            Value::from("Ordinal"),
        ]),
    ))
    .unwrap();
    let Value::Object(option_word_category) = option_word_category else {
        panic!("expected option-word categorical");
    };
    match option_word_category.properties().get("Categories").unwrap() {
        Value::StringArray(array) => assert_eq!(array.data(), vec!["Ordinal"]),
        other => panic!("expected option-word categories, got {other:?}"),
    }
    match option_word_category.properties().get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0]),
        other => panic!("expected option-word codes, got {other:?}"),
    }

    let whitespace = block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::StringArray(
            StringArray::new(vec!["a".into(), " a".into(), "a ".into()], vec![3, 1]).unwrap(),
        )]),
    ))
    .unwrap();
    let Value::Object(whitespace) = whitespace else {
        panic!("expected whitespace categorical");
    };
    match whitespace.properties().get("Categories").unwrap() {
        Value::StringArray(array) => assert_eq!(array.data(), vec!["a"]),
        other => panic!("expected whitespace categories, got {other:?}"),
    }
    match whitespace.properties().get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0, 1.0, 1.0]),
        other => panic!("expected whitespace codes, got {other:?}"),
    }

    let duplicate_valueset = block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("x"),
            Value::StringArray(StringArray::new(vec!["x".into(), "x".into()], vec![1, 2]).unwrap()),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        ]),
    ));
    assert!(duplicate_valueset.is_err());

    let merged_names = block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::StringArray(
                StringArray::new(vec!["same".into(), "same".into()], vec![1, 2]).unwrap(),
            ),
        ]),
    ))
    .unwrap();
    let Value::Object(merged_names) = merged_names else {
        panic!("expected merged categorical");
    };
    match merged_names.properties().get("Categories").unwrap() {
        Value::StringArray(array) => assert_eq!(array.data(), vec!["same"]),
        other => panic!("expected merged categories, got {other:?}"),
    }
    match merged_names.properties().get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0, 1.0]),
        other => panic!("expected merged codes, got {other:?}"),
    }

    let rounded = block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
            Tensor::new(vec![1.0, 1.234_567_89], vec![1, 2]).unwrap(),
        )]),
    ))
    .unwrap();
    let Value::Object(rounded) = rounded else {
        panic!("expected rounded numeric categories");
    };
    match rounded.properties().get("Categories").unwrap() {
        Value::StringArray(array) => assert_eq!(array.data(), vec!["1", "1.2346"]),
        other => panic!("expected rounded categories, got {other:?}"),
    }

    assert!(block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
            Tensor::new(vec![1.0, 1.000_01], vec![1, 2]).unwrap(),
        )])
    ))
    .is_err());

    let protected_ordinal = block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("low"),
            Value::from("Ordinal"),
            Value::Bool(true),
            Value::from("Protected"),
            Value::Bool(false),
        ]),
    ))
    .unwrap();
    let Value::Object(protected_ordinal) = protected_ordinal else {
        panic!("expected protected ordinal categorical");
    };
    assert_eq!(
        protected_ordinal.properties().get("Protected"),
        Some(&Value::Bool(true))
    );

    let low_high = block_on(ordinal_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("low"),
            Value::StringArray(
                StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
            ),
            Value::StringArray(
                StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
            ),
        ]),
    ))
    .unwrap();
    let high_low = block_on(ordinal_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("low"),
            Value::StringArray(
                StringArray::new(vec!["high".into(), "low".into()], vec![1, 2]).unwrap(),
            ),
            Value::StringArray(
                StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
            ),
        ]),
    ))
    .unwrap();
    assert!(block_on(crate::call_builtin_async(
        "lt",
        &[low_high.clone(), high_low]
    ))
    .is_err());

    let extra_category = block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("low"),
            Value::StringArray(
                StringArray::new(
                    vec!["low".into(), "medium".into(), "high".into()],
                    vec![1, 3],
                )
                .unwrap(),
            ),
        ]),
    ))
    .unwrap();
    assert!(block_on(crate::call_builtin_async("eq", &[low_high, extra_category])).is_err());
}

#[test]
fn categorical_integer_constructor_surface_is_exact_for_every_class() {
    let cases = [
        (
            IntegerStorage::I8(vec![-2, 10]),
            IntegerStorage::I8(vec![10, -2]),
        ),
        (
            IntegerStorage::I16(vec![-2, 10]),
            IntegerStorage::I16(vec![10, -2]),
        ),
        (
            IntegerStorage::I32(vec![-2, 10]),
            IntegerStorage::I32(vec![10, -2]),
        ),
        (
            IntegerStorage::I64(vec![-2, 10]),
            IntegerStorage::I64(vec![10, -2]),
        ),
        (
            IntegerStorage::U8(vec![2, 10]),
            IntegerStorage::U8(vec![10, 2]),
        ),
        (
            IntegerStorage::U16(vec![2, 10]),
            IntegerStorage::U16(vec![10, 2]),
        ),
        (
            IntegerStorage::U32(vec![2, 10]),
            IntegerStorage::U32(vec![10, 2]),
        ),
        (
            IntegerStorage::U64(vec![2, 10]),
            IntegerStorage::U64(vec![10, 2]),
        ),
    ];
    for (source, valueset) in cases {
        let source = Value::Tensor(Tensor::new_integer(source, vec![1, 2]).unwrap());
        let valueset = Value::Tensor(Tensor::new_integer(valueset, vec![1, 2]).unwrap());
        let Value::Object(result) = block_on(categorical_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                source,
                valueset,
                Value::StringArray(
                    StringArray::new(vec!["ten".into(), "small".into()], vec![1, 2]).unwrap(),
                ),
            ]),
        ))
        .unwrap() else {
            panic!("expected categorical object");
        };
        match result.properties().get("Codes").unwrap() {
            Value::Tensor(codes) => assert_eq!(codes.materialize_f64(), vec![2.0, 1.0]),
            other => panic!("expected categorical codes, got {other:?}"),
        }
    }

    let source = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .unwrap(),
    );
    let valueset = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            vec![1, 2],
        )
        .unwrap(),
    );
    let Value::Object(wide) = block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            source.clone(),
            valueset,
            Value::StringArray(
                StringArray::new(vec!["maximum".into(), "flint-plus-one".into()], vec![1, 2])
                    .unwrap(),
            ),
        ]),
    ))
    .unwrap() else {
        panic!("expected wide categorical object");
    };
    match wide.properties().get("Codes").unwrap() {
        Value::Tensor(codes) => assert_eq!(codes.materialize_f64(), vec![2.0, 1.0]),
        other => panic!("expected wide categorical codes, got {other:?}"),
    }

    let mismatched = block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            source,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I64(vec![1, 2]), vec![1, 2]).unwrap(),
            ),
        ]),
    ));
    assert!(mismatched.is_err());
}

#[test]
fn categorical_integer_flags_accept_only_exact_zero_or_one() {
    let flags = [
        (
            runmat_value::IntValue::I8(0),
            runmat_value::IntValue::I8(1),
            runmat_value::IntValue::I8(2),
        ),
        (
            runmat_value::IntValue::I16(0),
            runmat_value::IntValue::I16(1),
            runmat_value::IntValue::I16(2),
        ),
        (
            runmat_value::IntValue::I32(0),
            runmat_value::IntValue::I32(1),
            runmat_value::IntValue::I32(2),
        ),
        (
            runmat_value::IntValue::I64(0),
            runmat_value::IntValue::I64(1),
            runmat_value::IntValue::I64(2),
        ),
        (
            runmat_value::IntValue::U8(0),
            runmat_value::IntValue::U8(1),
            runmat_value::IntValue::U8(2),
        ),
        (
            runmat_value::IntValue::U16(0),
            runmat_value::IntValue::U16(1),
            runmat_value::IntValue::U16(2),
        ),
        (
            runmat_value::IntValue::U32(0),
            runmat_value::IntValue::U32(1),
            runmat_value::IntValue::U32(2),
        ),
        (
            runmat_value::IntValue::U64(0),
            runmat_value::IntValue::U64(1),
            runmat_value::IntValue::U64(2),
        ),
    ];
    for (zero, one, two) in flags {
        let Value::Object(nominal) = block_on(categorical_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("x"),
                Value::from("Ordinal"),
                Value::Int(zero),
            ]),
        ))
        .unwrap() else {
            panic!("expected nominal categorical");
        };
        assert_eq!(
            nominal.properties().get("Ordinal"),
            Some(&Value::Bool(false))
        );

        let Value::Object(ordinal) = block_on(categorical_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("x"),
                Value::from("Ordinal"),
                Value::Int(one),
            ]),
        ))
        .unwrap() else {
            panic!("expected ordinal categorical");
        };
        assert_eq!(
            ordinal.properties().get("Ordinal"),
            Some(&Value::Bool(true))
        );
        assert!(block_on(categorical_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("x"),
                Value::from("Protected"),
                Value::Int(two),
            ])
        ))
        .is_err());
    }
}

#[test]
fn categorical_resident_integer_input_is_mode_gated_and_gathered_exactly() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .unwrap();
        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(categorical_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                    handle.clone(),
                )]),
            ))
            .expect_err("MATLAB mode must reject resident categorical input before gather");
            assert_eq!(
                error.identifier(),
                CATEGORICAL_GPU_INPUT_EXTENSION.error_identifier
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let Value::Object(result) = block_on(categorical_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                    handle,
                )]),
            ))
            .unwrap() else {
                panic!("expected resident categorical object");
            };
            match result.properties().get("Codes").unwrap() {
                Value::Tensor(codes) => assert_eq!(codes.materialize_f64(), vec![1.0, 2.0]),
                other => panic!("expected categorical codes, got {other:?}"),
            }

            let flag = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();
            let flag_handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &flag).unwrap();
            let Value::Object(ordinal) = block_on(categorical_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("x"),
                    Value::from("Ordinal"),
                    Value::GpuTensor(flag_handle),
                ]),
            ))
            .unwrap() else {
                panic!("expected resident-flag categorical object");
            };
            assert_eq!(
                ordinal.properties().get("Ordinal"),
                Some(&Value::Bool(true))
            );
        }
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn categorical_wgpu_gathers_every_resident_integer_class_exactly() {
    let _guard = crate::builtins::common::test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = runmat_accelerate_api::provider().expect("WGPU provider");
    for storage in [
        IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![0, u8::MAX]),
        IntegerStorage::U16(vec![0, u16::MAX]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
    ] {
        let tensor = Tensor::new_integer(storage, vec![1, 2]).unwrap();
        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        let Value::Object(result) = block_on(categorical_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                handle,
            )]),
        ))
        .unwrap() else {
            panic!("expected categorical object");
        };
        match result.properties().get("Codes").unwrap() {
            Value::Tensor(codes) => assert_eq!(codes.materialize_f64(), vec![1.0, 2.0]),
            other => panic!("expected categorical codes, got {other:?}"),
        }
    }
}

#[test]
fn ordinal_min_max_preserve_categorical_results_for_missing_dims_and_indices() {
    let ordinal_with_missing = block_on(ordinal_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::StringArray(StringArray::new(vec!["x".into(), "z".into()], vec![2, 1]).unwrap()),
            Value::StringArray(StringArray::new(vec!["x".into(), "y".into()], vec![1, 2]).unwrap()),
            Value::StringArray(StringArray::new(vec!["x".into(), "y".into()], vec![1, 2]).unwrap()),
        ]),
    ))
    .unwrap();
    let default_max = block_on(crate::call_builtin_async(
        "max",
        std::slice::from_ref(&ordinal_with_missing),
    ))
    .unwrap();
    let Value::Object(default_max) = default_max.value() else {
        panic!("expected categorical max with missing");
    };
    match default_max.properties().get("Codes").unwrap() {
        Value::Tensor(tensor) => assert!(tensor.materialize_f64()[0].is_nan()),
        other => panic!("expected missing max code, got {other:?}"),
    }

    let omitnan_max = block_on(crate::call_builtin_async(
        "max",
        &[
            ordinal_with_missing.clone(),
            Value::Tensor(Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap()),
            Value::from("omitnan"),
        ],
    ))
    .unwrap();
    let Value::Object(omitnan_max) = omitnan_max.value() else {
        panic!("expected omitnan categorical max");
    };
    match omitnan_max.properties().get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0]),
        other => panic!("expected omitnan max code, got {other:?}"),
    }

    let two_outputs = block_on(crate::call_builtin_async_with_outputs(
        "max",
        std::slice::from_ref(&ordinal_with_missing),
        2,
    ))
    .unwrap();
    match two_outputs.value() {
        Value::OutputList(values) => {
            assert_eq!(values.len(), 2);
            assert!(matches!(&values[0], Value::Object(_)));
            match &values[1] {
                Value::Num(index) => assert_eq!(*index, 2.0),
                other => panic!("expected numeric index, got {other:?}"),
            }
        }
        other => panic!("expected two-output list, got {other:?}"),
    }

    let matrix = block_on(ordinal_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::StringArray(
                StringArray::new(
                    vec!["low".into(), "high".into(), "medium".into(), "low".into()],
                    vec![2, 2],
                )
                .unwrap(),
            ),
            Value::StringArray(
                StringArray::new(
                    vec!["low".into(), "medium".into(), "high".into()],
                    vec![1, 3],
                )
                .unwrap(),
            ),
            Value::StringArray(
                StringArray::new(
                    vec!["low".into(), "medium".into(), "high".into()],
                    vec![1, 3],
                )
                .unwrap(),
            ),
        ]),
    ))
    .unwrap();
    let row_max = block_on(crate::call_builtin_async(
        "max",
        &[
            matrix,
            Value::Tensor(Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap()),
            Value::Num(2.0),
        ],
    ))
    .unwrap();
    let Value::Object(row_max) = row_max.value() else {
        panic!("expected categorical row max");
    };
    match row_max.properties().get("Codes").unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2, 1]);
            assert_eq!(tensor.materialize_f64(), vec![2.0, 3.0]);
        }
        other => panic!("expected row max codes, got {other:?}"),
    }
}

#[test]
fn writetable_and_readcell_cover_delimited_interop() {
    let path = unique_path("writetable_round_trip").with_extension("csv");
    let table = table_from_columns(
        vec!["A".into(), "Name".into()],
        vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Value::StringArray(
                StringArray::new(vec!["item-a".into(), "item-b".into()], vec![2, 1]).unwrap(),
            ),
        ],
    )
    .unwrap();
    let bytes = block_on(writetable_builtin(
        None,
        table,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
            path.to_string_lossy().to_string(),
        )]),
    ))
    .unwrap();
    assert!(matches!(bytes, Value::Num(n) if n > 0.0));
    let cells = block_on(readcell_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        Vec::new(),
    ))
    .unwrap();
    match cells {
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 3);
            assert_eq!(cell.cols, 2);
            assert_eq!(cell.get(0, 0).unwrap(), Value::from("A"));
            assert_eq!(cell.get(0, 1).unwrap(), Value::from("Name"));
            assert_eq!(cell.get(1, 0).unwrap(), Value::Num(1.0));
            assert_eq!(cell.get(1, 1).unwrap(), Value::from("item-a"));
        }
        other => panic!("expected cell array, got {other:?}"),
    }
    let _ = fs::remove_file(path);
}

#[test]
fn timetable_rowtimes_option_keeps_all_variables_and_table2timetable_does_not_drop_data() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let a = Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap());
    let b = Value::Tensor(Tensor::new(vec![30.0, 40.0], vec![2, 1]).unwrap());
    let times = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let tt = block_on(timetable_builtin(vec![
        a,
        b,
        Value::from("RowTimes"),
        times,
        Value::from("VariableNames"),
        Value::Cell(CellArray::new(vec![Value::from("A"), Value::from("B")], 1, 2).unwrap()),
    ]))
    .unwrap();
    let tt_obj = object(tt);
    assert_eq!(
        table_variable_names_from_object(&tt_obj).unwrap(),
        vec!["A".to_string(), "B".to_string()]
    );

    let t = table_from_columns(
        vec!["A".into(), "B".into()],
        vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap()),
        ],
    )
    .unwrap();
    let converted = block_on(table2timetable_builtin(t, Vec::new())).unwrap();
    let converted_obj = object(converted);
    assert_eq!(
        table_variable_names_from_object(&converted_obj).unwrap(),
        vec!["A".to_string(), "B".to_string()]
    );
    match timetable_row_times(&converted_obj).unwrap().unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0]),
        other => panic!("expected generated row times, got {other:?}"),
    }

    let path = unique_path("readtimetable_rowtimes").with_extension("csv");
    fs::write(&path, "A\n10\n20\n").unwrap();
    let read = block_on(readtimetable_builtin(
        Value::from(path.to_string_lossy().to_string()),
        vec![
            Value::from("RowTimes"),
            Value::Tensor(Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap()),
        ],
    ))
    .unwrap();
    let read_obj = object(read);
    match timetable_row_times(&read_obj).unwrap().unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![5.0, 6.0]),
        other => panic!("expected explicit readtimetable row times, got {other:?}"),
    }
    let _ = fs::remove_file(path);
}

#[test]
fn table_selector_objects_filter_rows_and_variables() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let t = table_from_columns(
        vec!["A".into(), "Name".into()],
        vec![
            Value::Tensor(Tensor::new(vec![-1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
            Value::StringArray(
                StringArray::new(vec!["x".into(), "y".into(), "z".into()], vec![3, 1]).unwrap(),
            ),
        ],
    )
    .unwrap();
    let t_obj = object(t.clone());
    let numeric = block_on(vartype_builtin(Value::from("numeric"))).unwrap();
    let subset = object(
        table_paren_get(
            &t_obj,
            &Value::Cell(CellArray::new(vec![Value::from(":"), numeric], 1, 2).unwrap()),
        )
        .unwrap(),
    );
    assert_eq!(
        table_variable_names_from_object(&subset).unwrap(),
        vec!["A".to_string()]
    );

    let filter = block_on(rowfilter_builtin(vec![
        Value::Cell(CellArray::new(vec![Value::from("A")], 1, 1).unwrap()),
        Value::from("@gt0"),
    ]))
    .unwrap();
    let filtered = object(
        table_paren_get(
            &t_obj,
            &Value::Cell(CellArray::new(vec![filter, Value::from(":")], 1, 2).unwrap()),
        )
        .unwrap(),
    );
    assert_eq!(table_height(&filtered).unwrap(), 2);

    let tt = block_on(timetable_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap()),
        Value::from("VariableNames"),
        Value::Cell(CellArray::new(vec![Value::from("X")], 1, 1).unwrap()),
    ]))
    .unwrap();
    let tt_obj = object(tt);
    let range = block_on(timerange_builtin(vec![Value::Num(2.0), Value::Num(3.0)])).unwrap();
    let ranged = object(
        table_paren_get(
            &tt_obj,
            &Value::Cell(CellArray::new(vec![range, Value::from(":")], 1, 2).unwrap()),
        )
        .unwrap(),
    );
    assert!(ranged.class_name().is(TIMETABLE_CLASS));
    assert_eq!(table_height(&ranged).unwrap(), 1);

    let open_left = block_on(timerange_builtin(vec![
        Value::Num(2.0),
        Value::Num(3.0),
        Value::from("openleft"),
    ]))
    .unwrap();
    let ranged = object(
        table_paren_get(
            &tt_obj,
            &Value::Cell(CellArray::new(vec![open_left, Value::from(":")], 1, 2).unwrap()),
        )
        .unwrap(),
    );
    match table_member_get(&ranged, &Value::from("X")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![30.0]),
        other => panic!("expected openleft range result, got {other:?}"),
    }

    let open_right = block_on(timerange_builtin(vec![
        Value::Num(2.0),
        Value::Num(3.0),
        Value::from("openright"),
    ]))
    .unwrap();
    let ranged = object(
        table_paren_get(
            &tt_obj,
            &Value::Cell(CellArray::new(vec![open_right, Value::from(":")], 1, 2).unwrap()),
        )
        .unwrap(),
    );
    match table_member_get(&ranged, &Value::from("X")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![20.0]),
        other => panic!("expected openright range result, got {other:?}"),
    }

    let datetime_row_times = crate::builtins::datetime::datetime_object_from_serial_tensor(
        None,
        Tensor::new(vec![100.0, 200.0, 300.0], vec![3, 1]).unwrap(),
        "yyyy-MM-dd",
    )
    .unwrap();
    let datetime_selector = crate::builtins::datetime::datetime_object_from_serial_tensor(
        None,
        Tensor::new(vec![200.0], vec![1, 1]).unwrap(),
        "yyyy-MM-dd",
    )
    .unwrap();
    let tt = block_on(timetable_builtin(vec![
        datetime_row_times,
        Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap()),
        Value::from("VariableNames"),
        Value::Cell(CellArray::new(vec![Value::from("X")], 1, 1).unwrap()),
    ]))
    .unwrap();
    let tt_obj = object(tt);
    let selected = object(
        table_paren_get(
            &tt_obj,
            &Value::Cell(CellArray::new(vec![datetime_selector, Value::from(":")], 1, 2).unwrap()),
        )
        .unwrap(),
    );
    assert!(selected.class_name().is(TIMETABLE_CLASS));
    match table_member_get(&selected, &Value::from("X")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![20.0]),
        other => panic!("expected datetime row-time selection, got {other:?}"),
    }
}

#[test]
fn vartype_accepts_text_and_rejects_integer_or_resident_values_before_gather() {
    assert_eq!(
        VARTYPE_INTEGER_AUDIT.kind,
        runmat_builtins::BuiltinIntegerAuditKind::NotApplicable
    );
    let Value::Object(selector) = block_on(vartype_builtin(Value::from("integer"))).unwrap() else {
        panic!("expected vartype selector");
    };
    assert_eq!(
        selector.properties().get("Type"),
        Some(&Value::from("integer"))
    );

    let allocation = unregistered_table_allocation(vec![1, 1]);
    let resident = Value::GpuTensor(allocation.handle.clone());
    for invalid in [Value::Int(IntValue::U64(u64::MAX)), resident] {
        let error = block_on(vartype_builtin(invalid)).expect_err("invalid vartype input");
        assert!(error
            .message()
            .contains("host string scalar or character vector"));
        assert!(!error.message().to_ascii_lowercase().contains("provider"));
    }
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn timerange_defaults_to_half_open_and_orders_wide_integer_row_times_exactly() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let base = 1_u64 << 53;
    let tt = block_on(timetable_builtin(vec![
        Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![base, base + 1, base + 2]),
                vec![3, 1],
            )
            .unwrap(),
        ),
        Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap()),
        Value::from("VariableNames"),
        Value::Cell(CellArray::new(vec![Value::from("X")], 1, 1).unwrap()),
    ]))
    .unwrap();
    let selector = block_on(timerange_builtin(vec![
        Value::Int(IntValue::U64(base + 1)),
        Value::Int(IntValue::U64(base + 2)),
    ]))
    .unwrap();
    let timetable = object(tt);
    let selected = object(
        table_paren_get(
            &timetable,
            &Value::Cell(CellArray::new(vec![selector, Value::from(":")], 1, 2).unwrap()),
        )
        .unwrap(),
    );
    match table_member_get(&selected, &Value::from("X")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![20.0]),
        other => panic!("expected exact half-open range result, got {other:?}"),
    }
    let closed = block_on(timerange_builtin(vec![
        Value::Int(IntValue::U64(base + 1)),
        Value::Int(IntValue::U64(base + 2)),
        Value::from(" CLOSED "),
    ]))
    .unwrap();
    assert_eq!(
        object(closed.clone()).properties().get("Inclusivity"),
        Some(&Value::from("closed"))
    );
    let selected = object(
        table_paren_get(
            &timetable,
            &Value::Cell(CellArray::new(vec![closed, Value::from(":")], 1, 2).unwrap()),
        )
        .unwrap(),
    );
    match table_member_get(&selected, &Value::from("X")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![20.0, 30.0]),
        other => panic!("expected normalized closed range result, got {other:?}"),
    }
}

#[test]
fn timetable_preallocation_and_default_conversion_preserve_integer_variables() {
    let tt = block_on(timetable_builtin(vec![
        Value::from("Size"),
        Value::Tensor(Tensor::new_integer(IntegerStorage::U16(vec![2, 1]), vec![1, 2]).unwrap()),
        Value::from("VariableTypes"),
        Value::StringArray(StringArray::new(vec!["uint64".into()], vec![1, 1]).unwrap()),
        Value::from("SampleRate"),
        Value::Int(IntValue::U16(2)),
        Value::from("VariableNames"),
        Value::Cell(CellArray::new(vec![Value::from("Count")], 1, 1).unwrap()),
    ]))
    .expect("preallocated timetable");
    let tt_object = object(tt.clone());
    let variables = table_variables(&tt_object).unwrap();
    let Value::Tensor(counts) = variables.fields.get("Count").unwrap() else {
        panic!("expected integer Count variable")
    };
    assert_eq!(counts.numeric_dtype(), runmat_value::NumericDType::U64);

    let table = object(block_on(timetable2table_builtin(tt, Vec::new())).unwrap());
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["Time".to_string(), "Count".to_string()]
    );
    let variables = table_variables(&table).unwrap();
    let Value::Tensor(counts) = variables.fields.get("Count").unwrap() else {
        panic!("expected converted integer Count variable")
    };
    assert_eq!(counts.numeric_dtype(), runmat_value::NumericDType::U64);
}

#[test]
fn timetable_extensions_gate_numeric_times_and_integer_conversion_flags() {
    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let numeric_times = block_on(timetable_builtin(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap()),
        ]))
        .expect_err("numeric row times must gate");
        assert_eq!(
            numeric_times.identifier(),
            TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION.error_identifier
        );
        let numeric_range = block_on(timerange_builtin(vec![
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U8(2)),
        ]))
        .expect_err("numeric timerange must gate");
        assert_eq!(
            numeric_range.identifier(),
            TIMERANGE_NUMERIC_BOUNDS_EXTENSION.error_identifier
        );
    }

    let row_times = array2timetable_duration_row_times(vec![0.0]);
    let tt = block_on(timetable_builtin(vec![
        row_times,
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
        ),
    ]))
    .unwrap();
    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(timetable2table_builtin(
            tt.clone(),
            vec![Value::from("ConvertRowTimes"), Value::Int(IntValue::U8(0))],
        ))
        .expect_err("typed integer option must gate");
        assert_eq!(
            error.identifier(),
            TIMETABLE2TABLE_INTEGER_OPTION_EXTENSION.error_identifier
        );
    }
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let table = object(
        block_on(timetable2table_builtin(
            tt,
            vec![Value::from("ConvertRowTimes"), Value::Int(IntValue::U8(0))],
        ))
        .unwrap(),
    );
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["Var1".to_string()]
    );
}

#[test]
fn pivot_builds_wide_summary_table() {
    let t = table_from_columns(
        vec!["Group".into(), "Kind".into(), "Value".into()],
        vec![
            Value::StringArray(
                StringArray::new(vec!["A".into(), "A".into(), "B".into()], vec![3, 1]).unwrap(),
            ),
            Value::StringArray(
                StringArray::new(vec!["x".into(), "y".into(), "x".into()], vec![3, 1]).unwrap(),
            ),
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
        ],
    )
    .unwrap();
    let out = block_on(pivot_builtin(
        None,
        t,
        Value::from("Group"),
        Value::from("Kind"),
        Value::from("Value"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let out_obj = object(out);
    let names = table_variable_names_from_object(&out_obj).unwrap();
    assert!(names.contains(&"x_Value".to_string()));
    assert!(names.contains(&"y_Value".to_string()));
    assert_eq!(table_height(&out_obj).unwrap(), 2);
}

#[test]
fn pivot_preserves_wide_integer_groups_and_rejects_lossy_integer_data() {
    let base = 9_007_199_254_740_992_u64;
    let groups = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![base + 1, base, base + 1]),
            vec![3, 1],
        )
        .unwrap(),
    );
    let kinds = Value::StringArray(
        StringArray::new(vec!["x".into(), "x".into(), "x".into()], vec![3, 1]).unwrap(),
    );
    let values = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
    let table = table_from_columns(
        vec!["Group".into(), "Kind".into(), "Value".into()],
        vec![groups.clone(), kinds.clone(), values],
    )
    .unwrap();
    let output = object(
        block_on(pivot_builtin(
            None,
            table,
            Value::from("Group"),
            Value::from("Kind"),
            Value::from("Value"),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    let Value::Tensor(group_output) = table_member_get(&output, &Value::from("Group")).unwrap()
    else {
        panic!("expected integer grouping column");
    };
    assert_eq!(
        group_output.integer_storage(),
        Some(&IntegerStorage::U64(vec![base + 1, base]))
    );
    let too_wide = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![base + 1, base, base + 2]),
            vec![3, 1],
        )
        .unwrap(),
    );
    let table = table_from_columns(
        vec!["Group".into(), "Kind".into(), "Value".into()],
        vec![groups, kinds, too_wide],
    )
    .unwrap();
    let error = block_on(pivot_builtin(
        None,
        table,
        Value::from("Group"),
        Value::from("Kind"),
        Value::from("Value"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("wide integer data must not round through double");
    assert!(error.to_string().contains("exactly representable"));
}

#[test]
fn table2struct_defaults_to_row_structs_and_to_scalar_preserves_columns() {
    let t = table_from_columns(
        vec!["A".into(), "B".into()],
        vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap()),
        ],
    )
    .unwrap();
    match block_on(table2struct_builtin(t.clone(), Vec::new())).unwrap() {
        Value::StructArray(array) => {
            assert_eq!(array.shape(), [2, 1]);
            assert_eq!(
                array
                    .field_names()
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>(),
                ["A", "B"]
            );
            assert_eq!(
                array.get_linear(0).unwrap().fields.get("A"),
                Some(&Value::Num(1.0))
            );
            assert_eq!(
                array.get_linear(1).unwrap().fields.get("B"),
                Some(&Value::Num(4.0))
            );
        }
        other => panic!("expected typed structure array, got {other:?}"),
    }
    match block_on(table2struct_builtin(
        t,
        vec![Value::from("ToScalar"), Value::Bool(true)],
    ))
    .unwrap()
    {
        Value::Struct(st) => assert!(st.fields.contains_key("A")),
        other => panic!("expected scalar struct, got {other:?}"),
    }
}

#[test]
fn categorical_categories_and_dictionary_lookup_have_semantics() {
    let categorical = block_on(categorical_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::StringArray(
                StringArray::new(vec!["b".into(), "a".into(), "c".into()], vec![3, 1]).unwrap(),
            ),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
            Value::from("Ordinal"),
            Value::Bool(true),
        ]),
    ))
    .unwrap();
    let Value::Object(cat) = categorical else {
        panic!("expected categorical");
    };
    match cat.properties().get("Codes").unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.materialize_f64()[0], 2.0);
            assert_eq!(tensor.materialize_f64()[1], 1.0);
            assert!(tensor.materialize_f64()[2].is_nan());
        }
        other => panic!("expected categorical codes, got {other:?}"),
    }

    let dictionary = block_on(dictionary_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap()),
        ]),
    ))
    .unwrap();
    let value = block_on(dictionary_subsref(
        None,
        dictionary,
        crate::object::indexing::standard_substruct_fixture_from_parts(
            OBJECT_INDEX_PAREN,
            Value::Cell(CellArray::new(vec![Value::from("b")], 1, 1).unwrap()),
        )
        .unwrap(),
    ))
    .unwrap();
    assert_eq!(value, Value::Num(20.0));
}

#[test]
fn uitable_data_property_preserves_wide_integer_class_and_values() {
    let data = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .expect("integer table data"),
    );
    let output =
        block_on(uitable_builtin(vec![Value::from("Data"), data])).expect("uitable construction");
    let Value::Object(object) = output else {
        panic!("expected uitable object");
    };
    let Value::Tensor(stored) = object.properties().get("Data").expect("Data property") else {
        panic!("expected numeric Data property");
    };
    assert_eq!(
        stored.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
    );
    assert_eq!(stored.shape(), &[1, 2]);
}
