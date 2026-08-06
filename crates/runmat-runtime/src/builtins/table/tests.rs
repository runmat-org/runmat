use super::*;
use crate::RuntimeError;
#[cfg(not(target_arch = "wasm32"))]
use async_trait::async_trait;
use futures::executor::block_on;
use runmat_builtins::IntegerStorage;
#[cfg(not(target_arch = "wasm32"))]
use runmat_filesystem::{
    DirEntry, FileHandle, FsMetadata, FsProvider, NativeFsProvider, OpenFlags, SandboxFsProvider,
};
use runmat_time::unix_timestamp_ms;
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::io;
use std::io::Write;

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
        Value::from(path.to_string_lossy().to_string()),
        args,
    ))
    .expect("readtable")
}

fn read_table_err(path: &Path, args: Vec<Value>) -> RuntimeError {
    block_on(readtable_builtin(
        Value::from(path.to_string_lossy().to_string()),
        args,
    ))
    .expect_err("expected readtable failure")
}

fn spreadsheet_options(args: Vec<Value>) -> StructValue {
    match block_on(spreadsheet_import_options_builtin(args)).expect("spreadsheetImportOptions") {
        Value::Struct(options) => options,
        other => panic!("expected struct options, got {other:?}"),
    }
}

fn detect_options(path: &Path, args: Vec<Value>) -> StructValue {
    match block_on(detect_import_options_builtin(
        Value::from(path.to_string_lossy().to_string()),
        args,
    ))
    .expect("detectImportOptions")
    {
        Value::Struct(options) => options,
        other => panic!("expected struct options, got {other:?}"),
    }
}

fn char_row(array: &CharArray, row: usize) -> String {
    let start = row * array.cols;
    array.data[start..start + array.cols].iter().collect()
}

fn object(value: Value) -> ObjectInstance {
    match value {
        Value::Object(object) => object,
        other => panic!("expected table object, got {other:?}"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn write_sample_parquet(path: &Path) {
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
                Some("Ada"),
                Some("Grace"),
                Some("Linus"),
            ])),
            Arc::new(Float64Array::from(vec![Some(10.0), Some(12.5), None])),
            Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])),
            Arc::new(Int32Array::from(vec![Some(1), Some(1), Some(2)])),
        ],
    )
    .expect("create parquet batch");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
    fs::write(path, bytes).expect("write parquet sample");
}

#[cfg(not(target_arch = "wasm32"))]
fn write_integer_parquet(path: &Path) {
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
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
    fs::write(path, bytes).expect("write parquet integer sample");
}

#[test]
fn readtable_imports_headered_numeric_and_text_columns() {
    let path = unique_path("readtable_basic");
    fs::write(&path, "Name,Score\nAda,10\nGrace,12\n").expect("write sample");
    let table = object(read_table(&path, Vec::new()));
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["Name".to_string(), "Score".to_string()]
    );
    match table_member_get(&table, &Value::from("Score")).unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![2, 1]);
            assert_eq!(tensor.materialize_f64(), vec![10.0, 12.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    match table_member_get(&table, &Value::from("Name")).unwrap() {
        Value::StringArray(array) => {
            assert_eq!(array.data, vec!["Ada".to_string(), "Grace".to_string()]);
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
        block_on(parquetread_builtin(
            Value::from(path.to_string_lossy().to_string()),
            Vec::new(),
        ))
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
                array.data,
                vec!["Ada".to_string(), "Grace".to_string(), "Linus".to_string()]
            );
        }
        other => panic!("expected string column, got {other:?}"),
    }
    match table_member_get(&table, &Value::from("Score")).unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![3, 1]);
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
            assert_eq!(tensor.shape, vec![3, 1]);
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
    let path = unique_path("parquetread_integer_columns").with_extension("parquet");
    write_integer_parquet(&path);

    let table = object(
        block_on(parquetread_builtin(
            Value::from(path.to_string_lossy().to_string()),
            Vec::new(),
        ))
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
        let Value::Tensor(column) = table_member_get(&table, &Value::from(name)).unwrap() else {
            panic!("expected tensor column for {name}");
        };
        assert_eq!(column.shape, vec![2, 1], "{name}");
        assert_eq!(column.integer_storage(), Some(&expected), "{name}");
    }

    let _ = fs::remove_file(&path);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn parquetread_supports_selected_variables_and_row_groups() {
    let path = unique_path("parquetread_projection").with_extension("parquet");
    write_sample_parquet(&path);

    let selected = StringArray::new(vec!["Group".to_string(), "Score".to_string()], vec![1, 2])
        .expect("selected variables");
    let table = object(
        block_on(parquetread_builtin(
            Value::from(path.to_string_lossy().to_string()),
            vec![
                Value::from("SelectedVariableNames"),
                Value::StringArray(selected),
                Value::from("RowGroups"),
                Value::Num(2.0),
            ],
        ))
        .expect("parquetread selected"),
    );
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["Group".to_string(), "Score".to_string()]
    );
    match table_member_get(&table, &Value::from("Group")).unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 1]);
            assert_eq!(
                tensor.integer_storage(),
                Some(&IntegerStorage::I32(vec![2]))
            );
        }
        other => panic!("expected tensor column, got {other:?}"),
    }

    let _ = fs::remove_file(&path);
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
            array.data,
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
fn parquetread_rejects_rowfilter_until_predicates_are_available() {
    let path = unique_path("parquetread_rowfilter").with_extension("parquet");
    write_sample_parquet(&path);

    let err = block_on(parquetread_builtin(
        Value::from(path.to_string_lossy().to_string()),
        vec![Value::from("RowFilter"), Value::Bool(true)],
    ))
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
            array.data,
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
    fs::write(&path, "daily revenue,total orders\n100,10\n").expect("write sample");
    let table = object(read_table(
        &path,
        vec![Value::from("VariableNamingRule"), Value::from("preserve")],
    ));
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["daily revenue".to_string(), "total orders".to_string()]
    );
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
    assert!(runmat_builtins::builtin_function_by_name("spreadsheetImportOptions").is_some());
    let labels = SPREADSHEET_IMPORT_OPTIONS_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert!(labels.contains(&"opts = spreadsheetImportOptions()"));
    assert!(labels.contains(&"opts = spreadsheetImportOptions(nameValuePairs...)"));
}

#[test]
fn detect_import_options_registers_public_descriptor() {
    assert!(runmat_builtins::builtin_function_by_name("detectImportOptions").is_some());
    let labels = DETECT_IMPORT_OPTIONS_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert!(labels.contains(&"opts = detectImportOptions(filename)"));
    assert!(labels.contains(&"opts = detectImportOptions(filename, nameValuePairs...)"));
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
            array.data,
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
            array.data,
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
            Value::from(path.to_string_lossy().to_string()),
            vec![Value::Struct(options)],
        ),
    )
    .expect("readmatrix");
    match matrix {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![2, 2]);
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
            assert_eq!(array.data, vec!["A".to_string(), "B".to_string()])
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
    fs::write(&path, "Row,Name,Score\nr1,Ada,10\nr2,Grace,12\n").expect("write sample");
    let options = detect_options(&path, vec![Value::from("ReadRowNames"), Value::Bool(true)]);
    assert_eq!(options.fields.get("NumVariables"), Some(&Value::Num(2.0)));
    match options.fields.get("VariableNames").unwrap() {
        Value::StringArray(array) => assert_eq!(
            array.data,
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
            assert_eq!(array.data, vec!["r1".to_string(), "r2".to_string()])
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
            Value::from(path.to_string_lossy().to_string()),
            vec![Value::Struct(options)],
        ),
    )
    .expect("readmatrix");
    match matrix {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![2, 2]);
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
                Value::from("/provider_numeric.csv"),
                vec![Value::Struct(matrix_options)],
            ),
        )
        .expect("readmatrix");
        match matrix {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
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
            assert_eq!(array.data, vec!["Raw A".to_string(), "Raw B".to_string()])
        }
        other => panic!("expected string array, got {other:?}"),
    }
    match options.fields.get("VariableTypes").unwrap() {
        Value::StringArray(array) => {
            assert_eq!(
                array.data,
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
            assert_eq!(array.data, vec!["Var1".to_string(), "Var2".to_string()]);
            assert_eq!(array.shape, vec![1, 2]);
        }
        other => panic!("expected string array, got {other:?}"),
    }
    match options.fields.get("VariableTypes").unwrap() {
        Value::StringArray(array) => {
            assert_eq!(array.data, vec!["double".to_string(), "string".to_string()]);
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
    options.insert("Sheet", Value::from("Data"));
    options.insert("DataRange", Value::from("C2:C3"));
    options.insert(
        "VariableNames",
        Value::StringArray(StringArray::new(vec!["Amount".into()], vec![1, 1]).unwrap()),
    );
    options.insert(
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
            assert_eq!(tensor.shape, vec![2, 1]);
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
        Value::Object(object) => assert!(object.is_class("datetime")),
        other => panic!("expected datetime object, got {other:?}"),
    }
    match table_member_get(&table, &Value::from("Elapsed")).unwrap() {
        Value::Object(object) => assert!(object.is_class("duration")),
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
    fs::write(&path, "Name\nAda\nGrace\n").expect("write sample");
    let table = object(read_table(
        &path,
        vec![Value::from("TextType"), Value::from("char")],
    ));
    match table_member_get(&table, &Value::from("Name")).unwrap() {
        Value::CharArray(array) => {
            assert_eq!(array.rows, 2);
            assert_eq!(array.cols, 5);
            assert_eq!(char_row(&array, 0), "Ada  ");
            assert_eq!(char_row(&array, 1), "Grace");
        }
        other => panic!("expected char array, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[test]
fn readtable_variable_types_cellstr_imports_cell_column() {
    let path = unique_path("readtable_variable_types_cellstr");
    fs::write(&path, "Name\nAda\nGrace\n").expect("write sample");
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
                Value::CharArray(CharArray::new_row("Ada"))
            );
            assert_eq!(
                cell.get(1, 0).unwrap(),
                Value::CharArray(CharArray::new_row("Grace"))
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
    props.insert(
        VARIABLE_NAMES,
        Value::StringArray(StringArray::new(vec!["X".into(), "Y".into()], vec![1, 2]).unwrap()),
    );
    table_member_set(&mut table, PROPERTIES_MEMBER, Value::Struct(props)).unwrap();
    assert_eq!(
        table_variable_names_from_object(&table).unwrap(),
        vec!["X".to_string(), "Y".to_string()]
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
        Value::StringArray(array) => assert_eq!(array.data, vec!["r3", "r1"]),
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
            assert_eq!(array.data, vec!["first".to_string(), "second".to_string()]);
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
    let _guard = crate::output_count::push_output_count(Some(3));
    let result = grpstats_impl(x, group, vec![whichstats]).unwrap();
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 3);
    match &outputs[0] {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![2, 2]);
            assert_eq!(tensor.materialize_f64(), vec![3.5, 1.5, 35.0, 15.0]);
        }
        other => panic!("expected mean tensor, got {other:?}"),
    }
    match &outputs[1] {
        Value::Tensor(tensor) => {
            let root_half = 0.5_f64.sqrt();
            assert_eq!(tensor.shape, vec![2, 2]);
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
            assert_eq!(cell.data, vec![Value::from("1"), Value::from("2")]);
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
            assert_eq!(tensor.shape, vec![1, 1]);
            assert_eq!(tensor.materialize_f64(), vec![2.5]);
        }
        other => panic!("expected all-group mean tensor, got {other:?}"),
    }

    let group = Value::Tensor(Tensor::new(vec![f64::NAN, 2.0, 1.0, 2.0], vec![4, 1]).unwrap());
    let names = grpstats_impl(x.clone(), group.clone(), vec![Value::from("gname")]).unwrap();
    match names {
        Value::Cell(cell) => assert_eq!(cell.data, vec![Value::from("1"), Value::from("2")]),
        other => panic!("expected group names, got {other:?}"),
    }
    let means = grpstats_impl(x.clone(), group, vec![Value::from("mean")]).unwrap();
    match means {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0, 3.5]),
        other => panic!("expected means, got {other:?}"),
    }

    let _guard = crate::output_count::push_output_count(Some(1));
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
    let _guard = crate::output_count::push_output_count(Some(2));
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
        Value::Cell(cell) => assert_eq!(cell.data, vec![Value::from("b"), Value::from("a")]),
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
        Value::StringArray(array) => assert_eq!(array.data, vec!["b".to_string(), "a".to_string()]),
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
            assert_eq!(tensor.shape, vec![1, 2]);
            assert!(tensor.materialize_f64()[0] < 4.0);
            assert!(tensor.materialize_f64()[1] > 4.0);
        }
        other => panic!("expected interval tensor, got {other:?}"),
    }
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
            assert_eq!(column.shape, vec![2, 1]);
            assert_eq!(
                column.integer_storage().unwrap().numeric_dtype(),
                storage.numeric_dtype(),
                "{name}"
            );
        }
        let Value::Tensor(round_trip) = block_on(table2array_builtin(table)).unwrap() else {
            panic!("expected typed integer array");
        };
        assert_eq!(round_trip.shape, vec![2, 2]);
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
            block_on(timetable2table_builtin(timetable, Vec::new())).unwrap(),
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
        Tensor::new(vec![739_282.0], vec![1, 1]).unwrap(),
        "yyyy-MM-dd".to_string(),
    )
    .unwrap();
    let mut step = ObjectInstance::new("calendarDuration".to_string());
    step.properties.insert(
        "__months".to_string(),
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
    );
    step.properties.insert(
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
                    block_on(timetable2table_builtin(timetable, Vec::new())).unwrap(),
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
            block_on(timetable2table_builtin(timetable, Vec::new())).unwrap(),
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
    let duplicate = block_on(table_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap()),
        Value::from("VariableNames"),
        Value::Cell(CellArray::new(vec![Value::from("A"), Value::from("A")], 1, 2).unwrap()),
    ]));
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
}

#[test]
fn timetable_conversion_predicates_and_head_work() {
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
    let first_two = block_on(head_builtin(timetable.clone(), vec![Value::Num(2.0)])).unwrap();
    let first_two_object = object(first_two);
    assert_eq!(first_two_object.class_name, TIMETABLE_CLASS);
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
    assert_eq!(converted_object.class_name, TIMETABLE_CLASS);
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
fn categorical_dictionary_and_selector_objects_materialize() {
    let categorical = block_on(categorical_builtin(vec![Value::StringArray(
        StringArray::new(vec!["red".into(), "blue".into(), "red".into()], vec![3, 1]).unwrap(),
    )]))
    .unwrap();
    assert!(matches!(
        block_on(iscategorical_builtin(categorical.clone())).unwrap(),
        Value::Bool(true)
    ));
    let Value::Object(cat) = categorical else {
        panic!("expected categorical object");
    };
    match cat.properties.get("Categories").unwrap() {
        Value::StringArray(array) => assert_eq!(array.data, vec!["blue", "red"]),
        other => panic!("expected categories, got {other:?}"),
    }

    let dictionary = block_on(dictionary_builtin(vec![
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
    ]))
    .unwrap();
    assert!(matches!(dictionary, Value::Object(obj) if obj.class_name == DICTIONARY_CLASS));
    assert!(matches!(
        block_on(timerange_builtin(vec![Value::Num(1.0), Value::Num(3.0)])).unwrap(),
        Value::Object(obj) if obj.class_name == TIMERANGE_CLASS
    ));
    assert!(matches!(
        block_on(vartype_builtin(Value::from("numeric"))).unwrap(),
        Value::Object(obj) if obj.class_name == VARTYPE_CLASS
    ));
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
        assert_eq!(datastore.class_name, ARRAY_DATASTORE_CLASS);
        let Some(Value::Tensor(data)) = datastore.properties.get("Data") else {
            panic!("expected typed datastore data");
        };
        assert_eq!(data.integer_storage(), Some(&storage));
        assert_eq!(datastore.properties.get("ReadSize"), Some(&Value::Num(2.0)));
        assert_eq!(
            datastore.properties.get("IterationDimension"),
            Some(&Value::Num(1.0))
        );
        assert_eq!(
            datastore.properties.get("OutputType"),
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
                let Some(Value::Tensor(data)) = datastore.properties.get("Data") else {
                    panic!("expected gathered typed data");
                };
                assert_eq!(data.integer_storage(), Some(&storage));
            }
        }
    });
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
        let Some(Value::Tensor(data)) = datastore.properties.get("Data") else {
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
    assert_eq!(datastore.class_name, FILE_DATASTORE_CLASS);
    assert_eq!(datastore.properties.get("Files"), Some(&files));
    assert_eq!(
        datastore.properties.get("ReadFcn"),
        Some(&Value::FunctionHandle("readlines".into()))
    );
    assert_eq!(
        datastore.properties.get("FileExtensions"),
        Some(&extensions)
    );
    assert_eq!(
        datastore.properties.get("IncludeSubfolders"),
        Some(&Value::Bool(true))
    );
    assert_eq!(
        datastore.properties.get("ReadMode"),
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
fn ordinal_constructor_sets_ordered_categorical_semantics() {
    let default_ordered = block_on(ordinal_builtin(vec![Value::StringArray(
        StringArray::new(
            vec!["medium".into(), "low".into(), "high".into()],
            vec![3, 1],
        )
        .unwrap(),
    )]))
    .unwrap();
    let Value::Object(default_cat) = default_ordered else {
        panic!("expected default ordinal categorical object");
    };
    match default_cat.properties.get("Categories").unwrap() {
        Value::StringArray(array) => assert_eq!(array.data, vec!["high", "low", "medium"]),
        other => panic!("expected sorted categories, got {other:?}"),
    }

    let ordinal = block_on(ordinal_builtin(vec![
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
    ]))
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
    let Value::Object(max_cat) = max_value else {
        panic!("expected categorical max result");
    };
    match max_cat.properties.get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0]),
        other => panic!("expected categorical max code, got {other:?}"),
    }
    let min_value = block_on(crate::call_builtin_async(
        "min",
        std::slice::from_ref(&ordinal),
    ))
    .unwrap();
    let Value::Object(min_cat) = min_value else {
        panic!("expected categorical min result");
    };
    match min_cat.properties.get("Codes").unwrap() {
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
    let Value::Object(all_max_cat) = all_max else {
        panic!("expected categorical all max result");
    };
    match all_max_cat.properties.get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0]),
        other => panic!("expected categorical all max code, got {other:?}"),
    }

    let Value::Object(cat) = ordinal else {
        panic!("expected ordinal categorical object");
    };
    assert_eq!(cat.properties.get("Ordinal"), Some(&Value::Bool(true)));
    match cat.properties.get("Categories").unwrap() {
        Value::StringArray(array) => {
            assert_eq!(array.data, vec!["low", "medium", "high"]);
            assert_eq!(array.shape, vec![1, 3]);
        }
        other => panic!("expected categories, got {other:?}"),
    }
    match cat.properties.get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![2.0, 1.0, 3.0]),
        other => panic!("expected categorical codes, got {other:?}"),
    }

    let renamed = block_on(categorical_builtin(vec![
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
    ]))
    .unwrap();
    assert!(matches!(
        block_on(isordinal_builtin(renamed.clone())).unwrap(),
        Value::Bool(true)
    ));
    let Value::Object(renamed_cat) = renamed else {
        panic!("expected renamed categorical");
    };
    match renamed_cat.properties.get("Categories").unwrap() {
        Value::StringArray(array) => assert_eq!(array.data, vec!["child", "adult", "senior"]),
        other => panic!("expected renamed categories, got {other:?}"),
    }
    match renamed_cat.properties.get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![3.0, 1.0, 2.0]),
        other => panic!("expected renamed codes, got {other:?}"),
    }

    let nominal = block_on(categorical_builtin(vec![Value::StringArray(
        StringArray::new(vec!["low".into(), "high".into()], vec![2, 1]).unwrap(),
    )]))
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
fn categorical_compatibility_edges_match_matlab_surface() {
    let option_word_category = block_on(categorical_builtin(vec![
        Value::from("Ordinal"),
        Value::from("Ordinal"),
    ]))
    .unwrap();
    let Value::Object(option_word_category) = option_word_category else {
        panic!("expected option-word categorical");
    };
    match option_word_category.properties.get("Categories").unwrap() {
        Value::StringArray(array) => assert_eq!(array.data, vec!["Ordinal"]),
        other => panic!("expected option-word categories, got {other:?}"),
    }
    match option_word_category.properties.get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0]),
        other => panic!("expected option-word codes, got {other:?}"),
    }

    let whitespace = block_on(categorical_builtin(vec![Value::StringArray(
        StringArray::new(vec!["a".into(), " a".into(), "a ".into()], vec![3, 1]).unwrap(),
    )]))
    .unwrap();
    let Value::Object(whitespace) = whitespace else {
        panic!("expected whitespace categorical");
    };
    match whitespace.properties.get("Categories").unwrap() {
        Value::StringArray(array) => assert_eq!(array.data, vec![" a", "a", "a "]),
        other => panic!("expected whitespace categories, got {other:?}"),
    }

    let duplicate_valueset = block_on(categorical_builtin(vec![
        Value::from("x"),
        Value::StringArray(StringArray::new(vec!["x".into(), "x".into()], vec![1, 2]).unwrap()),
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
    ]));
    assert!(duplicate_valueset.is_err());

    let low_high = block_on(ordinal_builtin(vec![
        Value::from("low"),
        Value::StringArray(
            StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
        ),
    ]))
    .unwrap();
    let high_low = block_on(ordinal_builtin(vec![
        Value::from("low"),
        Value::StringArray(
            StringArray::new(vec!["high".into(), "low".into()], vec![1, 2]).unwrap(),
        ),
    ]))
    .unwrap();
    assert!(block_on(crate::call_builtin_async(
        "lt",
        &[low_high.clone(), high_low]
    ))
    .is_err());

    let extra_category = block_on(categorical_builtin(vec![
        Value::from("low"),
        Value::StringArray(
            StringArray::new(
                vec!["low".into(), "medium".into(), "high".into()],
                vec![1, 3],
            )
            .unwrap(),
        ),
    ]))
    .unwrap();
    assert!(block_on(crate::call_builtin_async("eq", &[low_high, extra_category])).is_err());
}

#[test]
fn ordinal_min_max_preserve_categorical_results_for_missing_dims_and_indices() {
    let ordinal_with_missing = block_on(ordinal_builtin(vec![
        Value::StringArray(StringArray::new(vec!["x".into(), "z".into()], vec![2, 1]).unwrap()),
        Value::StringArray(StringArray::new(vec!["x".into(), "y".into()], vec![1, 2]).unwrap()),
    ]))
    .unwrap();
    let default_max = block_on(crate::call_builtin_async(
        "max",
        std::slice::from_ref(&ordinal_with_missing),
    ))
    .unwrap();
    let Value::Object(default_max) = default_max else {
        panic!("expected categorical max with missing");
    };
    match default_max.properties.get("Codes").unwrap() {
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
    let Value::Object(omitnan_max) = omitnan_max else {
        panic!("expected omitnan categorical max");
    };
    match omitnan_max.properties.get("Codes").unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0]),
        other => panic!("expected omitnan max code, got {other:?}"),
    }

    let two_outputs = block_on(crate::call_builtin_async_with_outputs(
        "max",
        std::slice::from_ref(&ordinal_with_missing),
        2,
    ))
    .unwrap();
    match two_outputs {
        Value::OutputList(values) => {
            assert_eq!(values.len(), 2);
            assert!(matches!(values[0], Value::Object(_)));
            match &values[1] {
                Value::Num(index) => assert_eq!(*index, 2.0),
                other => panic!("expected numeric index, got {other:?}"),
            }
        }
        other => panic!("expected two-output list, got {other:?}"),
    }

    let matrix = block_on(ordinal_builtin(vec![
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
    ]))
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
    let Value::Object(row_max) = row_max else {
        panic!("expected categorical row max");
    };
    match row_max.properties.get("Codes").unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![2, 1]);
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
                StringArray::new(vec!["Ada".into(), "Grace".into()], vec![2, 1]).unwrap(),
            ),
        ],
    )
    .unwrap();
    let bytes = block_on(writetable_builtin(
        table,
        vec![Value::from(path.to_string_lossy().to_string())],
    ))
    .unwrap();
    assert!(matches!(bytes, Value::Num(n) if n > 0.0));
    let cells = block_on(readcell_builtin(
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
            assert_eq!(cell.get(1, 1).unwrap(), Value::from("Ada"));
        }
        other => panic!("expected cell array, got {other:?}"),
    }
    let _ = fs::remove_file(path);
}

#[test]
fn timetable_rowtimes_option_keeps_all_variables_and_table2timetable_does_not_drop_data() {
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
    assert_eq!(ranged.class_name, TIMETABLE_CLASS);
    assert_eq!(table_height(&ranged).unwrap(), 2);

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
        Tensor::new(vec![100.0, 200.0, 300.0], vec![3, 1]).unwrap(),
        "yyyy-MM-dd",
    )
    .unwrap();
    let datetime_selector = crate::builtins::datetime::datetime_object_from_serial_tensor(
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
    assert_eq!(selected.class_name, TIMETABLE_CLASS);
    match table_member_get(&selected, &Value::from("X")).unwrap() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![20.0]),
        other => panic!("expected datetime row-time selection, got {other:?}"),
    }
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
        t,
        Value::from("Group"),
        Value::from("Kind"),
        Value::from("Value"),
        Vec::new(),
    ))
    .unwrap();
    let out_obj = object(out);
    let names = table_variable_names_from_object(&out_obj).unwrap();
    assert!(names.contains(&"x_Value".to_string()));
    assert!(names.contains(&"y_Value".to_string()));
    assert_eq!(table_height(&out_obj).unwrap(), 2);
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
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 2);
            assert!(matches!(cell.data[0], Value::Struct(_)));
        }
        other => panic!("expected struct array cell, got {other:?}"),
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
    let categorical = block_on(categorical_builtin(vec![
        Value::StringArray(
            StringArray::new(vec!["b".into(), "a".into(), "c".into()], vec![3, 1]).unwrap(),
        ),
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        Value::from("Ordinal"),
        Value::Bool(true),
    ]))
    .unwrap();
    let Value::Object(cat) = categorical else {
        panic!("expected categorical");
    };
    match cat.properties.get("Codes").unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.materialize_f64()[0], 2.0);
            assert_eq!(tensor.materialize_f64()[1], 1.0);
            assert!(tensor.materialize_f64()[2].is_nan());
        }
        other => panic!("expected categorical codes, got {other:?}"),
    }

    let dictionary = block_on(dictionary_builtin(vec![
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap()),
    ]))
    .unwrap();
    let value = block_on(dictionary_subsref(
        dictionary,
        OBJECT_INDEX_PAREN.to_string(),
        Value::from("b"),
    ))
    .unwrap();
    assert_eq!(value, Value::Num(20.0));
}
