use super::range::column_index_from_letters;
use super::workbook::{checked_zip_slice, find_zip_eocd};
use super::*;
use runmat_value::IntegerStorage;

#[test]
fn text_controls_require_effective_scalar_shape() {
    let paged = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
            .expect("paged character array"),
    );
    assert!(value_to_string_scalar(&paged).is_err());
    let nonscalar = Value::StringArray(
        runmat_value::StringArray::new(vec!["x".into(), "y".into()], vec![1, 1, 2])
            .expect("paged string array"),
    );
    assert!(value_to_string_scalar(&nonscalar).is_err());
    let trailing = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['x'], vec![1, 1, 1])
            .expect("trailing singleton character row"),
    );
    assert_eq!(value_to_string_scalar(&trailing).unwrap(), "x");
    let singleton = Value::StringArray(
        runmat_value::StringArray::new(vec!["x".into()], vec![1, 1, 1])
            .expect("trailing singleton string"),
    );
    assert_eq!(value_to_string_scalar(&singleton).unwrap(), "x");
}

#[test]
fn workbook_and_sheet_codec_failures_retain_typed_sources() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    use std::error::Error;

    let invalid = unique_path("xlsread_invalid_codec_source");
    fs::write(&invalid, b"not a spreadsheet").expect("fixture");
    let error = block_on(xlsread_builtin(
        None,
        Value::from(invalid.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("invalid workbook");
    assert_eq!(error.identifier(), XLSREAD_ERROR_IO.identifier);
    assert!(error.source().is_some(), "retain the Calamine open error");
    let _ = fs::remove_file(&invalid);

    let workbook = unique_path("xlsread_missing_sheet_source");
    write_minimal_xlsx(&workbook);
    let error = block_on(xlsread_builtin(
        None,
        Value::from(workbook.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("Absent")]),
    ))
    .expect_err("missing worksheet");
    assert_eq!(error.identifier(), XLSREAD_ERROR_SHEET.identifier);
    assert!(error.source().is_some(), "retain the Calamine sheet error");
    let _ = fs::remove_file(&workbook);
}

#[test]
fn typed_sheet_selector_does_not_clamp_uint64() {
    let selected = parse_sheet_selector(&Value::Int(IntValue::U64(u64::MAX)));
    match usize::try_from(u64::MAX)
        .ok()
        .and_then(|value| value.checked_sub(1))
    {
        Some(index) => {
            assert!(matches!(selected, Ok(SheetSelector::Index(actual)) if actual == index))
        }
        None => assert!(selected.is_err()),
    }
    assert!(parse_sheet_selector(&Value::Int(IntValue::I64(-1))).is_err());

    let tensor =
        Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("typed sheet tensor");
    assert!(matches!(
        parse_sheet_selector(&Value::Tensor(tensor)),
        Ok(SheetSelector::Index(6))
    ));
    let too_large = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
        .expect("typed sheet tensor");
    let selected = parse_sheet_selector(&Value::Tensor(too_large));
    match usize::try_from(u64::MAX)
        .ok()
        .and_then(|value| value.checked_sub(1))
    {
        Some(index) => {
            assert!(matches!(selected, Ok(SheetSelector::Index(actual)) if actual == index))
        }
        None => assert!(selected.is_err()),
    }
    let negative =
        Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).expect("negative");
    assert!(parse_sheet_selector(&Value::Tensor(negative)).is_err());

    assert!(parse_sheet_selector(&Value::Num(usize::MAX as f64)).is_err());
    assert!(parse_sheet_selector(&Value::Num((usize::MAX as f64) + 1.0)).is_err());
}

#[test]
fn xlsread_numeric_range_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::U16(vec![2, 3, 4, 5]), vec![1, 4])
        .expect("typed range tensor");

    let range = parse_range(&Value::Tensor(tensor)).expect("numeric range");
    assert_eq!(range.start_row, 1);
    assert_eq!(range.start_col, 2);
    assert_eq!(range.end_row, Some(3));
    assert_eq!(range.end_col, Some(4));

    let invalid = Tensor::new_integer(IntegerStorage::I16(vec![-1, 1]), vec![1, 2])
        .expect("typed invalid range");
    assert!(parse_range(&Value::Tensor(invalid)).is_err());

    let too_large = Tensor::new_integer(
        IntegerStorage::U64(vec![MAX_EXCEL_ROW_INDEX as u64 + 2, 1]),
        vec![1, 2],
    )
    .expect("typed excessive range");
    assert!(parse_range(&Value::Tensor(too_large)).is_err());

    let boundary = Value::Tensor(
        Tensor::new(vec![usize::MAX as f64, 1.0], vec![1, 2]).expect("boundary range"),
    );
    assert!(parse_range(&boundary).is_err());
}

#[test]
fn xlsread_gates_numeric_range_before_file_access() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let range =
        Tensor::new_integer(IntegerStorage::U16(vec![1, 1]), vec![1, 2]).expect("numeric range");
    let error = futures::executor::block_on(xlsread_builtin(
        None,
        Value::from("definitely-missing.xlsx"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Sheet1"),
            Value::Tensor(range),
        ]),
    ))
    .expect_err("strict mode rejects numeric ranges");
    assert_eq!(
        error.identifier(),
        XLSREAD_NUMERIC_RANGE_EXTENSION.error_identifier
    );
    drop(strict);
}
use futures::executor::block_on;
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique_path(stem: &str) -> PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{id}.xlsx"))
}

fn write_zip_file<W: Write + std::io::Seek>(
    zip: &mut zip::ZipWriter<W>,
    name: &str,
    contents: &str,
) {
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file(name, options).expect("start xlsx part");
    zip.write_all(contents.as_bytes()).expect("write xlsx part");
}

fn write_minimal_xlsx(path: &Path) {
    write_named_xlsx(path, "Data");
}

fn write_named_xlsx(path: &Path, sheet_name: &str) {
    let file = std::fs::File::create(path).expect("create xlsx");
    let mut zip = zip::ZipWriter::new(file);
    fill_named_xlsx(&mut zip, sheet_name);
    zip.finish().expect("finish xlsx");
}

fn in_memory_xlsx(sheet_name: &str) -> Vec<u8> {
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    fill_named_xlsx(&mut zip, sheet_name);
    zip.finish().expect("finish xlsx").into_inner()
}

fn fill_named_xlsx<W: Write + std::io::Seek>(zip: &mut zip::ZipWriter<W>, sheet_name: &str) {
    write_zip_file(
        zip,
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
        zip,
        "_rels/.rels",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#,
    );
    write_zip_file(
        zip,
        "xl/workbook.xml",
        &format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
<sheet name="{sheet_name}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"#
        ),
    );
    write_zip_file(
        zip,
        "xl/_rels/workbook.xml.rels",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"#,
    );
    write_zip_file(
        zip,
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
        zip,
        "xl/worksheets/sheet1.xml",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
<row r="1">
  <c r="A1" t="inlineStr"><is><t>Voltage</t></is></c>
  <c r="B1" t="inlineStr"><is><t>Current</t></is></c>
  <c r="C1" t="inlineStr"><is><t>Label</t></is></c>
</row>
<row r="2">
  <c r="A2"><v>1.5</v></c>
  <c r="B2"><v>10</v></c>
  <c r="C2" t="inlineStr"><is><t>low</t></is></c>
</row>
<row r="3">
  <c r="A3"><v>2.5</v></c>
  <c r="B3"><v>20</v></c>
  <c r="C3" t="inlineStr"><is><t>high</t></is></c>
</row>
<row r="4">
  <c r="A4"><v>3.5</v></c>
  <c r="B4"><v>30</v></c>
  <c r="C4" t="b"><v>1</v></c>
</row>
  </sheetData>
</worksheet>"#,
    );
}

fn direct_xlsread(path: &Path, args: Vec<Value>, outputs: Option<usize>) -> Value {
    let _guard = outputs
        .map(|count| crate::output_count::push_output_count(Some(count)))
        .transpose()
        .expect("output-count scope");
    block_on(xlsread_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("xlsread")
}

#[path = "tests/owner.rs"]
mod owner;

#[test]
fn xlsread_registers_public_descriptor() {
    let entry = runmat_builtins::builtin_catalog_entry_by_name("xlsread")
        .expect("xlsread canonical catalog entry");
    assert_eq!(entry.identity.name, "xlsread");
    let labels = XLSREAD_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert!(labels.contains(&"num = xlsread(filename)"));
    assert!(labels.contains(&"num = xlsread(filename, sheetOrRange)"));
    assert!(labels.contains(&"[num, txt, raw] = xlsread(___)"));
}

#[test]
fn xlsread_reads_numeric_range_as_default_output() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("xlsread_numeric_range");
    write_minimal_xlsx(&path);
    let value = direct_xlsread(&path, vec![Value::from("A2:B4")], None);
    match value {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[3, 2]);
            assert_eq!(
                tensor.materialize_f64(),
                vec![1.5, 2.5, 3.5, 10.0, 20.0, 30.0]
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    let _ = fs::remove_file(path);
}

#[test]
fn xlsread_accepts_sheet_and_range() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("xlsread_sheet_range");
    write_minimal_xlsx(&path);
    let value = direct_xlsread(&path, vec![Value::from("Data"), Value::from("B2:B3")], None);
    match value {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2, 1]);
            assert_eq!(tensor.materialize_f64(), vec![10.0, 20.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    let _ = fs::remove_file(path);
}

#[test]
fn string_sheet_selector_is_not_numeric_index() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("xlsread_numeric_sheet_name");
    write_named_xlsx(&path, "1");
    let value = direct_xlsread(&path, vec![Value::from("1"), Value::from("A2:A2")], None);
    match value {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[1, 1]);
            assert_eq!(tensor.materialize_f64(), vec![1.5]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    let _ = fs::remove_file(path);
}

#[test]
fn single_selector_prefers_sheet_name_when_not_valid_excel_range() {
    let mut request = XlsReadRequest::default();
    parse_single_selector(&Value::from("Data1"), &mut request).expect("selector");
    match request.sheet {
        Some(SheetSelector::Name(name)) => assert_eq!(name, "Data1"),
        other => panic!("expected sheet name, got {other:?}"),
    }
    assert!(request.range.is_none());
}

#[test]
fn single_cell_range_selects_one_cell() {
    let spec = parse_range_string("B3").expect("single-cell range");
    assert_eq!(spec.start_row, 2);
    assert_eq!(spec.start_col, 1);
    assert_eq!(spec.end_row, Some(2));
    assert_eq!(spec.end_col, Some(1));
}

#[test]
fn r1c1_range_is_parsed_explicitly() {
    let spec = parse_range_string("R2C3:R4C5").expect("r1c1 range");
    assert_eq!(spec.start_row, 1);
    assert_eq!(spec.start_col, 2);
    assert_eq!(spec.end_row, Some(3));
    assert_eq!(spec.end_col, Some(4));
}

#[test]
fn a1_columns_starting_with_r_and_c_do_not_parse_as_r1c1() {
    let spec = parse_range_string("RC1:RBC10").expect("a1 range");
    assert_eq!(spec.start_row, 0);
    assert_eq!(spec.start_col, column_index_from_letters("RC").unwrap());
    assert_eq!(spec.end_row, Some(9));
    assert_eq!(
        spec.end_col,
        Some(column_index_from_letters("RBC").unwrap())
    );
}

#[test]
fn invalid_strict_r1c1_indices_are_rejected() {
    let err = parse_range_string("R0C1").expect_err("zero r1c1 row");
    assert!(err.message().contains("must be one-based"));
}

#[test]
fn interleaved_a1_tokens_are_rejected() {
    let err = parse_range_string("A1B2").expect_err("invalid token");
    assert!(err.message().contains("invalid range component"));
}

#[test]
fn xlsread_zip_preflight_rejects_oversized_entry() {
    let err = validate_zip_uncompressed_limits([MAX_XLSREAD_ZIP_ENTRY_UNCOMPRESSED_BYTES + 1])
        .expect_err("oversized entry");
    assert!(err.message().contains("entry exceeds"));
}

#[test]
fn xlsread_zip_preflight_rejects_oversized_total() {
    let part = MAX_XLSREAD_ZIP_ENTRY_UNCOMPRESSED_BYTES;
    let err =
        validate_zip_uncompressed_limits([part, part, part, part, 1]).expect_err("oversized total");
    assert!(err.message().contains("expanded size exceeds"));
}

#[test]
fn xlsread_zip_preflight_rejects_excessive_entry_count() {
    let err = validate_zip_entry_count(MAX_XLSREAD_ZIP_ENTRIES + 1).expect_err("too many entries");
    assert!(err.message().contains("more than"));
}

#[test]
fn xlsread_zip_preflight_rejects_raw_zip64_excessive_entry_count() {
    let bytes = zip64_entry_count_fixture((MAX_XLSREAD_ZIP_ENTRIES + 1) as u64);
    let err = preflight_zip_workbook(&bytes).expect_err("too many zip64 entries");
    assert!(err.message().contains("more than"));
}

#[test]
fn xlsread_zip_preflight_rejects_missing_zip64_locator() {
    let mut bytes = b"PK\x03\x04".to_vec();
    bytes.extend_from_slice(b"PK\x05\x06");
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&u16::MAX.to_le_bytes());
    bytes.extend_from_slice(&u16::MAX.to_le_bytes());
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());

    let err = preflight_zip_workbook(&bytes).expect_err("missing zip64 locator");
    assert!(err.message().contains("missing locator"));
}

#[test]
fn xlsread_zip_preflight_rejects_out_of_bounds_zip64_eocd_offset() {
    let mut bytes = zip64_entry_count_fixture(1);
    let eocd_offset = find_zip_eocd(&bytes).expect("eocd");
    let locator_record_offset = eocd_offset - 20 + 8;
    bytes[locator_record_offset..locator_record_offset + 8]
        .copy_from_slice(&u64::MAX.to_le_bytes());

    let err = preflight_zip_workbook(&bytes).expect_err("out-of-bounds zip64 eocd");
    assert!(err.message().contains("truncated directory"));
}

#[test]
fn xlsread_zip_preflight_rejects_truncated_zip64_eocd_offset() {
    let mut bytes = zip64_entry_count_fixture(1);
    let eocd_offset = find_zip_eocd(&bytes).expect("eocd");
    let locator_record_offset = eocd_offset - 20 + 8;
    let truncated_offset = (bytes.len() - 2) as u64;
    bytes[locator_record_offset..locator_record_offset + 8]
        .copy_from_slice(&truncated_offset.to_le_bytes());

    let err = preflight_zip_workbook(&bytes).expect_err("truncated zip64 eocd");
    assert!(err.message().contains("truncated directory"));
}

#[test]
fn xlsread_zip_eocd_scan_ignores_signature_inside_comment() {
    let mut bytes = b"PK\x03\x04".to_vec();
    let eocd_offset = bytes.len();
    let comment = b"comment with PK\x05\x06 marker";
    bytes.extend_from_slice(b"PK\x05\x06");
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&(comment.len() as u16).to_le_bytes());
    bytes.extend_from_slice(comment);

    assert_eq!(find_zip_eocd(&bytes), Some(eocd_offset));
}

#[test]
fn xlsread_zip_checked_slice_rejects_offset_overflow() {
    let err = checked_zip_slice(b"PK\x03\x04", usize::MAX, 4).expect_err("overflowing offset");
    assert!(err.message().contains("truncated directory"));
}

fn zip64_entry_count_fixture(entry_count: u64) -> Vec<u8> {
    let mut bytes = b"PK\x03\x04".to_vec();
    let zip64_eocd_offset = bytes.len() as u64;
    bytes.extend_from_slice(b"PK\x06\x06");
    bytes.extend_from_slice(&44u64.to_le_bytes());
    bytes.extend_from_slice(&45u16.to_le_bytes());
    bytes.extend_from_slice(&45u16.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&entry_count.to_le_bytes());
    bytes.extend_from_slice(&entry_count.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(b"PK\x06\x07");
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&zip64_eocd_offset.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(b"PK\x05\x06");
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&u16::MAX.to_le_bytes());
    bytes.extend_from_slice(&u16::MAX.to_le_bytes());
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes
}

#[test]
fn explicit_range_preserves_raw_shape_with_blank_borders() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("xlsread_blank_border_range");
    write_minimal_xlsx(&path);
    let value = direct_xlsread(&path, vec![Value::from("B1:D5")], Some(3));
    let Value::OutputList(outputs) = value else {
        panic!("expected output list");
    };
    match &outputs[2] {
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 5);
            assert_eq!(cell.cols, 3);
            assert_eq!(cell.get(0, 0).unwrap(), Value::from("Current"));
            match cell.get(4, 2).unwrap() {
                Value::Num(value) => assert!(value.is_nan()),
                other => panic!("expected NaN, got {other:?}"),
            }
        }
        other => panic!("expected raw cell, got {other:?}"),
    }
    let _ = fs::remove_file(path);
}

#[test]
fn oversized_explicit_range_is_rejected_before_allocation() {
    let spec = parse_range_string("A1:XFD1048576").expect("maximum Excel range parses");
    let err = super::range::validate_selected_shape(
        spec.start_row as u32,
        spec.start_col as u32,
        spec.end_row.expect("end row") as u32,
        spec.end_col.expect("end column") as u32,
    )
    .expect_err("oversized range");
    assert!(err.message().contains("selected range"));
}

#[test]
fn xlsread_returns_text_and_raw_outputs() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("xlsread_multi_output");
    write_minimal_xlsx(&path);
    let value = direct_xlsread(&path, vec![Value::from("A1:C3")], Some(3));
    let Value::OutputList(outputs) = value else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 3);
    match &outputs[0] {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2, 2]);
            assert_eq!(tensor.materialize_f64(), vec![1.5, 2.5, 10.0, 20.0]);
        }
        other => panic!("expected numeric tensor, got {other:?}"),
    }
    match &outputs[1] {
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 3);
            assert_eq!(cell.cols, 3);
            assert_eq!(cell.get(0, 0).unwrap(), Value::from("Voltage"));
            assert_eq!(cell.get(2, 2).unwrap(), Value::from("high"));
        }
        other => panic!("expected text cell, got {other:?}"),
    }
    match &outputs[2] {
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 3);
            assert_eq!(cell.cols, 3);
            assert_eq!(cell.get(0, 1).unwrap(), Value::from("Current"));
            assert_eq!(cell.get(1, 0).unwrap(), Value::Num(1.5));
            assert_eq!(cell.get(2, 2).unwrap(), Value::from("high"));
        }
        other => panic!("expected raw cell, got {other:?}"),
    }
    let _ = fs::remove_file(path);
}

#[test]
fn xlsread_rejects_too_many_outputs() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("xlsread_too_many_outputs");
    write_minimal_xlsx(&path);
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let err = block_on(xlsread_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("too many outputs");
    assert!(err.message().contains("at most 3 outputs"));
    let _ = fs::remove_file(path);
}
