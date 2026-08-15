#[path = "support/mod.rs"]
mod test_helpers;

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use runmat_builtins::{IntegerStorage, Value};
use test_helpers::execute_source;

fn temporary_csv(label: &str, contents: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "runmat-integer-import-{label}-{}-{nonce}.csv",
        std::process::id()
    ));
    fs::write(&path, contents).expect("write import fixture");
    path
}

fn source_path(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

fn contains_storage(values: &[Value], expected: &IntegerStorage) -> bool {
    values.iter().any(|value| match value {
        Value::Tensor(tensor) => tensor.integer_storage() == Some(expected),
        Value::Cell(cell) => contains_storage(&cell.data, expected),
        Value::Struct(structure) => contains_storage(
            &structure.fields.values().cloned().collect::<Vec<_>>(),
            expected,
        ),
        Value::Object(object) => contains_storage(
            &object.properties.values().cloned().collect::<Vec<_>>(),
            expected,
        ),
        _ => false,
    })
}

#[test]
fn compiled_documented_integer_import_forms_preserve_native_storage() {
    let path = temporary_csv("documented", "9007199254740993\n");
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let source = format!(
        "m=readmatrix('{}','OutputType','uint64'); c=textscan('9007199254740993 -128 FF 10','%u64 %d8 %xu8 %bu16'); a=c{{1}}; b=c{{2}}; h=c{{3}}; q=c{{4}};",
        source_path(&path)
    );
    let values = execute_source(&source).expect("compiled documented integer imports");
    assert!(contains_storage(
        &values,
        &IntegerStorage::U64(vec![9_007_199_254_740_993])
    ));
    assert!(contains_storage(&values, &IntegerStorage::I8(vec![-128])));
    assert!(contains_storage(&values, &IntegerStorage::U8(vec![255])));
    assert!(contains_storage(&values, &IntegerStorage::U16(vec![2])));
    fs::remove_file(path).expect("remove import fixture");
}

#[test]
fn compiled_import_extensions_reject_before_file_access_in_matlab_mode() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "m=readmatrix('definitely-missing.csv','NumHeaderLines',uint8(1));",
            "RunMat:compatibility:ReadmatrixTypedIntegerControlExtension",
        ),
        (
            "m=readmatrix('definitely-missing.csv','Like',uint16(0));",
            "RunMat:compatibility:ReadmatrixLikeOutputExtension",
        ),
        (
            "c=textscan('1','%f',uint8(1));",
            "RunMat:compatibility:TextscanTypedIntegerControlExtension",
        ),
        (
            "t=readtable('definitely-missing.csv','NumHeaderLines',uint8(1));",
            "RunMat:compatibility:ReadtableTypedIntegerControlExtension",
        ),
        (
            "t=readtimetable('definitely-missing.csv','Sheet',uint8(1));",
            "RunMat:compatibility:ReadtimetableTypedIntegerControlExtension",
        ),
        (
            "c=readcell('definitely-missing.csv','Range',uint8(1));",
            "RunMat:compatibility:ReadcellTypedIntegerControlExtension",
        ),
        (
            "o=spreadsheetImportOptions('DataRange',uint8(1));",
            "RunMat:compatibility:SpreadsheetImportOptionsTypedIntegerControlExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict import extension must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_runmat_import_extensions_preserve_selected_native_classes() {
    let path = temporary_csv("extensions", "header\n1\n");
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let source = format!(
        "m=readmatrix('{}','NumHeaderLines',uint8(1),'Like',uint16(0)); c=textscan(sprintf('header\\n2\\n'),'%u16','HeaderLines',uint8(1)); x=c{{1}};",
        source_path(&path)
    );
    let values = execute_source(&source).expect("compiled RunMat integer import extensions");
    assert!(contains_storage(&values, &IntegerStorage::U16(vec![1])));
    assert!(contains_storage(&values, &IntegerStorage::U16(vec![2])));
    fs::remove_file(path).expect("remove import fixture");
}
