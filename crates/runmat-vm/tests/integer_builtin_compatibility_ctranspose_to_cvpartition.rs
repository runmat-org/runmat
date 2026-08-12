#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::{IntegerStorage, Value};
use test_helpers::execute_source;

#[test]
fn compiled_ctranspose_and_datasample_preserve_integer_storage() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let values = execute_source(
        "rng(1); b = datasample(ctranspose(uint64([1 2])), uint8(2), uint8(1), 'Replace', false);",
    )
    .expect("compiled structural integer operations");
    assert!(values.iter().any(|value| {
        let Value::Tensor(tensor) = value else {
            return false;
        };
        let Some(IntegerStorage::U64(storage)) = tensor.integer_storage() else {
            return false;
        };
        storage.len() == 2 && storage.contains(&1) && storage.contains(&2)
    }));
}

#[test]
fn compiled_cumtrapz_and_cvpartition_integer_extensions_obey_mode() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "q = cumtrapz(uint8([1 2 3]));",
            "RunMat:compatibility:CumtrapzIntegerYExtension",
        ),
        (
            "c = cvpartition(uint8(6), 'KFold', 3);",
            "RunMat:compatibility:CvpartitionIntegerObservationCountExtension",
        ),
        (
            "y = datasample(uint8([1 2 3]), 2);",
            "RunMat:compatibility:DatasampleIntegerDataExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("MATLAB mode integer gate");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_graphics_metadata_integer_forms_execute() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let values = execute_source(
        "daspect(uint16([1 2 3])); r = daspect(); row = dataTipTextRow('Value', uint64([1 2])); v = row.Value;",
    )
    .expect("compiled graphics metadata integer forms");
    assert!(values.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.materialize_f64() == vec![1.0, 2.0, 3.0])
    }));
    assert!(values.iter().any(|value| {
        matches!(
            value,
            Value::Tensor(tensor)
                if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![1, 2]))
        )
    }));
}

#[test]
fn compiled_csv_extension_gates_precede_file_access() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let read_error =
        execute_source("x = csvread('__runmat_integer_missing__.csv', 0, 0, uint8([0 0]));")
            .expect_err("two-vector range gate");
    assert_eq!(
        read_error.identifier(),
        Some("RunMat:compatibility:CsvreadTwoVectorRangeExtension")
    );

    let write_error =
        execute_source("bytes = csvwrite('__runmat_integer_never_written__.csv', uint8([1 2]));")
            .expect_err("bytes-written gate");
    assert_eq!(
        write_error.identifier(),
        Some("RunMat:compatibility:CsvwriteBytesOutputExtension")
    );
}
