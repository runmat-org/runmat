#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::{IntegerStorage, Value};
use test_helpers::execute_source;

#[test]
fn compiled_diag_and_diff_preserve_every_integer_class_exactly() {
    for (constructor, diagonal, difference) in [
        (
            "int8",
            IntegerStorage::I8(vec![1, 0, 0, 3]),
            IntegerStorage::I8(vec![2]),
        ),
        (
            "int16",
            IntegerStorage::I16(vec![1, 0, 0, 3]),
            IntegerStorage::I16(vec![2]),
        ),
        (
            "int32",
            IntegerStorage::I32(vec![1, 0, 0, 3]),
            IntegerStorage::I32(vec![2]),
        ),
        (
            "int64",
            IntegerStorage::I64(vec![1, 0, 0, 3]),
            IntegerStorage::I64(vec![2]),
        ),
        (
            "uint8",
            IntegerStorage::U8(vec![1, 0, 0, 3]),
            IntegerStorage::U8(vec![2]),
        ),
        (
            "uint16",
            IntegerStorage::U16(vec![1, 0, 0, 3]),
            IntegerStorage::U16(vec![2]),
        ),
        (
            "uint32",
            IntegerStorage::U32(vec![1, 0, 0, 3]),
            IntegerStorage::U32(vec![2]),
        ),
        (
            "uint64",
            IntegerStorage::U64(vec![1, 0, 0, 3]),
            IntegerStorage::U64(vec![2]),
        ),
    ] {
        let source = format!("d = diag({constructor}([1 3])); delta = diff({constructor}([1 3]));");
        let values = execute_source(&source).expect("compiled structural integer semantics");
        assert!(values.iter().any(|value| {
            matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(&diagonal))
        }), "{constructor} diag result");
        let difference_scalar = difference
            .value_at(0)
            .expect("one exact expected difference value");
        assert!(
            values.iter().any(|value| match value {
                Value::Tensor(tensor) => tensor.integer_storage() == Some(&difference),
                Value::Int(value) => value == &difference_scalar,
                _ => false,
            }),
            "{constructor} diff result"
        );
    }
}

#[test]
fn compiled_floating_integer_extensions_reject_before_computation_in_matlab_mode() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "[q,r] = deconv(uint8([1 2]), uint8([1 1]));",
            "RunMat:compatibility:DeconvIntegerInputExtension",
        ),
        (
            "r = deg2rad(uint16(90));",
            "RunMat:compatibility:Deg2radIntegerInputExtension",
        ),
        (
            "d = det(int32([1 0; 0 1]));",
            "RunMat:compatibility:DetIntegerInputExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("MATLAB mode integer extension gate");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}
