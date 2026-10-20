use super::*;
use futures::executor::block_on;

fn resident_numeric() -> runmat_accelerate_api::test_support::TestAllocation {
    runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("test allocation")
}

#[test]
fn known_bad_format_rejects_before_resident_fid_download() {
    let allocation = resident_numeric();
    let wire = allocation.handle.wire_descriptor();
    let err = block_on(textscan_builtin(
        None,
        Value::GpuTensor(allocation.handle.clone()),
        Value::from("%z"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("invalid format");
    assert_eq!(err.identifier(), Some("RunMat:textscan:InvalidFormat"));
    assert_eq!(allocation.handle.wire_descriptor(), wire);
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn known_bad_option_rejects_before_resident_value_download() {
    let allocation = resident_numeric();
    let err = block_on(textscan_builtin(
        None,
        Value::from("1"),
        Value::from("%f"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("NoSuchOption"),
            Value::GpuTensor(allocation.handle.clone()),
        ]),
    ))
    .expect_err("invalid option");
    assert_eq!(err.identifier(), Some("RunMat:textscan:InvalidArgument"));
    assert!(err.message().contains("unsupported option"));
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn nested_resident_in_invalid_input_rejects_before_download() {
    let allocation = resident_numeric();
    let nested =
        CellArray::new(vec![Value::GpuTensor(allocation.handle.clone())], 1, 1).expect("cell");
    let err = block_on(textscan_builtin(
        None,
        Value::Cell(nested),
        Value::from("%f"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("cell is not text or file identifier");
    assert_eq!(err.identifier(), Some("RunMat:textscan:InvalidArgument"));
    assert_eq!(allocation.probe.retirements(), 0);
}
