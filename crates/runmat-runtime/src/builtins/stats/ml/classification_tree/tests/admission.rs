use super::*;

#[test]
fn fitctree_strict_mode_rejects_integer_numeric_extensions() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(fitctree_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), 4, 1),
        vec![tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1)],
    ))
    .expect_err("integer predictor extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FitctreeIntegerPredictorExtension")
    );

    let error = block_on(fitctree_builtin(
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        vec![
            tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
            Value::from("MaxNumSplits"),
            poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
        ],
    ))
    .expect_err("integer control extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FitctreeIntegerNumericArgumentExtension")
    );

    let poison = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![4, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle,
    );
    let error = block_on(fitctree_builtin(
        poison,
        vec![tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1)],
    ))
    .expect_err("resident gate before gather");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FitctreeResidentInputExtension")
    );
}

#[test]
fn fitctree_rejects_inexact_integer_predictors_and_weights() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let inexact = 9_007_199_254_740_993_u64;
    let error = block_on(fitctree_builtin(
        poisoned_int_tensor(IntegerStorage::U64(vec![0, 1, 2, inexact]), 4, 1),
        vec![tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1)],
    ))
    .expect_err("inexact predictor");
    assert!(error.message.contains("exactly representable"));

    let error = block_on(fitctree_builtin(
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        vec![
            tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
            Value::from("Weights"),
            poisoned_int_tensor(IntegerStorage::U64(vec![1, 1, 1, inexact]), 4, 1),
        ],
    ))
    .expect_err("inexact weights");
    assert!(error.message.contains("exactly representable"));
}
