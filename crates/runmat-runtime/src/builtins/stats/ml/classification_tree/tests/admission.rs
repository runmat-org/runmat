use super::*;

fn build_fit_spec(
    first: Value,
    rest: Vec<Value>,
    arguments: FitctreeArguments,
) -> BuiltinResult<FitSpec> {
    super::super::build_fit_spec(
        first,
        crate::call::FunctionArgumentView::from_slice(&rest),
        arguments,
    )
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn fitctree_dense_rank_boundaries_preserve_observation_geometry() {
    let rank_one = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    assert_eq!(classification_tree_matrix_dimensions(&rank_one), (1, 3));
    let error = build_fit_spec(
        Value::Tensor(rank_one),
        vec![tensor_value(vec![0.0, 1.0, 0.0], 3, 1)],
        FitctreeArguments {
            option_start: 1,
            options: Vec::new(),
        },
    )
    .expect_err("rank-one predictors are one observation, not three rows");
    assert!(error.message.contains("length(Y)"));

    let rank_zero = Tensor::new(vec![1.0], Vec::new()).unwrap();
    assert_eq!(classification_tree_matrix_dimensions(&rank_zero), (0, 0));
    let error = build_fit_spec(
        Value::Tensor(rank_zero),
        vec![tensor_value(vec![0.0], 1, 1)],
        FitctreeArguments {
            option_start: 1,
            options: Vec::new(),
        },
    )
    .expect_err("rank-zero predictors do not create a phantom observation");
    assert!(error.message.contains("length(Y)"));
}

#[test]
fn fitctree_header_retains_typed_options_and_rejects_malformed_pairs() {
    let first = tensor_value(vec![0.0, 1.0], 2, 1);
    let valid = vec![Value::Num(0.0), Value::from("MinLeafSize"), Value::Num(1.0)];
    let header = HeaderInspection::inspect(
        &first,
        crate::call::FunctionArgumentView::from_slice(&valid),
    )
    .unwrap()
    .validate(valid.len())
    .unwrap();
    assert_eq!(header.option_start, 1);
    assert_eq!(header.options.len(), 1);
    assert_eq!(header.options[0].name.identity, FitctreeOption::MinLeafSize);
    assert_eq!(header.options[0].name.index, 1);

    let odd = vec![Value::Num(0.0), Value::from("MinLeafSize")];
    assert!(
        HeaderInspection::inspect(&first, crate::call::FunctionArgumentView::from_slice(&odd))
            .unwrap()
            .validate(odd.len())
            .is_err()
    );
    let unknown = vec![Value::Num(0.0), Value::from("Unknown"), Value::Num(1.0)];
    assert!(HeaderInspection::inspect(
        &first,
        crate::call::FunctionArgumentView::from_slice(&unknown)
    )
    .unwrap()
    .validate(unknown.len())
    .is_err());
    let unsupported = vec![Value::Num(0.0), Value::from("Prune"), Value::Bool(true)];
    assert!(HeaderInspection::inspect(
        &first,
        crate::call::FunctionArgumentView::from_slice(&unsupported)
    )
    .unwrap()
    .validate(unsupported.len())
    .is_err());
}

#[test]
fn fitctree_rejects_known_rank_before_materialization() {
    let first = Value::Tensor(Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap());
    assert!(preflight_matrix_rank(&first).is_err());
}

#[test]
fn structural_counts_reject_the_first_value_beyond_platform_usize() {
    let first_out_of_range = (1u128 << usize::BITS) as f64;
    assert!(nonnegative_integer(&Value::Num(first_out_of_range), "MaxNumSplits").is_err());
    assert!(positive_integer(&Value::Num(first_out_of_range), "MinLeafSize").is_err());
    assert_eq!(
        nonnegative_integer(&Value::Num(0.0), "MaxNumSplits").unwrap(),
        0
    );
    assert_eq!(
        positive_integer(&Value::Num(1.0), "MinLeafSize").unwrap(),
        1
    );
}

#[test]
fn fitctree_strict_mode_rejects_integer_numeric_extensions() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(fitctree_builtin(
        None,
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), 4, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![tensor_value(
            vec![0.0, 0.0, 1.0, 1.0],
            4,
            1,
        )]),
    ))
    .expect_err("integer predictor extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FitctreeIntegerPredictorExtension")
    );

    let error = block_on(fitctree_builtin(
        None,
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
            Value::from("MaxNumSplits"),
            poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
        ]),
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
        None,
        poison,
        crate::builtins::common::test_support::builtin_arguments(vec![tensor_value(
            vec![0.0, 0.0, 1.0, 1.0],
            4,
            1,
        )]),
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
        None,
        poisoned_int_tensor(IntegerStorage::U64(vec![0, 1, 2, inexact]), 4, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![tensor_value(
            vec![0.0, 0.0, 1.0, 1.0],
            4,
            1,
        )]),
    ))
    .expect_err("inexact predictor");
    assert!(error.message.contains("exactly representable"));

    let error = block_on(fitctree_builtin(
        None,
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
            Value::from("Weights"),
            poisoned_int_tensor(IntegerStorage::U64(vec![1, 1, 1, inexact]), 4, 1),
        ]),
    ))
    .expect_err("inexact weights");
    assert!(error.message.contains("exactly representable"));
}
