use super::*;
use futures::executor::block_on;

#[test]
fn every_numeric_guard_checks_outputs_before_data_or_provider_access() {
    let allocation = resident_fixture(vec![1, 1], runmat_accelerate_api::NumericElementType::F64);
    let resident = Value::GpuTensor(allocation.handle);
    for name in [
        "mustBeFile",
        "mustBeFolder",
        "mustBeFinite",
        "mustBeInteger",
        "mustBeNonNan",
        "mustBeNonzero",
        "mustBeNonmissing",
        "mustBeReal",
        "mustBePositive",
        "mustBeNegative",
        "mustBeNonnegative",
        "mustBeNonpositive",
        "mustBeGreaterThan",
        "mustBeGreaterThanOrEqual",
        "mustBeLessThan",
        "mustBeLessThanOrEqual",
        "mustBeInRange",
        "mustBeMember",
    ] {
        let error = block_on(crate::call_builtin_async_with_outputs(
            name,
            std::slice::from_ref(&resident),
            1,
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some(format!("RunMat:{name}:TooManyOutputs").as_str()),
            "{name}: {error}"
        );
    }
    assert_eq!(allocation.probe.retirements(), 0);
    drop(resident);
    assert_eq!(allocation.probe.retirements(), 1);
}

fn nested(value: Value) -> Value {
    Value::Cell(
        CellArray::new(
            vec![Value::Cell(CellArray::new(vec![value], 1, 1).unwrap())],
            1,
            1,
        )
        .unwrap(),
    )
}

#[test]
fn nested_resident_missingness_checks_content_and_preserves_aliases() {
    runmat_gc::gc_test_context(|| {
        test_support::with_test_provider(|provider| {
            for (number, expected) in [(1.0, true), (f64::NAN, false)] {
                let tensor = Tensor::new(vec![number], vec![1, 1]).unwrap();
                let handle =
                    crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
                let value = nested(Value::GpuTensor(handle.clone()));
                assert_eq!(block_on(value_is_non_nan_async(&value)).unwrap(), expected);
                assert_eq!(
                    block_on(value_is_nonmissing_async(&value)).unwrap(),
                    expected
                );
                let download = block_on(provider.download_numeric(&handle)).unwrap();
                assert_eq!(download.shape, vec![1, 1]);
                let args = vec![value.clone()];
                // Abandoning an actual callable future never adopts or frees a borrowed input.
                drop(must_be_nonmissing::evaluate(
                    crate::builtins::common::validation::tests::retained_arguments(args),
                ));
                assert!(block_on(provider.download_numeric(&handle)).is_ok());
                let weak = handle.downgrade();
                drop(handle);
                assert_eq!(
                    block_on(value_is_nonmissing_async(&value)).unwrap(),
                    expected
                );
                assert!(weak.acquire().is_ok());
                drop(value);
                runmat_gc::gc_collect_major().unwrap();
                assert!(weak.acquire().is_err());
            }
        });
    });
}

#[test]
fn nested_integer_missingness_is_metadata_only_and_exact() {
    runmat_gc::gc_test_context(|| {
        download_lifecycle::without_downloads(|provider| {
            let tensor =
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap();
            let handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            let value = nested(Value::GpuTensor(handle.clone()));
            assert!(block_on(value_is_non_nan_async(&value)).unwrap());
            assert!(block_on(value_is_nonmissing_async(&value)).unwrap());
            let weak = handle.downgrade();
            drop(value);
            runmat_gc::gc_collect_major().unwrap();
            drop(handle);
            assert!(weak.acquire().is_err());
        });
    });
}

#[test]
fn invalid_range_controls_and_comparison_shapes_precede_downloads() {
    download_lifecycle::without_downloads(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        let value = Value::GpuTensor(handle.clone());
        let error = block_on(must_be_in_range::evaluate(
            crate::builtins::common::validation::tests::retained_arguments(vec![
                value.clone(),
                Value::Num(0.0),
                Value::Num(3.0),
                Value::String("bad-option".into()),
            ]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:mustBeInRange:InvalidArgument")
        );
        let error = block_on(value_is_greater_than_values_async(
            &value,
            &Value::Tensor(Tensor::new(vec![1.0; 3], vec![1, 3]).unwrap()),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:argumentValidation:InvalidArgument")
        );
        let weak = handle.downgrade();
        drop(value);
        drop(handle);
        assert!(weak.acquire().is_err());
    });
}

#[test]
fn exact_sign_range_and_membership_keep_wide_integer_boundaries() {
    let wide = Value::Int(IntValue::U64(9_007_199_254_740_993));
    let rounded = Value::Num(9_007_199_254_740_992.0);
    assert!(block_on(value_is_greater_than_values_async(&wide, &rounded)).unwrap());
    assert!(!block_on(value_is_member_async(&wide, &rounded)).unwrap());
    assert!(block_on(value_is_in_range_documented_async(
        &wide,
        &wide,
        &wide,
        RangeInclusivity::CLOSED
    ))
    .unwrap());
    assert!(block_on(value_is_positive_async(&wide)).unwrap());
    assert!(!block_on(value_is_negative_async(&wide)).unwrap());
}
