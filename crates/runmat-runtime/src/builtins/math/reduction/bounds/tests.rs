use super::*;
use runmat_value::{IntValue, IntegerStorage, Tensor};

async fn evaluate_bounds(value: Value, rest: &[Value]) -> BuiltinResult<BoundsEvaluation> {
    let rest = crate::call::BuiltinArguments::from_borrowed(rest, None)?;
    super::evaluate_bounds(None, value, rest).await
}

async fn bounds_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    super::bounds_builtin(None, value, rest.into()).await
}

#[tokio::test]
async fn missing_aliases_match_nan_flags() {
    for (alias, flag) in [("omitmissing", "omitnan"), ("includemissing", "includenan")] {
        let a = evaluate_bounds(
            tensor(vec![f64::NAN, 3.], vec![2, 1]),
            &[Value::from(alias)],
        )
        .await
        .unwrap();
        let b = evaluate_bounds(tensor(vec![f64::NAN, 3.], vec![2, 1]), &[Value::from(flag)])
            .await
            .unwrap();
        for (left, right) in [(a.smallest, b.smallest), (a.largest, b.largest)] {
            let (Value::Num(left), Value::Num(right)) = (left, right) else {
                panic!("expected scalar")
            };
            assert!(left == right || (left.is_nan() && right.is_nan()));
        }
    }
}

#[tokio::test]
async fn extra_outputs_require_extensions_before_provider_access() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident reduction admission fixture");
    let providerless = Value::GpuTensor(allocation.handle.clone());
    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let error = bounds_builtin(providerless.clone(), vec![])
            .await
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:BoundsExtraOutputsExtension")
        );
    }
    let error = bounds_builtin(providerless, vec![Value::Num(0.)])
        .await
        .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:bounds:InvalidArgument"));
    assert_eq!(allocation.probe.retirements(), 0);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    drop(error);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[tokio::test]
async fn integer_empty_shape_and_logical_output_class_are_explicit() {
    let empty =
        Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![]), vec![0, 3]).unwrap());
    let result = evaluate_bounds(empty, &[Value::from("all")]).await.unwrap();
    for value in [result.smallest, result.largest] {
        let Value::Tensor(value) = value else {
            panic!("expected empty integer tensor")
        };
        assert_eq!(value.shape(), vec![0, 3]);
        assert_eq!(value.integer_storage(), Some(&IntegerStorage::U64(vec![])));
    }
    let result = evaluate_bounds(Value::Bool(true), &[]).await.unwrap();
    assert_eq!(result.smallest, Value::Num(1.));
    assert_eq!(result.largest, Value::Num(1.));
}

#[tokio::test]
async fn empty_float_and_complex_bounds_preserve_shape_for_all_selection() {
    for input in [
        Value::Tensor(Tensor::new(vec![], vec![0, 3]).unwrap()),
        Value::ComplexTensor(runmat_value::ComplexTensor::new(vec![], vec![0, 3]).unwrap()),
    ] {
        let result = evaluate_bounds(input, &[Value::from("all")]).await.unwrap();
        for output in [result.smallest, result.largest] {
            match output {
                Value::Tensor(value) => assert_eq!(value.shape(), &[0, 3]),
                Value::ComplexTensor(value) => assert_eq!(value.shape(), &[0, 3]),
                _ => panic!("expected empty tensor"),
            }
        }
    }
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

#[test]
fn bounds_descriptor_declares_runtime_missingflag_default() {
    assert_eq!(BOUNDS_INPUT_NANFLAG.default, Some("\"omitmissing\""));
}

#[tokio::test]
async fn bounds_default_returns_min_and_max() {
    let input = tensor(vec![3.0, 1.0, 4.0, 2.0], vec![2, 2]);
    let result = evaluate_bounds(input, &[]).await.unwrap();
    let Value::Tensor(mins) = result.smallest else {
        panic!("expected min tensor");
    };
    assert_eq!(mins.materialize_f64(), vec![1.0, 2.0]);
    let Value::Tensor(maxes) = result.largest else {
        panic!("expected max tensor");
    };
    assert_eq!(maxes.materialize_f64(), vec![3.0, 4.0]);
}

#[tokio::test]
async fn bounds_supports_all_and_omitnan() {
    let input = tensor(vec![f64::NAN, 3.0, -1.0, 5.0], vec![2, 2]);
    let result = evaluate_bounds(input, &[Value::from("all"), Value::from("omitnan")])
        .await
        .unwrap();
    assert_eq!(result.smallest, Value::Num(-1.0));
    assert_eq!(result.largest, Value::Num(5.0));
}

#[tokio::test]
async fn bounds_omits_nan_by_default() {
    let input = tensor(vec![f64::NAN, 3.0, -1.0, 5.0], vec![2, 2]);
    let result = evaluate_bounds(input, &[Value::from("all")]).await.unwrap();
    assert_eq!(result.smallest, Value::Num(-1.0));
    assert_eq!(result.largest, Value::Num(5.0));
}

#[tokio::test]
async fn bounds_preserves_exact_integer_extrema_for_dimensions_and_all() {
    let large = 9_007_199_254_740_992_u64;
    let input = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![large + 1, large, u64::MAX, 7]),
            vec![2, 2],
        )
        .unwrap(),
    );

    let reduced = evaluate_bounds(input.clone(), &[]).await.unwrap();
    let Value::Tensor(minima) = reduced.smallest else {
        panic!("expected integer minima tensor");
    };
    let Value::Tensor(maxima) = reduced.largest else {
        panic!("expected integer maxima tensor");
    };
    assert_eq!(
        minima.integer_storage(),
        Some(&IntegerStorage::U64(vec![large, 7]))
    );
    assert_eq!(
        maxima.integer_storage(),
        Some(&IntegerStorage::U64(vec![large + 1, u64::MAX]))
    );

    let all = evaluate_bounds(input, &[Value::from("all")]).await.unwrap();
    assert_eq!(all.smallest, Value::Int(IntValue::U64(7)));
    assert_eq!(all.largest, Value::Int(IntValue::U64(u64::MAX)));
}

#[tokio::test]
async fn bounds_invalid_reduction_arguments_use_bounds_identifier() {
    let input = tensor(vec![1.0, 2.0], vec![2, 1]);
    let err = evaluate_bounds(input, &[Value::Num(0.0)])
        .await
        .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:bounds:InvalidArgument"));
}

#[tokio::test]
async fn bounds_builtin_pads_requested_outputs() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let input = tensor(vec![1.0, 4.0], vec![2, 1]);
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let result = bounds_builtin(input, Vec::new()).await.unwrap();
    let Value::OutputList(values) = result else {
        panic!("expected output list");
    };
    assert_eq!(values.len(), 3);
    assert_eq!(values[0], Value::Num(1.0));
    assert_eq!(values[1], Value::Num(4.0));
    assert_eq!(values[2], Value::Num(0.0));
}

#[test]
fn bounds_integer_gpu_fallback_preserves_exact_class_and_residency() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let wide = 9_007_199_254_740_993_u64;
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![wide + 2, wide, u64::MAX, 7]),
            vec![2, 2],
        )
        .expect("integer tensor");
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
        let result = futures::executor::block_on(evaluate_bounds(Value::GpuTensor(handle), &[]))
            .expect("bounds");
        for (value, expected) in [
            (result.smallest, vec![wide, 7]),
            (result.largest, vec![wide + 2, u64::MAX]),
        ] {
            let Value::GpuTensor(result_handle) = &value else {
                panic!("expected resident integer bound");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(result_handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let gathered =
                crate::builtins::common::test_support::gather(value).expect("gather bound");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(expected))
            );
        }
    });
}
