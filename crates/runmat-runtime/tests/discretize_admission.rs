use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
};
use runmat_runtime::call_builtin_async_with_outputs;
use runmat_value::{CharArray, NumericStorage, Tensor, Value};

fn array(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn invoke(
    args: &[Value],
    outputs: usize,
) -> Result<runmat_runtime::execution::RootedValues, Box<runmat_runtime::RuntimeError>> {
    block_on(call_builtin_async_with_outputs("discretize", args, outputs))
        .map_err(Box::new)?
        .with_owned(|value| {
            let values = match value {
                Value::OutputList(values) => values,
                value => vec![value].into(),
            };
            runmat_runtime::execution::RootedValues::from_output_storage(values)
                .map_err(|error| Box::new(error.into_runtime_error()))
        })
}

#[test]
fn numeric_bin_admission_precedes_any_resident_download() {
    let allocation = TestAllocationOwner::new()
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let physical = allocation.handle.allocation_lease().downgrade();
    let probe = allocation.probe.clone();
    let poisoned = Value::GpuTensor(allocation.handle);
    let edges = array(vec![0.0, 1.0, 2.0], vec![1, 3]);
    let cases = [
        (vec![poisoned.clone(), edges.clone()], 3),
        (vec![poisoned.clone(), edges.clone()], 2),
        (
            vec![
                poisoned.clone(),
                edges.clone(),
                Value::from("IncludedEdge"),
                Value::from("middle"),
            ],
            0,
        ),
        (vec![poisoned.clone(), array(vec![1.0, 0.0], vec![1, 2])], 1),
        (
            vec![
                poisoned.clone(),
                array(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
            ],
            1,
        ),
        (
            vec![poisoned, edges, array(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])],
            1,
        ),
    ];
    for (args, outputs) in cases {
        let error = invoke(&args, outputs).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:discretize:InvalidInput"));
        assert!(physical.acquire().is_some());
    }
    assert!(physical.acquire().is_none());
    assert_eq!(probe.retirements(), 1);
}

#[test]
fn scalar_tensor_counts_and_extreme_edges_preserve_shape_and_values() {
    for storage in [NumericStorage::F32(vec![2.0]), NumericStorage::U64(vec![2])] {
        let args = [
            array(vec![-f64::MAX, 0.0, f64::MAX], vec![1, 1, 3]),
            Value::Tensor(Tensor::from_numeric_storage(storage, vec![1, 1, 1]).unwrap()),
        ];
        let output = invoke(&args, 2).unwrap();
        let Value::Tensor(indices) = &output[0] else {
            panic!("dense indices")
        };
        let Value::Tensor(edges) = &output[1] else {
            panic!("dense edges")
        };
        assert_eq!(indices.shape(), &[1, 1, 3]);
        assert_eq!(indices.materialize_f64(), vec![1.0, 2.0, 2.0]);
        assert_eq!(edges.materialize_f64(), vec![-f64::MAX, 0.0, f64::MAX]);
        assert!(invoke(&args, 0).unwrap().is_empty());
    }
}

#[test]
fn text_options_use_effective_rank_and_scalar_string_arrays() {
    let input = array(vec![1.0], vec![1, 1]);
    let edges = array(vec![0.0, 1.0, 2.0], vec![1, 3, 1]);
    let option = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape(
            "IncludedEdge".chars().collect(),
            vec![1, 12, 1],
        )
        .unwrap(),
    );
    let output = invoke(
        &[input.clone(), edges.clone(), option, Value::from("right")],
        1,
    )
    .unwrap();
    let Value::Tensor(output) = &output[0] else {
        panic!("dense scalar output")
    };
    assert_eq!(output.materialize_f64(), vec![1.0]);
    let pages = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("IncludedEdge".chars().collect(), vec![1, 6, 2])
            .unwrap(),
    );
    assert!(invoke(&[input, edges, pages, Value::from("right")], 1).is_err());
}

#[test]
fn computed_double_edges_enclose_exact_wide_integer_extrema() {
    for (value, lower, upper) in [
        (
            9_007_199_254_740_993_u64,
            9_007_199_254_740_992.0,
            9_007_199_254_740_994.0,
        ),
        (
            u64::MAX,
            18_446_744_073_709_549_568.0,
            18_446_744_073_709_551_616.0,
        ),
    ] {
        for count in [1, 2] {
            let outputs = invoke(
                &[
                    Value::Tensor(
                        Tensor::from_numeric_storage(NumericStorage::U64(vec![value]), vec![1, 1])
                            .unwrap(),
                    ),
                    Value::Num(f64::from(count)),
                ],
                2,
            )
            .unwrap();
            let Value::Tensor(indices) = &outputs[0] else {
                panic!("indices")
            };
            let Value::Tensor(edges) = &outputs[1] else {
                panic!("double edges")
            };
            let index = indices.materialize_f64()[0];
            assert!(index >= 1.0 && index <= f64::from(count));
            let actual = edges.materialize_f64();
            assert_eq!(actual[0], lower);
            assert_eq!(*actual.last().unwrap(), upper);
            assert!(matches!(
                edges.clone().into_numeric_storage().unwrap(),
                NumericStorage::F64(_)
            ));
        }
    }
    for value in [f64::MAX, -f64::MAX] {
        let outputs = invoke(&[Value::Num(value), Value::Num(2.0)], 2).unwrap();
        let Value::Tensor(indices) = &outputs[0] else {
            panic!("indices")
        };
        let Value::Tensor(edges) = &outputs[1] else {
            panic!("edges")
        };
        assert!(indices.materialize_f64()[0].is_finite());
        assert!(edges.materialize_f64().iter().all(|edge| edge.is_finite()));
    }
}
