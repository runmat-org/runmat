use super::*;
use futures::executor::block_on;
use num_complex::Complex64;
use runmat_value::{SparseTensor, Tensor};
use std::{cell::RefCell, rc::Rc};
mod contracts;

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
}
fn call(runtime: &RuntimeContext, a: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::super::eigs_builtin_context(
        Some(runtime.clone()),
        a,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}
fn values(value: &Value) -> Vec<Complex64> {
    match value {
        Value::Tensor(t) => t
            .materialize_f64()
            .into_iter()
            .map(|v| Complex64::new(v, 0.0))
            .collect(),
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .into_iter()
            .map(|(r, i)| Complex64::new(r, i))
            .collect(),
        other => panic!("numeric output, got {other:?}"),
    }
}
#[test]
fn sparse_native_four_select_and_shift_without_dense_coefficients() {
    let runtime = runtime();
    let matrices = [
        SparseTensor::new_f32(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 3.0, 9.0]).unwrap(),
        SparseTensor::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 3.0, 9.0]).unwrap(),
        SparseTensor::new_complex_f32(
            3,
            3,
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![(1.0, 0.5), (3.0, 0.5), (9.0, 0.5)],
        )
        .unwrap(),
        SparseTensor::new_complex(
            3,
            3,
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![(1.0, 0.5), (3.0, 0.5), (9.0, 0.5)],
        )
        .unwrap(),
    ];
    for (index, matrix) in matrices.into_iter().enumerate() {
        let single = index.is_multiple_of(2);
        for generalized in [false, true] {
            for shifted in [false, true] {
                let sigma = if shifted {
                    Value::Num(2.5)
                } else {
                    Value::String("largestreal".into())
                };
                let mut arguments = Vec::new();
                if generalized {
                    arguments.push(Value::Tensor(
                        Tensor::new(
                            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                            vec![3, 3],
                        )
                        .unwrap(),
                    ));
                }
                arguments.extend([Value::Num(1.0), sigma]);
                let output =
                    call(&runtime, Value::SparseTensor(matrix.clone()), arguments).unwrap();
                let dtype = match &output {
                    Value::Tensor(t) => t.numeric_dtype(),
                    Value::ComplexTensor(t) => t.numeric_dtype(),
                    _ => panic!("numeric"),
                };
                assert_eq!(
                    dtype,
                    if single {
                        NumericDType::F32
                    } else {
                        NumericDType::F64
                    }
                );
                let expected = Complex64::new(
                    if shifted { 3.0 } else { 9.0 },
                    if index >= 2 { 0.5 } else { 0.0 },
                );
                assert!((values(&output)[0] - expected).norm() < if single { 2e-4 } else { 1e-10 });
            }
        }
    }
}
#[test]
fn tiny_complex_publication_and_huge_residual_scale_are_not_clipped() {
    let runtime = runtime();
    let skew = SparseTensor::new(2, 2, vec![0, 1, 2], vec![1, 0], vec![1e-100, -1e-100]).unwrap();
    let output = call(&runtime, Value::SparseTensor(skew), vec![Value::Num(2.0)]).unwrap();
    assert!(matches!(output, Value::ComplexTensor(_)));
    let mut imaginary = values(&output)
        .into_iter()
        .map(|v| v.im / 1e-100)
        .collect::<Vec<_>>();
    imaginary.sort_by(f64::total_cmp);
    assert!((imaginary[0] + 1.0).abs() < 1e-10 && (imaginary[1] - 1.0).abs() < 1e-10);
    use super::residual_norm::{converged, Norm};
    let norm = |value: f64| {
        let mut n = Norm::new();
        n.add(Complex64::new(value, 0.0));
        n
    };
    assert!(converged(
        norm(1e292),
        norm(1e308),
        Complex64::new(1e308, 0.0),
        norm(1.0),
        1e-14
    ));
    assert!(!converged(
        norm(1e300),
        norm(1e308),
        Complex64::new(1e308, 0.0),
        norm(1.0),
        1e-14
    ));
}
#[test]
fn successful_precision_probe_is_distinct_and_failed_probe_is_terminal() {
    for single in [false, true] {
        for complex_output in [false, true] {
            for fail in [false, true] {
                let runtime = runtime();
                let transcript = Rc::new(RefCell::new(Vec::new()));
                let observed = Rc::clone(&transcript);
                runtime.set_bound_function_invoker(Some(Rc::new(
                    move |_runtime, binding, arguments, requested, _environment| {
                        assert_eq!(binding.function, 9341);
                        assert_eq!(requested, 1);
                        assert_eq!(arguments.supplied().len(), 1);
                        let (dtype, complex, input) = match &arguments.supplied()[0] {
                            Value::Tensor(t) => {
                                assert_eq!(t.shape(), [2, 1]);
                                (
                                    t.numeric_dtype(),
                                    false,
                                    t.materialize_f64()
                                        .into_iter()
                                        .map(|v| Complex64::new(v, 0.0))
                                        .collect::<Vec<_>>(),
                                )
                            }
                            Value::ComplexTensor(t) => {
                                assert_eq!(t.shape(), [2, 1]);
                                (
                                    t.numeric_dtype(),
                                    true,
                                    t.materialize_f64()
                                        .into_iter()
                                        .map(|(r, i)| Complex64::new(r, i))
                                        .collect(),
                                )
                            }
                            _ => panic!("native input"),
                        };
                        observed.borrow_mut().push((dtype, complex));
                        crate::execution::admit_executable_sequence_with(
                            &_runtime.clone(),
                            move || async move {
                                if fail {
                                    return Err(crate::build_runtime_error("probe failed")
                                        .with_identifier("RunMat:Test:EigsProbe")
                                        .build());
                                }
                                let output: Vec<_> = input
                                    .into_iter()
                                    .enumerate()
                                    .map(|(i, v)| {
                                        let v = v * Complex64::new(
                                            if i == 0 { 1.0 } else { 3.0 },
                                            if complex_output { 0.25 } else { 0.0 },
                                        );
                                        (v.re, v.im)
                                    })
                                    .collect();
                                let output = match (single, complex_output) {
                                    (false, false) => Value::Tensor(
                                        Tensor::new(
                                            output.iter().map(|v| v.0).collect(),
                                            vec![2, 1],
                                        )
                                        .unwrap(),
                                    ),
                                    (true, false) => Value::Tensor(
                                        Tensor::from_f32(
                                            output.iter().map(|v| v.0 as f32).collect(),
                                            vec![2, 1],
                                        )
                                        .unwrap(),
                                    ),
                                    (false, true) => Value::ComplexTensor(
                                        runmat_value::ComplexTensor::new(output, vec![2, 1])
                                            .unwrap(),
                                    ),
                                    (true, true) => Value::ComplexTensor(
                                        runmat_value::ComplexTensor::from_f32(
                                            output
                                                .into_iter()
                                                .map(|v| (v.0 as f32, v.1 as f32))
                                                .collect(),
                                            vec![2, 1],
                                        )
                                        .unwrap(),
                                    ),
                                };
                                crate::user_functions::test_support::rooted_single_value_sequence(
                                    output,
                                )
                            },
                        )
                    },
                )));
                let target = crate::user_functions::test_support::exact_local_handle(
                    &runtime,
                    "eigs_probe",
                    9341,
                );
                let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
                let result = call(&runtime, target, vec![Value::Num(2.0), Value::Num(1.0)]);
                let transcript = transcript.borrow();
                assert_eq!(transcript[0], (NumericDType::F64, false));
                if fail {
                    let error = result.unwrap_err();
                    assert_eq!(error.identifier(), Some("RunMat:Test:EigsProbe"));
                    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                    assert_eq!(transcript.len(), 1);
                } else {
                    let output = result.unwrap();
                    let dtype = if single {
                        NumericDType::F32
                    } else {
                        NumericDType::F64
                    };
                    // An unchanged real-double probe is already the first
                    // native product. Class discovery cannot reuse it when
                    // precision or complex domain changes.
                    assert!(transcript.len() >= if single || complex_output { 3 } else { 2 });
                    assert!(transcript[1..]
                        .iter()
                        .all(|&class| class == (dtype, complex_output)));
                    let Value::OutputList(list) = &output else {
                        panic!("three actual eigs outputs expected")
                    };
                    assert_eq!(list.as_slice().len(), 3);
                    assert_eq!(list.as_slice()[2], Value::Num(0.0));
                    for (value, shape) in list.as_slice()[..2].iter().zip([[2, 1], [1, 1]]) {
                        match value {
                            Value::Tensor(t) => {
                                assert!(!complex_output);
                                assert_eq!(t.numeric_dtype(), dtype);
                                assert_eq!(t.shape(), shape);
                            }
                            Value::ComplexTensor(t) => {
                                assert!(complex_output);
                                assert_eq!(t.numeric_dtype(), dtype);
                                assert_eq!(t.shape(), shape);
                            }
                            _ => panic!("native eigenpair publication expected"),
                        }
                    }
                    // Independent diagonal-operator oracle: neither the selected
                    // eigenvalue nor residual is derived from the eigs recurrence.
                    let eigenvalue = values(&list.as_slice()[1])[0];
                    let imaginary = if complex_output { 0.25 } else { 0.0 };
                    let tolerance = if single { 2e-4 } else { 1e-11 };
                    assert!((eigenvalue - Complex64::new(3.0, imaginary)).norm() < tolerance);
                    let vector = values(&list.as_slice()[0]);
                    let length = vector.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                    assert!((length - 1.0).abs() < tolerance);
                    let residual = vector
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            ((Complex64::new(if i == 0 { 1.0 } else { 3.0 }, imaginary)
                                - eigenvalue)
                                * v)
                                .norm_sqr()
                        })
                        .sum::<f64>()
                        .sqrt();
                    assert!(residual < tolerance * length);
                }
            }
        }
    }
}
