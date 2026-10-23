use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts};
use futures::executor::block_on;
use num_complex::Complex64;
use runmat_value::{ComplexTensor, Tensor};
use std::rc::Rc;

struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: crate::context::ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let state = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(state.free_bytes()),
            epoch: state.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
fn runtime(bytes: u64) -> (RuntimeContext, ManagedHostEffectLedger) {
    let ledger = ManagedHostEffectLedger::new(bytes);
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
        );
    (context, ledger)
}
fn source(class: usize) -> SparseTensor {
    let p = vec![0, 2, 4];
    let r = vec![0, 2, 1, 2];
    match class {
        0 => SparseTensor::new_f32(3, 2, p, r, vec![1., 2., 3., 4.]),
        1 => SparseTensor::new(3, 2, p, r, vec![1., 2., 3., 4.]),
        2 => {
            SparseTensor::new_complex_f32(3, 2, p, r, vec![(1., 1.), (2., 0.), (3., -1.), (4., 1.)])
        }
        _ => SparseTensor::new_complex(3, 2, p, r, vec![(1., 1.), (2., 0.), (3., -1.), (4., 1.)]),
    }
    .unwrap()
}
fn value_at(v: &Value, row: usize, col: usize) -> Complex64 {
    let (r, i) = parts(v, row, col).unwrap();
    Complex64::new(r, i)
}
fn invoke(
    context: &RuntimeContext,
    a: SparseTensor,
    args: Vec<Value>,
    count: usize,
) -> BuiltinResult<Vec<Value>> {
    let _scope = crate::output_count::push_output_count(Some(count)).unwrap();
    let value = block_on(super::super::qr_builtin_context(
        Some(context.clone()),
        Value::SparseTensor(a),
        crate::execution::RootedValues::new(args)
            .map_err(|error| error.into_runtime_error())?
            .into(),
    ))?;
    let Value::OutputList(values) = value else {
        panic!("selected output list")
    };
    Ok(values.try_into_owned().unwrap().into_iter().collect())
}
fn assert_class(value: &Value, class: usize, complex: bool) {
    let dtype = if class.is_multiple_of(2) {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    match (value, complex) {
        (Value::SparseTensor(v), false) => assert_eq!(v.numeric_dtype(), Some(dtype)),
        (Value::SparseTensor(v), true) => assert!(if dtype == NumericDType::F32 {
            v.as_complex_f32_slice().is_some()
        } else {
            v.as_complex_f64_slice().is_some()
        }),
        (Value::Tensor(v), false) => assert_eq!(v.numeric_dtype(), dtype),
        (Value::ComplexTensor(v), true) => assert_eq!(v.numeric_dtype(), dtype),
        _ => panic!("incorrect native class/domain: {value:?}"),
    }
}

#[test]
fn sparse_qr_native_four_public_forms_transformed_rhs_and_retained_output_owners() {
    for class in 0..4 {
        let tolerance = if class % 2 == 0 { 3e-5 } else { 3e-12 };
        for economy in [false, true] {
            for vector in [false, true] {
                let (context, ledger) = runtime(2_000_000);
                let a = source(class);
                let a_value = Value::SparseTensor(a.clone());
                let mut args = Vec::new();
                if economy {
                    args.push(Value::from("econ"));
                }
                if vector {
                    args.push(Value::from("vector"));
                }
                let outputs = invoke(&context, a, args, 3).unwrap();
                let [q, r, p] = outputs.as_slice() else {
                    unreachable!()
                };
                assert_class(q, class, class >= 2);
                assert_class(r, class, class >= 2);
                let height = if economy { 2 } else { 3 };
                assert_eq!(shape(q), Some((3, height, class >= 2)));
                assert_eq!(shape(r), Some((height, 2, class >= 2)));
                assert!(matches!(r, Value::SparseTensor(_)));
                assert_eq!(matches!(p, Value::Tensor(_)), vector);
                for col in 0..2 {
                    let source_col = if vector {
                        value_at(p, col, 0).re as usize - 1
                    } else {
                        (0..2).find(|&row| value_at(p, row, col).re == 1.).unwrap()
                    };
                    for row in 0..3 {
                        let reconstructed = (0..height)
                            .map(|k| value_at(q, row, k) * value_at(r, k, col))
                            .sum::<Complex64>();
                        assert!(
                            (reconstructed - value_at(&a_value, row, source_col)).norm()
                                < tolerance * 10.
                        );
                    }
                }
                for col in 0..height {
                    for row in 0..height {
                        let inner = (0..3)
                            .map(|k| value_at(q, k, row).conj() * value_at(q, k, col))
                            .sum::<Complex64>();
                        assert!(
                            (inner - Complex64::new(if row == col { 1. } else { 0. }, 0.)).norm()
                                < tolerance
                        );
                    }
                }
                let retained = outputs.clone();
                drop(outputs);
                assert!(ledger.snapshot().live_claimed_bytes > 0);
                drop(retained);
                assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
                assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
            }
        }
        let (context, ledger) = runtime(2_000_000);
        let full = invoke(&context, source(class), vec![], 2).unwrap();
        for sparse_b in [false, true] {
            let b = if sparse_b {
                Value::SparseTensor(
                    SparseTensor::new_complex(
                        3,
                        1,
                        vec![0, 2],
                        vec![0, 2],
                        vec![(1., 2.), (3., -1.)],
                    )
                    .unwrap(),
                )
            } else {
                Value::ComplexTensor(
                    ComplexTensor::new(vec![(1., 2.), (0., 0.), (3., -1.)], vec![3, 1]).unwrap(),
                )
            };
            let transformed = invoke(&context, source(class), vec![b.clone()], 2).unwrap();
            assert_class(&transformed[0], class, true);
            for row in 0..3 {
                let expected = (0..3)
                    .map(|k| value_at(&full[0], k, row).conj() * value_at(&b, k, 0))
                    .sum::<Complex64>();
                assert!((value_at(&transformed[0], row, 0) - expected).norm() < tolerance * 10.);
            }
        }
        drop(full);
        let empty = Value::Tensor(Tensor::new(vec![], vec![3, 0]).unwrap());
        let outputs = invoke(&context, source(class), vec![empty, Value::from("econ")], 2).unwrap();
        assert_eq!(shape(&outputs[0]), Some((2, 0, class >= 2)));
        drop(outputs);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn sparse_qr_rejections_and_denied_growth_publish_nothing_and_never_retry() {
    let (context, ledger) = runtime(1);
    let error = invoke(&context, source(0), vec![], 3).unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let (context, ledger) = runtime(2_000_000);
    assert!(invoke(&context, source(0), vec![Value::from("vector")], 2).is_err());
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let a = SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![f64::NAN]).unwrap();
    let error = invoke(&context, a, vec![], 2).unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let context = context.with_cancellation_token(cancellation.token());
    cancellation.cancel(runmat_execution::CancellationReason::User);
    let error = invoke(&context, source(1), vec![], 2).unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn sparse_qr_empty_transformed_geometry_does_not_traverse_virtual_columns() {
    let (context, ledger) = runtime(100_000);
    let a = SparseTensor::new(0, 0, vec![0], vec![], vec![]).unwrap();
    let b = Value::Tensor(Tensor::new(vec![], vec![0, usize::MAX]).unwrap());
    let output = invoke(&context, a, vec![b], 2).unwrap();
    assert_eq!(shape(&output[0]), Some((0, usize::MAX, false)));
    assert_eq!(shape(&output[1]), Some((0, 0, false)));
    drop(output);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
