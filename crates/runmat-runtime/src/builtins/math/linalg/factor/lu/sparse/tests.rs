use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use num_complex::Complex64;
use std::rc::Rc;
struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: crate::context::ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let s = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(s.free_bytes()),
            epoch: s.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
fn runtime(managed: bool, bytes: u64) -> (RuntimeContext, ManagedHostEffectLedger) {
    let ledger = ManagedHostEffectLedger::new(bytes);
    let base = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    (
        if managed {
            base.with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
            )
        } else {
            base
        },
        ledger,
    )
}
fn source(class: usize, m: usize, n: usize, data: &[(f64, f64)]) -> SparseTensor {
    let mut ptr = vec![0];
    let mut rows = Vec::new();
    let mut values = Vec::new();
    for col in 0..n {
        for row in 0..m {
            let (r, i) = data[col * m + row];
            let i = if class >= 2 { i } else { 0. };
            if r != 0. || i != 0. {
                rows.push(row);
                values.push((r, i));
            }
        }
        ptr.push(rows.len());
    }
    match class {
        0 => SparseTensor::new_f32(
            m,
            n,
            ptr,
            rows,
            values.into_iter().map(|(r, _)| r as f32).collect(),
        ),
        1 => SparseTensor::new(
            m,
            n,
            ptr,
            rows,
            values.into_iter().map(|(r, _)| r).collect(),
        ),
        2 => SparseTensor::new_complex_f32(
            m,
            n,
            ptr,
            rows,
            values
                .into_iter()
                .map(|(r, i)| (r as f32, i as f32))
                .collect(),
        ),
        _ => SparseTensor::new_complex(m, n, ptr, rows, values),
    }
    .unwrap()
}
fn at(value: &Value, row: usize, col: usize) -> Complex64 {
    match value {
        Value::Tensor(t) => match t.numeric_value_at(col * t.shape()[0] + row).unwrap() {
            NumericScalar::F64(r) => Complex64::new(r, 0.),
            other => panic!("unexpected index class: {other:?}"),
        },
        Value::SparseTensor(t) => {
            let start = t.col_ptrs[col];
            let rows = &t.row_indices[start..t.col_ptrs[col + 1]];
            let Ok(index) = rows.binary_search(&row) else {
                return Complex64::new(0., 0.);
            };
            let index = start + index;
            if let Some(v) = t.as_complex_f32_slice() {
                Complex64::new(f64::from(v[index].0), f64::from(v[index].1))
            } else if let Some(v) = t.as_complex_f64_slice() {
                Complex64::new(v[index].0, v[index].1)
            } else {
                match t.numeric_value_at(index).unwrap() {
                    NumericScalar::F32(x) => Complex64::new(f64::from(x), 0.),
                    NumericScalar::F64(x) => Complex64::new(x, 0.),
                    other => panic!("unexpected factor class: {other:?}"),
                }
            }
        }
        other => panic!("unexpected LU output: {other:?}"),
    }
}
fn invoke_value(
    context: &RuntimeContext,
    a: SparseTensor,
    args: Vec<Value>,
    count: Option<usize>,
) -> BuiltinResult<Value> {
    let _scope = crate::output_count::push_output_count(count).unwrap();
    block_on(super::super::lu_builtin_context(
        Some(context.clone()),
        Value::SparseTensor(a),
        crate::execution::RootedValues::new(args)
            .map_err(|error| error.into_runtime_error())?
            .into(),
    ))
}
fn invoke(
    context: &RuntimeContext,
    a: SparseTensor,
    args: Vec<Value>,
    count: usize,
) -> BuiltinResult<Vec<Value>> {
    let output = invoke_value(context, a, args, Some(count))?;
    let Value::OutputList(values) = output else {
        panic!("selected list")
    };
    Ok(values.try_into_owned().unwrap().into_iter().collect())
}
fn assert_class(value: &Value, class: usize, complex: bool) {
    let Value::SparseTensor(s) = value else {
        panic!("sparse factor expected")
    };
    let dtype = if class.is_multiple_of(2) {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    assert_eq!(s.numeric_dtype(), Some(dtype));
    assert_eq!(
        s.as_complex_f32_slice().is_some() || s.as_complex_f64_slice().is_some(),
        complex
    );
}
#[test]
fn native_four_public_lu_sparse_forms_reconstruct_actual_scaled_permutations_in_both_policies() {
    for managed in [false, true] {
        for class in 0..4 {
            for count in [None, Some(1)] {
                let (context, ledger) = runtime(managed, 2_000_000);
                let a = source(class, 2, 2, &[(1., 1.), (2., 0.), (3., 0.), (4., -1.)]);
                let value = invoke_value(&context, a, vec![], count).unwrap();
                let packed = if let Value::OutputList(values) = value {
                    values.try_into_owned().unwrap().into_iter().next().unwrap()
                } else {
                    value
                };
                assert_class(&packed, class, class >= 2);
                let expected = if class >= 2 {
                    [(2., 0.), (0.5, 0.5), (4., -1.), (0.5, -1.5)]
                } else {
                    [(2., 0.), (0.5, 0.), (4., 0.), (1., 0.)]
                };
                for (index, (r, i)) in expected.into_iter().enumerate() {
                    assert!(
                        (at(&packed, index % 2, index / 2) - Complex64::new(r, i)).norm()
                            < if class % 2 == 0 { 1e-6 } else { 1e-12 }
                    );
                }
                drop(packed);
                assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
                assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
            }
            for (m, n, data) in [
                (
                    3,
                    2,
                    vec![(1., 1.), (3., 0.), (5., 0.), (2., 0.), (0., 4.), (6., 0.)],
                ),
                (
                    2,
                    3,
                    vec![(0., 0.), (0., 0.), (1., 1.), (2., 0.), (2., 2.), (4., 0.)],
                ),
            ] {
                for count in 2..=5 {
                    for vector in [false, true] {
                        let (context, ledger) = runtime(managed, 2_000_000);
                        let a = source(class, m, n, &data);
                        let original = Value::SparseTensor(a.clone());
                        let mut args = Vec::new();
                        if count >= 4 {
                            args.push(Value::Tensor(
                                runmat_value::Tensor::new(vec![0.1, 0.001], vec![1, 2, 1]).unwrap(),
                            ));
                        }
                        if vector {
                            args.push(Value::from("vector"));
                        }
                        let outputs = invoke(&context, a, args, count).unwrap();
                        let k = m.min(n);
                        let tolerance = if class % 2 == 0 { 7e-5 } else { 7e-12 };
                        assert_class(&outputs[0], class, class >= 2);
                        assert_class(&outputs[1], class, class >= 2);
                        let Value::SparseTensor(l) = &outputs[0] else {
                            unreachable!()
                        };
                        let Value::SparseTensor(u) = &outputs[1] else {
                            unreachable!()
                        };
                        assert_eq!((l.rows, l.cols), (m, k));
                        assert_eq!((u.rows, u.cols), (k, n));
                        if count == 5 {
                            assert_class(&outputs[4], class, false);
                        }
                        for col in 0..n {
                            for row in 0..m {
                                let original_row = if count < 3 {
                                    row
                                } else if vector {
                                    at(&outputs[2], row, 0).re as usize - 1
                                } else {
                                    (0..m)
                                        .find(|&original| at(&outputs[2], row, original).re == 1.)
                                        .unwrap()
                                };
                                let original_col = if count < 4 {
                                    col
                                } else if vector {
                                    at(&outputs[3], col, 0).re as usize - 1
                                } else {
                                    (0..n)
                                        .find(|&original| at(&outputs[3], original, col).re == 1.)
                                        .unwrap()
                                };
                                let scale = if count == 5 {
                                    at(&outputs[4], original_row, original_row).re
                                } else {
                                    1.
                                };
                                let expected = at(&original, original_row, original_col) / scale;
                                let product = (0..k)
                                    .map(|i| at(&outputs[0], row, i) * at(&outputs[1], i, col))
                                    .sum::<Complex64>();
                                assert!(
                                    (product - expected).norm()
                                        <= tolerance * (1. + expected.norm())
                                );
                            }
                        }
                        let clone = outputs.clone();
                        drop(outputs);
                        if managed {
                            assert!(ledger.snapshot().live_claimed_bytes > 0);
                        }
                        drop(clone);
                        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
                        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
                    }
                }
            }
        }
    }
}

#[test]
fn sparse_lu_empty_virtual_rows_denied_growth_and_pre_cancelled_control_have_no_leaks_or_replay() {
    let (context, ledger) = runtime(true, 100_000);
    let empty = SparseTensor::new(usize::MAX, 0, vec![0], vec![], vec![]).unwrap();
    let outputs = invoke(&context, empty, vec![], 2).unwrap();
    let Value::SparseTensor(l) = &outputs[0] else {
        unreachable!()
    };
    assert_eq!((l.rows, l.cols), (usize::MAX, 0));
    drop(outputs);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let (denied, small) = runtime(true, 1);
    let a = source(0, 2, 2, &[(1., 0.), (2., 0.), (3., 0.), (4., 0.)]);
    let error = invoke(&denied, a.clone(), vec![], 5).unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(small.snapshot().live_claimed_bytes, 0);
    assert_eq!(small.snapshot().live_reserved_bytes, 0);
    for args in [
        vec![Value::from("matrix"), Value::from("vector")],
        vec![Value::Tensor(
            runmat_value::Tensor::new(vec![0.1, 0.001], vec![1, 1, 2]).unwrap(),
        )],
    ] {
        let error = invoke(&context, a.clone(), args, 4).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:lu:InvalidArgument"));
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
    let source = runmat_execution::cancellation::CancellationSource::new();
    let context = context.with_cancellation_token(source.token());
    source.cancel(runmat_execution::CancellationReason::User);
    let error = invoke(&context, a, vec![], 2).unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
