use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServicePorts,
};
struct Managed(ManagedHostEffectLedger);
impl RuntimeParallelService for Managed {
    fn supports(&self, _: ParallelCapability) -> bool {
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
fn managed(ledger: &ManagedHostEffectLedger) -> RuntimeContext {
    runtime().with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
    )
}
fn empty(ledger: &ManagedHostEffectLedger) {
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
fn assert_retained(ledger: &ManagedHostEffectLedger, bytes: u64) {
    let snapshot = ledger.snapshot();
    assert_eq!(snapshot.live_claimed_bytes, bytes);
    // Retained output claims continue to consume capacity after invocation
    // headroom retires. Zero reservation here would allow double admission.
    assert_eq!(snapshot.live_reserved_bytes, bytes);
    assert_eq!(snapshot.free_bytes(), snapshot.capacity_bytes - bytes);
}
#[test]
fn supplied_cholesky_native_four_permutation_has_original_residual_and_b_orthogonality() {
    for single in [false, true] {
        for complex in [false, true] {
            for limit in [0, 1 << 20] {
                let ledger = ManagedHostEffectLedger::new(limit);
                let runtime = managed(&ledger);
                let a = if single {
                    SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 9.0]).unwrap()
                } else {
                    SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 9.0]).unwrap()
                };
                let r = if complex {
                    Value::ComplexTensor(
                        runmat_value::ComplexTensor::new(
                            vec![(2.0, 0.0), (0.0, 0.0), (1.0, 1.0), (3.0, 0.0)],
                            vec![2, 2],
                        )
                        .unwrap(),
                    )
                } else {
                    Value::Tensor(Tensor::new(vec![2.0, 0.0, 1.0, 3.0], vec![2, 2]).unwrap())
                };
                let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
                let result = call(
                    &runtime,
                    Value::SparseTensor(a),
                    vec![
                        r,
                        Value::Num(2.0),
                        Value::String("largestreal".into()),
                        Value::String("IsCholesky".into()),
                        Value::Bool(true),
                        Value::String("CholeskyPermutation".into()),
                        Value::Tensor(
                            Tensor::new_integer(
                                runmat_value::IntegerStorage::U16(vec![2, 1]),
                                vec![2, 1],
                            )
                            .unwrap(),
                        ),
                    ],
                );
                if limit == 0 {
                    let error = result.unwrap_err();
                    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                    let mut cause: &(dyn std::error::Error + 'static) = &error;
                    loop {
                        if cause.is::<crate::context::host_allocation::HostAllocationError>() {
                            break;
                        }
                        cause = cause.source().expect("original allocation cause retained");
                    }
                    empty(&ledger);
                    continue;
                }
                let output = result.unwrap();
                let Value::OutputList(list) = &output else {
                    panic!("three outputs");
                };
                assert_eq!(list.as_slice()[2], Value::Num(0.0));
                let v = values(&list.as_slice()[0]);
                let d = values(&list.as_slice()[1]);
                let dtype = match &list.as_slice()[0] {
                    Value::Tensor(t) => t.numeric_dtype(),
                    Value::ComplexTensor(t) => t.numeric_dtype(),
                    _ => panic!("vectors"),
                };
                assert_eq!(
                    dtype,
                    if single {
                        NumericDType::F32
                    } else {
                        NumericDType::F64
                    }
                );
                // Independent original B = P*(R'*R)*P' (not the solver action).
                let off = Complex64::new(2.0, if complex { -2.0 } else { 0.0 });
                let diagonal = if complex { 11.0 } else { 10.0 };
                let mass = |x: &[Complex64]| {
                    [diagonal * x[0] + off * x[1], off.conj() * x[0] + 4.0 * x[1]]
                };
                let tolerance = if single { 3e-4 } else { 3e-10 };
                for column in 0..2 {
                    let x = &v[column * 2..column * 2 + 2];
                    let bx = mass(x);
                    let lambda = d[column * 3];
                    assert!((2.0 * x[0] - lambda * bx[0]).norm() < tolerance * (1.0 + x[0].norm()));
                    assert!((9.0 * x[1] - lambda * bx[1]).norm() < tolerance * (1.0 + x[1].norm()));
                    for other in 0..2 {
                        let y = &v[other * 2..other * 2 + 2];
                        let inner = y[0].conj() * bx[0] + y[1].conj() * bx[1];
                        let expected = Complex64::new(if column == other { 1.0 } else { 0.0 }, 0.0);
                        assert!((inner - expected).norm() < tolerance);
                    }
                }
                let retained = list.as_slice().iter().fold(
                    crate::builtins::math::linalg::factor::iterative::output::retained_list_bytes(
                        3,
                    )
                    .unwrap(),
                    |total, value| {
                        use crate::context::host_allocation::HostAllocationPlan;
                        total
                            + match value {
                                Value::Tensor(tensor) => HostAllocationPlan::retained_tensor_bytes(
                                    tensor.numeric_dtype(),
                                    tensor.shape(),
                                )
                                .unwrap(),
                                Value::ComplexTensor(tensor) => {
                                    HostAllocationPlan::retained_complex_tensor_bytes(
                                        tensor.numeric_dtype(),
                                        tensor.shape(),
                                    )
                                    .unwrap()
                                }
                                Value::Num(_) => 0,
                                _ => panic!("numeric eigen publication"),
                            }
                    },
                );
                assert_retained(&ledger, retained);
                let clone = output.clone();
                drop(output);
                assert_retained(&ledger, retained);
                drop(clone);
                empty(&ledger);
            }
        }
    }
}
#[test]
fn control_during_output_is_terminal_and_releases_unpublished_claims() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use futures::task::noop_waker;
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::{
        future::Future,
        num::NonZeroU64,
        task::{Context, Poll},
    };
    for revoke in [false, true] {
        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let runtime = managed(&ledger);
        let geometry = geometry::Geometry::checked(2, 2, 2).unwrap();
        let receipt = HostAllocationReceipt::admit(
            Some(&runtime),
            geometry
                .quote::<f64>(Some(1), projected::Kind::General)
                .unwrap(),
        )
        .unwrap()
        .unwrap();
        let cancellation = CancellationSource::new();
        let quantum = NonZeroU64::new(5).unwrap();
        let active = ledger.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(quantum),
            cancellation.token(),
            move || {
                assert!(
                    active.snapshot().live_claimed_bytes > 0,
                    "output exists before this turn"
                );
                if revoke {
                    active.revoke();
                    Box::pin(async { Ok(()) })
                } else {
                    Box::pin(futures::future::pending())
                }
            },
        );
        let mut work = work::Work::with_test_budget(&runtime, &receipt, budget, quantum).unwrap();
        let eigenvalues = [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)];
        let vectors = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let mut future = Box::pin(publication::publish(
            &eigenvalues,
            &vectors,
            &[1, 1],
            2,
            NumericDType::F64,
            request::Failure::Keep,
            Some(1),
            &mut work,
        ));
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        let first = future.as_mut().poll(&mut context);
        let error = if revoke {
            match first {
                Poll::Ready(Err(error)) => error,
                other => panic!("authority must fail, got {other:?}"),
            }
        } else {
            assert!(first.is_pending());
            assert!(ledger.snapshot().live_claimed_bytes > 0);
            cancellation.cancel(CancellationReason::User);
            match future.as_mut().poll(&mut context) {
                Poll::Ready(Err(error)) => error,
                other => panic!("cancellation must fail, got {other:?}"),
            }
        };
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(
            error.identifier(),
            Some(if revoke {
                "RunMat:parallel:InfrastructureFailure"
            } else {
                "RunMat:Cancelled"
            })
        );
        drop(future);
        drop(work);
        drop(receipt);
        empty(&ledger);
    }
}
#[test]
fn semidefinite_metric_deflation_empty_geometry_and_dense_admission_are_explicit() {
    let ledger = ManagedHostEffectLedger::new(1 << 20);
    let runtime = managed(&ledger);
    // Coupled singular pencil: range(B) is e1, but the finite eigenspace is
    // span([3;-1]). A coordinate-nullspace deletion would return the wrong 2.
    let a = SparseTensor::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![2.0, 1.0, 1.0, 3.0],
    )
    .unwrap();
    let b = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap());
    let outputs = crate::output_count::push_output_count(Some(3)).unwrap();
    let output = call(
        &runtime,
        Value::SparseTensor(a),
        vec![
            b,
            Value::Num(1.0),
            Value::Num(0.0),
            Value::String("IsSymmetricDefinite".into()),
            Value::Bool(true),
        ],
    )
    .unwrap();
    let Value::OutputList(list) = &output else {
        panic!("coupled pencil outputs");
    };
    assert_eq!(list.as_slice()[2], Value::Num(0.0));
    let lambda = values(&list.as_slice()[1])[0];
    let v = values(&list.as_slice()[0]);
    assert!((lambda - Complex64::new(5.0 / 3.0, 0.0)).norm() < 1e-10);
    assert!((2.0 * v[0] + v[1] - lambda * v[0]).norm() < 1e-10);
    assert!((v[0] + 3.0 * v[1]).norm() < 1e-10);
    assert!((v[0].norm_sqr() - 1.0).abs() < 1e-10);
    drop(output);
    drop(outputs);
    empty(&ledger);
    for (index, source) in [
        SparseTensor::new_f32(0, 0, vec![0], vec![], vec![]).unwrap(),
        SparseTensor::new(0, 0, vec![0], vec![], vec![]).unwrap(),
        SparseTensor::new_complex_f32(0, 0, vec![0], vec![], vec![]).unwrap(),
        SparseTensor::new_complex(0, 0, vec![0], vec![], vec![]).unwrap(),
    ]
    .into_iter()
    .enumerate()
    {
        let dtype = if index.is_multiple_of(2) {
            NumericDType::F32
        } else {
            NumericDType::F64
        };
        for requested in [None, Some(0), Some(1), Some(2), Some(3)] {
            let _outputs = crate::output_count::push_output_count(requested).unwrap();
            let output = call(&runtime, Value::SparseTensor(source.clone()), vec![]).unwrap();
            let arrays = if let Value::OutputList(list) = &output {
                assert_eq!(list.as_slice().len(), requested.unwrap());
                if requested == Some(3) {
                    assert_eq!(list.as_slice()[2], Value::Num(0.0));
                }
                &list.as_slice()[..requested.unwrap().min(2)]
            } else {
                assert_eq!(requested, None);
                std::slice::from_ref(&output)
            };
            let shape = if requested.is_some_and(|n| n >= 2) {
                [0, 0]
            } else {
                [0, 1]
            };
            for value in arrays {
                let Value::Tensor(tensor) = value else {
                    panic!("real empty array");
                };
                assert_eq!(tensor.shape(), shape);
                assert_eq!(tensor.numeric_dtype(), dtype);
            }
            let retained = arrays.len() as u64
                * crate::context::host_allocation::HostAllocationPlan::retained_tensor_bytes(
                    dtype, &shape,
                )
                .unwrap() + match requested {
                Some(n @ 1..=3) => {
                    crate::builtins::math::linalg::factor::iterative::output::retained_list_bytes(n)
                        .unwrap()
                }
                _ => 0,
            };
            assert_retained(&ledger, retained);
            let clone = output.clone();
            drop(output);
            assert_retained(&ledger, retained);
            drop(clone);
            empty(&ledger);
        }
    }
    let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
    // Dense compatibility now uses the shared admitted full-EVD owner; it is
    // not a permanently refused DenseWork route. Qualify exact retained outputs
    // under this original finite owner, including alias/drop retirement.
    let output = call(
        &runtime,
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        vec![],
    )
    .expect("funded dense full EVD");
    let Value::OutputList(list) = &output else {
        panic!("three dense outputs");
    };
    assert_eq!(list.len(), 3);
    for value in &list.as_slice()[..2] {
        let Value::Tensor(tensor) = value else {
            panic!("real dense factors");
        };
        assert_eq!(tensor.shape(), [1, 1]);
        assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
    }
    assert!((values(&list.as_slice()[0])[0].norm_sqr() - 1.0).abs() < 1e-12);
    assert_eq!(values(&list.as_slice()[1]), [Complex64::new(1.0, 0.0)]);
    assert_eq!(list.as_slice()[2], Value::Num(0.0));
    let retained = 2 * crate::context::host_allocation::HostAllocationPlan::retained_tensor_bytes(
        NumericDType::F64,
        &[1, 1],
    )
    .unwrap()
        + crate::builtins::math::linalg::factor::iterative::output::retained_list_bytes(3).unwrap();
    assert_retained(&ledger, retained);
    let alias = output.clone();
    drop(output);
    assert_retained(&ledger, retained);
    drop(alias);
    empty(&ledger);

    // This root-free Tensor needs no GC registration. Zero capacity denies
    // the selected dense route's prefixed argument carrier, not its later EVD
    // workspace. The funded case above exercises the dense owner; neither
    // admission failure may trigger gather replay.
    let denied_ledger = ManagedHostEffectLedger::new(0);
    let denied_runtime = managed(&denied_ledger);
    let error = call(
        &denied_runtime,
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        vec![],
    )
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(error.identifier(), Some("RunMat:Call:Arguments"));
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    let allocation = loop {
        if let Some(allocation) =
            cause.downcast_ref::<crate::context::host_allocation::HostAllocationError>()
        {
            break allocation;
        }
        cause = cause.source().expect("original dense admission cause");
    };
    assert!(allocation.is_capacity_denial());
    empty(&denied_ledger);
}

#[test]
fn selector_aliases_inverse_scaling_and_nonconvergence_publication_are_explicit() {
    use super::super::super::Sigma;
    let runtime = runtime();
    let mut work = work::Work::new(Some(&runtime), None).unwrap();
    let spectrum = [
        Complex64::new(-3.0, -4.0),
        Complex64::new(2.0, 5.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(7.0, 2.0),
    ];
    let cases = [
        (Sigma::LargestAbs, [3, 1]),
        (Sigma::SmallestAbs, [2, 0]),
        (Sigma::LargestReal, [3, 1]),
        (Sigma::SmallestReal, [0, 2]),
        (Sigma::BothEndsReal, [0, 3]),
        (Sigma::LargestImag, [1, 3]),
        (Sigma::SmallestImag, [0, 2]),
        (Sigma::BothEndsImag, [1, 0]),
        (Sigma::Near(Complex64::new(1.0, 0.0)), [2, 1]),
    ];
    for (sigma, expected) in cases {
        let mut indices = [0; 4];
        let mut selected = [0; 2];
        block_on(selection::select(
            &spectrum,
            &mut indices,
            &mut selected,
            sigma,
            &mut work,
        ))
        .unwrap();
        assert_eq!(selected, expected);
    }
    use super::super::super::controls::{selector, ImaginaryAlias, Text};
    let (sigma, alias) = selector(Text::Utf8("li")).unwrap();
    assert_eq!(alias, ImaginaryAlias::Legacy);
    assert!(matches!(alias.resolve(sigma, false), Sigma::BothEndsImag));
    assert!(matches!(alias.resolve(sigma, true), Sigma::LargestImag));
    for value in [1e-308, 1e308] {
        let inverse = ritz::transformed(Complex64::new(value, 0.0), Sigma::SmallestAbs);
        assert!((inverse.re * value - 1.0).abs() < 1e-14);
        assert_eq!(inverse.im, 0.0);
    }
    drop(work);
    // Deliberately insufficient subspace/iterations; this tests real residual
    // failure rather than setting a status bit through a test-only hook.
    let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
    for failure in ["keep", "drop", "replacenan"] {
        let a = SparseTensor::new(
            8,
            8,
            (0..=8).collect::<Vec<usize>>(),
            (0..8).collect::<Vec<usize>>(),
            vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        )
        .unwrap();
        let output = call(
            &runtime,
            Value::SparseTensor(a),
            vec![
                Value::Num(1.0),
                Value::String("largestreal".into()),
                Value::String("SubspaceDimension".into()),
                Value::Num(3.0),
                Value::String("MaxIterations".into()),
                Value::Num(1.0),
                Value::String("Tolerance".into()),
                Value::Num(1e-30),
                Value::String("FailureTreatment".into()),
                Value::String(failure.into()),
            ],
        )
        .unwrap();
        let Value::OutputList(list) = output else {
            panic!("three outputs")
        };
        assert_eq!(list.as_slice()[2], Value::Num(1.0));
        let eigenvalues = values(&list.as_slice()[1]);
        match failure {
            "drop" => assert!(eigenvalues.is_empty()),
            "replacenan" => assert!(eigenvalues[0].re.is_nan()),
            _ => {
                assert!(eigenvalues[0].re.is_finite());
                let v = values(&list.as_slice()[0]);
                let residual = v
                    .iter()
                    .enumerate()
                    .map(|(row, &v)| {
                        ((2.0f64.powi(row as i32)) * v - eigenvalues[0] * v).norm_sqr()
                    })
                    .sum::<f64>()
                    .sqrt();
                assert!(residual > 1e-20);
            }
        }
    }
}
