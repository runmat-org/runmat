use std::rc::Rc;

use runmat_value::{Tensor, Value};

use super::{DenseFactor, DenseFactorError};
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;

trait RetainedCase:
    super::DenseScalar + crate::builtins::math::linalg::ops::triangular_operator::TriangularScalar
{
    const COMPLEX_CASE: bool;
    fn matrix(values: &[(f64, f64)], m: usize, n: usize) -> Value;
}
macro_rules! retained_real {
    ($ty:ty,$ctor:ident) => {
        impl RetainedCase for $ty {
            const COMPLEX_CASE: bool = false;
            fn matrix(v: &[(f64, f64)], m: usize, n: usize) -> Value {
                Value::Tensor(
                    Tensor::$ctor(v.iter().map(|v| v.0 as $ty).collect(), vec![m, n]).unwrap(),
                )
            }
        }
    };
}
retained_real!(f32, from_f32);
retained_real!(f64, new);
macro_rules! retained_complex {
    ($ty:ty,$ctor:ident) => {
        impl RetainedCase for runmat_value::ComplexElement<$ty> {
            const COMPLEX_CASE: bool = true;
            fn matrix(v: &[(f64, f64)], m: usize, n: usize) -> Value {
                Value::ComplexTensor(
                    runmat_value::ComplexTensor::$ctor(
                        v.iter().map(|v| (v.0 as $ty, v.1 as $ty)).collect(),
                        vec![m, n],
                    )
                    .unwrap(),
                )
            }
        }
    };
}
retained_complex!(f32, from_f32);
retained_complex!(f64, new);

#[test]
fn native_four_retained_dense_factors_use_compact_data_and_independent_solve_owners() {
    use super::{factor_dense, solve_dense, DenseFactorOptions};
    use crate::builtins::math::linalg::ops::sparse_factor::{
        FactorSolveOptions, SingularSolvePolicy,
    };
    use runmat_value::{NumericFactorCandidate as C, NumericFactorKind as K};
    async fn check<T: RetainedCase>(managed: bool) {
        let ledger = crate::context::host_effect::ManagedHostEffectLedger::new(8 * 1024 * 1024);
        let ports = if managed {
            crate::context::RuntimeServicePorts::default()
                .with_parallel(Rc::new(Managed(ledger.clone())))
        } else {
            crate::context::RuntimeServicePorts::default()
        };
        let runtime =
            RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(ports);
        let options = |kind, rank_tolerance| DenseFactorOptions {
            kind,
            rank_tolerance,
            triangle: None,
        };
        let solve_options = |adjoint| FactorSolveOptions {
            adjoint,
            singular: SingularSolvePolicy::PublicDecompositionIeee,
        };
        let lane = |v: (f64, f64)| T::from_parts(v.0, v.1).unwrap();
        // Rank-zero orthogonal solves consume no RHS/coordinate/StackReq
        // workspace or blocking job. The caller's stack sink is independent
        // of the old immutable factor owner and its retained allocation.
        let zero = T::matrix(&[(0.0, 0.0); 6], 2, 3);
        for kind in [K::Qr, K::Cod] {
            let data = factor_dense(&zero, options(kind, None), &runtime)
                .await
                .unwrap();
            let first = match data.parts() {
                C::DenseQr(first) | C::DenseCod { first, .. } => first,
                _ => panic!("orthogonal zero factor expected"),
            };
            assert_eq!(first.rank, 0);
            assert!(T::elements(&first.packed)
                .unwrap()
                .iter()
                .all(|v| v.parts() == (0.0, 0.0)));
            let (block, columns) = first.coefficients.dimensions();
            for (index, value) in T::elements(&first.coefficients).unwrap().iter().enumerate() {
                let expected = if index % block == (index / block) % block {
                    (f64::INFINITY, 0.0)
                } else {
                    (0.0, 0.0)
                };
                assert_eq!(value.parts(), expected);
            }
            assert_eq!(columns, 2);
            let no_scratch = crate::context::host_effect::ManagedHostEffectLedger::new(0);
            let current = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
                .with_service_ports(
                    crate::context::RuntimeServicePorts::default()
                        .with_parallel(Rc::new(Managed(no_scratch.clone()))),
                );
            let mut output = [lane((9.0, 0.0)); 3];
            let status = solve_dense::<T, _>(
                &data,
                &[lane((1.0, 0.0)), lane((2.0, 0.0))],
                1,
                solve_options(false),
                &current,
                |_, v| Some(v),
                |i, v| {
                    output[i] = v;
                    Ok(())
                },
            )
            .await
            .unwrap();
            assert_eq!(status.deficient_rank, Some(0));
            assert!(output.iter().all(|v| v.parts() == (0.0, 0.0)));
            assert_eq!(no_scratch.snapshot().live_claimed_bytes, 0);
            assert_eq!(no_scratch.snapshot().live_reserved_bytes, 0);
        }
        let tol = if T::DTYPE == runmat_value::NumericDType::F32 {
            2e-4
        } else {
            2e-11
        };
        async fn solve<T: RetainedCase>(
            data: &runmat_value::NumericFactorData,
            rhs: &[T],
            expected: &[(f64, f64)],
            options: FactorSolveOptions,
            runtime: &RuntimeContext,
            tol: f64,
        ) {
            let mut output = vec![T::default(); expected.len()];
            solve_dense::<T, _>(
                data,
                rhs,
                1,
                options,
                runtime,
                |_, v| Some(v),
                |i, v| {
                    output[i] = v;
                    Ok(())
                },
            )
            .await
            .unwrap();
            for (actual, expected) in output.iter().zip(expected) {
                let (re, im) = actual.parts();
                assert!(
                    (re - expected.0).hypot(im - expected.1) < tol,
                    "{re}+{im}i != {expected:?}"
                );
            }
        }
        let imaginary = if T::COMPLEX_CASE { 1.0 } else { 0.0 };
        let hermitian = T::matrix(
            &[(3.0, 0.0), (1.0, -imaginary), (1.0, imaginary), (4.0, 0.0)],
            2,
            2,
        );
        for kind in [K::Lu, K::Cholesky, K::Qr, K::Cod] {
            let data = factor_dense(&hermitian, options(kind, None), &runtime)
                .await
                .unwrap();
            if let C::DenseCholesky { lower } = data.parts() {
                let values = T::elements(lower).unwrap();
                assert_eq!(values[2].parts(), (0.0, 0.0));
                for index in [0, 3] {
                    let (real, imag) = values[index].parts();
                    assert!(real > 0.0 && real.is_finite());
                    assert_eq!(imag, 0.0);
                }
            }
            assert!(matches!(
                data.parts(),
                C::DenseLu { .. } | C::DenseCholesky { .. } | C::DenseQr(_) | C::DenseCod { .. }
            ));
            let rhs = [lane((5.0, 2.0 * imaginary)), lane((9.0, -imaginary))];
            solve::<T>(
                &data,
                &rhs,
                &[(1.0, 0.0), (2.0, 0.0)],
                solve_options(false),
                &runtime,
                tol,
            )
            .await;
            if kind != K::Qr {
                solve::<T>(
                    &data,
                    &rhs,
                    &[(1.0, 0.0), (2.0, 0.0)],
                    solve_options(true),
                    &runtime,
                    tol,
                )
                .await;
            }
        }
        let pivoted = T::matrix(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], 2, 2);
        let data = factor_dense(&pivoted, options(K::Lu, None), &runtime)
            .await
            .unwrap();
        solve::<T>(
            &data,
            &[lane((4.0, 0.0)), lane((7.0, 0.0))],
            &[(1.0, 0.0), (2.0, 0.0)],
            solve_options(false),
            &runtime,
            tol,
        )
        .await;
        solve::<T>(
            &data,
            &[lane((2.0, 0.0)), lane((8.0, 0.0))],
            &[(1.0, 0.0), (2.0, 0.0)],
            solve_options(true),
            &runtime,
            tol,
        )
        .await;
        drop(data);
        // Dependent rows and a nontrivial complex row. Independent minimum-
        // norm oracle x=A_row^H*6/||A_row||²; x is orthogonal to its nullspace.
        let wide = T::matrix(
            &[
                (1.0, imaginary),
                (2.0, 2.0 * imaginary),
                (2.0, 0.0),
                (4.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ],
            2,
            3,
        );
        let norm2 = 5.0 + imaginary * imaginary;
        let rhs = [lane((6.0, 0.0)), lane((12.0, 0.0))];
        let basic = factor_dense(&wide, options(K::Qr, None), &runtime)
            .await
            .unwrap();
        solve::<T>(
            &basic,
            &rhs,
            &[(0.0, 0.0), (3.0, 0.0), (0.0, 0.0)],
            solve_options(false),
            &runtime,
            tol,
        )
        .await;
        let cod = factor_dense(&wide, options(K::Cod, None), &runtime)
            .await
            .unwrap();
        solve::<T>(
            &cod,
            &rhs,
            &[
                (6.0 / norm2, -6.0 * imaginary / norm2),
                (12.0 / norm2, 0.0),
                (0.0, 0.0),
            ],
            solve_options(false),
            &runtime,
            tol,
        )
        .await;
        solve::<T>(
            &cod,
            &[lane((1.0, -imaginary)), lane((2.0, 0.0)), lane((0.0, 0.0))],
            &[(0.2, 0.0), (0.4, 0.0)],
            solve_options(true),
            &runtime,
            tol,
        )
        .await;
        drop(basic);
        drop(cod);
        let full_row = T::matrix(
            &[
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
            ],
            2,
            3,
        );
        let cod = factor_dense(&full_row, options(K::Cod, None), &runtime)
            .await
            .unwrap();
        solve::<T>(
            &cod,
            &[lane((1.0, 0.0)), lane((2.0, 0.0))],
            &[(0.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
            solve_options(false),
            &runtime,
            tol,
        )
        .await;
        solve::<T>(
            &cod,
            &[lane((2.0, 0.0)), lane((3.0, 0.0)), lane((5.0, 0.0))],
            &[(2.0, 0.0), (3.0, 0.0)],
            solve_options(true),
            &runtime,
            tol,
        )
        .await;
        drop(cod);
        let diagonal = T::matrix(&[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1e-6, 0.0)], 2, 2);
        for (threshold, last) in [(5e-7, 1.0), (2e-6, 0.0)] {
            let data = factor_dense(&diagonal, options(K::Cod, Some(threshold)), &runtime)
                .await
                .unwrap();
            solve::<T>(
                &data,
                &[lane((1.0, 0.0)), lane((1e-6, 0.0))],
                &[(1.0, 0.0), (last, 0.0)],
                solve_options(false),
                &runtime,
                tol,
            )
            .await;
        }
        // 17-by-17 selects faer's block_size=4. Rank five deliberately ends
        // inside a block, exercising the truncated coefficient block geometry.
        let mut entries = vec![(0.0, 0.0); 17 * 17];
        for i in 0..17 {
            entries[i * 17 + i] = (if i < 5 { 1.0 } else { 1e-6 }, 0.0);
        }
        let blocked = T::matrix(&entries, 17, 17);
        let cod = factor_dense(&blocked, options(K::Cod, Some(2e-6)), &runtime)
            .await
            .unwrap();
        let rhs: Vec<T> = (0..17)
            .map(|i| lane((if i < 5 { 1.0 } else { 1e-6 }, 0.0)))
            .collect();
        let expected: Vec<_> = (0..17)
            .map(|i| (if i < 5 { 1.0 } else { 0.0 }, 0.0))
            .collect();
        solve::<T>(&cod, &rhs, &expected, solve_options(false), &runtime, tol).await;
        solve::<T>(&cod, &rhs, &expected, solve_options(true), &runtime, tol).await;
        drop(cod);
        let data = factor_dense(&hermitian, options(K::Cholesky, None), &runtime)
            .await
            .unwrap();
        let clone = data.clone();
        drop(runtime);
        if managed {
            assert!(ledger.snapshot().live_claimed_bytes > 0);
            ledger.revoke();
        }
        let current = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let rhs = [lane((5.0, 2.0 * imaginary)), lane((9.0, -imaginary))];
        let mut a = [T::default(); 2];
        let mut b = [T::default(); 2];
        let results = futures::join!(
            solve_dense::<T, _>(
                &data,
                &rhs,
                1,
                solve_options(false),
                &current,
                |_, v| Some(v),
                |i, v| {
                    a[i] = v;
                    Ok(())
                }
            ),
            solve_dense::<T, _>(
                &clone,
                &rhs,
                1,
                solve_options(true),
                &current,
                |_, v| Some(v),
                |i, v| {
                    b[i] = v;
                    Ok(())
                }
            )
        );
        results.0.unwrap();
        results.1.unwrap();
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            assert!((a.parts().0 - (i + 1) as f64).abs() < tol);
            assert!((b.parts().0 - (i + 1) as f64).abs() < tol);
        }
        drop(data);
        drop(clone);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
    for managed in [false, true] {
        futures::executor::block_on(async {
            check::<f32>(managed).await;
            check::<f64>(managed).await;
            check::<runmat_value::ComplexElement<f32>>(managed).await;
            check::<runmat_value::ComplexElement<f64>>(managed).await;
        });
    }
}

struct Managed(crate::context::host_effect::ManagedHostEffectLedger);
impl crate::context::RuntimeParallelService for Managed {
    fn supports(&self, _: crate::context::ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> crate::context::RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        crate::context::RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(
        &self,
    ) -> Option<crate::context::host_effect::ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

#[test]
fn factor_exact_budget_reuses_storage_and_refunds_on_denial_or_drop() {
    let (plan, _) = super::kernel::quote::<f64>(2).unwrap();
    let bytes = plan.scratch_peak_bytes() + plan.retained_output_bytes();
    for indexed in [false, true] {
        for limit in [bytes - 1, bytes] {
            let ledger = crate::context::host_effect::ManagedHostEffectLedger::new(limit);
            let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
                .with_service_ports(
                    // This is the numerical workspace quote, not a quote for
                    // an independent hosted job/channel. Select the genuine
                    // inline NeverYield path without expanding its byte grant.
                    crate::context::RuntimeServicePorts::default()
                        .without_blocking_cpu()
                        .with_parallel(Rc::new(Managed(ledger.clone()))),
                );
            let matrix = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap());
            let reads = std::cell::Cell::new(0);
            let result = if indexed {
                futures::executor::block_on(DenseFactor::<f64>::factor_indexed(
                    2,
                    Some(&runtime),
                    async |index| {
                        reads.set(reads.get() + 1);
                        Ok([0.0, 1.0, 2.0, 3.0][index])
                    },
                ))
            } else {
                futures::executor::block_on(DenseFactor::<f64>::factor(&matrix, 2, &runtime))
            };
            assert_eq!(reads.get(), if indexed && limit == bytes { 4 } else { 0 });
            if limit < bytes {
                assert!(matches!(result, Err(DenseFactorError::Admission(_))));
            } else {
                let mut factor = result.unwrap();
                let claimed = ledger.snapshot().live_claimed_bytes;
                assert!(claimed > 0);
                for (input, expected) in [([10.0, 19.0], [4.0, 5.0]), ([-2.0, -1.0], [2.0, -1.0])] {
                    let mut output = [0.0; 2];
                    futures::executor::block_on(factor.solve_into(&input, &mut output, &runtime))
                        .unwrap();
                    for (actual, expected) in output.iter().zip(expected) {
                        assert!((actual - expected).abs() < 1e-12);
                    }
                    assert_eq!(ledger.snapshot().live_claimed_bytes, claimed);
                }
                drop(factor);
            }
            assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
    }
}

#[test]
fn pivoted_native_factor_reuses_one_owner_for_two_right_hand_sides() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    // Column-major A = [0 2; 1 3], requiring a row pivot.
    let matrix = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap());
    let mut factor =
        futures::executor::block_on(DenseFactor::<f64>::factor(&matrix, 2, &runtime)).unwrap();
    let mut first = [0.0; 2];
    futures::executor::block_on(factor.solve_into(&[10.0, 19.0], &mut first, &runtime)).unwrap();
    assert!((first[0] - 4.0).abs() < 1e-12);
    assert!((first[1] - 5.0).abs() < 1e-12);
    let mut second = [0.0; 2];
    futures::executor::block_on(factor.solve_into(&[-2.0, -1.0], &mut second, &runtime)).unwrap();
    assert!((second[0] - 2.0).abs() < 1e-12);
    assert!((second[1] + 1.0).abs() < 1e-12);

    let singular = Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
    assert!(matches!(
        futures::executor::block_on(DenseFactor::<f64>::factor(&singular, 1, &runtime)),
        Err(DenseFactorError::Singular)
    ));
}
