use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServicePorts,
};
use runmat_value::{NumericDType, Value};
use std::rc::Rc;

struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
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
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
        );
    (runtime, ledger)
}
fn empty(ledger: &ManagedHostEffectLedger) {
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn full_evd_exact_owners_preserve_complex_pencil_residuals_and_retire_on_control() {
    futures::executor::block_on(async {
        let a = [
            Complex64::new(1., 1.),
            Complex64::new(0., 0.),
            Complex64::new(2., -0.5),
            Complex64::new(3., 2.),
        ];
        let b = [
            Complex64::new(2., -1.),
            Complex64::new(0., 0.),
            Complex64::new(0., 0.),
            Complex64::new(1., 0.25),
        ];
        for problem in [Problem::Standard, Problem::Generalized] {
            for vectors in [Vectors::Right, Vectors::LeftAndRight] {
                let (context, ledger) = runtime(1 << 24);
                let mut prepared = PreparedEvd::new(2, problem, vectors, Some(&context))
                    .await
                    .unwrap();
                prepared.matrix_mut().copy_from_slice(&a);
                if let Some(mass) = prepared.mass_mut() {
                    mass.copy_from_slice(&b);
                }
                let result = prepared.compute(Some(&context)).await.unwrap();
                result.check_owner(Some(&context)).unwrap();
                for column in 0..2 {
                    let lambda = result.values()[column];
                    for row in 0..2 {
                        let av = (0..2)
                            .map(|j| a[row + 2 * j] * result.right()[j + 2 * column])
                            .sum::<Complex64>();
                        let bv = if problem == Problem::Generalized {
                            (0..2)
                                .map(|j| b[row + 2 * j] * result.right()[j + 2 * column])
                                .sum()
                        } else {
                            result.right()[row + 2 * column]
                        };
                        assert!((av - lambda * bv).norm() < 1e-9);
                    }
                    if let Some(left) = result.left() {
                        for j in 0..2 {
                            let wa = (0..2)
                                .map(|row| left[row + 2 * column].conj() * a[row + 2 * j])
                                .sum::<Complex64>();
                            let wb = if problem == Problem::Generalized {
                                (0..2)
                                    .map(|row| left[row + 2 * column].conj() * b[row + 2 * j])
                                    .sum()
                            } else {
                                left[j + 2 * column].conj()
                            };
                            assert!((wa - lambda * wb).norm() < 1e-9);
                        }
                        let pairing = (0..2)
                            .map(|row| {
                                let bv = if problem == Problem::Generalized {
                                    (0..2)
                                        .map(|j| b[row + 2 * j] * result.right()[j + 2 * column])
                                        .sum()
                                } else {
                                    result.right()[row + 2 * column]
                                };
                                left[row + 2 * column].conj() * bv
                            })
                            .sum::<Complex64>();
                        assert!((pairing - Complex64::new(1., 0.)).norm() < 1e-9);
                    }
                }
                let output =
                    publication::matrix(Some(&context), NumericDType::F32, 2, 2, |index| {
                        result.right()[index]
                    })
                    .await
                    .unwrap();
                let dtype = match &output {
                    Value::Tensor(v) => v.numeric_dtype(),
                    Value::ComplexTensor(v) => v.numeric_dtype(),
                    _ => panic!("numeric EVD output"),
                };
                assert_eq!(dtype, NumericDType::F32);
                let retained = output.clone();
                drop((output, result));
                assert!(ledger.snapshot().live_claimed_bytes > 0);
                drop(retained);
                empty(&ledger);
            }
        }
        // A coupled singular regular pencil: one finite and one infinite
        // eigenvalue. Inspect original alpha/beta before in-place projection,
        // through the same admitted CPU job, without retaining an alpha mirror.
        #[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
        for scale in [0., 1e-200] {
            let mass = [
                Complex64::new(2., -0.25),
                Complex64::new(0., 0.),
                Complex64::new(1., 1.),
                Complex64::new(scale, 0.),
            ];
            let (context, ledger) = runtime(1 << 24);
            let mut prepared = PreparedEvd::new(
                2,
                Problem::Generalized,
                Vectors::LeftAndRight,
                Some(&context),
            )
            .await
            .unwrap();
            prepared.matrix_mut().copy_from_slice(&a);
            prepared.mass_mut().unwrap().copy_from_slice(&mass);
            let PreparedEvd { workspace, receipt } = prepared;
            let (workspace, ()) =
                run_job(Some(&context), receipt.as_ref(), workspace, move |owner| {
                    generalized::compute_homogeneous(owner)?;
                    let beta = owner.generalized.as_ref().unwrap().beta.get();
                    assert_eq!(
                        beta.iter()
                            .filter(|&&z| z == Complex64::new(0., 0.))
                            .count(),
                        usize::from(scale == 0.)
                    );
                    for column in 0..2 {
                        let alpha = owner.values.get()[column];
                        for row in 0..2 {
                            let av = (0..2)
                                .map(|j| a[row + 2 * j] * owner.right.get()[j + 2 * column])
                                .sum::<Complex64>();
                            let bv = (0..2)
                                .map(|j| mass[row + 2 * j] * owner.right.get()[j + 2 * column])
                                .sum::<Complex64>();
                            assert!((beta[column] * av - alpha * bv).norm() < 1e-9);
                        }
                        let left = owner.left.as_ref().unwrap().get();
                        for j in 0..2 {
                            let wa = (0..2)
                                .map(|row| left[row + 2 * column].conj() * a[row + 2 * j])
                                .sum::<Complex64>();
                            let wb = (0..2)
                                .map(|row| left[row + 2 * column].conj() * mass[row + 2 * j])
                                .sum::<Complex64>();
                            assert!((beta[column] * wa - alpha * wb).norm() < 1e-9);
                        }
                    }
                    Ok(())
                })
                .await
                .unwrap();
            drop((workspace, receipt));
            empty(&ledger);
        }
        // Cross an indeterminate trailing Schur block while recovering both
        // finite and infinite vectors. The pinned GEVD private backsolve used
        // reciprocal(0) even for a consistent zero equation here. Coupling is
        // nonzero in the second row; no diagonal-only shortcut is sufficient.
        #[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
        for coupling in [0., 0.5] {
            for vectors in [Vectors::Right, Vectors::LeftAndRight] {
                let zero = Complex64::new(0., 0.);
                let a = [
                    Complex64::new(2., 1.),
                    zero,
                    zero,
                    Complex64::new(0.25, -0.5),
                    Complex64::new(4., -1.),
                    zero,
                    zero,
                    Complex64::new(coupling, 0.25 * coupling),
                    zero,
                ];
                let b = [
                    Complex64::new(1., 0.),
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                ];
                let (context, ledger) = runtime(1 << 24);
                let mut prepared =
                    PreparedEvd::new(3, Problem::Generalized, vectors, Some(&context))
                        .await
                        .unwrap();
                prepared.matrix_mut().copy_from_slice(&a);
                prepared.mass_mut().unwrap().copy_from_slice(&b);
                let PreparedEvd { workspace, receipt } = prepared;
                let (workspace, ()) =
                    run_job(Some(&context), receipt.as_ref(), workspace, move |owner| {
                        generalized::compute_homogeneous(owner)?;
                        let beta = owner.generalized.as_ref().unwrap().beta.get();
                        let alpha = owner.values.get();
                        assert_eq!(
                            (0..3)
                                .filter(|&i| alpha[i] == zero && beta[i] == zero)
                                .count(),
                            1
                        );
                        assert_eq!(
                            (0..3)
                                .filter(|&i| alpha[i] != zero && beta[i] == zero)
                                .count(),
                            1
                        );
                        assert!(owner
                            .right
                            .get()
                            .iter()
                            .chain(owner.left.iter().flat_map(workspace::Buffer::get))
                            .all(|z| z.re.is_finite() && z.im.is_finite()));
                        for column in 0..3 {
                            for row in 0..3 {
                                let av = (0..3)
                                    .map(|j| a[row + 3 * j] * owner.right.get()[j + 3 * column])
                                    .sum::<Complex64>();
                                let bv = (0..3)
                                    .map(|j| b[row + 3 * j] * owner.right.get()[j + 3 * column])
                                    .sum::<Complex64>();
                                assert!((beta[column] * av - alpha[column] * bv).norm() < 1e-9);
                            }
                            if let Some(left) = &owner.left {
                                for j in 0..3 {
                                    let wa = (0..3)
                                        .map(|row| {
                                            left.get()[row + 3 * column].conj() * a[row + 3 * j]
                                        })
                                        .sum::<Complex64>();
                                    let wb = (0..3)
                                        .map(|row| {
                                            left.get()[row + 3 * column].conj() * b[row + 3 * j]
                                        })
                                        .sum::<Complex64>();
                                    assert!((beta[column] * wa - alpha[column] * wb).norm() < 1e-9);
                                }
                            }
                        }
                        Ok(())
                    })
                    .await
                    .unwrap();
                drop((workspace, receipt));
                empty(&ledger);
            }
        }
        // Mixed scales in different columns must not erase the tiny pairing.
        // These are exact eigenvectors of A=B, so no separate eigensolver is
        // involved in this normalization regression. Use admitted owner arrays.
        {
            let (context, ledger) = runtime(1 << 24);
            let mut prepared = PreparedEvd::new(
                2,
                Problem::Generalized,
                Vectors::LeftAndRight,
                Some(&context),
            )
            .await
            .unwrap();
            let owner = &mut prepared.workspace;
            owner.mass.as_mut().unwrap().get_mut().copy_from_slice(&[
                Complex64::new(1e200, 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(1e-200, 0.),
            ]);
            owner.right.get_mut().copy_from_slice(&[
                Complex64::new(0., 1.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(1., 0.),
            ]);
            owner.left.as_mut().unwrap().get_mut().copy_from_slice(&[
                Complex64::new(1., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 1.),
            ]);
            owner
                .generalized
                .as_mut()
                .unwrap()
                .beta
                .get_mut()
                .fill(Complex64::new(1., 0.));
            normalization::left(
                owner.left.as_mut().unwrap().get_mut(),
                owner.right.get(),
                2,
                normalization::Metric::Pencil {
                    mass: owner.mass.as_ref().unwrap().get(),
                    beta: owner.generalized.as_ref().unwrap().beta.get(),
                },
            );
            let left = owner.left.as_ref().unwrap().get();
            for column in 0..2 {
                let pairing = left[column + 2 * column].conj()
                    * owner.mass.as_ref().unwrap().get()[column + 2 * column]
                    * owner.right.get()[column + 2 * column];
                assert!((pairing - Complex64::new(1., 0.)).norm() < 1e-12);
            }
            assert!(left[0].im.abs() < 1e-199 && left[3].re > 1e199);
            // Independent real/imaginary exponent families: the large real
            // terms cancel, leaving exactly i*1e-200. Common complex-factor
            // scaling would erase that entire representable pairing.
            owner.mass.as_mut().unwrap().get_mut().copy_from_slice(&[
                Complex64::new(1e200, 1e-200),
                Complex64::new(0., 0.),
                Complex64::new(-1e200, 0.),
                Complex64::new(1., 0.),
            ]);
            owner.right.get_mut().copy_from_slice(&[
                Complex64::new(1., 0.),
                Complex64::new(1., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
            ]);
            owner.left.as_mut().unwrap().get_mut().copy_from_slice(&[
                Complex64::new(1., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
            ]);
            normalization::left(
                owner.left.as_mut().unwrap().get_mut(),
                owner.right.get(),
                2,
                normalization::Metric::Pencil {
                    mass: owner.mass.as_ref().unwrap().get(),
                    beta: owner.generalized.as_ref().unwrap().beta.get(),
                },
            );
            let scaled = owner.left.as_ref().unwrap().get()[0];
            assert_eq!(scaled.re, 0.);
            assert!((scaled.im / 1e200 - 1.).abs() < 1e-12);
            prepared.check_owner(Some(&context)).unwrap();
            drop(prepared);
            empty(&ledger);
        }
        // Small nonzero B is not an infinite eigenvalue and does not waive
        // representable left pairing. This checks the actual final owner on
        // both feature-selected kernels without divided-by-infinity residuals.
        {
            let (context, ledger) = runtime(1 << 24);
            let mut prepared = PreparedEvd::new(
                2,
                Problem::Generalized,
                Vectors::LeftAndRight,
                Some(&context),
            )
            .await
            .unwrap();
            prepared.matrix_mut().copy_from_slice(&a);
            let mass = b.map(|value| value * 1e-200);
            prepared.mass_mut().unwrap().copy_from_slice(&mass);
            let result = prepared.compute(Some(&context)).await.unwrap();
            assert!(result
                .values()
                .iter()
                .all(|z| z.re.is_finite() && z.im.is_finite()));
            let left = result.left().unwrap();
            for column in 0..2 {
                let pair = (0..2)
                    .map(|row| {
                        let bv = (0..2)
                            .map(|j| mass[row + 2 * j] * result.right()[j + 2 * column])
                            .sum::<Complex64>();
                        left[row + 2 * column].conj() * bv
                    })
                    .sum::<Complex64>();
                assert!((pair - Complex64::new(1., 0.)).norm() < 1e-9);
            }
            drop(result);
            empty(&ledger);
        }
        // Full eig and eigs subset publication intentionally differ for an
        // imaginary NaN (reachable in the existing singular-pencil policy).
        for projection in [
            publication::RealProjection::EigFactor,
            publication::RealProjection::EigsSubset,
        ] {
            let (context, ledger) = runtime(1 << 20);
            let output = publication::matrix_with_projection(
                Some(&context),
                NumericDType::F64,
                1,
                1,
                projection,
                |_| Complex64::new(2., f64::NAN),
            )
            .await
            .unwrap();
            match projection {
                publication::RealProjection::EigFactor => {
                    assert!(matches!(&output, Value::Tensor(value)
                        if value.numeric_value_at(0).unwrap().materialize_f64() == 2.));
                }
                publication::RealProjection::EigsSubset => {
                    assert!(matches!(&output, Value::ComplexTensor(value)
                        if value.numeric_value_at(0).unwrap().1.materialize_f64().is_nan()));
                }
            }
            drop(output);
            empty(&ledger);
        }
        for problem in [Problem::Standard, Problem::Generalized] {
            let (denied, ledger) = runtime(1);
            assert!(matches!(
                PreparedEvd::new(2, problem, Vectors::Right, Some(&denied)).await,
                Err(DenseEvdError::Admission(_))
            ));
            empty(&ledger);
            let (cancelled, ledger) = runtime(1 << 20);
            let cancellation = runmat_execution::cancellation::CancellationSource::new();
            let cancelled = cancelled.with_cancellation_token(cancellation.token());
            let prepared = PreparedEvd::new(2, problem, Vectors::Right, Some(&cancelled))
                .await
                .unwrap();
            assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
            assert!(matches!(
                prepared.compute(Some(&cancelled)).await,
                Err(DenseEvdError::Admission(HostAllocationError::Cancelled))
            ));
            empty(&ledger);
        }
        let ordinary_cancellation = runmat_execution::cancellation::CancellationSource::new();
        let ordinary = RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            ordinary_cancellation.token(),
        );
        check(None, Some(&ordinary)).unwrap();
        assert!(ordinary_cancellation.cancel(runmat_execution::CancellationReason::User));
        assert!(ordinary.is_cancelled());
        assert!(matches!(
            check(None, Some(&ordinary)),
            Err(DenseEvdError::Execution(ExecutionServiceError::Cancelled))
        ));
    });
}
