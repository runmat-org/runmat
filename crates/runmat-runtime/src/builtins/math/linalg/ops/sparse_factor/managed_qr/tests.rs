use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    RuntimeContext, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use runmat_value::SparseTensor;
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

trait Case: SparseQrScalar {
    fn native(real: f64, imag: f64) -> Self;
    fn sparse(m: usize, n: usize, p: Vec<usize>, r: Vec<usize>, v: Vec<Self>) -> SparseTensor;
    const TOLERANCE: f64;
    const TINY: f64;
}

fn public_factor_cases<T: Case>() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    // A zero leading column must not stop unpivoted public QR. A subsequent
    // nonzero column below the COD tolerance must remain in the factor.
    for pivot in [false, true] {
        let a = T::sparse(
            3,
            3,
            vec![0, 0, 1, 2],
            vec![1, 2],
            vec![T::native(1.0, 0.0), T::native(T::TINY, 0.0)],
        );
        let factor = block_on(SparseQrDecomposition::<T>::factor(&a, pivot, &runtime)).unwrap();
        let mut q = vec![T::default(); 9];
        for i in 0..3 {
            q[i * 3 + i] = T::one();
        }
        block_on(factor.apply_q(&mut q, 3, false, &runtime)).unwrap();
        let p = factor.permutation(&runtime).unwrap();
        for column in 0..3 {
            let (rows, values) = factor.r_column(column, &runtime).unwrap();
            for row in 0..3 {
                let mut reconstructed = T::default();
                for (&k, &r) in rows.iter().zip(values) {
                    reconstructed = reconstructed + q[k * 3 + row] * r;
                }
                let expected = if (p[column], row) == (1, 1) {
                    T::one()
                } else if (p[column], row) == (2, 2) {
                    T::native(T::TINY, 0.0)
                } else {
                    T::default()
                };
                let scale = if expected == T::default() {
                    1.0
                } else {
                    expected.parts().0.abs()
                };
                let (re, im) = (reconstructed - expected).parts();
                assert!(re.hypot(im) <= T::TOLERANCE * scale);
            }
        }
    }
}

#[test]
fn untruncated_public_qr_native_four_zero_leading_and_tiny_columns() {
    public_factor_cases::<f32>();
    public_factor_cases::<f64>();
    public_factor_cases::<faer::c32>();
    public_factor_cases::<faer::c64>();
}
macro_rules! real {
    ($t:ty,$ctor:ident,$tol:expr,$tiny:expr) => {
        impl Case for $t {
            fn native(r: f64, _: f64) -> Self {
                r as Self
            }
            fn sparse(
                m: usize,
                n: usize,
                p: Vec<usize>,
                r: Vec<usize>,
                v: Vec<Self>,
            ) -> SparseTensor {
                SparseTensor::$ctor(m, n, p, r, v).unwrap()
            }
            const TOLERANCE: f64 = $tol;
            const TINY: f64 = $tiny;
        }
    };
}
real!(f32, new_f32, 2e-5, 1e-30);
real!(f64, new, 2e-12, 1e-250);
macro_rules! complex {
    ($t:ty,$lane:ty,$ctor:ident,$tol:expr,$tiny:expr) => {
        impl Case for $t {
            fn native(r: f64, i: f64) -> Self {
                Self::new(r as $lane, i as $lane)
            }
            fn sparse(
                m: usize,
                n: usize,
                p: Vec<usize>,
                r: Vec<usize>,
                v: Vec<Self>,
            ) -> SparseTensor {
                SparseTensor::$ctor(m, n, p, r, v.into_iter().map(|x| (x.re, x.im)).collect())
                    .unwrap()
            }
            const TOLERANCE: f64 = $tol;
            const TINY: f64 = $tiny;
        }
    };
}
complex!(faer::c32, f32, new_complex_f32, 3e-5, 1e-30);
complex!(faer::c64, f64, new_complex, 3e-12, 1e-250);

macro_rules! retained_complex_case {
    ($real:ty, $ctor:ident, $tol:expr, $tiny:expr) => {
        impl Case for runmat_value::ComplexElement<$real> {
            fn native(real: f64, imag: f64) -> Self {
                Self(real as $real, imag as $real)
            }
            fn sparse(
                m: usize,
                n: usize,
                p: Vec<usize>,
                r: Vec<usize>,
                v: Vec<Self>,
            ) -> SparseTensor {
                SparseTensor::$ctor(
                    m,
                    n,
                    p,
                    r,
                    v.into_iter().map(|value| (value.0, value.1)).collect(),
                )
                .unwrap()
            }
            const TOLERANCE: f64 = $tol;
            const TINY: f64 = $tiny;
        }
    };
}
retained_complex_case!(f32, new_complex_f32, 3e-5, 1e-30);
retained_complex_case!(f64, new_complex, 3e-12, 1e-250);

fn check<T: Case>(
    shape: [usize; 2],
    structure: (Vec<usize>, Vec<usize>),
    values: &[(f64, f64)],
    b: &[(f64, f64)],
    expected: &[(f64, f64)],
    rank: usize,
    scale: f64,
) {
    let [m, n] = shape;
    let (p, rows) = structure;
    for managed in [false, true] {
        let ledger = ManagedHostEffectLedger::new(2_000_000);
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let runtime = if managed {
            runtime.with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
            )
        } else {
            runtime
        };
        let a = T::sparse(
            m,
            n,
            p.clone(),
            rows.clone(),
            values
                .iter()
                .map(|&(r, i)| T::native(r * scale, i * scale))
                .collect(),
        );
        let pointers = a.col_ptrs.as_ptr();
        let factor = block_on(SparseCompleteQr::<T>::factor(&a, &runtime)).unwrap();
        assert_eq!(factor.first.rank, rank);
        assert_eq!(
            a.col_ptrs.as_ptr(),
            pointers,
            "coefficient topology remains borrowed"
        );
        let input: Vec<T> = b
            .iter()
            .map(|&(r, i)| T::native(r * scale, i * scale))
            .collect();
        let mut output = vec![T::default(); n];
        block_on(factor.solve_with_output(
            &input,
            1,
            &runtime,
            std::convert::identity,
            |index, value| {
                output[index] = value;
                Ok(())
            },
        ))
        .unwrap();
        for (&actual, &(r, i)) in output.iter().zip(expected) {
            let (re, im) = (actual - T::native(r, i)).parts();
            assert!(
                re.hypot(im) < T::TOLERANCE,
                "minimum-norm coordinate mismatch: {re}+{im}i"
            );
        }
        // Independently check the least-squares normal equations from borrowed A.
        let mut residual = input.clone();
        for col in 0..n {
            for at in a.col_ptrs[col]..a.col_ptrs[col + 1] {
                let row = a.row_indices[at];
                residual[row] = residual[row] - T::read(&a, at) * output[col];
            }
        }
        for col in 0..n {
            let mut normal = T::default();
            for at in a.col_ptrs[col]..a.col_ptrs[col + 1] {
                normal = normal + T::read(&a, at).conjugate() * residual[a.row_indices[at]];
            }
            let (re, im) = normal.parts();
            assert!(re.hypot(im) <= T::TOLERANCE * scale * scale * 100.0);
        }
        drop(factor);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}
fn native_cases<T: Case>() {
    for scale in [1.0, T::TINY] {
        // Tall inconsistent system: normal-equation residual is zero.
        check::<T>(
            [3, 2],
            (vec![0, 2, 4], vec![0, 2, 1, 2]),
            &[(1., 0.), (1., 0.), (1., 0.), (1., 0.)],
            &[(-1., 0.), (4., 0.), (4., 0.)],
            &[(-2. / 3., 0.), (13. / 3., 0.)],
            2,
            scale,
        );
        // Rank-deficient tall: x is orthogonal to null(A)=span([-2;1]).
        check::<T>(
            [3, 2],
            (vec![0, 3, 6], vec![0, 1, 2, 0, 1, 2]),
            &[
                (1., 1.),
                (2., 2.),
                (-1., -1.),
                (2., 2.),
                (4., 4.),
                (-2., -2.),
            ],
            &[(5., 5.), (10., 10.), (-5., -5.)],
            &[(1., 0.), (2., 0.)],
            1,
            scale,
        );
        // Wide full-row-rank and wide rank-deficient minimum norms.
        check::<T>(
            [2, 3],
            (vec![0, 1, 2, 4], vec![0, 1, 0, 1]),
            &[(1., 0.), (1., 0.), (1., 0.), (1., 0.)],
            &[(1., 0.), (1., 0.)],
            &[(1. / 3., 0.), (1. / 3., 0.), (2. / 3., 0.)],
            2,
            scale,
        );
        check::<T>(
            [2, 3],
            (vec![0, 2, 4, 6], vec![0, 1, 0, 1, 0, 1]),
            &[(1., 0.), (2., 0.), (2., 0.), (4., 0.), (3., 0.), (6., 0.)],
            &[(14., 0.), (28., 0.)],
            &[(1., 0.), (2., 0.), (3., 0.)],
            1,
            scale,
        );
    }
}
#[test]
fn native_four_tall_wide_rank_deficient_and_scale_invariant() {
    native_cases::<f32>();
    native_cases::<f64>();
    native_cases::<faer::c32>();
    native_cases::<faer::c64>();
}

fn decomposition_rank_cases<T: Case>() {
    for managed in [false, true] {
        let ledger = ManagedHostEffectLedger::new(2_000_000);
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let runtime = if managed {
            runtime.with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
            )
        } else {
            runtime
        };
        let matrix = T::sparse(
            1,
            2,
            vec![0, 1, 2],
            vec![0, 0],
            vec![T::native(1.0, 0.0), T::native(2.0, 0.0)],
        );
        for complete in [false, true] {
            let factor = block_on(SparseCompleteQr::<T>::factor_for_decomposition(
                &matrix, None, complete, &runtime,
            ))
            .unwrap();
            assert_eq!(factor.first.rank, 1);
            assert_eq!(factor.cod.is_some(), complete);
            let mut answer = [T::default(); 2];
            block_on(factor.solve_with_output(
                &[T::native(5.0, 0.0)],
                1,
                &runtime,
                std::convert::identity,
                |index, value| {
                    answer[index] = value;
                    Ok(())
                },
            ))
            .unwrap();
            let expected = if complete { [1.0, 2.0] } else { [0.0, 2.5] };
            for (actual, expected) in answer.into_iter().zip(expected) {
                let (real, imag) = (actual - T::native(expected, 0.0)).parts();
                assert!(real.hypot(imag) < T::TOLERANCE);
            }
        }
        let matrix = T::sparse(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![T::one(), T::native(T::TINY, 0.0)],
        );
        for (tolerance, expected) in [
            (None, 1),
            (Some(T::TINY / 2.0), 2),
            (Some(T::TINY * 2.0), 1),
            (Some(0.0), 2),
            (Some(f64::INFINITY), 0),
        ] {
            for complete in [false, true] {
                let factor = block_on(SparseCompleteQr::<T>::factor_for_decomposition(
                    &matrix, tolerance, complete, &runtime,
                ))
                .unwrap();
                assert_eq!(factor.first.rank, expected);
            }
        }
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn native_four_basic_qr_cod_and_absolute_rank_tolerance_have_distinct_oracles() {
    decomposition_rank_cases::<f32>();
    decomposition_rank_cases::<f64>();
    decomposition_rank_cases::<faer::c32>();
    decomposition_rank_cases::<faer::c64>();
}

fn retained_data_cases<T: Case + super::super::retained::RetainedFactorScalar>() {
    use super::super::managed_lu::FactorSolveOptions;
    use super::super::managed_lu::SingularSolvePolicy::PublicDecompositionIeee;
    use super::super::managed_lu::SingularSolvePolicy::Reject;
    use super::super::retained::{factor_sparse, solve_factor_native};
    use runmat_value::NumericFactorKind;
    let options = |adjoint| FactorSolveOptions {
        adjoint,
        singular: PublicDecompositionIeee,
    };
    for managed in [false, true] {
        for kind in [
            NumericFactorKind::Lu,
            NumericFactorKind::Cholesky,
            NumericFactorKind::Qr,
            NumericFactorKind::Cod,
        ] {
            let ledger = ManagedHostEffectLedger::new(2_000_000);
            let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
            let runtime = if managed {
                runtime.with_service_ports(
                    RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
                )
            } else {
                runtime
            };
            // Hermitian positive definite, with genuinely complex native
            // coefficients in complex classes. Exact solution is [1;2].
            let matrix = T::sparse(
                2,
                2,
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![
                    T::native(3., 0.),
                    T::native(1., -1.),
                    T::native(1., 1.),
                    T::native(4., 0.),
                ],
            );
            let data = block_on(factor_sparse(&matrix, kind, None, false, &runtime)).unwrap();
            assert_eq!(data.kind(), kind);
            let clone = data.clone();
            drop(runtime);
            if managed {
                assert!(ledger.snapshot().live_claimed_bytes > 0);
                ledger.revoke();
            }
            let caller = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
            for adjoint in [false, true] {
                let mut answer = [T::default(); 2];
                let result = block_on(solve_factor_native::<T, T>(
                    &clone,
                    &[T::native(5., 2.), T::native(9., -1.)],
                    1,
                    options(adjoint),
                    &caller,
                    |_, value| Some(value),
                    |index, value| {
                        answer[index] = value;
                        Ok(())
                    },
                ));
                if kind == NumericFactorKind::Qr && adjoint {
                    assert!(result.is_err());
                    continue;
                }
                result.unwrap();
                for (actual, expected) in answer.into_iter().zip([1., 2.]) {
                    let (real, imag) = (actual - T::native(expected, 0.)).parts();
                    assert!(real.hypot(imag) < T::TOLERANCE);
                }
            }
            // Reentrant independent futures share only immutable factor DATA,
            // never a RefCell/mutex borrow or mutable retained RHS.
            let (mut left, mut right) = ([T::default(); 2], [T::default(); 2]);
            let input = [T::native(5., 2.), T::native(9., -1.)];
            let (a, b) = block_on(futures::future::join(
                solve_factor_native::<T, T>(
                    &clone,
                    &input,
                    1,
                    options(false),
                    &caller,
                    |_, value| Some(value),
                    |i, value| {
                        left[i] = value;
                        Ok(())
                    },
                ),
                solve_factor_native::<T, T>(
                    &clone,
                    &input,
                    1,
                    options(false),
                    &caller,
                    |_, value| Some(value),
                    |i, value| {
                        right[i] = value;
                        Ok(())
                    },
                ),
            ));
            a.unwrap();
            b.unwrap();
            for (a, b) in left.into_iter().zip(right) {
                let (real, imag) = (a - b).parts();
                assert!(real.hypot(imag) < T::TOLERANCE);
            }
            drop(data);
            if managed {
                assert!(ledger.snapshot().live_claimed_bytes > 0);
            }
            drop(clone);
            assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
        let ledger = ManagedHostEffectLedger::new(2_000_000);
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let runtime = if managed {
            runtime.with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
            )
        } else {
            runtime
        };
        let matrix = T::sparse(
            2,
            3,
            vec![0, 1, 2, 2],
            vec![0, 0],
            vec![T::native(1., 1.), T::native(2., 0.)],
        );
        let data = block_on(factor_sparse(
            &matrix,
            NumericFactorKind::Cod,
            None,
            false,
            &runtime,
        ))
        .unwrap();
        let mut answer = [T::default(); 2];
        let runmat_value::NumericFactorCandidate::Cod { first, .. } = data.parts() else {
            panic!("retained COD expected");
        };
        assert_eq!(first.rank, 1);
        assert_eq!(
            first.reflectors.col_ptrs.last().copied(),
            Some(first.reflectors.nnz())
        );
        assert_eq!(first.reflectors.col_ptrs[1], first.reflectors.col_ptrs[2]);
        block_on(solve_factor_native::<T, T>(
            &data,
            &[T::native(1., -1.), T::native(2., 0.), T::default()],
            1,
            options(true),
            &runtime,
            |_, value| Some(value),
            |index, value| {
                answer[index] = value;
                Ok(())
            },
        ))
        .unwrap();
        for (actual, expected) in answer.into_iter().zip([1., 0.]) {
            let (real, imag) = (actual - T::native(expected, 0.)).parts();
            assert!(real.hypot(imag) < T::TOLERANCE);
        }
        drop(data);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        let matrix = T::sparse(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![T::one(); 4]);
        let data = block_on(factor_sparse(
            &matrix,
            NumericFactorKind::Lu,
            None,
            false,
            &runtime,
        ))
        .unwrap();
        let input = [T::one(), T::native(2., 0.)];
        let mut writes = 0;
        assert!(block_on(solve_factor_native::<T, T>(
            &data,
            &input,
            1,
            FactorSolveOptions {
                adjoint: false,
                singular: Reject
            },
            &runtime,
            |_, value| Some(value),
            |_, _| {
                writes += 1;
                Ok(())
            }
        ))
        .is_err());
        assert_eq!(writes, 0);
        let mut answer = [T::default(); 2];
        let status = block_on(solve_factor_native::<T, T>(
            &data,
            &input,
            1,
            options(false),
            &runtime,
            |_, value| Some(value),
            |index, value| {
                answer[index] = value;
                Ok(())
            },
        ))
        .unwrap();
        assert!(status.singular);
        assert!(answer.iter().any(|value| !value.finite()));
        // Scaling changes coefficients, not the singularity policy or kind.
        for policy in [
            Reject,
            super::super::SingularSolvePolicy::PublicDecompositionIeee,
        ] {
            let mut writes = 0;
            let mut nonfinite = false;
            let result = block_on(super::super::retained::solve_factor_indexed_scaled::<T>(
                &data,
                super::super::FactorRhsGeometry { len: 2, columns: 1 },
                FactorSolveOptions {
                    adjoint: false,
                    singular: policy,
                },
                (-2.0, 0.0),
                &runtime,
                |index| Some(input[index]),
                |_, value| {
                    writes += 1;
                    nonfinite |= !value.finite();
                    Ok(())
                },
            ));
            if policy == Reject {
                assert!(result.is_err());
                assert_eq!(writes, 0);
            } else {
                assert!(result.unwrap().singular);
                assert_eq!(writes, 2);
                assert!(nonfinite);
            }
        }
        assert_eq!(
            data.kind(),
            NumericFactorKind::Lu,
            "no COD switch after a singular LU solve"
        );
        assert!(block_on(solve_factor_native::<T, T>(
            &data,
            &[T::native(f64::NAN, 0.), T::one()],
            1,
            options(false),
            &runtime,
            |_, value| Some(value),
            |_, _| {
                writes += 1;
                Ok(())
            }
        ))
        .is_err());
        assert_eq!(
            writes, 0,
            "IEEE singular policy never admits nonfinite inputs"
        );
        drop(data);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn native_four_immutable_data_survives_original_issuer_and_uses_independent_solve_workspaces() {
    retained_data_cases::<f32>();
    retained_data_cases::<f64>();
    retained_data_cases::<runmat_value::ComplexElement<f32>>();
    retained_data_cases::<runmat_value::ComplexElement<f64>>();
    scaled_data_cases::<f32, f32>();
    scaled_data_cases::<f64, f64>();
    scaled_data_cases::<runmat_value::ComplexElement<f32>, runmat_value::ComplexElement<f32>>();
    scaled_data_cases::<runmat_value::ComplexElement<f64>, runmat_value::ComplexElement<f64>>();
    scaled_data_cases::<f32, runmat_value::ComplexElement<f32>>();
    scaled_data_cases::<f64, runmat_value::ComplexElement<f64>>();
}

fn scaled_data_cases<
    S: Case + super::super::retained::RetainedFactorScalar,
    T: super::super::retained::RetainedFactorScalar,
>() {
    use super::super::retained::{factor_sparse, solve_factor_indexed_scaled};
    use super::super::{FactorRhsGeometry, FactorSolveOptions, SingularSolvePolicy};
    use runmat_value::{ExactHostSparseClass as C, NumericDType as D, NumericFactorKind as K};
    let single = matches!(S::storage_class(), C::Numeric(D::F32) | C::Complex(D::F32));
    let complex = T::from_parts(0.0, 1.0).is_some();
    let large = if single { 1e30 } else { 1e308 };
    let small = if single { 1e-30 } else { 1e-308 };
    let down = if single { 1e-40 } else { 1e-309 };
    let maximum = if single {
        f64::from(f32::MAX)
    } else {
        f64::MAX
    };
    let tolerance = if single { 5e-4 } else { 3e-12 };
    for managed in [false, true] {
        let ledger = ManagedHostEffectLedger::new(2_000_000);
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let runtime = if managed {
            runtime.with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
            )
        } else {
            runtime
        };
        for (a, scale, b) in [
            (large, (down, 0.0), 1.0),
            (small, (large, 0.0), small),
            (0.5, (-2.0, 0.0), 1.0),
            (0.5, (1.0, 1.0), 0.75 * maximum),
        ] {
            if scale.1 != 0.0 && !complex {
                continue;
            }
            let coefficient = S::native(a, 0.0);
            let a = coefficient.parts().0;
            let rhs = T::from_parts(b, 0.0).unwrap();
            let b = rhs.parts().0;
            // Independent scalar quotient. Forming the effective coefficient
            // first is representable in these witnesses; b/scale is not.
            let expected = runmat_value::divide_complex_f64((b, 0.0), (a * scale.0, a * scale.1));
            for kind in [K::Lu, K::Cholesky, K::Qr, K::Cod] {
                let matrix = S::sparse(1, 1, vec![0, 1], vec![0], vec![coefficient]);
                let data = block_on(factor_sparse(&matrix, kind, None, false, &runtime)).unwrap();
                for adjoint in [false, true] {
                    let mut writes = 0;
                    let mut output = T::default();
                    let result = block_on(solve_factor_indexed_scaled::<T>(
                        &data,
                        FactorRhsGeometry { len: 1, columns: 1 },
                        FactorSolveOptions {
                            adjoint,
                            singular: SingularSolvePolicy::Reject,
                        },
                        scale,
                        &runtime,
                        |_| Some(rhs),
                        |_, v| {
                            writes += 1;
                            output = v;
                            Ok(())
                        },
                    ));
                    if kind == K::Qr && adjoint {
                        assert!(result.is_err());
                        assert_eq!(writes, 0);
                        continue;
                    }
                    let status = result.unwrap();
                    assert!(!status.singular);
                    assert_eq!(status.deficient_rank, None);
                    assert_eq!(writes, 1);
                    assert!(output.finite());
                    let actual = output.parts();
                    for (actual, expected) in [(actual.0, expected.0), (actual.1, expected.1)] {
                        if expected == 0.0 {
                            assert_eq!(actual, 0.0);
                        } else {
                            assert!(
                                (actual / expected - 1.0).abs() < tolerance,
                                "native scaled {kind:?} adjoint={adjoint}: {actual} vs {expected}"
                            );
                        }
                    }
                }
                drop(data);
                assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
                assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
            }
        }
        // A finite scale may round a coefficient to physical zero. Preserve
        // scalar IEEE semantics and report singular for that effective U.
        let matrix = S::sparse(1, 1, vec![0, 1], vec![0], vec![S::native(small, 0.0)]);
        let data = block_on(factor_sparse(&matrix, K::Lu, None, false, &runtime)).unwrap();
        for singular in [
            SingularSolvePolicy::Reject,
            SingularSolvePolicy::PublicDecompositionIeee,
        ] {
            let mut writes = 0;
            let mut nonfinite = false;
            let result = block_on(solve_factor_indexed_scaled::<T>(
                &data,
                FactorRhsGeometry { len: 1, columns: 1 },
                FactorSolveOptions {
                    adjoint: false,
                    singular,
                },
                (down, 0.0),
                &runtime,
                |_| T::from_parts(1.0, 0.0),
                |_, value| {
                    writes += 1;
                    nonfinite |= !value.finite();
                    Ok(())
                },
            ));
            if singular == SingularSolvePolicy::Reject {
                assert!(result.is_err());
                assert_eq!(writes, 0);
            } else {
                assert!(result.unwrap().singular);
                assert_eq!(writes, 1);
                assert!(nonfinite);
            }
        }
        drop(data);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        // A wide row exercises the distinct COD completion stage and QR's
        // basic pivot solution. The adjoint RHS is A^H, so its exact answer
        // is 1/scale rather than a self-derived factor reconstruction.
        let second = S::native(2.0, 1.0);
        let (ar, ai) = second.parts();
        let scale = if complex { (2.0, 1.0) } else { (-2.0, 0.0) };
        let inverse = runmat_value::divide_complex_f64((1.0, 0.0), scale);
        let norm_squared = 1.0 + ar * ar + ai * ai;
        for kind in [K::Qr, K::Cod] {
            let matrix = S::sparse(
                1,
                2,
                vec![0, 1, 2],
                vec![0, 0],
                vec![S::native(1.0, 0.0), second],
            );
            let data = block_on(factor_sparse(&matrix, kind, None, false, &runtime)).unwrap();
            for adjoint in [false, true] {
                let mut output = [T::default(); 2];
                let mut writes = 0;
                let result = block_on(solve_factor_indexed_scaled::<T>(
                    &data,
                    FactorRhsGeometry {
                        len: if adjoint { 2 } else { 1 },
                        columns: 1,
                    },
                    FactorSolveOptions {
                        adjoint,
                        singular: SingularSolvePolicy::Reject,
                    },
                    scale,
                    &runtime,
                    |index| {
                        T::from_parts(
                            if index == 0 { 1.0 } else { ar },
                            if index == 0 { 0.0 } else { -ai },
                        )
                    },
                    |index, value| {
                        output[index] = value;
                        writes += 1;
                        Ok(())
                    },
                ));
                if adjoint && kind == K::Qr {
                    assert!(result.is_err());
                    assert_eq!(writes, 0);
                    continue;
                }
                result.unwrap();
                let expected = if adjoint {
                    [inverse, (0.0, 0.0)]
                } else if kind == K::Qr {
                    [
                        (0.0, 0.0),
                        runmat_value::divide_complex_f64(inverse, (ar, ai)),
                    ]
                } else {
                    [
                        (inverse.0 / norm_squared, inverse.1 / norm_squared),
                        (
                            (ar * inverse.0 + ai * inverse.1) / norm_squared,
                            (ar * inverse.1 - ai * inverse.0) / norm_squared,
                        ),
                    ]
                };
                assert_eq!(writes, if adjoint { 1 } else { 2 });
                for (value, expected) in output.into_iter().zip(expected).take(writes) {
                    let value = value.parts();
                    assert!((value.0 - expected.0).hypot(value.1 - expected.1) < tolerance);
                }
            }
            drop(data);
            assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
    }
}
#[test]
fn complex_wide_uses_conjugate_complete_decomposition() {
    check::<faer::c64>(
        [2, 3],
        (vec![0, 1, 2, 4], vec![0, 1, 0, 1]),
        &[(1., 0.), (1., 0.), (0., 1.), (0., 1.)],
        &[(1., 0.), (1., 0.)],
        &[(1. / 3., 0.), (1. / 3., 0.), (0., -2. / 3.)],
        2,
        1.0,
    );
    check::<faer::c32>(
        [2, 3],
        (vec![0, 1, 2, 4], vec![0, 1, 0, 1]),
        &[(1., 0.), (1., 0.), (0., 1.), (0., 1.)],
        &[(1., 0.), (1., 0.)],
        &[(1. / 3., 0.), (1. / 3., 0.), (0., -2. / 3.)],
        2,
        1.0,
    );
}
