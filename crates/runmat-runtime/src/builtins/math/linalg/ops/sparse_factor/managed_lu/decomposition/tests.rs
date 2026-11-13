use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use num_complex::Complex64;
use runmat_value::SparseTensor;
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
trait Case: ManagedSparseScalar + std::fmt::Debug {
    fn source(m: usize, n: usize, p: Vec<usize>, r: Vec<usize>, v: Vec<Self>) -> SparseTensor;
    const TOLERANCE: f64;
}
macro_rules! real {
    ($t:ty,$ctor:ident,$tol:expr) => {
        impl Case for $t {
            fn source(
                m: usize,
                n: usize,
                p: Vec<usize>,
                r: Vec<usize>,
                v: Vec<Self>,
            ) -> SparseTensor {
                SparseTensor::$ctor(m, n, p, r, v).unwrap()
            }
            const TOLERANCE: f64 = $tol;
        }
    };
}
real!(f32, new_f32, 5e-5);
real!(f64, new, 5e-12);
macro_rules! complex {
    ($t:ty,$ctor:ident,$tol:expr) => {
        impl Case for $t {
            fn source(
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
        }
    };
}
complex!(faer::c32, new_complex_f32, 7e-5);
complex!(faer::c64, new_complex, 7e-12);
fn complex<T: ManagedSparseScalar>(value: T) -> Complex64 {
    let (r, i) = value.parts();
    Complex64::new(r, i)
}
fn source<T: Case>(m: usize, n: usize, data: &[(f64, f64)]) -> SparseTensor {
    let mut p = vec![0];
    let mut rows = Vec::new();
    let mut values = Vec::new();
    for col in 0..n {
        for row in 0..m {
            let (r, i) = data[col * m + row];
            let value = T::from_parts(r, i)
                .or_else(|| T::from_parts(r, 0.))
                .unwrap();
            if value != T::default() {
                rows.push(row);
                values.push(value);
            }
        }
        p.push(values.len());
    }
    T::source(m, n, p, rows, values)
}
fn native_cases<T: Case>() {
    for managed in [false, true] {
        let ledger = ManagedHostEffectLedger::new(2_000_000);
        let base = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let runtime = if managed {
            base.with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
            )
        } else {
            base
        };
        for (m, n, data) in [
            (
                3,
                2,
                vec![(1., 1.), (3., 0.), (5., 0.), (2., 0.), (0., 4.), (6., 0.)],
            ),
            (
                2,
                3,
                vec![(1., 0.), (3., 0.), (2., 0.), (4., 1.), (0., 0.), (5., 0.)],
            ),
            (
                2,
                3,
                vec![(0., 0.), (0., 0.), (1., 0.), (2., 0.), (2., 0.), (4., 0.)],
            ),
            (3, 2, vec![(0., 0.); 6]),
            (0, 0, vec![]),
            (3, 0, vec![]),
        ] {
            let a = source::<T>(m, n, &data);
            let ptr = a.col_ptrs.as_ptr();
            for count in 2..=5 {
                let options = SparseLuOptions {
                    column_ordering: count >= 4,
                    row_scaling: count == 5,
                    unpermute_lower: count == 2,
                    threshold: if count >= 4 { [0.1, 0.001] } else { [1., 1.] },
                };
                let factor = block_on(PublicSparseLu::<T>::factor(&a, options, &runtime)).unwrap();
                assert_eq!(factor.dimensions(), (m, n));
                assert_eq!(a.col_ptrs.as_ptr(), ptr);
                let p = factor.row_permutation(&runtime).unwrap();
                let q = factor.column_permutation(&runtime).unwrap();
                let k = m.min(n);
                for col in 0..n {
                    for row in 0..m {
                        let mut actual = Complex64::new(0., 0.);
                        let (ur, uv) = factor.upper_column(col, &runtime).unwrap();
                        for (&inner, &value) in ur.iter().zip(uv) {
                            let (lr, lv) = factor.lower_column(inner, &runtime).unwrap();
                            if let Ok(index) = lr.binary_search(&row) {
                                actual += complex(lv[index]) * complex(value);
                            }
                        }
                        let original_row = if count == 2 { row } else { p.at(row) };
                        let original_col = q.at(col);
                        let mut expected = T::default();
                        for at in a.col_ptrs[original_col]..a.col_ptrs[original_col + 1] {
                            if a.row_indices[at] == original_row {
                                expected = T::read(&a, at);
                            }
                        }
                        let scale = if count == 5 {
                            complex(factor.scaling(&runtime).unwrap()[original_row]).re
                        } else {
                            1.
                        };
                        let expected = complex(expected) / scale;
                        assert!(
                            (actual - expected).norm() <= T::TOLERANCE * (1. + expected.norm()),
                            "{m}x{n} count{count}: {row},{col}: {actual} vs {expected}"
                        );
                    }
                }
                for col in 0..k {
                    let (rows, values) = factor.lower_column(col, &runtime).unwrap();
                    assert!(rows.windows(2).all(|x| x[0] < x[1]));
                    if count >= 3 {
                        assert!(rows.iter().all(|&row| row >= col));
                        assert_eq!(values[rows.binary_search(&col).unwrap()], T::one());
                    }
                }
                drop(factor);
                assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
                assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
            }
        }
    }
}
#[test]
fn native_four_rectangular_singular_and_scaling_factors_have_independent_product_oracles() {
    native_cases::<f32>();
    native_cases::<f64>();
    native_cases::<faer::c32>();
    native_cases::<faer::c64>();
}
#[test]
fn retained_public_factor_denied_growth_and_revoked_original_owner_release_exact_claims() {
    let options = SparseLuOptions {
        column_ordering: false,
        row_scaling: false,
        unpermute_lower: false,
        threshold: [1., 1.],
    };
    let a = source::<f64>(2, 2, &[(1., 0.), (2., 0.), (3., 0.), (4., 0.)]);
    for capacity in [1, 2_000_000] {
        let ledger = ManagedHostEffectLedger::new(capacity);
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
            );
        let result = block_on(PublicSparseLu::<f64>::factor(&a, options, &runtime));
        if capacity == 1 {
            assert!(matches!(result, Err(Error::Admission(_))));
        } else {
            let factor = result.unwrap();
            assert!(ledger.snapshot().live_claimed_bytes > 0);
            ledger.revoke();
            assert!(factor.check_owner(&runtime).is_err());
            drop(factor);
        }
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}
