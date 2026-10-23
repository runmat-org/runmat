use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use crate::execution::{
    CpuCooperationBudget, OwnedCpuJob, RuntimeBlockingCpuService, RuntimeExecutionService,
};
use futures::{executor::block_on, task::noop_waker};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use runmat_value::SparseTensor;
use std::{cell::RefCell, future::Future, num::NonZeroU64, rc::Rc, task::Context};

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
fn runtime(ledger: &ManagedHostEffectLedger) -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
    )
}
fn empty(ledger: &ManagedHostEffectLedger) {
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

trait Case: ManagedSparseScalar {
    fn native(value: f64) -> Self;
    fn sparse(
        order: usize,
        pointers: Vec<usize>,
        rows: Vec<usize>,
        values: Vec<f64>,
    ) -> SparseTensor;
    const TOLERANCE: f64;
}
macro_rules! real_case {
    ($ty:ty,$ctor:ident,$tol:expr) => {
        impl Case for $ty {
            fn native(v: f64) -> Self {
                v as Self
            }
            fn sparse(n: usize, p: Vec<usize>, r: Vec<usize>, v: Vec<f64>) -> SparseTensor {
                SparseTensor::$ctor(n, n, p, r, v.into_iter().map(|v| v as Self).collect()).unwrap()
            }
            const TOLERANCE: f64 = $tol;
        }
    };
}
real_case!(f32, new_f32, 2e-5);
real_case!(f64, new, 1e-12);
macro_rules! complex_case {
    ($ty:ty,$ctor:ident,$lane:ty,$tol:expr) => {
        impl Case for $ty {
            fn native(v: f64) -> Self {
                Self::new(v as $lane, 0.0)
            }
            fn sparse(n: usize, p: Vec<usize>, r: Vec<usize>, v: Vec<f64>) -> SparseTensor {
                SparseTensor::$ctor(
                    n,
                    n,
                    p,
                    r,
                    v.into_iter()
                        .map(|v| (v as $lane, (v * 0.25) as $lane))
                        .collect(),
                )
                .unwrap()
            }
            const TOLERANCE: f64 = $tol;
        }
    };
}
complex_case!(faer::c32, new_complex_f32, f32, 2e-5);
complex_case!(faer::c64, new_complex, f64, 1e-12);
fn coefficient<T: Case>() -> SparseTensor {
    // [0 2 1; 1 3 0; 4 0 5]: needs pivoting and creates actual fill.
    T::sparse(
        3,
        vec![0, 2, 4, 6],
        vec![1, 2, 0, 1, 0, 2],
        vec![1.0, 4.0, 2.0, 3.0, 1.0, 5.0],
    )
}
fn check_case<T: Case>() {
    let ledger = ManagedHostEffectLedger::new(2_000_000);
    let runtime = runtime(&ledger);
    let sparse = coefficient::<T>();
    let mut factor = block_on(ManagedSparseLu::<T>::factor(&sparse, &runtime)).unwrap();
    assert_eq!(factor.order(), 3);
    assert!(factor.l_rows.len() + factor.u_rows.len() > sparse.nnz());
    let expected = [
        T::native(1.0),
        T::native(-2.0),
        T::native(3.0),
        T::native(2.0),
        T::native(0.0),
        T::native(-1.0),
    ];
    let mut input = [T::default(); 6];
    for rhs in 0..2 {
        for col in 0..3 {
            for i in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
                let row = sparse.row_indices[i];
                input[rhs * 3 + row] =
                    input[rhs * 3 + row] + T::read(&sparse, i) * expected[rhs * 3 + col];
            }
        }
    }
    let before = ledger.snapshot().live_claimed_bytes;
    for _ in 0..2 {
        let mut output = [T::default(); 6];
        block_on(factor.solve_into(&input, &mut output, 2, &runtime)).unwrap();
        for (actual, expected) in output.into_iter().zip(expected) {
            let (r, i) = (actual - expected).parts();
            assert!(r.hypot(i) < T::TOLERANCE);
        }
        assert_eq!(
            ledger.snapshot().live_claimed_bytes,
            before,
            "no per-solve allocation"
        );
    }
    drop(factor);
    empty(&ledger);
}

#[test]
fn pivot_fill_native_classes_and_reused_multiple_rhs() {
    check_case::<f32>();
    check_case::<f64>();
    check_case::<faer::c32>();
    check_case::<faer::c64>();
    for managed in [false, true] {
        check_shift_case::<f32>(managed);
        check_shift_case::<f64>(managed);
        check_shift_case::<faer::c32>(managed);
        check_shift_case::<faer::c64>(managed);
    }
}

fn check_shift_case<T: Case>(managed: bool) {
    let ledger = ManagedHostEffectLedger::new(2_000_000);
    let runtime = if managed {
        runtime(&ledger)
    } else {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    };
    let a = coefficient::<T>();
    let b = T::sparse(3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 2.0, 4.0]);
    let shift = T::native(0.5);
    let expected = [T::native(1.0), T::native(-2.0), T::native(3.0)];
    for b in [None, Some(&b)] {
        let mut input = [T::default(); 3];
        for col in 0..3 {
            for index in a.col_ptrs[col]..a.col_ptrs[col + 1] {
                let row = a.row_indices[index];
                input[row] = input[row] + T::read(&a, index) * expected[col];
            }
            if let Some(b) = b {
                for index in b.col_ptrs[col]..b.col_ptrs[col + 1] {
                    let row = b.row_indices[index];
                    input[row] = input[row] - shift * T::read(b, index) * expected[col];
                }
            } else {
                input[col] = input[col] - shift * expected[col];
            }
        }
        let mut factor =
            block_on(ManagedSparseLu::<T>::factor_shifted(&a, b, shift, &runtime)).unwrap();
        let before = ledger.snapshot().live_claimed_bytes;
        for _ in 0..2 {
            let mut output = [T::default(); 3];
            block_on(factor.solve_into(&input, &mut output, 1, &runtime)).unwrap();
            for (actual, expected) in output.into_iter().zip(expected) {
                let (r, i) = (actual - expected).parts();
                assert!(r.hypot(i) < T::TOLERANCE);
            }
            assert_eq!(
                ledger.snapshot().live_claimed_bytes,
                before,
                "shifted RHS must reuse its factor and scratch"
            );
        }
        if !managed {
            let replacement = super::tests::runtime(&ledger);
            assert!(matches!(
                factor.check_owner(&replacement),
                Err(HostAllocationError::StaleAuthority)
            ));
        }
        drop(factor);
        empty(&ledger);
    }
}

#[test]
fn geometric_growth_overlaps_exact_claims_and_denial_keeps_old_backing() {
    use crate::context::host_allocation::collection_block_retained_bytes;
    use std::alloc::Layout;
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let runtime = runtime(&ledger);
    let fence = admit(&runtime, 0, 0).unwrap();
    let mut work = Work::new(&runtime, &fence);
    let mut values = block_on(ClaimedVec::<usize>::zeroed(4, &mut work)).unwrap();
    values.as_mut_slice().copy_from_slice(&[1, 2, 3, 4]);
    let old = ledger.snapshot().live_claimed_bytes;
    let next = collection_block_retained_bytes(Layout::array::<usize>(8).unwrap()).unwrap();
    block_on(values.ensure(1, &mut work)).unwrap();
    assert_eq!(values.as_slice(), &[1, 2, 3, 4]);
    assert!(
        ledger.snapshot().peak_claimed_bytes >= old + next,
        "old and new backing must overlap"
    );
    let before = ledger.snapshot().live_claimed_bytes;
    let pressure = admit(&runtime, ledger.snapshot().free_bytes(), 0).unwrap();
    let error = ManagedSparseLuError::from(block_on(values.ensure(100, &mut work)).unwrap_err());
    assert!(matches!(
        error,
        ManagedSparseLuError::Admission(HostAllocationError::Resource(_))
    ));
    assert_eq!(values.as_slice(), &[1, 2, 3, 4]);
    assert_eq!(ledger.snapshot().live_claimed_bytes, before);
    drop(pressure);
    drop(values);
    drop(work);
    drop(fence);
    empty(&ledger);
}

#[test]
fn numeric_failures_and_revoked_or_substituted_authority_never_publish_success() {
    let ledger = ManagedHostEffectLedger::new(2_000_000);
    let runtime = runtime(&ledger);
    let singular = SparseTensor::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    assert!(matches!(
        block_on(ManagedSparseLu::<f64>::factor(&singular, &runtime)),
        Err(ManagedSparseLuError::Singular { .. })
    ));
    empty(&ledger);
    let invalid = SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![f64::NAN]).unwrap();
    assert!(matches!(
        block_on(ManagedSparseLu::<f64>::factor(&invalid, &runtime)),
        Err(ManagedSparseLuError::NonFiniteInput)
    ));
    empty(&ledger);
    let sparse = coefficient::<f64>();
    let mut factor = block_on(ManagedSparseLu::<f64>::factor(&sparse, &runtime)).unwrap();
    let other = ManagedHostEffectLedger::new(2_000_000);
    let other_runtime = super::tests::runtime(&other);
    let mut output = [91.0; 3];
    assert!(matches!(
        block_on(factor.solve_into(&[1.0; 3], &mut output, 1, &other_runtime)),
        Err(ManagedSparseLuError::Admission(
            HostAllocationError::StaleAuthority
        ))
    ));
    assert_eq!(output, [91.0; 3]);
    empty(&other);
    ledger.revoke();
    assert!(matches!(
        block_on(factor.solve_into(&[1.0; 3], &mut output, 1, &runtime)),
        Err(ManagedSparseLuError::Admission(
            HostAllocationError::StaleAuthority
        ))
    ));
    assert_eq!(output, [91.0; 3]);
    drop(factor);
    empty(&ledger);
}

#[test]
fn finite_extreme_complex_pivots_use_robust_division() {
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let runtime = runtime(&ledger);
    let value = faer::c64::new(f64::MAX * 0.75, f64::MAX * 0.75);
    let sparse =
        SparseTensor::new_complex(1, 1, vec![0, 1], vec![0], vec![(value.re, value.im)]).unwrap();
    let mut factor = block_on(ManagedSparseLu::<faer::c64>::factor(&sparse, &runtime)).unwrap();
    let mut output = [faer::c64::default()];
    block_on(factor.solve_into(&[value], &mut output, 1, &runtime)).unwrap();
    assert!((output[0] - faer::c64::new(1.0, 0.0)).norm() < 1e-14);
    drop(factor);
    empty(&ledger);
}

#[test]
fn banded_sparse_work_does_not_probe_absent_rows_quadratically() {
    let count = |n: usize| {
        let ledger = ManagedHostEffectLedger::new(16_000_000);
        let runtime = runtime(&ledger);
        let mut pointers = vec![0];
        let mut rows = Vec::new();
        let mut values = Vec::new();
        for col in 0..n {
            if col > 0 {
                rows.push(col - 1);
                values.push(-1.0);
            }
            rows.push(col);
            values.push(4.0);
            if col + 1 < n {
                rows.push(col + 1);
                values.push(-1.0);
            }
            pointers.push(rows.len());
        }
        let sparse = SparseTensor::new(n, n, pointers, rows, values).unwrap();
        let factor = block_on(ManagedSparseLu::<f64>::factor(&sparse, &runtime)).unwrap();
        let units = factor.work_units;
        assert!(factor.l_rows.len() + factor.u_rows.len() < 8 * n);
        drop(factor);
        empty(&ledger);
        units
    };
    let small = count(128);
    let large = count(512);
    assert!(
        large < small * 6,
        "fourfold order must not cause quadratic structural probing: {small} -> {large}"
    );
}

#[test]
fn abandoning_a_pending_growth_turn_releases_replacement_without_losing_original() {
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let runtime = runtime(&ledger);
    let fence = admit(&runtime, 0, 0).unwrap();
    let mut work = Work::new(&runtime, &fence);
    let mut values = block_on(ClaimedVec::<usize>::zeroed(300, &mut work)).unwrap();
    let before = ledger.snapshot().live_claimed_bytes;
    let cancel = CancellationSource::new();
    let quantum = NonZeroU64::new(7).unwrap();
    let budget = CpuCooperationBudget::with_test_turn(
        crate::execution::CpuCooperationPolicy::WorkQuantum(quantum),
        cancel.token(),
        || Box::pin(futures::future::pending()),
    );
    work = Work::with_test_budget(&runtime, &fence, budget, quantum);
    block_on(work.tick()).unwrap();
    block_on(work.flush()).unwrap();
    assert_eq!(
        work.capacity(),
        6,
        "partial flush preserves the remaining selected quantum"
    );
    let mut future = Box::pin(values.ensure(1, &mut work));
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&noop_waker()))
        .is_pending());
    assert!(ledger.snapshot().live_claimed_bytes > before);
    cancel.cancel(CancellationReason::User);
    assert!(matches!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(&noop_waker())),
        std::task::Poll::Ready(Err(SparseWorkError::Execution(
            ExecutionServiceError::Cancelled
        )))
    ));
    drop(future);
    assert_eq!(values.len(), 300);
    assert_eq!(ledger.snapshot().live_claimed_bytes, before);
    drop(values);
    drop(work);
    drop(fence);
    empty(&ledger);
}

#[derive(Default)]
struct HeldCpu(RefCell<Option<OwnedCpuJob>>);
impl RuntimeBlockingCpuService for HeldCpu {
    fn submit(
        &self,
        runtime: Option<&RuntimeContext>,
        job: OwnedCpuJob,
    ) -> crate::execution::OwnedCpuSubmission {
        let submitted = crate::execution::admit_optional_hostcall_with(
            runtime,
            ExecutionServiceError::HostAllocation,
            || async { Ok(()) },
        )?;
        assert!(self.0.borrow_mut().replace(job).is_none());
        Ok(submitted)
    }
}
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn ordering_claims_survive_await_drop_or_cancel_until_physical_job_finishes() {
    for cancel_await in [false, true] {
        let ledger = ManagedHostEffectLedger::new(2_000_000);
        let cancel = CancellationSource::new();
        let service = Rc::new(HeldCpu::default());
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            cancel.token(),
        )
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_parallel(Rc::new(Parallel(ledger.clone())))
                .with_blocking_cpu(service.clone()),
        );
        let sparse = coefficient::<f64>();
        let mut future = Box::pin(ManagedSparseLu::<f64>::factor(&sparse, &runtime));
        assert!(future
            .as_mut()
            .poll(&mut Context::from_waker(&noop_waker()))
            .is_pending());
        assert!(service.0.borrow().is_some());
        if cancel_await {
            cancel.cancel(CancellationReason::User);
            assert!(matches!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(&noop_waker())),
                std::task::Poll::Ready(Err(ManagedSparseLuError::Execution(
                    ExecutionServiceError::Cancelled
                )))
            ));
        }
        drop(future);
        assert!(
            ledger.snapshot().live_claimed_bytes > 0,
            "physical work still owns topology and scratch"
        );
        assert!(ledger.snapshot().live_reserved_bytes > 0);
        service.0.borrow_mut().take().unwrap().run();
        empty(&ledger);
    }
}
