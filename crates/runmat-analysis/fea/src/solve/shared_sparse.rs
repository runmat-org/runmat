//! The one bridge between FEA operators and the shared sparse solver layer
//! (`runmat-sparse-solvers`). Every family's linear and eigen solve goes through here, so solver
//! policy, threading, cancellation and provenance are decided in one place.

use std::borrow::Cow;

use runmat_sparse_solvers::{control::SolveControl, CsrRef};

use crate::{algebra::RealSparseMatrix, planning::resource::LocalExecutionPlan};

/// Index arrays of a FEA matrix as `usize` slices (zero-copy on 64-bit hosts).
fn as_usize(values: &[u64]) -> Cow<'_, [usize]> {
    #[cfg(target_pointer_width = "64")]
    {
        // SAFETY: u64 and usize have identical size and alignment on 64-bit targets and every
        // bit pattern is valid for both.
        Cow::Borrowed(unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<usize>(), values.len()) })
    }
    #[cfg(not(target_pointer_width = "64"))]
    {
        Cow::Owned(values.iter().map(|&value| value as usize).collect())
    }
}

/// Borrowed CSR arrays of a canonical FEA matrix.
pub(crate) struct CsrArrays<'a> {
    rows: usize,
    columns: usize,
    row_ptr: Cow<'a, [usize]>,
    col_idx: Cow<'a, [usize]>,
    values: &'a [f64],
}

impl<'a> CsrArrays<'a> {
    pub(crate) fn of(matrix: &'a RealSparseMatrix) -> Self {
        Self {
            rows: matrix.rows(),
            columns: matrix.columns(),
            row_ptr: as_usize(matrix.pattern().row_offsets()),
            col_idx: as_usize(matrix.pattern().column_indices()),
            values: matrix.values(),
        }
    }
    /// Canonical patterns are validated at construction (sorted, in-bounds), so the view needs no
    /// further checks.
    pub(crate) fn view(&self) -> CsrRef<'_, f64> {
        CsrRef::new_unchecked(self.rows, self.columns, &self.row_ptr, &self.col_idx, self.values)
    }
}

/// Runs shared-solver work on the admitted CPU plan (inline when serial or already inside an
/// admitted pool).
pub(crate) fn with_plan<R: Send>(plan: LocalExecutionPlan, work: impl FnOnce() -> R + Send) -> R {
    let threads = usize::from(effective_plan(plan).thread_count().get());
    if threads <= 1 || runmat_cpu_execution::current_threads() > 1 {
        return work();
    }
    // Cancellation and progress are thread-bound; carry them onto the pool worker.
    let context = crate::progress::capture_fea_progress_context();
    match runmat_cpu_execution::with_cpu_pool(threads, move || {
        let _guard = crate::progress::replace_fea_progress_context(context);
        work()
    }) {
        Ok(result) => result,
        Err(error) => panic!("admitted CPU pool unavailable for the sparse solve: {error}"),
    }
}

/// Cancellation hook bound to the FEA progress context.
pub(crate) fn fea_cancelled() -> bool {
    crate::is_fea_cancelled()
}

pub(crate) fn control() -> SolveControl<'static> {
    SolveControl {
        cancelled: Some(&fea_cancelled),
        progress: None,
        progress_stride: 0,
    }
}

thread_local! {
    static AMBIENT_PLAN: std::cell::Cell<Option<LocalExecutionPlan>> = const { std::cell::Cell::new(None) };
}

/// Installs the admitted CPU plan for every shared solve issued by `work` on this thread (family
/// pipelines whose solver call chain carries no plan of its own: modal, buckling, harmonic,
/// electromagnetic, acoustic, fluid).
pub fn with_solver_plan<R>(plan: LocalExecutionPlan, work: impl FnOnce() -> R) -> R {
    let previous = AMBIENT_PLAN.with(|slot| slot.replace(Some(plan)));
    struct Restore(Option<LocalExecutionPlan>);
    impl Drop for Restore {
        fn drop(&mut self) {
            AMBIENT_PLAN.with(|slot| slot.set(self.0));
        }
    }
    let _restore = Restore(previous);
    work()
}

/// The plan installed by [`with_solver_plan`], serial otherwise.
pub(crate) fn ambient_plan() -> LocalExecutionPlan {
    AMBIENT_PLAN
        .with(|slot| slot.get())
        .unwrap_or_else(LocalExecutionPlan::serial)
}

/// `requested` when it is parallel, otherwise the ambient plan.
pub(crate) fn effective_plan(requested: LocalExecutionPlan) -> LocalExecutionPlan {
    if requested.is_serial() {
        ambient_plan()
    } else {
        requested
    }
}

