//! Local numerical phase admission shared by solver and recovery owners.

use super::coercivity::{NumericalCapacityError, NumericalCapacityPhase};
use runmat_simulation_usage::{
    AsyncCapacityOwner, LedgerError, OperationCount, UsageCategory, UsageContext,
};
use std::{cell::RefCell, future::Future, marker::PhantomData, pin::Pin};

pub(crate) struct PhaseCapacityOwner<'a, O, F, A> {
    before: RefCell<&'a mut O>,
    phase: NumericalCapacityPhase,
    marker: PhantomData<fn() -> (F, A)>,
}
impl<'a, O, F, A> PhaseCapacityOwner<'a, O, F, A> {
    pub fn new(before: &'a mut O, phase: NumericalCapacityPhase) -> Self {
        Self {
            before: RefCell::new(before),
            phase,
            marker: PhantomData,
        }
    }
}
impl<O, F, A> AsyncCapacityOwner<A> for PhaseCapacityOwner<'_, O, F, A>
where
    O: FnMut(NumericalCapacityPhase, OperationCount) -> F,
    F: Future<Output = Result<(), A>>,
{
    fn ensure_capacity(
        &self,
        additional: OperationCount,
    ) -> Pin<Box<dyn Future<Output = Result<(), A>> + '_>> {
        let future = (self.before.borrow_mut())(self.phase.clone(), additional);
        Box::pin(future)
    }
}

/// The host wait occurs before reserving or running this batch. The successful
/// batch records its own work; enclosing callers must not charge it again.
pub(crate) async fn measured<T, E, A>(
    owner: Option<&dyn AsyncCapacityOwner<A>>,
    category: UsageCategory,
    work: OperationCount,
    run: impl FnOnce() -> Result<T, E>,
) -> Result<T, NumericalCapacityError<E, A>>
where
    E: From<LedgerError>,
{
    let context = UsageContext::capture();
    if let Some(owner) = owner {
        owner
            .ensure_capacity(work)
            .await
            .map_err(NumericalCapacityError::Admission)?;
    }
    let mut reservation = context
        .as_ref()
        .map(|context| context.ledger().reserve(category, work))
        .transpose()
        .map_err(|error| NumericalCapacityError::Numerical(E::from(error)))?;
    let result = run().map_err(NumericalCapacityError::Numerical)?;
    if let Some(reservation) = &mut reservation {
        reservation
            .complete(work)
            .map_err(|error| NumericalCapacityError::Numerical(E::from(error)))?;
    }
    Ok(result)
}

/// Native numerical batches share reservation and completion semantics with
/// awaited owners. Without an async owner there is no suspension point.
pub(crate) fn measured_sync<T, E>(
    category: UsageCategory,
    work: OperationCount,
    run: impl FnOnce() -> Result<T, E>,
) -> Result<T, E>
where
    E: From<LedgerError>,
{
    use futures::FutureExt;
    measured::<T, E, std::convert::Infallible>(None, category, work, run)
        .now_or_never()
        .expect("native numerical batch has no awaited capacity owner")
        .map_err(|error| match error {
            NumericalCapacityError::Numerical(error) => error,
            NumericalCapacityError::Admission(never) => match never {},
        })
}

/// Reference work of one real CSR product `y = A x`: a length-k dot per stored row
/// (`k` products and `k - 1` additions). Empty rows cost nothing.
pub(crate) fn real_spmv_work(
    matrix: &super::RealSparseMatrix,
) -> Result<OperationCount, LedgerError> {
    matrix
        .pattern()
        .row_offsets()
        .windows(2)
        .try_fold(OperationCount::ZERO, |total, row| {
            total
                .checked_add(OperationCount::direct_real_dot(row[1] - row[0])?)
                .map_err(LedgerError::from)
        })
}
