use crate::{LedgerError, OperationCount};
use std::{future::Future, pin::Pin};

/// Local, asynchronous capacity acquisition before a numerical batch executes.
/// Implementations retain the original ledger owner and fixed spending cap.
/// Acquiring capacity does not reserve or complete numerical work.
pub trait AsyncCapacityOwner<E = LedgerError> {
    fn ensure_capacity(
        &self,
        additional: OperationCount,
    ) -> Pin<Box<dyn Future<Output = Result<(), E>> + '_>>;
}
