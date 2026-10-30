//! Issued futures and the output contract each handle was issued with.
//!
//! A handle names a future by id, but its output contract travels with the handle and can be
//! altered by its holder. The table keeps the contract issued by `create_future` beside the
//! future's state, so every operation can reject a handle whose contract differs.
use std::collections::HashMap;

use runmat_execution::{FutureId, OutputContract};
use runmat_runtime::execution::ExecutionServiceError;

use super::FutureState;

pub(super) struct FutureRecord {
    outputs: OutputContract,
    state: FutureState,
}

#[derive(Default)]
pub(super) struct FutureTable(HashMap<FutureId, FutureRecord>);

impl FutureTable {
    pub(super) fn issue(&mut self, id: FutureId, outputs: OutputContract, state: FutureState) {
        let previous = self.0.insert(id, FutureRecord { outputs, state });
        debug_assert!(previous.is_none(), "future ids are issued once");
    }

    /// The stored state of the future this exact handle was issued for.
    pub(super) fn validate(
        &self,
        id: &FutureId,
        outputs: &OutputContract,
    ) -> Result<&FutureState, ExecutionServiceError> {
        let record = self.0.get(id).ok_or(ExecutionServiceError::UnknownHandle)?;
        if &record.outputs != outputs {
            return Err(ExecutionServiceError::InvalidOutputContract);
        }
        Ok(&record.state)
    }

    pub(super) fn get(&self, id: &FutureId) -> Option<&FutureState> {
        self.0.get(id).map(|record| &record.state)
    }

    pub(super) fn get_mut(&mut self, id: &FutureId) -> Option<&mut FutureState> {
        self.0.get_mut(id).map(|record| &mut record.state)
    }

    /// Replaces the state of an issued future and returns the retired state. A future that is
    /// not (or no longer) issued is left absent: a state without its issued contract is never
    /// created.
    pub(super) fn insert(&mut self, id: FutureId, state: FutureState) -> Option<FutureState> {
        self.0
            .get_mut(&id)
            .map(|record| std::mem::replace(&mut record.state, state))
    }

    pub(super) fn remove(&mut self, id: &FutureId) -> Option<FutureState> {
        self.0.remove(id).map(|record| record.state)
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &FutureState> {
        self.0.values().map(|record| &record.state)
    }

    pub(super) fn values_mut(&mut self) -> impl Iterator<Item = &mut FutureState> {
        self.0.values_mut().map(|record| &mut record.state)
    }

    pub(super) fn iter_mut(&mut self) -> impl Iterator<Item = (&FutureId, &mut FutureState)> {
        self.0
            .iter_mut()
            .map(|(id, record)| (id, &mut record.state))
    }

    #[cfg(all(test, unix))]
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
}
