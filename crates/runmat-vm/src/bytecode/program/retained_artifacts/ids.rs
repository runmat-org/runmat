use std::collections::HashSet;

use runmat_hir::FunctionId;

use super::{FunctionRegistry, RetainedCallableStageError};

pub(super) struct DestinationFunctionIds {
    occupied: HashSet<usize>,
    next: usize,
}

impl DestinationFunctionIds {
    pub(super) fn new(registry: &FunctionRegistry) -> Self {
        let occupied = registry
            .functions
            .keys()
            .chain(registry.retired.keys())
            .map(|function| function.0)
            .collect::<HashSet<_>>();
        Self { occupied, next: 0 }
    }

    pub(super) fn allocate(&mut self) -> Result<FunctionId, RetainedCallableStageError> {
        while self.occupied.contains(&self.next) {
            self.next = self
                .next
                .checked_add(1)
                .ok_or(RetainedCallableStageError::FunctionIdentityExhausted)?;
        }
        let function = FunctionId(self.next);
        self.occupied.insert(self.next);
        self.next = self
            .next
            .checked_add(1)
            .ok_or(RetainedCallableStageError::FunctionIdentityExhausted)?;
        Ok(function)
    }
}
