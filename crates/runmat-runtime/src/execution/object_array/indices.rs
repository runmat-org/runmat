//! Temporary index metadata, never a second Value payload representation.
use super::*;
use crate::context::host_allocation::HostTypedWorkspace;
enum Storage {
    Exact(HostTypedWorkspace<usize>),
    Ordinary(Vec<usize>),
}
pub(crate) struct Indices<'a> {
    storage: Storage,
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a RuntimeContext>,
}
impl<'a> Indices<'a> {
    pub(crate) fn new(
        runtime: Option<&'a RuntimeContext>,
        count: usize,
    ) -> Result<Self, RuntimeError> {
        let bytes =
            HostTypedWorkspace::<usize>::quoted_bytes(count).map_err(|e| allocation(runtime, e))?;
        let receipt = HostAllocationReceipt::admit(
            runtime,
            HostAllocationPlan::new(bytes, 0).map_err(|e| allocation(runtime, e))?,
        )
        .map_err(|e| allocation(runtime, e))?;
        let storage = match &receipt {
            Some(receipt) => Storage::Exact(
                receipt
                    .temporary_typed(runtime.expect("selected context"), count)
                    .map_err(|e| allocation(runtime, e))?,
            ),
            None => Storage::Ordinary(vec![0; count]),
        };
        let result = Self {
            storage,
            receipt,
            runtime,
        };
        result.checkpoint()?;
        Ok(result)
    }
    pub(crate) fn values(&self) -> &[usize] {
        match &self.storage {
            Storage::Exact(s) => s.as_slice(),
            Storage::Ordinary(s) => s,
        }
    }
    pub(crate) fn values_mut(&mut self) -> &mut [usize] {
        match &mut self.storage {
            Storage::Exact(s) => s.as_mut_slice(),
            Storage::Ordinary(s) => s,
        }
    }
    pub(crate) fn checkpoint(&self) -> Result<(), RuntimeError> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("selected context"))
                .map_err(|e| allocation(self.runtime, e))?;
        }
        Ok(())
    }
}
