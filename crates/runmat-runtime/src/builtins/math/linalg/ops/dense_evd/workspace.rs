//! Numerical backing only. Every managed buffer owns its physical claim;
//! the operation receipt remains outside CPU jobs and outlives publication.
use faer::dyn_stack::{MemBuffer, StackReq};
use num_complex::Complex64;
use std::mem::MaybeUninit;

use super::DenseEvdError;
use crate::context::{
    host_allocation::{
        AlignedHostWorkspace, HostAllocationError, HostAllocationReceipt, HostTypedElement,
        HostTypedWorkspace,
    },
    RuntimeContext,
};
use runmat_value::ExactHostNumericAllocationError;

pub(super) enum Buffer<T: HostTypedElement> {
    Managed(HostTypedWorkspace<T>),
    Ordinary(Vec<T>),
}
impl<T: HostTypedElement> Buffer<T> {
    pub(super) async fn new(
        count: usize,
        runtime: Option<&RuntimeContext>,
        receipt: Option<&HostAllocationReceipt>,
    ) -> Result<Self, DenseEvdError> {
        if let (Some(runtime), Some(receipt)) = (runtime, receipt) {
            return Ok(Self::Managed(receipt.temporary_typed(runtime, count)?));
        }
        super::check(None, runtime)?;
        let mut values = Vec::new();
        values.try_reserve_exact(count).map_err(|_| {
            HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed)
        })?;
        let mut cooperation = runtime.map(RuntimeContext::cpu_cooperation);
        while values.len() < count {
            let end = values.len() + (count - values.len()).min(4096);
            values.resize(end, T::default());
            if let Some(cooperation) = &mut cooperation {
                cooperation.checkpoint(4096).await?;
            }
        }
        super::check(None, runtime)?;
        Ok(Self::Ordinary(values))
    }
    pub(super) fn get(&self) -> &[T] {
        match self {
            Self::Managed(v) => v.as_slice(),
            Self::Ordinary(v) => v,
        }
    }
    pub(super) fn get_mut(&mut self) -> &mut [T] {
        match self {
            Self::Managed(v) => v.as_mut_slice(),
            Self::Ordinary(v) => v,
        }
    }
}

pub(super) enum Scratch {
    Managed(AlignedHostWorkspace),
    Ordinary(MemBuffer),
}
impl Scratch {
    pub(super) fn new(
        requirement: StackReq,
        runtime: Option<&RuntimeContext>,
        receipt: Option<&HostAllocationReceipt>,
    ) -> Result<Self, DenseEvdError> {
        if let (Some(runtime), Some(receipt)) = (runtime, receipt) {
            return Ok(Self::Managed(AlignedHostWorkspace::new(
                requirement
                    .layout()
                    .map_err(|_| HostAllocationError::Overflow)?,
                receipt,
                runtime,
            )?));
        }
        super::check(None, runtime)?;
        Ok(Self::Ordinary(MemBuffer::try_new(requirement).map_err(
            |_| HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed),
        )?))
    }
    pub(super) fn get_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        match self {
            Self::Managed(v) => v.as_mut_uninit(),
            Self::Ordinary(v) => v,
        }
    }
}

pub(super) struct Workspace {
    pub order: usize,
    pub matrix: Buffer<Complex64>,
    pub mass: Option<Buffer<Complex64>>,
    pub values: Buffer<Complex64>,
    pub right: Buffer<Complex64>,
    pub left: Option<Buffer<Complex64>>,
    pub scratch: Scratch,
    pub generalized: Option<super::generalized::Workspace>,
}
