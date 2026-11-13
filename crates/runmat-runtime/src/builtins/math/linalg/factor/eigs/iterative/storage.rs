//! Exact scratch owners; ordinary storage is permitted only under the
//! caller's captured unrestricted policy, never as a Managed fallback.
use super::scalar::Scalar;
use crate::context::{
    host_allocation::{
        HostAllocationError, HostAllocationReceipt, HostNumericWorkspace, HostTypedElement,
        HostTypedWorkspace,
    },
    RuntimeContext,
};
use crate::BuiltinResult;

pub(super) enum Buffer<T: HostTypedElement> {
    Managed(HostTypedWorkspace<T>),
    Unbounded(Vec<T>),
}
impl<T: HostTypedElement> Buffer<T> {
    pub async fn new(
        count: usize,
        runtime: Option<&RuntimeContext>,
        receipt: Option<&HostAllocationReceipt>,
    ) -> BuiltinResult<Self> {
        if let (Some(runtime), Some(receipt)) = (runtime, receipt) {
            return receipt
                .temporary_typed(runtime, count)
                .map(Self::Managed)
                .map_err(|e| super::admission(Some(runtime), e));
        }
        crate::builtins::math::linalg::factor::iterative::copy::unbounded_checkpoint(
            runtime, "eigs",
        )?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| super::invalid("unable to reserve iterative scratch"))?;
        let mut work = super::work::Work::new(runtime, None)?;
        while values.len() < count {
            values.push(T::default());
            work.tick().await?;
        }
        work.flush().await?;
        Ok(Self::Unbounded(values))
    }
    pub fn get(&self) -> &[T] {
        match self {
            Self::Managed(v) => v.as_slice(),
            Self::Unbounded(v) => v,
        }
    }
    pub fn get_mut(&mut self) -> &mut [T] {
        match self {
            Self::Managed(v) => v.as_mut_slice(),
            Self::Unbounded(v) => v,
        }
    }
}
pub(super) enum Native<T: Scalar> {
    Managed(HostNumericWorkspace),
    Unbounded(Vec<T>),
}
impl<T: Scalar> Native<T> {
    pub async fn new(
        count: usize,
        runtime: Option<&RuntimeContext>,
        receipt: Option<&HostAllocationReceipt>,
    ) -> BuiltinResult<Self> {
        if let (Some(runtime), Some(receipt)) = (runtime, receipt) {
            let lanes = count
                .checked_mul(T::LANES)
                .ok_or_else(|| super::admission(Some(runtime), HostAllocationError::Overflow))?;
            return receipt
                .temporary_numeric(runtime, T::DTYPE, lanes)
                .map(Self::Managed)
                .map_err(|e| super::admission(Some(runtime), e));
        }
        crate::builtins::math::linalg::factor::iterative::copy::unbounded_checkpoint(
            runtime, "eigs",
        )?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| super::invalid("unable to reserve native eigs scratch"))?;
        let mut work = super::work::Work::new(runtime, None)?;
        while values.len() < count {
            values.push(T::zero());
            work.tick().await?;
        }
        work.flush().await?;
        Ok(Self::Unbounded(values))
    }
    pub fn get_mut(&mut self) -> &mut [T] {
        match self {
            Self::Managed(v) => T::scratch(v).expect("native recurrence allocation class"),
            Self::Unbounded(v) => v,
        }
    }
}
