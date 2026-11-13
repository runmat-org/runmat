//! Reusable native-class dense LU owner for matrix solves and preconditioners. The
//! partial-pivot factorization and solves use faer's existing low-level
//! kernels over caller-owned, exactly quoted arrays and MemStack scratch.

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::lu::partial_pivoting::{factor, solve};
use faer::mat::{MatMut, MatRef};
use faer::perm::PermRef;
use faer::traits::ComplexField;
use faer::{Par, Spec};
use runmat_execution::resource::RetainedHostAllocationClaim;
use runmat_value::Value;
use std::sync::Arc;

use crate::context::host_allocation::{
    AlignedHostWorkspace, HostAllocationError, HostAllocationPlan, HostAllocationPolicy,
    HostAllocationReceipt, HostTypedElement, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationPolicy, ExecutionServiceError};

mod native_view;
mod retained;
mod scalar;
pub(crate) mod system;
#[cfg(test)]
pub(in crate::builtins::math::linalg) use retained::solve_dense;
pub(in crate::builtins::math::linalg) use retained::{
    factor_dense, solve_dense_indexed, DenseFactorOptions, DenseScalar,
};
pub(in crate::builtins::math::linalg) use scalar::FaerFactorScalar;

#[derive(Debug)]
pub(in crate::builtins::math::linalg) enum DenseFactorError {
    Admission(HostAllocationError),
    InvalidShape,
    UnsupportedClass,
    AllocationFailed,
    Singular,
    NotPositiveDefinite,
    NonFinite,
    InvalidData(String),
    Numeric(runmat_value::ExactHostNumericAllocationError),
    Execution(ExecutionServiceError),
    Operator(super::sparse_operator::SparseOperatorError),
}
impl std::fmt::Display for DenseFactorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidData(source) => write!(f, "invalid dense factor data: {source}"),
            Self::Operator(source) => write!(f, "dense factor cooperation failed: {source:?}"),
            source => write!(f, "{source:?}"),
        }
    }
}
impl std::error::Error for DenseFactorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(e) => Some(e),
            Self::Execution(e) => Some(e),
            Self::Numeric(e) => Some(e),
            Self::Operator(super::sparse_operator::SparseOperatorError::Admission(e)) => Some(e),
            _ => None,
        }
    }
}
impl From<runmat_value::ExactHostNumericAllocationError> for DenseFactorError {
    fn from(e: runmat_value::ExactHostNumericAllocationError) -> Self {
        Self::Numeric(e)
    }
}

impl From<HostAllocationError> for DenseFactorError {
    fn from(value: HostAllocationError) -> Self {
        Self::Admission(value)
    }
}

enum FactorBacking<E: HostTypedElement> {
    Managed {
        lu: HostTypedWorkspace<E>,
        perm: HostTypedWorkspace<usize>,
        perm_inv: HostTypedWorkspace<usize>,
        rhs: HostTypedWorkspace<E>,
        scratch: AlignedHostWorkspace,
    },
    Unbounded {
        lu: Vec<E>,
        perm: Vec<usize>,
        perm_inv: Vec<usize>,
        rhs: Vec<E>,
        scratch: MemBuffer,
    },
}

/// Factor, permutation, reusable RHS, scratch, and receipt have one lifetime.
/// Solves do not allocate or materialize another coefficient payload.
pub(in crate::builtins::math::linalg) struct DenseFactor<T: FaerFactorScalar> {
    order: usize,
    backing: Option<FactorBacking<T::Element>>,
    receipt: Option<HostAllocationReceipt>,
    _metadata_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

mod input;
mod kernel;
mod work;
use kernel::{factor_body, quote, solve_body, unbounded_vec};

impl<T: FaerFactorScalar> DenseFactor<T> {
    fn checkpoint(&self, runtime: Option<&RuntimeContext>) -> Result<(), DenseFactorError> {
        if let Some(receipt) = self.receipt.as_ref() {
            let runtime = runtime.ok_or(DenseFactorError::Admission(
                HostAllocationError::MissingManagedOwner,
            ))?;
            receipt.checkpoint(runtime)?;
        } else {
            Self::check_unbounded(runtime)?;
        }
        Ok(())
    }

    fn check_unbounded(runtime: Option<&RuntimeContext>) -> Result<(), DenseFactorError> {
        if HostAllocationReceipt::policy(runtime)? != HostAllocationPolicy::Unbounded {
            return Err(DenseFactorError::Admission(
                HostAllocationError::StaleAuthority,
            ));
        }
        Ok(())
    }

    pub async fn factor<'a>(
        value: &Value,
        order: usize,
        runtime: impl Into<Option<&'a RuntimeContext>>,
    ) -> Result<Box<Self>, DenseFactorError> {
        Self::factor_with_policy(value, order, runtime.into(), input::Policy::Exact).await
    }

    pub(in crate::builtins::math::linalg) async fn factor_converted(
        value: &Value,
        order: usize,
        runtime: Option<&RuntimeContext>,
        convert: fn(num_complex::Complex64) -> Option<T>,
    ) -> Result<Box<Self>, DenseFactorError> {
        Self::factor_with_policy(value, order, runtime, input::Policy::Converted(convert)).await
    }

    async fn factor_with_policy(
        value: &Value,
        order: usize,
        runtime: Option<&RuntimeContext>,
        policy: input::Policy<T>,
    ) -> Result<Box<Self>, DenseFactorError> {
        let shape = match value {
            Value::Tensor(tensor) => tensor.shape(),
            Value::ComplexTensor(tensor) => tensor.shape(),
            _ => return Err(DenseFactorError::UnsupportedClass),
        };
        if order == 0
            || order > isize::MAX as usize
            || shape.first() != Some(&order)
            || shape.get(1) != Some(&order)
            || shape.iter().skip(2).any(|&extent| extent != 1)
        {
            return Err(DenseFactorError::InvalidShape);
        }
        let source = input::Input::borrow(value, policy)?;
        let count = order
            .checked_mul(order)
            .ok_or(DenseFactorError::InvalidShape)?;
        if source.len() != count {
            return Err(DenseFactorError::InvalidShape);
        }
        Self::factor_indexed(order, runtime, async |index| source.read(index)).await
    }

    /// Build directly in the admitted LU backing. Numerical model owners do
    /// not need a duplicate coefficient tensor just to call this factorizer.
    async fn factor_indexed(
        order: usize,
        runtime: Option<&RuntimeContext>,
        read: impl std::ops::AsyncFnMut(usize) -> Result<T, DenseFactorError>,
    ) -> Result<Box<Self>, DenseFactorError> {
        if order == 0 || order > isize::MAX as usize {
            return Err(DenseFactorError::InvalidShape);
        }
        let count = order
            .checked_mul(order)
            .ok_or(DenseFactorError::InvalidShape)?;
        let (plan, scratch_req) = quote::<T>(order)?;
        let receipt = HostAllocationReceipt::admit(runtime, plan)?;
        // Only general dense factors need a stable heap owner. Quote its
        // control state separately from the numeric workspaces, and retain
        // that claim until the owner itself is released. No solve allocates
        // another factor or another coefficient payload.
        let metadata_claim = receipt
            .as_ref()
            .map(|receipt| {
                receipt.claim_retained_metadata(
                    runtime.ok_or(HostAllocationError::MissingManagedOwner)?,
                    std::mem::size_of::<Self>() as u64,
                )
            })
            .transpose()?;
        let backing = if let Some(receipt) = receipt.as_ref() {
            let runtime = runtime.ok_or(DenseFactorError::Admission(
                HostAllocationError::MissingManagedOwner,
            ))?;
            let lu = receipt.temporary_typed::<T::Element>(runtime, count)?;
            let perm = receipt.temporary_typed::<usize>(runtime, order)?;
            let perm_inv = receipt.temporary_typed::<usize>(runtime, order)?;
            let rhs = receipt.temporary_typed::<T::Element>(runtime, order)?;
            let layout = scratch_req
                .layout()
                .map_err(|_| HostAllocationError::Overflow)?;
            let scratch = AlignedHostWorkspace::new(layout, receipt, runtime)?;
            FactorBacking::Managed {
                lu,
                perm,
                perm_inv,
                rhs,
                scratch,
            }
        } else {
            let lu = unbounded_vec(count, runtime).await?;
            let perm = unbounded_vec(order, runtime).await?;
            let perm_inv = unbounded_vec(order, runtime).await?;
            let rhs = unbounded_vec(order, runtime).await?;
            let scratch = MemBuffer::new(scratch_req);
            FactorBacking::Unbounded {
                lu,
                perm,
                perm_inv,
                rhs,
                scratch,
            }
        };
        let mut owner = Box::new(Self {
            order,
            backing: Some(backing),
            receipt,
            _metadata_claim: metadata_claim,
        });
        owner.copy_indexed(read, runtime).await?;
        owner.run_factor(runtime).await?;
        Ok(owner)
    }

    async fn copy_indexed(
        &mut self,
        mut read: impl std::ops::AsyncFnMut(usize) -> Result<T, DenseFactorError>,
        runtime: Option<&RuntimeContext>,
    ) -> Result<(), DenseFactorError> {
        let mut work = work::Work::new(runtime, self.receipt.as_ref())?;
        let lu = match self.backing.as_mut().expect("factor backing is installed") {
            FactorBacking::Managed { lu, .. } => lu.as_mut_slice(),
            FactorBacking::Unbounded { lu, .. } => lu.as_mut_slice(),
        };
        for (index, target) in lu.iter_mut().enumerate() {
            *target = read(index).await?.to_factor();
            if !T::finite(*target) {
                return Err(DenseFactorError::NonFinite);
            }
            work.tick().await?;
        }
        work.flush().await
    }

    async fn run_factor(
        &mut self,
        runtime: Option<&RuntimeContext>,
    ) -> Result<(), DenseFactorError> {
        let order = self.order;
        self.checkpoint(runtime)?;
        let backing = self.backing.take().expect("factor backing is installed");
        let work = move || {
            let mut backing = backing;
            match &mut backing {
                FactorBacking::Managed {
                    lu,
                    perm,
                    perm_inv,
                    scratch,
                    ..
                } => factor_body(
                    lu.as_mut_slice(),
                    perm.as_mut_slice(),
                    perm_inv.as_mut_slice(),
                    scratch.as_mut_uninit(),
                    order,
                ),
                FactorBacking::Unbounded {
                    lu,
                    perm,
                    perm_inv,
                    scratch,
                    ..
                } => factor_body(lu, perm, perm_inv, scratch, order),
            }
            backing
        };
        let backing = if let Some(runtime) = runtime {
            if runtime.service_ports().blocking_cpu().is_none()
                && runtime.execution().cpu_cooperation_policy() == CpuCooperationPolicy::NeverYield
            {
                self.checkpoint(Some(runtime))?;
                work()
            } else {
                runtime
                    .cpu_cooperation()
                    .run_owned_blocking_in_context(
                        runtime,
                        runtime
                            .service_ports()
                            .blocking_cpu()
                            .map(|service| service.as_ref()),
                        self.receipt
                            .as_ref()
                            .map(HostAllocationReceipt::retain_scratch_for_cpu_work),
                        work,
                    )
                    .await
                    .map_err(DenseFactorError::Execution)?
            }
        } else {
            work()
        };
        self.backing = Some(backing);
        self.checkpoint(runtime)?;
        let lu = match self.backing.as_ref().expect("factor backing returned") {
            FactorBacking::Managed { lu, .. } => lu.as_slice(),
            FactorBacking::Unbounded { lu, .. } => lu.as_slice(),
        };
        let mut work = work::Work::new(runtime, self.receipt.as_ref())?;
        for diagonal in 0..order {
            if lu[diagonal * order + diagonal] == T::Element::default() {
                return Err(DenseFactorError::Singular);
            }
            work.tick().await?;
        }
        for &value in lu {
            if !T::finite(value) {
                return Err(DenseFactorError::NonFinite);
            }
            work.tick().await?;
        }
        work.flush().await
    }

    pub async fn solve_into<'a>(
        &mut self,
        input: &[T],
        output: &mut [T],
        runtime: impl Into<Option<&'a RuntimeContext>>,
    ) -> Result<(), DenseFactorError> {
        let runtime = runtime.into();
        if input.len() != self.order || output.len() != self.order {
            return Err(DenseFactorError::InvalidShape);
        }
        let order = self.order;
        let mut work = work::Work::new(runtime, self.receipt.as_ref())?;
        let rhs = match self.backing.as_mut().expect("factor backing is installed") {
            FactorBacking::Managed { rhs, .. } => rhs.as_mut_slice(),
            FactorBacking::Unbounded { rhs, .. } => rhs.as_mut_slice(),
        };
        for (target, &value) in rhs.iter_mut().zip(input) {
            *target = value.to_factor();
            if !T::finite(*target) {
                return Err(DenseFactorError::NonFinite);
            }
            work.tick().await?;
        }
        work.flush().await?;
        // The immutable receipt borrow ends before moving backing to the CPU job.
        drop(work);
        self.checkpoint(runtime)?;
        let backing = self.backing.take().expect("factor backing is installed");
        let work = move || {
            let mut backing = backing;
            match &mut backing {
                FactorBacking::Managed {
                    lu,
                    perm,
                    perm_inv,
                    rhs,
                    scratch,
                } => solve_body(
                    lu.as_slice(),
                    perm.as_slice(),
                    perm_inv.as_slice(),
                    rhs.as_mut_slice(),
                    scratch.as_mut_uninit(),
                    order,
                ),
                FactorBacking::Unbounded {
                    lu,
                    perm,
                    perm_inv,
                    rhs,
                    scratch,
                } => solve_body(lu, perm, perm_inv, rhs, scratch, order),
            }
            backing
        };
        let backing = if let Some(runtime) = runtime {
            if runtime.service_ports().blocking_cpu().is_none()
                && runtime.execution().cpu_cooperation_policy() == CpuCooperationPolicy::NeverYield
            {
                self.checkpoint(Some(runtime))?;
                work()
            } else {
                runtime
                    .cpu_cooperation()
                    .run_owned_blocking_in_context(
                        runtime,
                        runtime
                            .service_ports()
                            .blocking_cpu()
                            .map(|service| service.as_ref()),
                        self.receipt
                            .as_ref()
                            .map(HostAllocationReceipt::retain_scratch_for_cpu_work),
                        work,
                    )
                    .await
                    .map_err(DenseFactorError::Execution)?
            }
        } else {
            work()
        };
        self.backing = Some(backing);
        self.checkpoint(runtime)?;
        let rhs = match self.backing.as_ref().expect("factor backing returned") {
            FactorBacking::Managed { rhs, .. } => rhs.as_slice(),
            FactorBacking::Unbounded { rhs, .. } => rhs.as_slice(),
        };
        let mut work = work::Work::new(runtime, self.receipt.as_ref())?;
        for (&source, target) in rhs.iter().zip(output.iter_mut()) {
            *target = T::from_factor(source).ok_or(DenseFactorError::NonFinite)?;
            work.tick().await?;
        }
        work.flush().await
    }
}

#[cfg(test)]
mod tests;
