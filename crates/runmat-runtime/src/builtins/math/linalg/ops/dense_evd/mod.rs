//! Shared caller-owned full/projected EVD physical boundary. Language parsing,
//! residency intent, diagnostics and subset policy remain with their consumers.
use crate::context::{
    host_allocation::{
        AlignedHostWorkspace, HostAllocationError, HostAllocationPlan, HostAllocationPolicy,
        HostAllocationReceipt, HostTypedWorkspace,
    },
    RuntimeContext,
};
use crate::execution::{CpuCooperationPolicy, ExecutionServiceError};
use num_complex::Complex64;

mod generalized;
#[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
mod generalized_kernel;
#[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
pub(crate) use generalized_kernel::GevdFailure;
mod homogeneous;
pub(crate) use homogeneous::homogeneous_eigenvalue;
pub(crate) mod kernel;
mod normalization;
pub(crate) mod publication;
#[cfg(test)]
mod tests;
mod workspace;
use workspace::{Buffer, Scratch, Workspace};

pub(crate) const REAL_EPS: f64 = 1e-12;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Problem {
    Standard,
    Generalized,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Vectors {
    Right,
    LeftAndRight,
}

#[derive(Debug)]
pub(crate) struct EvdFailure(pub faer::linalg::evd::EvdError);
impl std::fmt::Display for EvdFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "eigendecomposition failed: {:?}", self.0)
    }
}
impl std::error::Error for EvdFailure {}

#[derive(Debug, thiserror::Error)]
pub(crate) enum DenseEvdError {
    #[error("EVD allocation failed: {0}")]
    Admission(#[from] HostAllocationError),
    #[error("EVD execution failed: {0}")]
    Execution(#[from] ExecutionServiceError),
    #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
    #[error("EVD matrix/workspace dimensions do not match")]
    DimensionMismatch,
    #[error("EVD requires finite coefficient components")]
    NonFiniteInput,
    #[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
    #[error("{0}")]
    GeneralizedNumerical(#[source] GevdFailure),
    #[error("{0}")]
    Numerical(#[source] EvdFailure),
    #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
    #[error("LAPACK ZGGEV workspace query failed with info = {0}")]
    LapackQuery(i32),
    #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
    #[error("LAPACK ZGGEV failed with info = {0}")]
    LapackFailure(i32),
}

// Numerical worker results do not carry Runtime-owned diagnostic objects.
enum NumericalFailure {
    Evd(EvdFailure),
    #[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
    Gevd(GevdFailure),
    #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
    Dimension,
    #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
    LapackQuery(i32),
    #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
    LapackFailure(i32),
}
impl From<NumericalFailure> for DenseEvdError {
    fn from(error: NumericalFailure) -> Self {
        match error {
            NumericalFailure::Evd(e) => Self::Numerical(e),
            #[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
            NumericalFailure::Gevd(e) => Self::GeneralizedNumerical(e),
            #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
            NumericalFailure::Dimension => Self::DimensionMismatch,
            #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
            NumericalFailure::LapackQuery(info) => Self::LapackQuery(info),
            #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
            NumericalFailure::LapackFailure(info) => Self::LapackFailure(info),
        }
    }
}

pub(crate) struct PreparedEvd {
    workspace: Workspace,
    // Numerical allocations physically retire before the operation receipt.
    receipt: Option<HostAllocationReceipt>,
}
pub(crate) struct Eigenpairs {
    workspace: Workspace,
    receipt: Option<HostAllocationReceipt>,
}

pub(super) fn add(left: u64, right: u64) -> Result<u64, HostAllocationError> {
    left.checked_add(right).ok_or(HostAllocationError::Overflow)
}
pub(super) fn check(
    receipt: Option<&HostAllocationReceipt>,
    runtime: Option<&RuntimeContext>,
) -> Result<(), DenseEvdError> {
    match (receipt, runtime) {
        (Some(receipt), Some(runtime)) => receipt.checkpoint(runtime)?,
        (Some(_), None) => return Err(HostAllocationError::StaleAuthority.into()),
        (None, runtime) => {
            if runtime.is_some_and(RuntimeContext::is_cancelled) {
                return Err(ExecutionServiceError::Cancelled.into());
            }
            if HostAllocationReceipt::policy(runtime)? != HostAllocationPolicy::Unbounded {
                return Err(HostAllocationError::StaleAuthority.into());
            }
        }
    }
    Ok(())
}

impl PreparedEvd {
    pub(crate) async fn new(
        order: usize,
        problem: Problem,
        vectors: Vectors,
        runtime: Option<&RuntimeContext>,
    ) -> Result<Self, DenseEvdError> {
        let square = order
            .checked_mul(order)
            .ok_or(HostAllocationError::Overflow)?;
        let left = vectors == Vectors::LeftAndRight;
        let mut bytes = add(
            HostTypedWorkspace::<Complex64>::quoted_bytes(square)?,
            HostTypedWorkspace::<Complex64>::quoted_bytes(square)?,
        )?;
        bytes = add(bytes, HostTypedWorkspace::<Complex64>::quoted_bytes(order)?)?;
        if left {
            bytes = add(
                bytes,
                HostTypedWorkspace::<Complex64>::quoted_bytes(square)?,
            )?;
        }
        let requirement = if problem == Problem::Generalized {
            bytes = add(
                bytes,
                HostTypedWorkspace::<Complex64>::quoted_bytes(square)?,
            )?;
            bytes = add(bytes, generalized::bytes(order, left)?)?;
            generalized::requirement(order, left)
        } else {
            kernel::requirement(order, kernel::Kind::General, left)
        };
        bytes = add(
            bytes,
            AlignedHostWorkspace::quoted_bytes(
                requirement
                    .layout()
                    .map_err(|_| HostAllocationError::Overflow)?,
            )?,
        )?;
        let receipt = HostAllocationReceipt::admit(runtime, HostAllocationPlan::new(bytes, 0)?)?;
        let workspace = Workspace {
            order,
            matrix: Buffer::new(square, runtime, receipt.as_ref()).await?,
            mass: if problem == Problem::Generalized {
                Some(Buffer::new(square, runtime, receipt.as_ref()).await?)
            } else {
                None
            },
            values: Buffer::new(order, runtime, receipt.as_ref()).await?,
            right: Buffer::new(square, runtime, receipt.as_ref()).await?,
            left: if left {
                Some(Buffer::new(square, runtime, receipt.as_ref()).await?)
            } else {
                None
            },
            scratch: Scratch::new(requirement, runtime, receipt.as_ref())?,
            generalized: if problem == Problem::Generalized {
                Some(generalized::Workspace::new(order, left, runtime, receipt.as_ref()).await?)
            } else {
                None
            },
        };
        check(receipt.as_ref(), runtime)?;
        Ok(Self { workspace, receipt })
    }
    pub(crate) fn matrix_mut(&mut self) -> &mut [Complex64] {
        self.workspace.matrix.get_mut()
    }
    pub(crate) fn check_owner(
        &self,
        runtime: Option<&RuntimeContext>,
    ) -> Result<(), DenseEvdError> {
        check(self.receipt.as_ref(), runtime)
    }
    pub(crate) fn mass_mut(&mut self) -> Option<&mut [Complex64]> {
        self.workspace.mass.as_mut().map(Buffer::get_mut)
    }

    pub(crate) async fn compute(
        self,
        runtime: Option<&RuntimeContext>,
    ) -> Result<Eigenpairs, DenseEvdError> {
        let Self { workspace, receipt } = self;
        #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
        let mut workspace = workspace;
        check(receipt.as_ref(), runtime)?;
        // Validation is bounded by actual coefficient work, not just one gate.
        let mut cooperation = runtime.map(RuntimeContext::cpu_cooperation);
        for values in
            std::iter::once(workspace.matrix.get()).chain(workspace.mass.iter().map(Buffer::get))
        {
            for chunk in values.chunks(4096) {
                if chunk.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
                    return Err(DenseEvdError::NonFiniteInput);
                }
                if let Some(cooperation) = &mut cooperation {
                    cooperation.checkpoint(chunk.len() as u64).await?;
                }
                check(receipt.as_ref(), runtime)?;
            }
        }
        if workspace.order == 0 {
            check(receipt.as_ref(), runtime)?;
            return Ok(Eigenpairs { workspace, receipt });
        }
        #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
        if workspace.generalized.is_some() {
            let (queried, count) =
                run_job(runtime, receipt.as_ref(), workspace, generalized::query).await?;
            workspace = queried;
            let bytes = HostTypedWorkspace::<Complex64>::quoted_bytes(count)?;
            let query_receipt =
                HostAllocationReceipt::admit(runtime, HostAllocationPlan::new(bytes, 0)?)?;
            let work = Buffer::new(count, runtime, query_receipt.as_ref()).await?;
            workspace
                .generalized
                .as_mut()
                .expect("generalized query")
                .work = Some(work);
            check(query_receipt.as_ref(), runtime)?;
        }
        let (workspace, ()) = run_job(runtime, receipt.as_ref(), workspace, |workspace| {
            if workspace.generalized.is_some() {
                generalized::compute(workspace)?;
            } else {
                kernel::compute(
                    workspace.order,
                    kernel::Kind::General,
                    workspace.matrix.get(),
                    workspace.values.get_mut(),
                    workspace.right.get_mut(),
                    workspace.left.as_mut().map(Buffer::get_mut),
                    workspace.scratch.get_mut(),
                )
                .map_err(NumericalFailure::Evd)?;
                normalize_right(workspace.right.get_mut(), workspace.order);
                if let Some(left) = &mut workspace.left {
                    normalization::left(
                        left.get_mut(),
                        workspace.right.get(),
                        workspace.order,
                        normalization::Metric::Identity,
                    );
                }
            }
            Ok(())
        })
        .await?;
        check(receipt.as_ref(), runtime)?;
        Ok(Eigenpairs { workspace, receipt })
    }
}

async fn run_job<T: Send + 'static>(
    runtime: Option<&RuntimeContext>,
    receipt: Option<&HostAllocationReceipt>,
    workspace: Workspace,
    compute: impl FnOnce(&mut Workspace) -> Result<T, NumericalFailure> + Send + 'static,
) -> Result<(Workspace, T), DenseEvdError> {
    check(receipt, runtime)?;
    let work = move || {
        let mut workspace = workspace;
        let result = compute(&mut workspace);
        (workspace, result)
    };
    let (workspace, result) = if let Some(runtime) = runtime {
        if runtime.service_ports().blocking_cpu().is_none()
            && runtime.execution().cpu_cooperation_policy() == CpuCooperationPolicy::NeverYield
        {
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
                    receipt.map(HostAllocationReceipt::retain_scratch_for_cpu_work),
                    work,
                )
                .await?
        }
    } else {
        work()
    };
    check(receipt, runtime)?;
    Ok((workspace, result.map_err(DenseEvdError::from)?))
}

impl Eigenpairs {
    pub(crate) fn order(&self) -> usize {
        self.workspace.order
    }
    pub(crate) fn values(&self) -> &[Complex64] {
        self.workspace.values.get()
    }
    pub(crate) fn right(&self) -> &[Complex64] {
        self.workspace.right.get()
    }
    pub(crate) fn left(&self) -> Option<&[Complex64]> {
        self.workspace.left.as_ref().map(Buffer::get)
    }
    pub(crate) fn check_owner(
        &self,
        runtime: Option<&RuntimeContext>,
    ) -> Result<(), DenseEvdError> {
        check(self.receipt.as_ref(), runtime)
    }
}

fn normalize_right(right: &mut [Complex64], n: usize) {
    if n == 0 {
        return;
    }
    for column in right.chunks_exact_mut(n) {
        let norm = column.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        if norm > REAL_EPS {
            for value in column {
                *value /= norm;
            }
        }
    }
}
