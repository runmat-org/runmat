//! One original-context sparse QR/COD factor for equality correction,
//! null-space projection and adjoint multiplier recovery. No Gram matrix.
use super::{SparseCompleteQr, SparseCompleteQrError as Error};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::BuiltinCatalogIdentity;

pub(crate) struct EqualityFactor {
    factor: SparseCompleteQr<f64>,
    rows: usize,
    columns: usize,
    runtime: Option<RuntimeContext>,
    owner: BuiltinCatalogIdentity,
}
fn error(
    source: Error,
    runtime: Option<&RuntimeContext>,
    owner: BuiltinCatalogIdentity,
) -> RuntimeError {
    let failure = match source {
        Error::Execution(source) => {
            return crate::builtins::common::map_control_flow_with_builtin(
                source.into_runtime_error_in_context(runtime),
                owner.name,
            )
        }
        Error::Admission(source) => crate::context::host_allocation::host_allocation_runtime_error(
            runtime,
            "RunMat:LinearEquality:Allocation",
            "original equality factor ownership",
            source,
        ),
        source => {
            crate::build_runtime_error(format!("{}: equality QR/COD failed: {source}", owner.name))
                .with_builtin(owner.name)
                .with_source(source)
                .build()
        }
    };
    let mut failure = crate::builtins::common::map_control_flow_with_builtin(failure, owner.name);
    failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
    failure
}
impl EqualityFactor {
    pub(crate) async fn factor(
        rows: usize,
        columns: usize,
        pointers: &[usize],
        indices: &[usize],
        values: &[f64],
        runtime: Option<&RuntimeContext>,
        owner: BuiltinCatalogIdentity,
    ) -> BuiltinResult<Option<Self>> {
        crate::context::host_allocation::require_unbounded_workspace(runtime,owner.name,"finite host-memory execution is unsupported for equality QR/COD topology validation and fill")?;
        let factor = match SparseCompleteQr::factor_borrowed(
            rows, columns, pointers, indices, values, runtime,
        )
        .await
        {
            Ok(factor) => factor,
            Err(Error::NumericalBreakdown) => return Ok(None),
            Err(source) => return Err(error(source, runtime, owner)),
        };
        Ok(Some(Self {
            factor,
            rows,
            columns,
            runtime: runtime.cloned(),
            owner,
        }))
    }
    pub(crate) fn rank(&self) -> usize {
        self.factor.rank()
    }
    pub(crate) async fn solve(
        &self,
        rhs: &[f64],
        adjoint: bool,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        let runtime = self.runtime.as_ref();
        let input = if adjoint { self.columns } else { self.rows };
        let output = if adjoint { self.rows } else { self.columns };
        if rhs.len() != input {
            return Err(error(Error::DimensionMismatch, runtime, self.owner));
        }
        let mut values = runmat_value::exact_capacity_vec(output).map_err(|source| {
            error(
                Error::Admission(
                    crate::context::host_allocation::HostAllocationError::Numeric(source),
                ),
                runtime,
                self.owner,
            )
        })?;
        values.resize(output, 0.0);
        match self.solve_into(rhs, adjoint, &mut values).await? {
            Some(()) => Ok(Some(values)),
            None => Ok(None),
        }
    }
    pub(crate) async fn solve_into(
        &self,
        rhs: &[f64],
        adjoint: bool,
        values: &mut [f64],
    ) -> BuiltinResult<Option<()>> {
        let runtime = self.runtime.as_ref();
        let input = if adjoint { self.columns } else { self.rows };
        let output = if adjoint { self.rows } else { self.columns };
        if rhs.len() != input || values.len() != output {
            return Err(error(Error::DimensionMismatch, runtime, self.owner));
        }
        let result = self
            .factor
            .solve_optional_with_output(
                rhs,
                1,
                adjoint,
                runtime,
                std::convert::identity,
                |index, value| {
                    let slot = values.get_mut(index).ok_or(Error::DimensionMismatch)?;
                    *slot = value;
                    Ok(())
                },
            )
            .await;
        match result {
            Ok(()) => Ok(values.iter().all(|value| value.is_finite()).then_some(())),
            Err(Error::NumericalBreakdown) => Ok(None),
            Err(source) => Err(error(source, runtime, self.owner)),
        }
    }
}
