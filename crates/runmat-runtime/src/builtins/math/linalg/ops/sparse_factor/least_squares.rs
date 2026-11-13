//! Selected-column least squares through the existing owned RRQR/COD authority.
//! This facade reports only a passive solution/rank, never an optimization status.
use super::{SparseCompleteQr, SparseCompleteQrError};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::BuiltinCatalogIdentity;
use runmat_value::SparseTensor;

pub(crate) struct PassiveSolution {
    pub values: Vec<f64>,
    pub rank: usize,
}

fn qr_error(
    error: SparseCompleteQrError,
    runtime: Option<&RuntimeContext>,
    owner: BuiltinCatalogIdentity,
) -> RuntimeError {
    let mut error = match error {
        SparseCompleteQrError::Admission(source) => {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:LeastSquares:Allocation",
                "selected-column QR storage",
                source,
            )
        }
        SparseCompleteQrError::Execution(source) => source.into_runtime_error_in_context(runtime),
        source => crate::build_runtime_error(format!(
            "{}: passive QR/COD solve failed: {source}",
            owner.name
        ))
        .with_builtin(owner.name)
        .with_source(source)
        .build(),
    };
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    crate::builtins::common::map_control_flow_with_builtin(error, owner.name)
}
fn allocation(
    source: runmat_value::ExactHostNumericAllocationError,
    owner: BuiltinCatalogIdentity,
) -> RuntimeError {
    crate::build_runtime_error(format!(
        "{}: passive-column storage allocation failed",
        owner.name
    ))
    .with_builtin(owner.name)
    .with_source(source)
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}
use super::super::linear_operator::CooperativeOperatorError;
use super::super::sparse_operator::{CscOperator, SparseLinearOperator, SparseOperatorError};

/// One immutable, already-validated original CSC and its invocation authority.
/// No coefficients, factor, solver state or callback data are copied here.
pub(crate) struct PassiveMatrix<'a> {
    matrix: CscOperator<'a, f64>,
    runtime: &'a RuntimeContext,
    owner: BuiltinCatalogIdentity,
}
impl<'a> PassiveMatrix<'a> {
    pub(crate) async fn borrow(
        original: &'a SparseTensor,
        runtime: &'a RuntimeContext,
        owner: BuiltinCatalogIdentity,
    ) -> BuiltinResult<Self> {
        crate::context::host_allocation::require_unbounded_workspace(Some(runtime),owner.name,
            "finite host-memory execution is unsupported for selected-column assembly and passive-factor workspace")?;
        let operator = SparseLinearOperator::borrow_cooperatively(original, runtime)
            .await
            .map_err(|error| {
                let error = match error {
                    CooperativeOperatorError::Execution(e) => SparseCompleteQrError::Execution(e),
                    CooperativeOperatorError::Operator(SparseOperatorError::Admission(e)) => {
                        SparseCompleteQrError::Admission(e)
                    }
                    CooperativeOperatorError::Operator(SparseOperatorError::UnsupportedClass) => {
                        SparseCompleteQrError::UnsupportedClass
                    }
                    CooperativeOperatorError::Operator(SparseOperatorError::InvalidTopology(
                        detail,
                    )) => SparseCompleteQrError::InvalidTopology(detail.to_owned()),
                    CooperativeOperatorError::Operator(SparseOperatorError::DimensionMismatch) => {
                        SparseCompleteQrError::DimensionMismatch
                    }
                    CooperativeOperatorError::Operator(SparseOperatorError::Overflow) => {
                        SparseCompleteQrError::Admission(
                            crate::context::host_allocation::HostAllocationError::Overflow,
                        )
                    }
                };
                qr_error(error, Some(runtime), owner)
            })?;
        let SparseLinearOperator::RealF64(matrix) = operator else {
            return Err(qr_error(
                SparseCompleteQrError::UnsupportedClass,
                Some(runtime),
                owner,
            ));
        };
        Ok(Self {
            matrix,
            runtime,
            owner,
        })
    }
    /// Only selected-column storage is assembled; QR/COD owns its numerical
    /// fill and solve workspace under the original issuer. No public status.
    pub(crate) async fn solve(
        &self,
        columns: &[usize],
        rhs: &[f64],
    ) -> BuiltinResult<PassiveSolution> {
        let runtime = self.runtime;
        let owner = self.owner;
        let original = &self.matrix;
        crate::context::host_allocation::require_unbounded_workspace(Some(runtime),owner.name,
            "finite host-memory execution is unsupported for selected-column assembly and passive-factor workspace")?;
        let invalid = |e| qr_error(e, Some(runtime), owner);
        if rhs.len() != original.rows()
            || columns.iter().any(|&c| c >= original.cols())
            || columns.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(invalid(SparseCompleteQrError::DimensionMismatch));
        }
        let mut cooperation = runtime.cpu_cooperation();
        cooperation
            .checkpoint(0)
            .await
            .map_err(|e| qr_error(e.into(), Some(runtime), owner))?;
        if columns.is_empty() {
            return Ok(PassiveSolution {
                values: Vec::new(),
                rank: 0,
            });
        }
        let capacity = columns
            .iter()
            .try_fold(0usize, |sum, &col| {
                sum.checked_add(original.column(col).0.len())
            })
            .ok_or_else(|| invalid(SparseCompleteQrError::DimensionMismatch))?;
        let mut pointers = runmat_value::exact_capacity_vec(
            columns
                .len()
                .checked_add(1)
                .ok_or_else(|| invalid(SparseCompleteQrError::DimensionMismatch))?,
        )
        .map_err(|e| allocation(e, owner))?;
        let mut rows =
            runmat_value::exact_capacity_vec(capacity).map_err(|e| allocation(e, owner))?;
        let mut coefficients =
            runmat_value::exact_capacity_vec(capacity).map_err(|e| allocation(e, owner))?;
        pointers.push(0);
        for &col in columns {
            let (source_rows, source_values) = original.column(col);
            for (row_chunk, value_chunk) in source_rows.chunks(1024).zip(source_values.chunks(1024))
            {
                rows.extend_from_slice(row_chunk);
                coefficients.extend_from_slice(value_chunk);
                cooperation
                    .checkpoint(row_chunk.len() as u64)
                    .await
                    .map_err(|e| qr_error(e.into(), Some(runtime), owner))?;
            }
            pointers.push(rows.len());
            cooperation
                .checkpoint(1)
                .await
                .map_err(|e| qr_error(e.into(), Some(runtime), owner))?;
        }
        let selected =
            SparseTensor::new(original.rows(), columns.len(), pointers, rows, coefficients)
                .map_err(|e| invalid(SparseCompleteQrError::InvalidTopology(e)))?;
        let factor = SparseCompleteQr::<f64>::factor(&selected, runtime)
            .await
            .map_err(invalid)?;
        drop(selected);
        factor
            .check_owner(runtime)
            .map_err(|e| qr_error(e.into(), Some(runtime), owner))?;
        let rank = factor.rank();
        let mut output =
            runmat_value::exact_capacity_vec(columns.len()).map_err(|e| allocation(e, owner))?;
        output.resize(columns.len(), 0.0);
        factor
            .solve_with_output(rhs, 1, runtime, std::convert::identity, |index, value| {
                if let Some(slot) = output.get_mut(index) {
                    *slot = value;
                    Ok(())
                } else {
                    Err(SparseCompleteQrError::DimensionMismatch)
                }
            })
            .await
            .map_err(invalid)?;
        factor
            .check_owner(runtime)
            .map_err(|e| qr_error(e.into(), Some(runtime), owner))?;
        Ok(PassiveSolution {
            values: output,
            rank,
        })
    }
}

mod borrowed;
pub(crate) use borrowed::{solve_borrowed, BorrowedProblem};
