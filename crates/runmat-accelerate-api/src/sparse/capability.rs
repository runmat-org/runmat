use anyhow::ensure;
use runmat_execution::CandidatePreparationState;
use serde::{Deserialize, Serialize};

use crate::{
    GpuTensorStorage, NumericElementType, ProviderCostEstimate, ProviderRejection,
    ProviderRejectionCode,
};

use super::{
    SparseFormat, SparseIndexWidth, SparseKrylovMethod, SparseMatrixDescriptor,
    SparsePatternDescriptor, SparsePreconditionerKind, SparseTranspose,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum SparseCancellationCapability {
    Unsupported,
    BeforeDispatch,
    Cooperative {
        maximum_work_units_between_checks: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseKrylovCapability {
    pub method: SparseKrylovMethod,
    pub preconditioner: Option<SparsePreconditionerKind>,
}

impl SparseCancellationCapability {
    pub const fn supports_resident_solve(self) -> bool {
        matches!(
            self,
            Self::Cooperative {
                maximum_work_units_between_checks: 1..
            }
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderSparseCapabilities {
    pub formats: Vec<SparseFormat>,
    pub index_widths: Vec<SparseIndexWidth>,
    pub element_types: Vec<NumericElementType>,
    pub storage: Vec<GpuTensorStorage>,
    pub transpose_modes: Vec<SparseTranspose>,
    pub preconditioners: Vec<SparsePreconditionerKind>,
    pub krylov: Vec<SparseKrylovCapability>,
    pub max_host_synchronizations_per_solve: Option<u32>,
    pub cancellation: SparseCancellationCapability,
    pub transactional_results: bool,
    pub max_rows: Option<u64>,
    pub max_columns: Option<u64>,
    pub max_nnz: Option<u64>,
}

impl ProviderSparseCapabilities {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure_unique(&self.formats, "sparse formats")?;
        ensure_unique(&self.index_widths, "sparse index widths")?;
        ensure_unique(&self.element_types, "sparse element types")?;
        ensure_unique(&self.storage, "sparse storage layouts")?;
        ensure_unique(&self.transpose_modes, "sparse transpose modes")?;
        ensure_unique(&self.preconditioners, "sparse preconditioners")?;
        ensure_unique(&self.krylov, "sparse Krylov capabilities")?;
        ensure!(
            self.element_types
                .iter()
                .all(|element| element.precision().is_some()),
            "sparse capabilities may advertise only floating-point values"
        );
        ensure!(
            self.krylov.iter().all(|capability| capability
                .preconditioner
                .is_none_or(|kind| self.preconditioners.contains(&kind))),
            "sparse Krylov capability references an unsupported preconditioner"
        );
        ensure!(
            self.krylov.is_empty() || self.max_host_synchronizations_per_solve.is_some(),
            "resident sparse Krylov capability requires a synchronization bound"
        );
        ensure!(
            self.krylov.is_empty() || self.cancellation.supports_resident_solve(),
            "resident sparse Krylov capability requires cooperative cancellation"
        );
        ensure!(
            self.krylov.is_empty() || self.transactional_results,
            "resident sparse Krylov capability requires transactional results"
        );
        ensure!(
            !matches!(
                self.cancellation,
                SparseCancellationCapability::Cooperative {
                    maximum_work_units_between_checks: 0
                }
            ),
            "cooperative sparse cancellation requires a positive work bound"
        );
        ensure!(
            [self.max_rows, self.max_columns, self.max_nnz]
                .into_iter()
                .flatten()
                .all(|limit| limit > 0),
            "sparse capability limits must be positive"
        );
        Ok(())
    }

    pub fn supports(&self, query: SparseFeasibilityQuery) -> bool {
        if self.validate().is_err() || query.validate().is_err() {
            return false;
        }
        let operation_supported = match query.operation {
            SparseOperation::UploadPattern | SparseOperation::BindValues => true,
            SparseOperation::Spmv(transpose) => self.transpose_modes.contains(&transpose),
            SparseOperation::PreparePreconditioner(kind)
            | SparseOperation::ApplyPreconditioner(kind) => self.preconditioners.contains(&kind),
            SparseOperation::KrylovSolve {
                method,
                preconditioner,
            } => self.krylov.contains(&SparseKrylovCapability {
                method,
                preconditioner,
            }),
        };
        operation_supported
            && self.formats.contains(&query.pattern.format)
            && self.index_widths.contains(&query.pattern.index_width)
            && self.element_types.contains(&query.element_type)
            && self.storage.contains(&query.storage)
            && self
                .max_rows
                .is_none_or(|limit| query.pattern.rows <= limit)
            && self
                .max_columns
                .is_none_or(|limit| query.pattern.columns <= limit)
            && self.max_nnz.is_none_or(|limit| query.pattern.nnz <= limit)
    }
}

fn ensure_unique<T: Eq>(values: &[T], description: &str) -> anyhow::Result<()> {
    ensure!(
        values
            .iter()
            .enumerate()
            .all(|(index, value)| !values[index + 1..].contains(value)),
        "{description} must not contain duplicates"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum SparseOperation {
    UploadPattern,
    BindValues,
    Spmv(SparseTranspose),
    PreparePreconditioner(SparsePreconditionerKind),
    ApplyPreconditioner(SparsePreconditionerKind),
    KrylovSolve {
        method: SparseKrylovMethod,
        preconditioner: Option<SparsePreconditionerKind>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseFeasibilityQuery {
    pub operation: SparseOperation,
    pub pattern: SparsePatternDescriptor,
    pub element_type: NumericElementType,
    pub storage: GpuTensorStorage,
    pub vector_count: u32,
    pub vector_length: u64,
    pub max_iterations: Option<u32>,
}

impl SparseFeasibilityQuery {
    pub fn validate(self) -> anyhow::Result<()> {
        self.pattern.validate()?;
        SparseMatrixDescriptor {
            pattern_id: 0,
            element_type: self.element_type,
            storage: self.storage,
        }
        .validate()?;
        match self.operation {
            SparseOperation::UploadPattern
            | SparseOperation::BindValues
            | SparseOperation::PreparePreconditioner(_) => {
                ensure!(
                    self.vector_count == 0 && self.vector_length == 0,
                    "sparse preparation feasibility cannot claim resident vectors"
                );
                ensure!(
                    self.max_iterations.is_none(),
                    "sparse preparation feasibility cannot claim iterations"
                );
            }
            SparseOperation::Spmv(transpose) => {
                let expected_length = match transpose {
                    SparseTranspose::None => self.pattern.columns,
                    SparseTranspose::Transpose | SparseTranspose::ConjugateTranspose => {
                        self.pattern.rows
                    }
                };
                ensure!(
                    self.vector_count > 0 && self.vector_length == expected_length,
                    "sparse SpMV feasibility vector inventory does not match the operation"
                );
                ensure!(
                    self.max_iterations.is_none(),
                    "sparse SpMV feasibility cannot claim iterations"
                );
            }
            SparseOperation::ApplyPreconditioner(_) => {
                ensure!(
                    self.pattern.rows == self.pattern.columns,
                    "sparse preconditioner feasibility requires a square pattern"
                );
                ensure!(
                    self.vector_count > 0 && self.vector_length == self.pattern.rows,
                    "sparse preconditioner feasibility vector inventory does not match the matrix"
                );
                ensure!(
                    self.max_iterations.is_none(),
                    "sparse preconditioner application cannot claim iterations"
                );
            }
            SparseOperation::KrylovSolve { .. } => {
                ensure!(
                    self.pattern.rows == self.pattern.columns,
                    "sparse Krylov feasibility requires a square pattern"
                );
                ensure!(
                    self.vector_count > 0 && self.vector_length == self.pattern.rows,
                    "sparse Krylov feasibility vector inventory does not match the matrix"
                );
                ensure!(
                    self.max_iterations.is_some_and(|count| count > 0),
                    "sparse Krylov feasibility requires a positive iteration bound"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseResourceEstimate {
    pub persistent_pattern_bytes: u64,
    pub persistent_matrix_bytes: u64,
    pub persistent_preconditioner_bytes: u64,
    pub workspace_bytes: u64,
    pub output_bytes: u64,
    pub dispatch_upper_bound: u64,
    pub host_synchronization_upper_bound: u32,
}

impl SparseResourceEstimate {
    pub fn checked_peak_bytes(self) -> Option<u64> {
        self.persistent_pattern_bytes
            .checked_add(self.persistent_matrix_bytes)?
            .checked_add(self.persistent_preconditioner_bytes)?
            .checked_add(self.workspace_bytes)?
            .checked_add(self.output_bytes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum SparseFeasibility {
    Supported { estimate: SparseResourceEstimate },
    Rejected { rejection: ProviderRejection },
}

impl SparseFeasibility {
    pub fn rejected(code: ProviderRejectionCode, detail: &'static str) -> Self {
        Self::Rejected {
            rejection: ProviderRejection {
                code,
                detail: Some(detail.to_string()),
            },
        }
    }

    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported { .. })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseCostQuery {
    pub operation: SparseFeasibilityQuery,
    pub preparation: CandidatePreparationState,
    pub required_upload_bytes: u64,
    pub required_download_bytes: u64,
    pub downstream_materialization: bool,
}

pub(crate) fn unsupported_sparse_future<T>() -> crate::AccelProviderFuture<'static, T> {
    super::super::unsupported_future("sparse operation not supported by provider")
}

pub(crate) fn unsupported_sparse_cost(_query: &SparseCostQuery) -> Option<ProviderCostEstimate> {
    None
}
