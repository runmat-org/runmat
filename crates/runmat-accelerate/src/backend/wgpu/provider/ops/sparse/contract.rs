use runmat_accelerate_api::{
    GpuTensorStorage, NumericElementType, ProviderCostEstimate, ProviderRejectionCode,
    ProviderSparseCapabilities, SparseCancellationCapability, SparseCostQuery, SparseFeasibility,
    SparseFeasibilityQuery, SparseFormat, SparseIndexWidth, SparseKrylovCapability,
    SparseOperation, SparseResourceEstimate, SparseTranspose,
};
use runmat_execution::ExecutionCostComponents;

use super::super::WgpuProvider;
use super::storage_binding_limit;

pub(crate) fn capabilities(provider: &WgpuProvider) -> ProviderSparseCapabilities {
    let binding_limit = storage_binding_limit(provider);
    let scalar_limit = binding_limit / provider.element_size as u64;
    let index_limit = binding_limit / 4;
    let dispatch_row_limit =
        u64::from(provider.adapter_limits.max_compute_workgroups_per_dimension)
            .saturating_mul(u64::from(provider.workgroup_config.scalar.max(1)));
    let max_rows = index_limit
        .saturating_sub(1)
        .min(dispatch_row_limit)
        .min(u64::from(u32::MAX));
    let max_nnz = index_limit.min(scalar_limit).min(u64::from(u32::MAX));
    ProviderSparseCapabilities {
        formats: vec![SparseFormat::Csr],
        index_widths: vec![SparseIndexWidth::U32],
        element_types: vec![match provider.precision {
            super::super::NumericPrecision::F32 => NumericElementType::F32,
            super::super::NumericPrecision::F64 => NumericElementType::F64,
        }],
        storage: vec![GpuTensorStorage::Real],
        transpose_modes: vec![SparseTranspose::None],
        preconditioners: vec![runmat_accelerate_api::SparsePreconditionerKind::Jacobi],
        krylov: vec![SparseKrylovCapability {
            method: runmat_accelerate_api::SparseKrylovMethod::ConjugateGradient,
            preconditioner: Some(runmat_accelerate_api::SparsePreconditionerKind::Jacobi),
        }],
        max_host_synchronizations_per_solve: Some(
            1 + super::cg::MAXIMUM_ITERATIONS.div_ceil(super::cg::ITERATIONS_PER_CHECKPOINT),
        ),
        cancellation: SparseCancellationCapability::Cooperative {
            maximum_work_units_between_checks: u64::from(super::cg::ITERATIONS_PER_CHECKPOINT)
                .saturating_mul(
                    max_nnz
                        .saturating_mul(2)
                        .saturating_add(max_rows.saturating_mul(8)),
                ),
        },
        transactional_results: true,
        max_rows: Some(max_rows),
        max_columns: Some(u64::from(u32::MAX)),
        max_nnz: Some(max_nnz),
    }
}

pub(crate) fn feasibility(
    provider: &WgpuProvider,
    query: &SparseFeasibilityQuery,
) -> SparseFeasibility {
    if query.validate().is_err() {
        return SparseFeasibility::rejected(
            ProviderRejectionCode::InvalidShape,
            "provider.sparse.query.invalid",
        );
    }
    let capabilities = capabilities(provider);
    if !capabilities.element_types.contains(&query.element_type) {
        return SparseFeasibility::rejected(
            ProviderRejectionCode::UnsupportedElementType,
            "provider.sparse.element.unsupported",
        );
    }
    if !capabilities.storage.contains(&query.storage) {
        return SparseFeasibility::rejected(
            ProviderRejectionCode::UnsupportedStorage,
            "provider.sparse.storage.unsupported",
        );
    }
    if !capabilities.formats.contains(&query.pattern.format)
        || !capabilities
            .index_widths
            .contains(&query.pattern.index_width)
    {
        return SparseFeasibility::rejected(
            ProviderRejectionCode::UnsupportedLayout,
            "provider.sparse.layout.unsupported",
        );
    }
    if capabilities
        .max_rows
        .is_some_and(|limit| query.pattern.rows > limit)
        || capabilities
            .max_columns
            .is_some_and(|limit| query.pattern.columns > limit)
        || capabilities
            .max_nnz
            .is_some_and(|limit| query.pattern.nnz > limit)
    {
        return SparseFeasibility::rejected(
            ProviderRejectionCode::ResourceLimit,
            "provider.sparse.resource.limit",
        );
    }
    if query
        .max_iterations
        .is_some_and(|count| count > super::cg::MAXIMUM_ITERATIONS)
    {
        return SparseFeasibility::rejected(
            ProviderRejectionCode::UnsupportedOperation,
            "provider.sparse.krylov.envelope.unsupported",
        );
    }
    if !capabilities.supports(*query) {
        return SparseFeasibility::rejected(
            ProviderRejectionCode::UnsupportedOperation,
            "provider.sparse.operation.unsupported",
        );
    }
    estimate(provider, query).map_or_else(
        || {
            SparseFeasibility::rejected(
                ProviderRejectionCode::ResourceLimit,
                "provider.sparse.resource.overflow",
            )
        },
        |estimate| SparseFeasibility::Supported { estimate },
    )
}

pub(crate) fn cost(
    provider: &WgpuProvider,
    query: &SparseCostQuery,
) -> Option<ProviderCostEstimate> {
    if !feasibility(provider, &query.operation).is_supported() {
        return None;
    }
    let nnz_work = query.operation.pattern.nnz.checked_mul(2)?;
    let compile_or_prepare_ns = match query.preparation {
        runmat_execution::CandidatePreparationState::Ready
        | runmat_execution::CandidatePreparationState::Warm => 0,
        runmat_execution::CandidatePreparationState::Cold
        | runmat_execution::CandidatePreparationState::Preparing => nnz_work / 4,
        // Like dense WGPU costing, an uncertified cache phase yields no evidence.
        runmat_execution::CandidatePreparationState::Unknown => return None,
    };
    Some(ProviderCostEstimate::static_prior(
        ExecutionCostComponents {
            compile_or_prepare_ns,
            upload_ns: query.required_upload_bytes,
            allocation_ns: query.operation.pattern.rows,
            queue_ns: 1,
            execution_ns: nnz_work,
            synchronization_ns: u64::from(
                estimate(provider, &query.operation)?.host_synchronization_upper_bound,
            ),
            download_ns: query.required_download_bytes,
            downstream_ns: u64::from(query.downstream_materialization),
        },
        estimate(provider, &query.operation)?.workspace_bytes,
    ))
}

fn estimate(
    provider: &WgpuProvider,
    query: &SparseFeasibilityQuery,
) -> Option<SparseResourceEstimate> {
    let row_offset_bytes = query.pattern.rows.checked_add(1)?.checked_mul(4)?;
    let column_index_bytes = query.pattern.nnz.checked_mul(4)?.max(4);
    let pattern_bytes = row_offset_bytes.checked_add(column_index_bytes)?;
    let element_bytes = query.element_type.element_size() as u64;
    let value_bytes = query
        .pattern
        .nnz
        .checked_mul(element_bytes)?
        .max(element_bytes);
    let preconditioner_bytes = match query.operation {
        SparseOperation::PreparePreconditioner(_)
        | SparseOperation::ApplyPreconditioner(_)
        | SparseOperation::KrylovSolve {
            preconditioner: Some(_),
            ..
        } => query
            .pattern
            .rows
            .checked_mul(element_bytes)?
            .max(element_bytes),
        _ => 0,
    };
    let output_bytes = match query.operation {
        SparseOperation::Spmv(_) => query
            .pattern
            .rows
            .checked_mul(element_bytes)?
            .max(element_bytes),
        SparseOperation::ApplyPreconditioner(_) => query
            .pattern
            .rows
            .checked_mul(element_bytes)?
            .max(element_bytes),
        SparseOperation::KrylovSolve { .. } => query
            .pattern
            .rows
            .checked_mul(element_bytes)?
            .max(element_bytes),
        SparseOperation::UploadPattern
        | SparseOperation::BindValues
        | SparseOperation::PreparePreconditioner(_) => 0,
    };
    let workspace_bytes = match query.operation {
        SparseOperation::KrylovSolve { .. } => {
            let vector_bytes = query.vector_length.checked_mul(element_bytes)?;
            let reduction_groups = query
                .vector_length
                .div_ceil(u64::from(provider.workgroup_config.scalar.max(1)));
            let state_bytes: u64 = match query.element_type {
                NumericElementType::F32 => 44,
                NumericElementType::F64 => 80,
                _ => return None,
            };
            vector_bytes
                .checked_mul(4)?
                .checked_add(reduction_groups.checked_mul(element_bytes)?)?
                .checked_add(element_bytes.checked_mul(3)?)?
                .checked_add(state_bytes.checked_mul(2)?)?
        }
        _ => 0,
    };
    let binding_limit = storage_binding_limit(provider);
    if [
        row_offset_bytes,
        column_index_bytes,
        value_bytes,
        output_bytes,
    ]
    .into_iter()
    .any(|bytes| bytes > binding_limit)
    {
        return None;
    }
    Some(SparseResourceEstimate {
        persistent_pattern_bytes: pattern_bytes,
        persistent_matrix_bytes: value_bytes,
        persistent_preconditioner_bytes: preconditioner_bytes,
        workspace_bytes,
        output_bytes,
        dispatch_upper_bound: match query.operation {
            SparseOperation::KrylovSolve { .. } => {
                7_u64.checked_add(u64::from(query.max_iterations?).checked_mul(12)?)?
            }
            SparseOperation::Spmv(_)
            | SparseOperation::PreparePreconditioner(_)
            | SparseOperation::ApplyPreconditioner(_) => 1,
            SparseOperation::UploadPattern | SparseOperation::BindValues => 0,
        },
        host_synchronization_upper_bound: match query.operation {
            SparseOperation::PreparePreconditioner(_) => 1,
            SparseOperation::KrylovSolve { .. } => {
                1 + query
                    .max_iterations?
                    .div_ceil(super::cg::ITERATIONS_PER_CHECKPOINT)
            }
            _ => 0,
        },
    })
}
