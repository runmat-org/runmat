use runmat_accelerate_api::{
    GpuTensorStorage, NumericElementType, ProviderCostEstimate, ProviderRejectionCode,
    ProviderSparseCapabilities, SparseCancellationCapability, SparseCostQuery, SparseFeasibility,
    SparseFeasibilityQuery, SparseFormat, SparseIndexWidth, SparseOperation,
    SparseResourceEstimate, SparseTranspose,
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
        krylov_methods: Vec::new(),
        resident_krylov: false,
        max_host_synchronizations_per_solve: None,
        cancellation: SparseCancellationCapability::BeforeDispatch,
        transactional_results: true,
        max_rows: Some(
            index_limit
                .saturating_sub(1)
                .min(dispatch_row_limit)
                .min(u64::from(u32::MAX)),
        ),
        max_columns: Some(u64::from(u32::MAX)),
        max_nnz: Some(index_limit.min(scalar_limit).min(u64::from(u32::MAX))),
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
    Some(ProviderCostEstimate::static_prior(
        ExecutionCostComponents {
            compile_or_prepare_ns: match query.preparation {
                runmat_execution::CandidatePreparationState::Ready
                | runmat_execution::CandidatePreparationState::Warm => 0,
                runmat_execution::CandidatePreparationState::Cold
                | runmat_execution::CandidatePreparationState::Preparing => nnz_work / 4,
            },
            upload_ns: query.required_upload_bytes,
            allocation_ns: query.operation.pattern.rows,
            queue_ns: 1,
            execution_ns: nnz_work,
            synchronization_ns: 0,
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
        SparseOperation::UploadPattern
        | SparseOperation::BindValues
        | SparseOperation::PreparePreconditioner(_) => 0,
        _ => return None,
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
        workspace_bytes: 0,
        output_bytes,
        dispatch_upper_bound: u64::from(matches!(
            query.operation,
            SparseOperation::Spmv(_)
                | SparseOperation::PreparePreconditioner(_)
                | SparseOperation::ApplyPreconditioner(_)
        )),
        host_synchronization_upper_bound: u32::from(matches!(
            query.operation,
            SparseOperation::PreparePreconditioner(_)
        )),
    })
}
