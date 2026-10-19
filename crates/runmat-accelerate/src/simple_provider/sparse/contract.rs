use runmat_accelerate_api::{
    ProviderCostEstimate, ProviderRejectionCode, ProviderSparseCapabilities,
    SparseCancellationCapability, SparseCostQuery, SparseFeasibility, SparseFeasibilityQuery,
    SparseFormat, SparseIndexWidth, SparseKrylovCapability, SparseKrylovMethod, SparseOperation,
    SparsePreconditionerKind, SparseResourceEstimate, SparseTranspose,
};
use runmat_execution::ExecutionCostComponents;

pub(in crate::simple_provider) fn capabilities() -> ProviderSparseCapabilities {
    ProviderSparseCapabilities {
        formats: vec![SparseFormat::Csr],
        index_widths: vec![SparseIndexWidth::U32, SparseIndexWidth::U64],
        element_types: vec![runmat_accelerate_api::NumericElementType::F64],
        storage: vec![runmat_accelerate_api::GpuTensorStorage::Real],
        transpose_modes: vec![SparseTranspose::None],
        preconditioners: vec![SparsePreconditionerKind::Jacobi],
        krylov: vec![
            SparseKrylovCapability {
                method: SparseKrylovMethod::ConjugateGradient,
                preconditioner: None,
            },
            SparseKrylovCapability {
                method: SparseKrylovMethod::ConjugateGradient,
                preconditioner: Some(SparsePreconditionerKind::Jacobi),
            },
        ],
        max_host_synchronizations_per_solve: Some(0),
        cancellation: SparseCancellationCapability::Cooperative {
            maximum_work_units_between_checks:
                super::cancellation::MAXIMUM_WORK_UNITS_BETWEEN_CHECKS,
        },
        transactional_results: true,
        max_rows: None,
        max_columns: None,
        max_nnz: None,
    }
}

pub(in crate::simple_provider) fn feasibility(query: &SparseFeasibilityQuery) -> SparseFeasibility {
    if query.validate().is_err() {
        return SparseFeasibility::rejected(
            ProviderRejectionCode::InvalidShape,
            "provider.sparse.query.invalid",
        );
    }
    if !capabilities().supports(*query) {
        return SparseFeasibility::rejected(
            ProviderRejectionCode::UnsupportedOperation,
            "provider.sparse.envelope.unsupported",
        );
    }
    estimate(query).map_or_else(
        || {
            SparseFeasibility::rejected(
                ProviderRejectionCode::ResourceLimit,
                "provider.sparse.resource.overflow",
            )
        },
        |estimate| SparseFeasibility::Supported { estimate },
    )
}

pub(in crate::simple_provider) fn cost(query: &SparseCostQuery) -> Option<ProviderCostEstimate> {
    if !feasibility(&query.operation).is_supported() {
        return None;
    }
    let work = operation_work(&query.operation)?;
    let preparation = match query.preparation {
        runmat_execution::CandidatePreparationState::Cold
        | runmat_execution::CandidatePreparationState::Preparing => work / 4,
        runmat_execution::CandidatePreparationState::Ready
        | runmat_execution::CandidatePreparationState::Warm => 0,
        // An uncertified cache phase yields no cost evidence rather than a guess.
        runmat_execution::CandidatePreparationState::Unknown => return None,
    };
    Some(ProviderCostEstimate::static_prior(
        ExecutionCostComponents {
            compile_or_prepare_ns: preparation,
            upload_ns: query.required_upload_bytes,
            execution_ns: work,
            synchronization_ns: 0,
            download_ns: query.required_download_bytes,
            downstream_ns: u64::from(query.downstream_materialization),
            ..ExecutionCostComponents::default()
        },
        estimate(&query.operation)?.workspace_bytes,
    ))
}

fn estimate(query: &SparseFeasibilityQuery) -> Option<SparseResourceEstimate> {
    let pattern_bytes = query.pattern.checked_storage_bytes()?;
    let value_bytes = query
        .pattern
        .nnz
        .checked_mul(u64::try_from(query.element_type.element_size()).ok()?)?;
    let vector_bytes = query
        .vector_length
        .checked_mul(u64::from(query.vector_count))?
        .checked_mul(u64::try_from(query.element_type.element_size()).ok()?)?;
    let dimension_bytes = query
        .pattern
        .rows
        .checked_mul(u64::try_from(query.element_type.element_size()).ok()?)?;
    let preconditioner_bytes = match query.operation {
        SparseOperation::PreparePreconditioner(_)
        | SparseOperation::ApplyPreconditioner(_)
        | SparseOperation::KrylovSolve {
            preconditioner: Some(_),
            ..
        } => dimension_bytes,
        _ => 0,
    };
    let output_bytes = match query.operation {
        SparseOperation::Spmv(_)
        | SparseOperation::ApplyPreconditioner(_)
        | SparseOperation::KrylovSolve { .. } => dimension_bytes,
        SparseOperation::UploadPattern
        | SparseOperation::BindValues
        | SparseOperation::PreparePreconditioner(_) => 0,
    };
    Some(SparseResourceEstimate {
        persistent_pattern_bytes: pattern_bytes,
        persistent_matrix_bytes: value_bytes,
        persistent_preconditioner_bytes: preconditioner_bytes,
        workspace_bytes: vector_bytes,
        output_bytes,
        dispatch_upper_bound: 0,
        host_synchronization_upper_bound: 0,
    })
}

fn operation_work(query: &SparseFeasibilityQuery) -> Option<u64> {
    let iterations = u64::from(query.max_iterations.unwrap_or(1));
    match query.operation {
        SparseOperation::UploadPattern => query.pattern.rows.checked_add(query.pattern.nnz),
        SparseOperation::BindValues => Some(query.pattern.nnz),
        SparseOperation::Spmv(_) => query.pattern.nnz.checked_mul(2),
        SparseOperation::PreparePreconditioner(_) => Some(query.pattern.nnz),
        SparseOperation::ApplyPreconditioner(_) => query.pattern.rows.checked_mul(2),
        SparseOperation::KrylovSolve { .. } => query
            .pattern
            .nnz
            .checked_mul(2)?
            .checked_add(query.pattern.rows.checked_mul(8)?)?
            .checked_mul(iterations),
    }
}
