use runmat_accelerate_api::{
    provider, provider_for_device, AccelProvider, GpuTensorStorage, HostCsrPatternView,
    HostNumericDataView, HostNumericTensorView, NumericElementType, ProviderCapabilitySnapshot,
    SparseCostQuery, SparseFeasibility, SparseFeasibilityQuery, SparseIndexWidth,
    SparseKrylovMethod, SparseMatrixHandle, SparseOperation, SparsePatternDescriptor,
    SparsePatternHandle, SparsePreconditionerHandle, SparsePreconditionerKind,
    SparsePreconditionerRequest,
};
use runmat_execution::{CandidatePreparationState, Digest};

use crate::algebra::{
    validate_prepared_solver_compatibility, PreparedRealLinearSystem, RealPreconditioner,
    RealSolver,
};

use super::contract::AcceleratorAdapterError;

const KRYLOV_RESIDENT_VECTOR_COUNT: u32 = 6;

pub struct AcceleratorPreparedLinearSystem {
    pub(super) prepared: PreparedRealLinearSystem,
    pub(super) operator_identity: Digest,
    pub(super) capability: ProviderCapabilitySnapshot,
    pub(super) maximum_iterations: u32,
    resources: SparseProviderResources,
}

impl AcceleratorPreparedLinearSystem {
    pub(super) fn provider(&self) -> Result<&'static dyn AccelProvider, AcceleratorAdapterError> {
        provider_for_device(self.capability.device.device_id).ok_or_else(|| {
            AcceleratorAdapterError::execution(
                "the provider owning the prepared sparse handles is unavailable",
            )
        })
    }

    pub(super) fn matrix(&self) -> &SparseMatrixHandle {
        self.resources
            .matrix
            .as_ref()
            .expect("prepared sparse matrix resource")
    }

    pub(super) fn preconditioner(&self) -> &SparsePreconditionerHandle {
        self.resources
            .preconditioner
            .as_ref()
            .expect("prepared sparse preconditioner resource")
    }
}

struct SparseProviderResources {
    device_id: u32,
    pattern: Option<SparsePatternHandle>,
    matrix: Option<SparseMatrixHandle>,
    preconditioner: Option<SparsePreconditionerHandle>,
}

impl SparseProviderResources {
    fn new(device_id: u32) -> Self {
        Self {
            device_id,
            pattern: None,
            matrix: None,
            preconditioner: None,
        }
    }

    fn release(&mut self) {
        let Some(provider) = provider_for_device(self.device_id) else {
            return;
        };
        if let Some(preconditioner) = self.preconditioner.take() {
            let _ = provider.free_sparse_preconditioner(&preconditioner);
        }
        if let Some(matrix) = self.matrix.take() {
            let _ = provider.free_sparse_matrix(&matrix);
        }
        if let Some(pattern) = self.pattern.take() {
            let _ = provider.free_sparse_pattern(&pattern);
        }
    }
}

impl Drop for SparseProviderResources {
    fn drop(&mut self) {
        self.release();
    }
}

pub fn prepare_accelerator_linear_system(
    prepared: PreparedRealLinearSystem,
    maximum_iterations: u32,
) -> Result<AcceleratorPreparedLinearSystem, AcceleratorAdapterError> {
    validate_prepared_solver_compatibility(
        &prepared,
        RealSolver::ConjugateGradient,
        RealPreconditioner::Jacobi,
    )
    .map_err(|error| AcceleratorAdapterError::admission(error.to_string()))?;
    if prepared.dof_count() == 0 || maximum_iterations == 0 {
        return Err(AcceleratorAdapterError::admission(
            "accelerator CG requires a nonempty operator and positive iteration bound",
        ));
    }
    let provider = provider().ok_or_else(|| {
        AcceleratorAdapterError::admission("no accelerator provider is assigned to this worker")
    })?;
    let capability = provider.capability_snapshot();
    if capability.schema_version != runmat_accelerate_api::PROVIDER_CAPABILITY_SCHEMA_VERSION
        || capability.revision == 0
        || capability.device.device_id != provider.device_id()
    {
        return Err(AcceleratorAdapterError::admission(
            "assigned provider returned an invalid capability identity",
        ));
    }
    let sparse = capability.sparse.as_ref().ok_or_else(|| {
        AcceleratorAdapterError::admission("assigned provider has no native sparse capability")
    })?;
    sparse
        .validate()
        .map_err(|error| AcceleratorAdapterError::admission(error.to_string()))?;
    let matrix = prepared.matrix();
    let index_width = select_index_width(matrix, sparse)?;
    let pattern = SparsePatternDescriptor {
        format: runmat_accelerate_api::SparseFormat::Csr,
        index_width,
        rows: u64::try_from(matrix.rows()).map_err(|_| {
            AcceleratorAdapterError::admission("operator row count does not fit u64")
        })?,
        columns: u64::try_from(matrix.columns()).map_err(|_| {
            AcceleratorAdapterError::admission("operator column count does not fit u64")
        })?,
        nnz: u64::try_from(matrix.values().len()).map_err(|_| {
            AcceleratorAdapterError::admission("operator nonzero count does not fit u64")
        })?,
    };
    let query = SparseFeasibilityQuery {
        operation: SparseOperation::KrylovSolve {
            method: SparseKrylovMethod::ConjugateGradient,
            preconditioner: Some(SparsePreconditionerKind::Jacobi),
        },
        pattern,
        element_type: NumericElementType::F64,
        storage: GpuTensorStorage::Real,
        vector_count: KRYLOV_RESIDENT_VECTOR_COUNT,
        vector_length: pattern.rows,
        max_iterations: Some(maximum_iterations),
    };
    let estimate = match provider.query_sparse_feasibility(&query) {
        SparseFeasibility::Supported { estimate } => estimate,
        SparseFeasibility::Rejected { rejection } => {
            return Err(AcceleratorAdapterError::admission(format!(
                "assigned provider rejected native f64 CG: {:?} ({})",
                rejection.code,
                rejection.detail.as_deref().unwrap_or("no detail")
            )))
        }
    };
    let vector_bytes = pattern
        .rows
        .checked_mul(8)
        .ok_or_else(|| AcceleratorAdapterError::admission("sparse vector budget overflows"))?;
    let upload_bytes = estimate
        .persistent_pattern_bytes
        .checked_add(estimate.persistent_matrix_bytes)
        .and_then(|value| value.checked_add(vector_bytes.checked_mul(2)?))
        .ok_or_else(|| AcceleratorAdapterError::admission("sparse upload budget overflows"))?;
    if provider
        .estimate_sparse_cost(&SparseCostQuery {
            operation: query,
            preparation: CandidatePreparationState::Cold,
            required_upload_bytes: upload_bytes,
            required_download_bytes: vector_bytes,
            downstream_materialization: true,
        })
        .is_none()
    {
        return Err(AcceleratorAdapterError::admission(
            "assigned provider has no cost evidence for native f64 CG",
        ));
    }
    admit_resources(
        provider,
        estimate.checked_peak_bytes().ok_or_else(|| {
            AcceleratorAdapterError::admission("sparse peak resource estimate overflows")
        })?,
        estimate.workspace_bytes,
    )?;

    let operator_identity = matrix
        .values_identity()
        .map_err(|error| AcceleratorAdapterError::preparation(error.to_string()))?;
    let mut resources = SparseProviderResources::new(provider.device_id());
    resources.pattern = Some(
        provider
            .upload_sparse_pattern(HostCsrPatternView {
                rows: pattern.rows,
                columns: pattern.columns,
                row_offsets: matrix.pattern().row_offsets(),
                column_indices: matrix.pattern().column_indices(),
                index_width,
            })
            .map_err(|error| AcceleratorAdapterError::preparation(error.to_string()))?,
    );
    resources.matrix = Some(
        provider
            .bind_sparse_values(
                resources.pattern.as_ref().expect("uploaded sparse pattern"),
                &HostNumericTensorView {
                    data: HostNumericDataView::F64(matrix.values()),
                    shape: &[matrix.values().len()],
                    storage: GpuTensorStorage::Real,
                },
            )
            .map_err(|error| AcceleratorAdapterError::preparation(error.to_string()))?,
    );
    resources.preconditioner = Some(
        futures::executor::block_on(provider.prepare_sparse_preconditioner(
            &SparsePreconditionerRequest {
                matrix: resources.matrix.as_ref().expect("bound sparse matrix"),
                kind: SparsePreconditionerKind::Jacobi,
            },
        ))
        .map_err(|error| AcceleratorAdapterError::preparation(error.to_string()))?,
    );

    Ok(AcceleratorPreparedLinearSystem {
        prepared,
        operator_identity,
        capability,
        maximum_iterations,
        resources,
    })
}

fn select_index_width(
    matrix: &crate::algebra::RealSparseMatrix,
    capability: &runmat_accelerate_api::ProviderSparseCapabilities,
) -> Result<SparseIndexWidth, AcceleratorAdapterError> {
    let fits_u32 = matrix
        .pattern()
        .row_offsets()
        .iter()
        .chain(matrix.pattern().column_indices())
        .all(|value| *value <= u64::from(u32::MAX));
    if fits_u32 && capability.index_widths.contains(&SparseIndexWidth::U32) {
        return Ok(SparseIndexWidth::U32);
    }
    if capability.index_widths.contains(&SparseIndexWidth::U64) {
        return Ok(SparseIndexWidth::U64);
    }
    Err(AcceleratorAdapterError::admission(
        "canonical operator indices exceed the provider's native sparse index envelope",
    ))
}

fn admit_resources(
    provider: &dyn AccelProvider,
    required_peak_bytes: u64,
    required_scratch_bytes: u64,
) -> Result<(), AcceleratorAdapterError> {
    let resources = provider.placement_resources();
    if resources.device_id != provider.device_id() || resources.lost {
        return Err(AcceleratorAdapterError::admission(
            "assigned provider resource snapshot is stale or lost",
        ));
    }
    if resources
        .queue_limit
        .zip(resources.queue_depth)
        .is_some_and(|(limit, depth)| depth >= limit)
    {
        return Err(AcceleratorAdapterError::admission(
            "assigned provider queue has no available slot",
        ));
    }
    if resources
        .available_after_eviction_bytes()
        .is_some_and(|available| required_peak_bytes > available)
    {
        return Err(AcceleratorAdapterError::admission(
            "native sparse solve exceeds available provider memory",
        ));
    }
    if resources
        .scratch_available_bytes
        .is_some_and(|available| required_scratch_bytes > available)
    {
        return Err(AcceleratorAdapterError::admission(
            "native sparse solve exceeds available provider scratch memory",
        ));
    }
    Ok(())
}
