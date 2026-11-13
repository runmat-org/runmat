use runmat_meshing_core::SolverMeshArtifact;
use runmat_simulation_usage::OperationCount;

use crate::{excitation::RealizedExcitation, topology::AnalysisProjection};

use super::StructuralLoadPlanError;

pub(crate) enum MixedLoadCapacityError<A> {
    Admission(A),
    Projection(crate::topology::AnalysisProjectionError),
    Plan(StructuralLoadPlanError),
}

pub(crate) enum MixedLoadPhase {
    Projection,
    BodyLoadQuadrature,
}

pub(super) async fn derive_solid_projection<O, F, A>(
    mesh: &SolverMeshArtifact,
    awaited_elements: bool,
    before: &mut O,
) -> Result<AnalysisProjection, MixedLoadCapacityError<A>>
where
    O: FnMut(MixedLoadPhase, OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
{
    if awaited_elements {
        let work = AnalysisProjection::derivation_work_upper_bound(mesh)
            .map_err(MixedLoadCapacityError::Projection)?;
        if work > OperationCount::ZERO {
            before(MixedLoadPhase::Projection, work)
                .await
                .map_err(MixedLoadCapacityError::Admission)?;
        }
    }
    AnalysisProjection::derive(mesh).map_err(MixedLoadCapacityError::Projection)
}

pub(super) async fn admit_body_quadrature<O, F, A>(
    projection: &AnalysisProjection,
    load: &RealizedExcitation,
    awaited_elements: bool,
    before: &mut O,
) -> Result<(), MixedLoadCapacityError<A>>
where
    O: FnMut(MixedLoadPhase, OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
{
    if awaited_elements {
        let work = crate::assembly::solver_solid_boundary::body_load_quadrature_work_upper_bound(
            projection, load,
        )
        .map_err(|error| {
            MixedLoadCapacityError::Plan(StructuralLoadPlanError::Accounting(error))
        })?;
        if work > OperationCount::ZERO {
            before(MixedLoadPhase::BodyLoadQuadrature, work)
                .await
                .map_err(MixedLoadCapacityError::Admission)?;
        }
    }
    Ok(())
}

pub(super) fn solid_load_error(
    load: &RealizedExcitation,
    error: crate::assembly::SolverMeshRegionMappingError,
) -> StructuralLoadPlanError {
    match error {
        crate::assembly::SolverMeshRegionMappingError::Projection(
            crate::topology::AnalysisProjectionError::Accounting(cause),
        ) => StructuralLoadPlanError::Accounting(cause),
        crate::assembly::SolverMeshRegionMappingError::Projection(
            crate::topology::AnalysisProjectionError::Cancelled,
        ) => StructuralLoadPlanError::Cancelled,
        other => StructuralLoadPlanError::UnresolvedTarget {
            excitation_id: load.excitation_id.0.clone(),
            reason: other.to_string(),
        },
    }
}
