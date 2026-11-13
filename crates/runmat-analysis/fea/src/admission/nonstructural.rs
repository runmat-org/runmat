use runmat_analysis_core::{
    AnalysisProcedure, BoundaryTransferImplementation, PhysicsFamily, SolutionSpace,
    SourceImplementation, TemporalImplementation, TopologyImplementation,
};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

use super::{
    admit_formulation, registered_formulations, AnalysisRunIdentity, FormulationAdmissionRequest,
};
use crate::contracts::FeaRunError;

/// Admit the exact nonstructural realization selected by mesh topology and transfer implementations.
pub fn admit_nonstructural(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solver_mesh: Option<&SolverMeshArtifact>,
    source: SourceImplementation,
    boundary_transfer: BoundaryTransferImplementation,
) -> Result<AnalysisRunIdentity, FeaRunError> {
    admit_nonstructural_with_temporal(
        family,
        procedure,
        solver_mesh,
        source,
        boundary_transfer,
        None,
    )
}

pub(crate) fn admit_nonstructural_with_temporal(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    solver_mesh: Option<&SolverMeshArtifact>,
    source: SourceImplementation,
    boundary_transfer: BoundaryTransferImplementation,
    temporal: Option<TemporalImplementation>,
) -> Result<AnalysisRunIdentity, FeaRunError> {
    if family == PhysicsFamily::Fluid {
        return Err(FeaRunError::InvalidModel(
            "canonical fluid admission requires its explicit steady or transient operator entry"
                .into(),
        ));
    }
    if family == PhysicsFamily::Electromagnetic && procedure == AnalysisProcedure::Static {
        return Err(FeaRunError::InvalidModel(
            "canonical static electrical admission requires an explicit equation; use admit_canonical_electrical_static".into(),
        ));
    }
    let topology = if let Some(mesh) = solver_mesh {
        match mesh.resolved_request.element_order {
            ElementOrder::Tet4 => TopologyImplementation::CanonicalSolverMeshTet4,
            ElementOrder::Tet10 => TopologyImplementation::CanonicalSolverMeshTet10,
        }
    } else {
        match family {
            PhysicsFamily::Electromagnetic => TopologyImplementation::DofAdjacency,
            _ => {
                return Err(FeaRunError::InvalidModel(format!(
                    "no meshless FEA pipeline admission branch is defined for {family:?}"
                )))
            }
        }
    };
    let realization = registered_formulations(family, procedure)
        .into_iter()
        .find(|entry| {
            let realization = entry.realization();
            let has_topology = realization
                .components
                .iter()
                .any(|component| component.topologies.contains(&topology));
            let has_source = realization
                .components
                .iter()
                .any(|component| component.sources.contains(&source));
            has_topology
                && has_source
                && realization.boundary_transfers.contains(&boundary_transfer)
                && temporal.is_none_or(|selected| realization.temporals == vec![selected])
        })
        .map(|entry| entry.realization().clone())
        .ok_or_else(|| {
            FeaRunError::InvalidModel(format!(
                "no exact formulation is registered for {family:?} {procedure:?} topology {topology:?}, source {source:?}, and boundary transfer {boundary_transfer:?}"
            ))
        })?;
    admit_formulation(FormulationAdmissionRequest {
        family,
        procedure,
        solution_space: SolutionSpace::Direct,
        realization,
    })
    .map(|receipt| receipt.into_identity())
    .map_err(Into::into)
}

#[cfg(test)]
mod tests;
