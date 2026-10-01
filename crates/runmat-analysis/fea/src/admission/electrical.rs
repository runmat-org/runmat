use runmat_analysis_core::{
    AnalysisProcedure, BoundaryTransferImplementation, ConstitutiveImplementation,
    CouplingImplementation, ExactFormulationComponent, ExactFormulationRealization,
    FormulationRequirement, PhysicsFamily, RecoveryImplementation, SolutionSpace,
    SourceImplementation, SpatialOperatorImplementation, StaticElectricalEquation,
    TemporalImplementation, TopologyImplementation,
};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

use super::{admit_formulation, AnalysisRunIdentity, FormulationAdmissionRequest};
use crate::contracts::FeaRunError;
use crate::physics::electrical::CanonicalElectricalSource;

/// Admit canonical scalar potential FEM, with equation and source kind chosen explicitly.
/// This is experimental identity admission, not model validation or production qualification.
pub fn admit_canonical_electrical_static(
    equation: StaticElectricalEquation,
    source: CanonicalElectricalSource,
    mesh: &SolverMeshArtifact,
    requirement: FormulationRequirement,
) -> Result<AnalysisRunIdentity, FeaRunError> {
    admit_formulation(FormulationAdmissionRequest {
        family: PhysicsFamily::Electromagnetic,
        procedure: AnalysisProcedure::Static,
        solution_space: SolutionSpace::Direct,
        requirement,
        realization: realization(equation, mesh.resolved_request.element_order, source),
    })
    .map(|receipt| receipt.into_identity())
    .map_err(Into::into)
}

pub(super) fn registered_realizations() -> Vec<ExactFormulationRealization> {
    [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ]
    .into_iter()
    .flat_map(|equation| {
        [ElementOrder::Tet4, ElementOrder::Tet10]
            .into_iter()
            .flat_map(move |order| {
                [
                    CanonicalElectricalSource::SourceFree,
                    CanonicalElectricalSource::VolumeCharge,
                ]
                .into_iter()
                .filter(move |source| {
                    equation == StaticElectricalEquation::Electrostatics
                        || *source == CanonicalElectricalSource::SourceFree
                })
                .map(move |source| realization(equation, order, source))
            })
    })
    .collect()
}

pub(crate) fn realization(
    equation: StaticElectricalEquation,
    order: ElementOrder,
    source: CanonicalElectricalSource,
) -> ExactFormulationRealization {
    let (constitutive, recovery, temporal) = match equation {
        StaticElectricalEquation::Electrostatics => (
            ConstitutiveImplementation::ElementOwnedIsotropicPermittivity,
            RecoveryImplementation::ElectricalSolverMeshFieldDisplacementEnergy,
            TemporalImplementation::ElectrostaticPotentialLinearSolve,
        ),
        StaticElectricalEquation::SteadyConduction => (
            ConstitutiveImplementation::ElementOwnedIsotropicElectricalConductivity,
            RecoveryImplementation::ElectricalSolverMeshFieldCurrentJoulePower,
            TemporalImplementation::SteadyConductionPotentialLinearSolve,
        ),
    };
    let topology = match order {
        ElementOrder::Tet4 => TopologyImplementation::CanonicalSolverMeshTet4,
        ElementOrder::Tet10 => TopologyImplementation::CanonicalSolverMeshTet10,
    };
    ExactFormulationRealization {
        components: vec![ExactFormulationComponent {
            spatial_operator: SpatialOperatorImplementation::ElectricalGalerkinPotentialGradient,
            constitutive,
            recovery,
            topologies: vec![topology],
            sources: vec![match source {
                CanonicalElectricalSource::SourceFree => SourceImplementation::NotApplicable,
                CanonicalElectricalSource::VolumeCharge => {
                    SourceImplementation::CanonicalVolumeProjection
                }
            }],
        }],
        temporals: vec![temporal],
        coupling: CouplingImplementation::Uncoupled,
        boundary_transfers: vec![BoundaryTransferImplementation::CanonicalBoundaryProjection],
    }
}

#[cfg(test)]
mod tests;
