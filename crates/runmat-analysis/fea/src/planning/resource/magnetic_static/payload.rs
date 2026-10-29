//! Borrowed wire-shape projection. Repeated slots serialize without allocating
//! field arrays; codec-wide numeric bounds cover every retained SI value.
use super::*;
use crate::results::field::{RepeatedShape, SampledTopologyShape, VolumeFieldSamplingRule};
use runmat_types::physical::{kind, Quantity};
use serde::Serialize;

/// Retained response: per-sample A and B (three each), per-element mean B and permeability,
/// and one circulation per planned edge slot. Mirrors `MagnetostaticResponse` field order.
#[derive(Serialize)]
struct Response {
    #[serde(with = "runmat_types::physical::canonical_storage")]
    edge_circulations_wb: Quantity<kind::MagneticFlux, RepeatedShape<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    element_flux_density_t: Quantity<kind::MagneticFluxDensity, RepeatedShape<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    element_permeability_h_per_m: Quantity<kind::MagneticPermeability, RepeatedShape<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    flux_density_t: Quantity<kind::MagneticFluxDensity, RepeatedShape<f64>>,
    iterations: u32,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    magnetic_energy_j: Quantity<kind::Energy, f64>,
    relative_energy_work_error: f64,
    relative_gauge_residual: f64,
    relative_physical_residual: f64,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    source_work_j: Quantity<kind::Energy, f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    vector_potential_wb_per_m: Quantity<kind::MagneticVectorPotential, RepeatedShape<f64>>,
}
#[derive(Serialize)]
struct ResultProjection<'a> {
    element_order: runmat_meshing_core::ElementOrder,
    formulation: crate::admission::AnalysisRunIdentity,
    mesh_digest: Digest,
    producer: crate::execution::StepIdentity,
    projection_digest: Digest,
    publication_declarations: Vec<&'a runmat_analysis_core::StateOutput>,
    relative_residual_target: f64,
    response: Response,
    sampling_topology: SampledTopologyShape,
    source_workload: crate::execution::AnalysisWorkloadIdentity,
}
pub(super) fn profile(
    workload: &MagnetostaticWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<runmat_canonical_codec::SerializedCanonicalProfile, ResourcePlanningError> {
    let elements = mesh.topology.volume_elements.len();
    let edges = elements
        .checked_mul(6)
        .ok_or(ResourcePlanningError::Overflow("magnetic edge slots"))?;
    let components = elements
        .checked_mul(crate::physics::magnetic::MAGNETIC_SAMPLES_PER_ELEMENT * 3)
        .ok_or(ResourcePlanningError::Overflow("magnetic sample slots"))?;
    let element_components = elements
        .checked_mul(3)
        .ok_or(ResourcePlanningError::Overflow("magnetic element slots"))?;
    let digest = Digest::from_bytes([0xff; 32]);
    let value = ResultProjection {
        source_workload: workload.identity().map_err(invalid)?,
        producer: workload.producer_identity().map_err(invalid)?,
        formulation: workload.formulation(mesh).map_err(invalid)?,
        mesh_digest: digest,
        projection_digest: digest,
        sampling_topology: SampledTopologyShape::new(
            mesh.topology.nodes.len(),
            elements,
            VolumeFieldSamplingRule::TetrahedronKeastFifteen,
        ),
        publication_declarations: workload.model.sequence.steps[0]
            .state_outputs
            .iter()
            .filter(|d| d.persistence != runmat_analysis_core::StatePersistence::CheckpointOnly)
            .collect(),
        element_order: mesh.resolved_request.element_order,
        relative_residual_target: workload.relative_residual_target,
        response: Response {
            edge_circulations_wb: Quantity::new(RepeatedShape::new(0.0, edges)),
            element_flux_density_t: Quantity::new(RepeatedShape::new(0.0, element_components)),
            element_permeability_h_per_m: Quantity::new(RepeatedShape::new(0.0, elements)),
            flux_density_t: Quantity::new(RepeatedShape::new(0.0, components)),
            iterations: 0,
            magnetic_energy_j: Quantity::new(0.0),
            relative_energy_work_error: 0.0,
            relative_gauge_residual: 0.0,
            relative_physical_residual: 0.0,
            source_work_j: Quantity::new(0.0),
            vector_potential_wb_per_m: Quantity::new(RepeatedShape::new(0.0, components)),
        },
    };
    runmat_canonical_codec::profile_serialized_shape(&value, LIMITS).map_err(invalid)
}
