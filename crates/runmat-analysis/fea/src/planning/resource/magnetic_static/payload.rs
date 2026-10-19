//! Borrowed wire-shape projection. Repeated slots serialize without allocating
//! field arrays; codec-wide numeric bounds cover every retained SI value.
use super::*;
use crate::physics::magnetic::MagneticFieldSample;
use crate::results::field::VolumeFieldSamplingRule;
use runmat_meshing_core::StableDigest;
use serde::{Serialize, Serializer};

struct Repeated<'a, T> {
    value: &'a T,
    count: usize,
}
impl<T: Serialize> Serialize for Repeated<'_, T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(std::iter::repeat_n(self.value, self.count))
    }
}
#[derive(Serialize)]
struct SamplingData<'a> {
    schema_version: u16,
    mesh_digest: StableDigest,
    projection_digest: Digest,
    rule: VolumeFieldSamplingRule,
    node_identities: Repeated<'a, StableDigest>,
    element_identities: Repeated<'a, StableDigest>,
    element_corner_identities: Repeated<'a, [StableDigest; 4]>,
    oriented_edges: Repeated<'a, crate::results::field::SampledOrientedEdge>,
    integration_samples: Repeated<'a, crate::results::field::VolumeFieldIntegrationSample>,
}
#[derive(Serialize)]
struct Sampling<'a> {
    data: SamplingData<'a>,
    sampling_digest: Digest,
}
#[derive(Serialize)]
struct Response<'a> {
    edge_node_identities: Repeated<'a, [StableDigest; 2]>,
    edge_circulations_wb: Repeated<'a, f64>,
    samples: Repeated<'a, MagneticFieldSample>,
    magnetic_energy_j: f64,
    source_work_j: f64,
    relative_physical_residual: f64,
    relative_gauge_residual: f64,
    relative_energy_work_error: f64,
    iterations: u32,
}
#[derive(Serialize)]
struct ResultProjection<'a> {
    source_workload: crate::execution::AnalysisWorkloadIdentity,
    producer: crate::execution::StepIdentity,
    formulation: crate::admission::AnalysisRunIdentity,
    mesh_digest: Digest,
    projection_digest: Digest,
    sampling_topology: Sampling<'a>,
    publication_declarations: Vec<&'a runmat_analysis_core::StateOutput>,
    element_order: runmat_meshing_core::ElementOrder,
    relative_residual_target: f64,
    response: Response<'a>,
}
pub(super) fn profile(
    workload: &MagnetostaticWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<runmat_canonical_codec::SerializedCanonicalProfile, ResourcePlanningError> {
    let elements = mesh.topology.volume_elements.len();
    let edges = elements
        .checked_mul(6)
        .ok_or(ResourcePlanningError::Overflow("magnetic edge slots"))?;
    let samples = elements
        .checked_mul(15)
        .ok_or(ResourcePlanningError::Overflow("magnetic sample slots"))?;
    let stable = StableDigest::from_bytes([0; 32]);
    let digest = Digest::from_bytes([0; 32]);
    let pair = [stable; 2];
    let corners = [stable; 4];
    let scalar = 0.0;
    let edge = crate::results::field::SampledOrientedEdge {
        identity: stable,
        node_identities: pair,
    };
    let sample = crate::results::field::VolumeFieldIntegrationSample {
        identity: stable,
        element_identity: stable,
        ordinal: 0,
        position_m: [0.0; 3],
        volume_weight_m3: 0.0,
    };
    let physical = MagneticFieldSample {
        element_identity: stable,
        position_m: [0.0; 3],
        volume_weight_m3: 0.0,
        vector_potential_wb_per_m: [0.0; 3],
        flux_density_t: [0.0; 3],
        magnetic_field_a_per_m: [0.0; 3],
    };
    let value = ResultProjection {
        source_workload: workload.identity().map_err(invalid)?,
        producer: workload.producer_identity().map_err(invalid)?,
        formulation: workload.formulation(mesh).map_err(invalid)?,
        mesh_digest: digest,
        projection_digest: digest,
        sampling_topology: Sampling {
            sampling_digest: digest,
            data: SamplingData {
                schema_version: 1,
                mesh_digest: stable,
                projection_digest: digest,
                rule: VolumeFieldSamplingRule::TetrahedronKeastFifteen,
                node_identities: Repeated {
                    value: &stable,
                    count: mesh.topology.nodes.len(),
                },
                element_identities: Repeated {
                    value: &stable,
                    count: elements,
                },
                element_corner_identities: Repeated {
                    value: &corners,
                    count: elements,
                },
                oriented_edges: Repeated {
                    value: &edge,
                    count: edges,
                },
                integration_samples: Repeated {
                    value: &sample,
                    count: samples,
                },
            },
        },
        publication_declarations: workload.model.sequence.steps[0]
            .state_outputs
            .iter()
            .filter(|d| d.persistence != runmat_analysis_core::StatePersistence::CheckpointOnly)
            .collect(),
        element_order: mesh.resolved_request.element_order,
        relative_residual_target: workload.relative_residual_target,
        response: Response {
            edge_node_identities: Repeated {
                value: &pair,
                count: edges,
            },
            edge_circulations_wb: Repeated {
                value: &scalar,
                count: edges,
            },
            samples: Repeated {
                value: &physical,
                count: samples,
            },
            magnetic_energy_j: 0.0,
            source_work_j: 0.0,
            relative_physical_residual: 0.0,
            relative_gauge_residual: 0.0,
            relative_energy_work_error: 0.0,
            iterations: 0,
        },
    };
    runmat_canonical_codec::profile_serialized_shape(&value, LIMITS).map_err(invalid)
}
