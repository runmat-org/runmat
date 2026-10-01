//! Source-bound retained native coefficients and SI samples. Cold validation does
//! not assemble an operator, reconstruct a missing field or rerun the solve.
use super::*;
use crate::{
    execution::SolverResultIdentity, physics::magnetic::MagnetostaticResponse,
    topology::AnalysisProjection,
};
use std::collections::BTreeSet;

const PREFIX: &[u8] = b"runmat-magnetostatic-result/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64);

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct MagnetostaticSolverResult {
    data: ResultData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultData {
    source_workload: AnalysisWorkloadIdentity,
    producer: StepIdentity,
    formulation: AnalysisRunIdentity,
    mesh_digest: Digest,
    projection_digest: Digest,
    sampling_topology: crate::results::field::SampledVolumeFieldTopology,
    publication_declarations: Vec<runmat_analysis_core::StateOutput>,
    element_order: runmat_meshing_core::ElementOrder,
    relative_residual_target: f64,
    response: MagnetostaticResponse,
}

impl<'de> Deserialize<'de> for MagnetostaticSolverResult {
    fn deserialize<D: serde::Deserializer<'de>>(decoder: D) -> Result<Self, D::Error> {
        let value = Self {
            data: ResultData::deserialize(decoder)?,
        };
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

impl MagnetostaticSolverResult {
    pub(super) fn new(
        workload: &MagnetostaticWorkload,
        mesh: &SolverMeshArtifact,
        formulation: AnalysisRunIdentity,
        response: MagnetostaticResponse,
    ) -> AnalysisExecutionResult<Self> {
        let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
        let result = Self {
            data: ResultData {
                source_workload: workload.identity()?,
                producer: workload.producer_identity()?,
                formulation,
                mesh_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
                projection_digest: projection.projection_digest,
                sampling_topology: crate::results::field::SampledVolumeFieldTopology::derive(
                    mesh,
                    crate::results::field::VolumeFieldSamplingRule::TetrahedronKeastFifteen,
                )
                .map_err(invalid)?,
                publication_declarations: workload.model.sequence.steps[0]
                    .state_outputs
                    .iter()
                    .filter(|d| {
                        d.persistence != runmat_analysis_core::StatePersistence::CheckpointOnly
                    })
                    .cloned()
                    .collect(),
                element_order: mesh.resolved_request.element_order,
                relative_residual_target: workload.relative_residual_target,
                response,
            },
        };
        result.validate_for(workload, mesh)?;
        Ok(result)
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        use runmat_analysis_core::{AnalysisProcedure, PhysicsFamily, SolutionSpace};
        let d = &self.data;
        let r = &d.response;
        let t = d.relative_residual_target;
        d.sampling_topology.validate().map_err(invalid)?;
        if d.sampling_topology.mesh_digest() != d.mesh_digest
            || d.sampling_topology.projection_digest() != d.projection_digest
            || d.sampling_topology.rule()
                != crate::results::field::VolumeFieldSamplingRule::TetrahedronKeastFifteen
            || d.publication_declarations.iter().any(|output| {
                output.kind != runmat_analysis_core::StateKind::ElectromagneticStatic
                    || output.completeness != runmat_analysis_core::StateCompleteness::Complete
                    || output.persistence == runmat_analysis_core::StatePersistence::CheckpointOnly
                    || output.state_id.0.trim().is_empty()
            })
            || d.publication_declarations
                .iter()
                .map(|d| &d.state_id)
                .collect::<BTreeSet<_>>()
                .len()
                != d.publication_declarations.len()
            || !r.edge_node_identities.iter().copied().eq(d
                .sampling_topology
                .oriented_edges()
                .iter()
                .map(|e| e.node_identities))
            || r.samples.len() != d.sampling_topology.integration_samples().len()
            || r.samples
                .iter()
                .zip(d.sampling_topology.integration_samples())
                .any(|(a, b)| {
                    a.element_identity != b.element_identity
                        || a.position_m != b.position_m
                        || a.volume_weight_m3 != b.volume_weight_m3
                })
        {
            return Err(invalid(
                "magnetic retained sampling topology or authored declarations differ",
            ));
        }
        if d.formulation.family() != PhysicsFamily::Electromagnetic
            || d.formulation.procedure() != AnalysisProcedure::Static
            || d.formulation.solution_space() != SolutionSpace::Direct
            || d.formulation.realization()
                != &crate::admission::canonical_magnetic_realization(d.element_order)
            || !t.is_finite()
            || t <= 0.0
            || t >= 1.0
            || !within(r.relative_physical_residual, 8.0 * t)
            || !within(r.relative_gauge_residual, 8.0 * t)
            || !within(r.relative_energy_work_error, 16.0 * t)
            || !r.magnetic_energy_j.is_finite()
            || r.magnetic_energy_j < 0.0
            || !r.source_work_j.is_finite()
            || r.edge_node_identities.is_empty()
            || r.edge_circulations_wb.len() != r.edge_node_identities.len()
            || r.edge_circulations_wb.iter().any(|v| !v.is_finite())
            || r.edge_node_identities.iter().any(|[a, b]| a >= b)
            || r.edge_node_identities
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || r.samples.is_empty()
            || !r.samples.len().is_multiple_of(15)
        {
            return Err(invalid(
                "magnetic formulation, closure or edge inventory is invalid",
            ));
        }
        let mut elements = BTreeSet::new();
        let mut energy = 0.0;
        for group in r.samples.chunks_exact(15) {
            if !elements.insert(group[0].element_identity) {
                return Err(invalid("duplicate magnetic sample element"));
            }
            for s in group {
                if s.element_identity != group[0].element_identity
                    || !s.volume_weight_m3.is_finite()
                    || s.volume_weight_m3 <= 0.0
                    || s.position_m
                        .iter()
                        .chain(&s.vector_potential_wb_per_m)
                        .chain(&s.flux_density_t)
                        .chain(&s.magnetic_field_a_per_m)
                        .any(|v| !v.is_finite())
                {
                    return Err(invalid("invalid retained magnetic sample inventory"));
                }
                let density =
                    crate::algebra::dot_canonical(&s.flux_density_t, &s.magnetic_field_a_per_m);
                if !density.is_finite() || density < 0.0 {
                    return Err(invalid("negative magnetic energy density"));
                }
                energy += 0.5 * s.volume_weight_m3 * density;
            }
        }
        if !close(energy, r.magnetic_energy_j, 256.0 * f64::EPSILON)
            || !close(2.0 * energy, r.source_work_j, 16.0 * t)
            || !close(
                relative_error(2.0 * energy, r.source_work_j),
                r.relative_energy_work_error,
                256.0 * f64::EPSILON,
            )
        {
            return Err(invalid("retained magnetic energy/work summaries disagree"));
        }
        Ok(())
    }
    pub fn validate_for(
        &self,
        workload: &MagnetostaticWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let d = &self.data;
        d.sampling_topology
            .validate_for_mesh(mesh)
            .map_err(invalid)?;
        let compiled = crate::physics::magnetic::compile_canonical_input(
            &workload.model,
            mesh,
            workload.step_id.as_str(),
            workload.formulation_requirement,
            workload.relative_residual_target,
        )
        .map_err(invalid)?;
        let p = &compiled.projection;
        if d.source_workload != workload.identity()?
            || d.producer != workload.producer_identity()?
            || d.formulation != workload.formulation(mesh)?
            || d.relative_residual_target != workload.relative_residual_target
            || d.mesh_digest != Digest::from_bytes(*mesh.canonical_digest.bytes())
            || d.projection_digest != p.projection_digest
            || d.element_order != mesh.resolved_request.element_order
            || !d
                .publication_declarations
                .iter()
                .eq(workload.model.sequence.steps[0]
                    .state_outputs
                    .iter()
                    .filter(|d| {
                        d.persistence != runmat_analysis_core::StatePersistence::CheckpointOnly
                    }))
        {
            return Err(invalid(
                "magnetic result belongs to a different workload or topology",
            ));
        }
        let edges = crate::physics::magnetic::validate_projection_envelope(p).map_err(invalid)?;
        if !d
            .response
            .edge_node_identities
            .iter()
            .copied()
            .eq(edges.edges().iter().map(|e| e.node_identities))
            || d.response.samples.len()
                != p.elements
                    .len()
                    .checked_mul(15)
                    .ok_or_else(|| invalid("sample count overflow"))?
        {
            return Err(invalid(
                "magnetic result edge/sample inventory differs from mesh",
            ));
        }
        let mut work = 0.0;
        for (index, law) in compiled.volumes.iter().enumerate() {
            let mut sample_index = 0;
            let mut mismatch = false;
            edges
                .visit_element_quadrature(p, index, |point| {
                    let s = &d.response.samples[index * 15 + sample_index];
                    sample_index += 1;
                    mismatch |= s.element_identity != law.element_identity
                        || s.position_m != point.position_m
                        || s.volume_weight_m3 != point.volume_weight_m3
                        || s.flux_density_t
                            .iter()
                            .zip(s.magnetic_field_a_per_m)
                            .any(|(b, h)| {
                                !close(*b / law.permeability_h_per_m, h, 256.0 * f64::EPSILON)
                            });
                    work += s.volume_weight_m3
                        * crate::algebra::dot_canonical(
                            &s.vector_potential_wb_per_m,
                            &law.current_density_a_per_m2,
                        );
                })
                .map_err(invalid)?;
            if mismatch || sample_index != 15 {
                return Err(invalid(
                    "magnetic sample geometry or constitutive law substitution",
                ));
            }
        }
        if !close(work, d.response.source_work_j, 256.0 * f64::EPSILON) {
            return Err(invalid(
                "magnetic retained source work differs from original volume current",
            ));
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "magnetostatic-result",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(PREFIX, "magnetostatic-result", bytes, LIMITS)?;
        value.validate()?;
        Ok(value)
    }
    /// An authenticated result digest is mandatory at cold import; source binding
    /// alone is not a signature of a producer's numerical field values.
    pub fn from_canonical_bytes_for(
        bytes: &[u8],
        expected_digest: Digest,
        workload: &MagnetostaticWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        if Digest::sha256(bytes) != expected_digest {
            return Err(invalid("magnetic retained result digest mismatch"));
        }
        let value = Self::from_canonical_bytes(bytes)?;
        value.validate_for(workload, mesh)?;
        Ok(value)
    }
    pub fn source_result_identity(&self) -> AnalysisExecutionResult<SolverResultIdentity> {
        SolverResultIdentity::from_step_result(
            self.data.producer,
            Digest::sha256(self.canonical_bytes()?),
        )
    }
    pub fn response(&self) -> &MagnetostaticResponse {
        &self.data.response
    }
    pub fn producer(&self) -> StepIdentity {
        self.data.producer
    }
    pub fn source_workload(&self) -> AnalysisWorkloadIdentity {
        self.data.source_workload
    }
    pub fn mesh_digest(&self) -> Digest {
        self.data.mesh_digest
    }
    pub fn element_order(&self) -> runmat_meshing_core::ElementOrder {
        self.data.element_order
    }
    pub fn sampling_topology(&self) -> &crate::results::field::SampledVolumeFieldTopology {
        &self.data.sampling_topology
    }
    pub fn publication_declarations(&self) -> &[runmat_analysis_core::StateOutput] {
        &self.data.publication_declarations
    }
}

fn within(v: f64, t: f64) -> bool {
    v.is_finite() && v >= 0.0 && v <= t
}
fn relative_error(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs());
    if scale == 0.0 {
        0.0
    } else {
        (a - b).abs() / scale
    }
}
fn close(a: f64, b: f64, t: f64) -> bool {
    a.is_finite() && b.is_finite() && relative_error(a, b) <= t
}
