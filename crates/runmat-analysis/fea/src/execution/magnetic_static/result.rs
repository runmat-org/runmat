//! Source-bound retained native coefficients and SI samples. Cold validation does
//! not assemble an operator, reconstruct a missing field or rerun the solve.
use super::*;
use crate::{execution::SolverResultIdentity, physics::magnetic::MagnetostaticResponse};
use std::collections::BTreeSet;

const PREFIX: &[u8] = b"runmat-magnetostatic-result/v2\0";
/// The retained result grows with the mesh (edges and fifteen samples per element). Its size is
/// admitted by the magnetostatic resource plan (`result_output`) and checked against it by the
/// object inventory, so the codec bounds it only by the wire range; its identity is digested
/// from the streamed encoding.
const SAMPLES: usize = crate::physics::magnetic::MAGNETIC_SAMPLES_PER_ELEMENT;
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct MagnetostaticSolverResult {
    data: ResultData,
}

/// Fields are declared in canonical (UTF-8 key) order so the streamed encoding holds neither
/// the sampled topology nor the response back for sorting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultData {
    element_order: runmat_meshing_core::ElementOrder,
    formulation: AnalysisRunIdentity,
    mesh_digest: Digest,
    producer: StepIdentity,
    projection_digest: Digest,
    publication_declarations: Vec<runmat_analysis_core::StateOutput>,
    relative_residual_target: f64,
    response: MagnetostaticResponse,
    sampling_topology: crate::results::field::SampledVolumeFieldTopology,
    source_workload: AnalysisWorkloadIdentity,
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
    /// `sampling_topology` is the mesh's canonical fifteen-point layout (derived by the caller
    /// from the solve's projection); it records the projection digest.
    pub(super) fn new(
        workload: &MagnetostaticWorkload,
        mesh: &SolverMeshArtifact,
        formulation: AnalysisRunIdentity,
        sampling_topology: crate::results::field::SampledVolumeFieldTopology,
        response: MagnetostaticResponse,
    ) -> AnalysisExecutionResult<Self> {
        if sampling_topology.rule()
            != crate::results::field::VolumeFieldSamplingRule::TetrahedronKeastFifteen
        {
            return Err(invalid("magnetic results use the fifteen-point layout"));
        }
        let result = Self {
            data: ResultData {
                source_workload: workload.identity()?,
                producer: workload.producer_identity()?,
                formulation,
                mesh_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
                projection_digest: sampling_topology.projection_digest(),
                sampling_topology,
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
            || r.edge_circulations_wb.canonical_values().len()
                != d.sampling_topology.oriented_edges().len()
            || r.sample_count() != d.sampling_topology.integration_sample_count()
            || r.flux_density_t.canonical_values().len() != 3 * r.sample_count()
            || r.vector_potential_wb_per_m.canonical_values().len() != 3 * r.sample_count()
            || r.element_count() != d.sampling_topology.element_identities().len()
            || r.element_flux_density_t.canonical_values().len() != 3 * r.element_count()
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
            || !r.magnetic_energy_j.canonical_value().is_finite()
            || r.magnetic_energy_j.canonical_value() < 0.0
            || !r.source_work_j.canonical_value().is_finite()
            || r.edge_circulations_wb.canonical_values().is_empty()
            || [
                r.edge_circulations_wb.canonical_values(),
                r.flux_density_t.canonical_values(),
                r.vector_potential_wb_per_m.canonical_values(),
                r.element_flux_density_t.canonical_values(),
            ]
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite())
            || r.element_permeability_h_per_m
                .canonical_values()
                .iter()
                .any(|mu| !mu.is_finite() || *mu <= 0.0)
            || r.sample_count() == 0
        {
            return Err(invalid(
                "magnetic formulation, closure or edge inventory is invalid",
            ));
        }
        // The energy is re-integrated from the samples against mesh weights in `validate_for`;
        // here the summaries must agree with each other.
        if !close(
            2.0 * r.magnetic_energy_j.canonical_value(),
            r.source_work_j.canonical_value(),
            16.0 * t,
        ) || !close(
            relative_error(
                2.0 * r.magnetic_energy_j.canonical_value(),
                r.source_work_j.canonical_value(),
            ),
            r.relative_energy_work_error,
            256.0 * f64::EPSILON,
        ) {
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
        // One canonical projection serves the layout, input and edge checks below.
        let compiled = crate::physics::magnetic::compile_canonical_input(
            &workload.model,
            mesh,
            workload.step_id.as_str(),
            workload.relative_residual_target,
        )
        .map_err(invalid)?;
        let p = &compiled.projection;
        d.sampling_topology
            .validate_for_projection(mesh, p)
            .map_err(invalid)?;
        if d.source_workload != workload.identity()?
            || d.producer != workload.producer_identity()?
            || d.formulation
                != workload.formulation_with_sampling(mesh, Some(&d.sampling_topology))?
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
        let edges =
            crate::physics::magnetic::validate_projection_envelope(p, &compiled.grounded_faces)
                .map_err(invalid)?;
        let r = &d.response;
        if !d
            .sampling_topology
            .oriented_edges()
            .iter()
            .map(|e| e.node_identities)
            .eq(edges.edges().iter().map(|e| e.node_identities))
            || compiled.volumes.len() != d.sampling_topology.element_identities().len()
            || r.sample_count()
                != p.elements
                    .len()
                    .checked_mul(SAMPLES)
                    .ok_or_else(|| invalid("sample count overflow"))?
        {
            return Err(invalid(
                "magnetic result edge/sample inventory differs from mesh",
            ));
        }
        // Samples are named by (element, rule point): the element's law and the rule's
        // geometry come from the original mesh, so energy, work and element means are
        // re-integrated here exactly as the solver did.
        let mut work = 0.0;
        let mut energy = 0.0;
        for (index, law) in compiled.volumes.iter().enumerate() {
            let permeability = law.permeability_h_per_m.canonical_value();
            if law.element_identity != d.sampling_topology.element_identities()[index]
                || r.element_permeability_h_per_m.canonical_values()[index] != permeability
            {
                return Err(invalid(
                    "magnetic sample geometry or constitutive law substitution",
                ));
            }
            let mut weights = [0.0; SAMPLES];
            let mut ordinal = 0;
            edges
                .visit_element_quadrature(p, index, |point| {
                    if ordinal < SAMPLES {
                        let sample = index * SAMPLES + ordinal;
                        let flux = r.flux_density(sample);
                        let field = flux.map(|v| v / permeability);
                        weights[ordinal] = point.volume_weight_m3;
                        energy += 0.5
                            * point.volume_weight_m3
                            * crate::algebra::dot_canonical(&flux, &field);
                        work += point.volume_weight_m3
                            * crate::algebra::dot_canonical(
                                &r.vector_potential(sample),
                                &law.current_density_a_per_m2.canonical_value(),
                            );
                    }
                    ordinal += 1;
                })
                .map_err(invalid)?;
            let mean = crate::physics::magnetic::element_mean(&weights, |o| {
                r.flux_density(index * SAMPLES + o)
            });
            if ordinal != SAMPLES || mean != r.element_flux_density(index) {
                return Err(invalid(
                    "magnetic sample geometry or constitutive law substitution",
                ));
            }
        }
        if !close(
            energy,
            r.magnetic_energy_j.canonical_value(),
            256.0 * f64::EPSILON,
        ) {
            return Err(invalid("retained magnetic energy/work summaries disagree"));
        }
        if !close(
            work,
            r.source_work_j.canonical_value(),
            256.0 * f64::EPSILON,
        ) {
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
    /// SHA-256 of `canonical_bytes`, from the streamed encoding: identifying a result never
    /// materializes its mesh-sized wire.
    pub fn payload_digest(&self) -> AnalysisExecutionResult<Digest> {
        self.validate()?;
        Ok(
            crate::contracts::canonical_digest::canonical_contract_digest(
                PREFIX,
                "magnetostatic-result",
                self,
                LIMITS,
            )?,
        )
    }
    pub fn source_result_identity(&self) -> AnalysisExecutionResult<SolverResultIdentity> {
        SolverResultIdentity::from_step_result(self.data.producer, self.payload_digest()?)
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
