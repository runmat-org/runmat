//! Retained original complex fields. Cold import never solves or recreates lost fields.
use super::*;
use crate::{
    execution::SolverResultIdentity,
    physics::maxwell::{MaxwellInput, MaxwellResponse},
};
use runmat_analysis_core::{
    AnalysisProcedure, PhysicsFamily, SolutionSpace, StateOutput, StatePersistence,
};
use std::collections::BTreeMap;
const PREFIX: &[u8] = b"runmat-maxwell-harmonic-result/v2\0";
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct MaxwellHarmonicSolverResult {
    data: ResultData,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultData {
    source_workload: AnalysisWorkloadIdentity,
    producer: StepIdentity,
    formulation: AnalysisRunIdentity,
    relative_residual_target: f64,
    element_order: runmat_meshing_core::ElementOrder,
    publication_declarations: Vec<StateOutput>,
    input: MaxwellInput,
    response: MaxwellResponse,
}
impl<'de> Deserialize<'de> for MaxwellHarmonicSolverResult {
    fn deserialize<D: serde::Deserializer<'de>>(decoder: D) -> Result<Self, D::Error> {
        let value = Self {
            data: ResultData::deserialize(decoder)?,
        };
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}
impl MaxwellHarmonicSolverResult {
    pub(super) fn new(
        workload: &MaxwellHarmonicWorkload,
        mesh: &SolverMeshArtifact,
        formulation: AnalysisRunIdentity,
        input: MaxwellInput,
        response: MaxwellResponse,
    ) -> AnalysisExecutionResult<Self> {
        let value = Self {
            data: ResultData {
                source_workload: workload.identity()?,
                producer: workload.producer_identity()?,
                formulation,
                relative_residual_target: workload.relative_residual_target,
                element_order: mesh.resolved_request.element_order,
                publication_declarations: declarations(workload),
                input,
                response,
            },
        };
        value.validate_for(workload, mesh)?;
        Ok(value)
    }
    /// Structural/retained summaries only. Source authentication additionally needs
    /// original workload, mesh and authenticated expected digest at cold import.
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let d = &self.data;
        let r = &d.response;
        let input = &d.input;
        let sampling = &input.sampling;
        let t = d.relative_residual_target;
        sampling.validate().map_err(invalid)?;
        if d.formulation.family() != PhysicsFamily::Electromagnetic
            || d.formulation.procedure() != AnalysisProcedure::HarmonicResponse
            || d.formulation.solution_space() != SolutionSpace::Direct
            || d.formulation.realization()
                != &crate::admission::canonical_maxwell_realization(d.element_order)
            || !t.is_finite()
            || t <= 0.0
            || t >= 1.0
            || r.input_identity != input.identity().map_err(invalid)?
            || sampling.mesh_digest() != Digest::from_bytes(*r.mesh_digest.bytes())
            || sampling.projection_digest() != r.projection_digest
            || sampling.sampling_digest() != r.sampling_digest
            || sampling.rule()
                != crate::results::field::VolumeFieldSamplingRule::TetrahedronKeastFifteen
            || r.angular_frequency_rad_per_s.canonical_value()
                != input.angular_frequency_rad_per_s.canonical_value()
            || !r.angular_frequency_rad_per_s.canonical_value().is_finite()
            || r.angular_frequency_rad_per_s.canonical_value() <= 0.0
            || !within(r.relative_physical_residual, 8.0 * t)
            || !within(r.relative_complex_power_error, 16.0 * t)
            || !within(r.relative_charge_continuity_residual, 16.0 * t)
            || r.edge_circulations_v.canonical_values().len() != r.edge_node_identities.len()
            || r.edge_circulations_v
                .canonical_values()
                .iter()
                .any(|v| !v.is_finite())
            || !r
                .edge_node_identities
                .iter()
                .copied()
                .eq(sampling.oriented_edges().iter().map(|e| e.node_identities))
            || r.samples.len() != sampling.integration_sample_count()
            || input.source_current.len() != r.samples.len()
            || d.publication_declarations.iter().any(|o| {
                o.kind != runmat_analysis_core::StateKind::ElectromagneticComplex
                    || o.completeness != runmat_analysis_core::StateCompleteness::Complete
                    || o.persistence == StatePersistence::CheckpointOnly
            })
        {
            return Err(invalid(
                "Maxwell retained identity/topology/physical acceptance mismatch",
            ));
        }
        let laws: BTreeMap<_, _> = input
            .volumes
            .iter()
            .map(|v| (v.element_identity, v))
            .collect();
        if laws.len() != input.volumes.len()
            || !sampling
                .element_identities()
                .iter()
                .all(|id| laws.contains_key(id))
            || laws.len() != sampling.element_identities().len()
        {
            return Err(invalid("Maxwell retained material inventory mismatch"));
        }
        let mut wm = 0.0;
        let mut ohmic = 0.0;
        let points = sampling.rule().points_per_element();
        for (index, (s, j)) in r.samples.iter().zip(&input.source_current).enumerate() {
            let law = laws
                .get(&s.element_identity)
                .ok_or_else(|| invalid("foreign retained element"))?;
            // Sample geometry is derived from the mesh; `validate_for` re-evaluates it.
            let identity = sampling
                .integration_sample_identity(index)
                .map_err(invalid)?;
            if s.sample_identity != identity
                || j.sample_identity != identity
                || s.element_identity != sampling.element_identities()[index / points]
                || s.ordinal != index % points
                || !s.volume_weight_m3.is_finite()
                || s.volume_weight_m3 <= 0.0
                || s.position_m.iter().any(|v| !v.is_finite())
                || !law.permeability_h_per_m.canonical_value().is_finite()
                || law.permeability_h_per_m.canonical_value() <= 0.0
                || !law.permittivity_f_per_m.canonical_value().is_finite()
                || law.permittivity_f_per_m.canonical_value() <= 0.0
                || !law.conductivity_s_per_m.canonical_value().is_finite()
                || law.conductivity_s_per_m.canonical_value() < 0.0
                || [
                    s.electric_field_v_per_m.canonical_value(),
                    s.flux_density_t.canonical_value(),
                    s.magnetic_field_a_per_m.canonical_value(),
                    s.electric_displacement_c_per_m2.canonical_value(),
                    s.conduction_current_density_a_per_m2.canonical_value(),
                    j.current_density_a_per_m2.canonical_value(),
                ]
                .into_iter()
                .flatten()
                .any(|v| !v.is_finite())
            {
                return Err(invalid("Maxwell retained sample/law substitution"));
            }
            let mut ee = 0.0;
            let mut hh = 0.0;
            for axis in 0..3 {
                ee += s.electric_field_v_per_m.canonical_value()[axis].norm_squared();
                hh += s.magnetic_field_a_per_m.canonical_value()[axis].norm_squared();
            }
            let weight = s.volume_weight_m3;
            wm += 0.25 * weight * law.permeability_h_per_m.canonical_value() * hh;
            ohmic += 0.5 * weight * law.conductivity_s_per_m.canonical_value() * ee;
        }
        // Electric energy, source power and closure are re-derived from the original
        // equations in `validate_for`; the sample summaries checked here are B/H/J/loss.
        if !close(wm, r.magnetic_energy_j.canonical_value())
            || !close(ohmic, r.ohmic_power_w.canonical_value())
            || !close(
                2.0 * r.angular_frequency_rad_per_s.canonical_value()
                    * (r.magnetic_energy_j.canonical_value()
                        - r.electric_energy_j.canonical_value()),
                r.reactive_power_var.canonical_value(),
            )
        {
            return Err(invalid("Maxwell retained power/energy summaries disagree"));
        }
        Ok(())
    }
    pub fn validate_for(
        &self,
        workload: &MaxwellHarmonicWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let input = workload.compile_input(mesh)?;
        let plan = crate::physics::maxwell::MaxwellSystemPlan::derive(mesh, &input.grounded_faces)
            .map_err(invalid)?;
        if self.data.source_workload != workload.identity()?
            || self.data.producer != workload.producer_identity()?
            || self.data.formulation != workload.formulation(mesh)?
            || self.data.relative_residual_target != workload.relative_residual_target
            || self.data.element_order != mesh.resolved_request.element_order
            || self.data.publication_declarations != declarations(workload)
            || self.data.input != input
            || self.data.response.system_plan_identity != plan.identity().map_err(invalid)?
            || self.data.response.mesh_digest != mesh.canonical_digest
        {
            return Err(invalid(
                "Maxwell retained result differs from original authored closure",
            ));
        }
        crate::physics::maxwell::validate_retained_consistency(
            mesh,
            &self.data.input,
            &self.data.response,
            self.data.relative_residual_target,
        )
        .map_err(invalid)?;
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "maxwell-harmonic-result",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(PREFIX, "maxwell-harmonic-result", bytes, LIMITS)?;
        value.validate()?;
        Ok(value)
    }
    pub fn from_canonical_bytes_for(
        bytes: &[u8],
        expected: Digest,
        workload: &MaxwellHarmonicWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        if Digest::sha256(bytes) != expected {
            return Err(invalid("Maxwell authenticated result digest mismatch"));
        }
        let value = Self::from_canonical_bytes(bytes)?;
        value.validate_for(workload, mesh)?;
        Ok(value)
    }
    pub fn identity(&self) -> AnalysisExecutionResult<SolverResultIdentity> {
        SolverResultIdentity::derive(
            "maxwell-harmonic-result",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }
    pub fn response(&self) -> &MaxwellResponse {
        &self.data.response
    }
    /// Exact original integration-point and globally oriented edge authority.
    pub fn sampling_topology(&self) -> &crate::results::field::SampledVolumeFieldTopology {
        &self.data.input.sampling
    }
    pub fn input(&self) -> &MaxwellInput {
        &self.data.input
    }
    /// Cycle-averaged Joule loss density per volume element, (1/V)∫(1/2)sigma|E|^2 dV with
    /// the total field, in the mesh's canonical element order. This is the heat source a
    /// thermal analysis receives from this harmonic state; its integral is `ohmic_power_w`.
    pub fn element_loss_density_w_per_m3(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<
        runmat_types::physical::Quantity<
            runmat_types::physical::kind::HeatGenerationDensity,
            Vec<f64>,
        >,
    > {
        let laws: BTreeMap<_, _> = self
            .data
            .input
            .volumes
            .iter()
            .map(|v| (v.element_identity, v.conductivity_s_per_m))
            .collect();
        let mut loss: BTreeMap<runmat_meshing_core::StableDigest, (f64, f64)> = BTreeMap::new();
        for sample in &self.data.response.samples {
            let sigma = laws
                .get(&sample.element_identity)
                .ok_or_else(|| invalid("foreign retained element"))?;
            let squared: f64 = sample
                .electric_field_v_per_m
                .canonical_value()
                .iter()
                .map(|v| v.norm_squared())
                .sum();
            let entry = loss.entry(sample.element_identity).or_default();
            entry.0 += 0.5 * sigma.canonical_value() * squared * sample.volume_weight_m3;
            entry.1 += sample.volume_weight_m3;
        }
        mesh.topology
            .volume_elements
            .iter()
            .map(|element| {
                loss.get(&element.stable_identity)
                    .map(|(power, volume)| power / volume)
                    .ok_or_else(|| invalid("Maxwell loss is missing a mesh element"))
            })
            .collect::<AnalysisExecutionResult<Vec<f64>>>()
            .map(runmat_types::physical::Quantity::new)
    }
    pub fn source_workload(&self) -> AnalysisWorkloadIdentity {
        self.data.source_workload
    }
    pub fn producer_identity(&self) -> StepIdentity {
        self.data.producer
    }
    pub fn formulation(&self) -> &AnalysisRunIdentity {
        &self.data.formulation
    }
    pub fn publication_declarations(&self) -> &[StateOutput] {
        &self.data.publication_declarations
    }
}

#[cfg(test)]
mod consistency_tests;
fn declarations(workload: &MaxwellHarmonicWorkload) -> Vec<StateOutput> {
    workload.model.sequence.steps[0]
        .state_outputs
        .iter()
        .filter(|o| o.persistence != StatePersistence::CheckpointOnly)
        .cloned()
        .collect()
}
fn within(value: f64, upper: f64) -> bool {
    value.is_finite() && value >= 0.0 && value <= upper
}
fn close(a: f64, b: f64) -> bool {
    a.is_finite()
        && b.is_finite()
        && (a == b || (a - b).abs() <= 256.0 * f64::EPSILON * a.abs().max(b.abs()))
}
