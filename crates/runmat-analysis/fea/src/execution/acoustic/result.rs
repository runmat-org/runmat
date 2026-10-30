//! Source/mesh-bound fields and algebraic/physical closure, not independent accuracy evidence.
use super::super::SolverResultIdentity;
use super::*;
use crate::{
    excitation::{ProjectionScalar, ProjectionVector3, SpatialProjectionEvidence},
    physics::acoustic::CanonicalAcousticResponse,
    topology::AnalysisProjection,
};
use runmat_meshing_core::StableDigest;

const PREFIX: &[u8] = b"runmat-acoustic-harmonic-result/v2\0";
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct AcousticHarmonicSolverResult {
    data: ResultData,
}

impl<'de> Deserialize<'de> for AcousticHarmonicSolverResult {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::try_from(ResultData::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultData {
    source_workload: AnalysisWorkloadIdentity,
    producer: super::super::StepIdentity,
    publication_declaration: Option<runmat_analysis_core::StateOutput>,
    formulation: AnalysisRunIdentity,
    mesh_digest: StableDigest,
    projection_digest: Digest,
    frequency: Frequency,
    solver: AcousticHarmonicPolicy,
    power_tolerance: f64,
    ordered_node_ids: Vec<u64>,
    ordered_element_ids: Vec<u64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pressure_pa: runmat_types::physical::Quantity<
        runmat_types::physical::kind::Pressure,
        Vec<ProjectionScalar>,
    >,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    velocity_m_per_s: runmat_types::physical::Quantity<
        runmat_types::physical::kind::Velocity,
        Vec<ProjectionVector3>,
    >,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    intensity_w_per_m2: runmat_types::physical::Quantity<
        runmat_types::physical::kind::AcousticIntensity,
        Vec<[f64; 3]>,
    >,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    port_intensity_w_per_m2: runmat_types::physical::Quantity<
        runmat_types::physical::kind::AcousticIntensity,
        Vec<f64>,
    >,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    energy_density_j_per_m3:
        runmat_types::physical::Quantity<runmat_types::physical::kind::EnergyDensity, Vec<f64>>,
    iterations: u32,
    relative_residual: f64,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    mean_input_power_w: runmat_types::physical::Quantity<runmat_types::physical::kind::Power, f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    mean_impedance_loss_w:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Power, f64>,
    power_relative_error: f64,
    source_evidence: Vec<SpatialProjectionEvidence>,
    certificate: crate::physics::acoustic::AcousticSolveCertificate,
}
impl TryFrom<ResultData> for AcousticHarmonicSolverResult {
    type Error = AnalysisExecutionError;
    fn try_from(data: ResultData) -> AnalysisExecutionResult<Self> {
        let result = Self { data };
        result.validate()?;
        Ok(result)
    }
}
/// Scalar closure and identity of one acoustic harmonic result. Per-node values are published
/// as field partitions and the native state; this summary is what run records retain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticHarmonicResultSummary {
    pub source_workload: AnalysisWorkloadIdentity,
    pub producer: super::super::StepIdentity,
    pub source_result: SolverResultIdentity,
    pub formulation: AnalysisRunIdentity,
    pub mesh_digest: StableDigest,
    pub frequency: Frequency,
    pub iterations: u32,
    pub relative_residual: f64,
    pub mean_input_power_w: f64,
    pub mean_impedance_loss_w: f64,
    pub power_relative_error: f64,
}

impl AcousticHarmonicSolverResult {
    pub fn summary(&self) -> AnalysisExecutionResult<AcousticHarmonicResultSummary> {
        let d = &self.data;
        Ok(AcousticHarmonicResultSummary {
            source_workload: d.source_workload,
            producer: d.producer,
            source_result: self.source_result_identity()?,
            formulation: d.formulation.clone(),
            mesh_digest: d.mesh_digest,
            frequency: d.frequency,
            iterations: d.iterations,
            relative_residual: d.relative_residual,
            mean_input_power_w: d.mean_input_power_w.canonical_value(),
            mean_impedance_loss_w: d.mean_impedance_loss_w.canonical_value(),
            power_relative_error: d.power_relative_error,
        })
    }

    pub(super) fn new(
        workload: &AcousticHarmonicWorkload,
        mesh: &SolverMeshArtifact,
        formulation: AnalysisRunIdentity,
        response: CanonicalAcousticResponse,
    ) -> AnalysisExecutionResult<Self> {
        let data = ResultData {
            source_workload: workload.identity()?,
            producer: workload.producer_identity()?,
            publication_declaration: workload.publication_declaration()?.cloned(),
            formulation,
            mesh_digest: response.projection.mesh_digest,
            projection_digest: response.projection.projection_digest,
            frequency: Frequency::new(response.frequency_hz),
            solver: workload.solver,
            power_tolerance: workload.power_balance_relative_tolerance,
            ordered_node_ids: response
                .projection
                .nodes
                .iter()
                .map(|n| n.node_id)
                .collect(),
            ordered_element_ids: response
                .projection
                .elements
                .iter()
                .map(|e| e.element_id)
                .collect(),
            pressure_pa: response.pressure_pa,
            velocity_m_per_s: response.element_velocity_m_per_s,
            intensity_w_per_m2: response.element_mean_intensity_w_per_m2,
            port_intensity_w_per_m2: response.element_port_intensity_w_per_m2,
            energy_density_j_per_m3: response.element_mean_energy_density_j_per_m3,
            iterations: response.iterations,
            relative_residual: response.relative_residual,
            mean_input_power_w: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Power,
                _,
            >::new(response.mean_input_power_w.canonical_value()),
            mean_impedance_loss_w: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Power,
                _,
            >::new(
                response.mean_impedance_loss_w.canonical_value()
            ),
            power_relative_error: response.power_relative_error,
            source_evidence: response.source_evidence,
            certificate: response.certificate,
        };
        let result = Self::try_from(data)?;
        result.validate_for(workload, mesh)?;
        Ok(result)
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let d = &self.data;
        if let Some(declaration) = &d.publication_declaration {
            crate::execution::SemanticStatePublicationRequest::new(
                crate::execution::SemanticStatePublicationSource::AcousticNativeFields,
                d.producer,
                declaration.clone(),
                Digest::from_bytes(*d.mesh_digest.bytes()),
                super::super::SolverResultIdentity::from_step_result(
                    d.producer,
                    Digest::sha256([]),
                )?,
            )?;
        }
        d.solver.validate(d.frequency).map_err(invalid)?;
        validate_solver_formulation_compatibility(
            &d.formulation,
            SolverMethod::AcousticCanonicalHelmholtzCertifiedLu,
            crate::contracts::SolverPreconditioner::None,
        )
        .map_err(invalid)?;
        d.certificate
            .validate(
                d.solver.contraction_limit,
                d.ordered_node_ids
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| invalid("acoustic certificate dimension overflow"))?,
            )
            .map_err(invalid)?;
        if let Some(reason) = closure_rejection_reason(d) {
            return Err(invalid(format!(
                "acoustic result rejected: {reason}; family={:?} procedure={:?} mesh={:?} frequency_hz={} refinement_steps={} relative_residual={:e} solver_relative_tolerance={:e} power_relative_error={:e} power_relative_tolerance={:e} mean_input_power_w={:e} mean_impedance_loss_w={:e}",
                d.formulation.family(), d.formulation.procedure(), d.mesh_digest,
                d.frequency.canonical_value(), d.iterations,
                d.relative_residual, d.solver.relative_tolerance,
                d.power_relative_error, d.power_tolerance,
                d.mean_input_power_w.canonical_value(), d.mean_impedance_loss_w.canonical_value(),
            )));
        }
        let nodes = d.ordered_node_ids.len();
        let elements = d.ordered_element_ids.len();
        let unique = |ids: &[u64]| {
            let set = ids.iter().collect::<std::collections::BTreeSet<_>>();
            !ids.is_empty() && set.len() == ids.len() && !ids.contains(&0)
        };
        if !unique(&d.ordered_node_ids)
            || !unique(&d.ordered_element_ids)
            || d.pressure_pa.canonical_values().len() != nodes
            || d.velocity_m_per_s.canonical_values().len() != elements
            || d.intensity_w_per_m2.canonical_values().len() != elements
            || d.port_intensity_w_per_m2.canonical_values().len() != elements
            || d.port_intensity_w_per_m2
                .canonical_values()
                .iter()
                .any(|x| !x.is_finite() || *x < 0.0)
            || d.energy_density_j_per_m3.canonical_values().len() != elements
            || d.pressure_pa
                .canonical_values()
                .iter()
                .any(|p| !p.real.is_finite() || !p.imaginary.is_finite())
            || d.velocity_m_per_s
                .canonical_values()
                .iter()
                .any(|v| v.real.iter().chain(&v.imaginary).any(|x| !x.is_finite()))
            || d.intensity_w_per_m2
                .canonical_values()
                .iter()
                .flatten()
                .any(|x| !x.is_finite())
            || d.energy_density_j_per_m3
                .canonical_values()
                .iter()
                .any(|x| !x.is_finite() || *x < 0.0)
            || d.source_evidence
                .iter()
                .any(|e| !e.accepted || e.validate().is_err())
        {
            return Err(invalid(
                "acoustic result field shape, finiteness or source evidence is invalid",
            ));
        }
        Ok(())
    }
    /// Lineage/compatibility checks. Does not recompute physics or claim accuracy.
    pub fn validate_for(
        &self,
        workload: &AcousticHarmonicWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let d = &self.data;
        let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
        if d.source_workload != workload.identity()?
            || d.producer != workload.producer_identity()?
            || d.publication_declaration.as_ref() != workload.publication_declaration()?
            || d.formulation != workload.formulation(mesh)?
            || d.mesh_digest != mesh.canonical_digest
            || d.projection_digest != projection.projection_digest
            || d.frequency != workload.frequency
            || d.solver != workload.solver
            || d.power_tolerance != workload.power_balance_relative_tolerance
            || !d
                .ordered_node_ids
                .iter()
                .copied()
                .eq(projection.nodes.iter().map(|n| n.node_id))
            || !d
                .ordered_element_ids
                .iter()
                .copied()
                .eq(projection.elements.iter().map(|e| e.element_id))
        {
            return Err(invalid(
                "acoustic result does not belong to the workload and canonical mesh",
            ));
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "acoustic-harmonic-result",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let result: Self = decode_contract(PREFIX, "acoustic-harmonic-result", bytes, LIMITS)?;
        result.validate()?;
        Ok(result)
    }
    pub fn source_result_identity(&self) -> AnalysisExecutionResult<SolverResultIdentity> {
        SolverResultIdentity::derive(
            "acoustic-harmonic-result",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }
    pub fn pressure_pa(&self) -> &[ProjectionScalar] {
        self.data.pressure_pa.canonical_values()
    }
    pub fn velocity_m_per_s(&self) -> &[ProjectionVector3] {
        self.data.velocity_m_per_s.canonical_values()
    }
    pub fn source_evidence(&self) -> &[SpatialProjectionEvidence] {
        &self.data.source_evidence
    }
    /// Validated producer claim, not an independently replayed cold proof.
    pub const fn certificate(&self) -> crate::physics::acoustic::AcousticSolveCertificate {
        self.data.certificate
    }
    pub fn intensity_w_per_m2(&self) -> &[[f64; 3]] {
        self.data.intensity_w_per_m2.canonical_values()
    }
    /// Element plane-wave port intensity (incident wave on incident faces, outgoing wave on
    /// anechoic faces), W/m².
    pub fn port_intensity_w_per_m2(&self) -> &[f64] {
        self.data.port_intensity_w_per_m2.canonical_values()
    }
    pub fn energy_density_j_per_m3(&self) -> &[f64] {
        self.data.energy_density_j_per_m3.canonical_values()
    }
    pub fn ordered_node_ids(&self) -> &[u64] {
        &self.data.ordered_node_ids
    }
    pub fn ordered_element_ids(&self) -> &[u64] {
        &self.data.ordered_element_ids
    }
    pub fn formulation(&self) -> &AnalysisRunIdentity {
        &self.data.formulation
    }
    pub fn producer(&self) -> super::super::StepIdentity {
        self.data.producer
    }
    pub fn source_workload(&self) -> AnalysisWorkloadIdentity {
        self.data.source_workload
    }
    pub fn relative_residual(&self) -> f64 {
        self.data.relative_residual
    }
    pub fn power_relative_error(&self) -> f64 {
        self.data.power_relative_error
    }
    /// Accepted net input power, including reflected-wave subtraction at incident faces.
    pub fn mean_input_power_w(&self) -> f64 {
        self.data.mean_input_power_w.canonical_value()
    }
    /// Accepted passive boundary loss, excluding incident-inlet Robin bookkeeping.
    pub fn mean_impedance_loss_w(&self) -> f64 {
        self.data.mean_impedance_loss_w.canonical_value()
    }
    pub fn frequency(&self) -> Frequency {
        self.data.frequency
    }
    pub fn mesh_digest(&self) -> StableDigest {
        self.data.mesh_digest
    }
    pub(super) fn publication_declaration(&self) -> Option<&runmat_analysis_core::StateOutput> {
        self.data.publication_declaration.as_ref()
    }
}

fn closure_rejection_reason(d: &ResultData) -> Option<&'static str> {
    if d.formulation.family() != PhysicsFamily::Acoustic
        || d.formulation.procedure() != AnalysisProcedure::HarmonicResponse
    {
        Some("incompatible family or procedure")
    } else if !d.power_tolerance.is_finite() || d.power_tolerance <= 0.0 || d.power_tolerance >= 1.0
    {
        Some("invalid power tolerance")
    } else if !d.relative_residual.is_finite() || d.relative_residual < 0.0 {
        Some("invalid algebraic residual")
    } else if d.relative_residual > d.solver.relative_tolerance {
        Some("algebraic residual exceeds tolerance")
    } else if !d.power_relative_error.is_finite() || d.power_relative_error < 0.0 {
        Some("invalid power error")
    } else if d.power_relative_error > d.power_tolerance {
        Some("power closure exceeds tolerance")
    } else if !d.mean_input_power_w.canonical_value().is_finite() {
        Some("nonfinite input power")
    } else if !d.mean_impedance_loss_w.canonical_value().is_finite()
        || d.mean_impedance_loss_w.canonical_value() < 0.0
    {
        Some("invalid impedance loss")
    } else {
        None
    }
}
