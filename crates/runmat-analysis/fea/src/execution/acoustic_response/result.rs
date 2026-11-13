//! Retained real pressure samples with validated lineage; cold import never re-solves.
use super::*;
use crate::{
    execution::SolverResultIdentity,
    physics::acoustic::{CanonicalAcousticModes, CanonicalAcousticTransient},
    results::field::RecoveredFieldSample,
    topology::AnalysisProjection,
};

const PREFIX: &[u8] = b"runmat-acoustic-response-result/v3\0";
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

pub(super) enum SolvedResponse {
    Modal(CanonicalAcousticModes),
    Transient(CanonicalAcousticTransient),
}

/// Procedure-specific scalar evidence retained beside the pressure samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "procedure", deny_unknown_fields)]
pub enum AcousticResponseDiagnostics {
    Modal {
        /// Ascending natural frequencies (Hz), one per mode sample.
        frequencies_hz: Vec<f64>,
        /// |K phi - lambda M phi| / ((|lambda| + s) |M phi|) per mode.
        relative_residuals: Vec<f64>,
        mass_orthogonality_error: f64,
        operator_applications: usize,
        linear_iterations: usize,
    },
    Transient {
        /// Every accepted time point; pressure fields are retained at the sampled subset.
        time_points_s: Vec<f64>,
        /// 1/2 p'ᵀ M p' + 1/2 pᵀ K p (J) per accepted time point.
        acoustic_energy_j: Vec<f64>,
        /// Cumulative energy absorbed by impedance and radiation boundaries (J).
        boundary_absorbed_energy_j: Vec<f64>,
        linear_iterations: usize,
        maximum_linear_relative_residual: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct AcousticResponseSolverResult {
    data: ResultData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultData {
    source_workload: AnalysisWorkloadIdentity,
    producer: StepIdentity,
    formulation: AnalysisRunIdentity,
    solver: AcousticResponseSolver,
    publication_declaration: runmat_analysis_core::StateOutput,
    publication_observations: Vec<runmat_analysis_core::ObservationRequest>,
    mesh_digest: runmat_meshing_core::StableDigest,
    projection_digest: Digest,
    ordered_node_ids: Vec<u64>,
    /// Volume elements of the projection the nodal pressure is interpolated over.
    ordered_element_ids: Vec<u64>,
    /// Mode (1-based) or time coordinates, in ascending order.
    samples: Vec<RecoveredFieldSample>,
    /// Real nodal pressure (Pa, or mass-normalized mode amplitude) per sample.
    pressure_pa: Vec<Vec<f64>>,
    diagnostics: AcousticResponseDiagnostics,
}

impl<'de> Deserialize<'de> for AcousticResponseSolverResult {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Self {
            data: ResultData::deserialize(deserializer)?,
        };
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

impl AcousticResponseSolverResult {
    pub(super) fn new(
        workload: &AcousticResponseWorkload,
        mesh: &SolverMeshArtifact,
        formulation: AnalysisRunIdentity,
        response: SolvedResponse,
    ) -> AnalysisExecutionResult<Self> {
        let declaration = workload.step()?.state_outputs[0].clone();
        let (projection, samples, pressure_pa, diagnostics) = match response {
            SolvedResponse::Modal(modes) => (
                modes.projection,
                (1..=modes.frequencies_hz.len())
                    .map(RecoveredFieldSample::ModeIndex)
                    .collect(),
                modes.mode_pressures,
                AcousticResponseDiagnostics::Modal {
                    frequencies_hz: modes.frequencies_hz,
                    relative_residuals: modes.relative_residuals,
                    mass_orthogonality_error: modes.mass_orthogonality_error,
                    operator_applications: modes.operator_applications,
                    linear_iterations: modes.linear_iterations,
                },
            ),
            SolvedResponse::Transient(transient) => (
                transient.projection,
                transient
                    .retained_time_points_s
                    .iter()
                    .copied()
                    .map(RecoveredFieldSample::TimeSeconds)
                    .collect(),
                transient.pressure_pa,
                AcousticResponseDiagnostics::Transient {
                    time_points_s: transient.time_points_s,
                    acoustic_energy_j: transient.acoustic_energy_j,
                    boundary_absorbed_energy_j: transient.boundary_absorbed_energy_j,
                    linear_iterations: transient.linear_iterations,
                    maximum_linear_relative_residual: transient.maximum_linear_relative_residual,
                },
            ),
        };
        let value = Self {
            data: ResultData {
                source_workload: workload.identity()?,
                producer: workload.producer_identity()?,
                formulation,
                solver: workload.solver,
                publication_declaration: declaration,
                publication_observations: workload.step()?.observations.clone(),
                mesh_digest: mesh.canonical_digest,
                projection_digest: projection.projection_digest,
                ordered_node_ids: projection.nodes.iter().map(|node| node.node_id).collect(),
                ordered_element_ids: projection
                    .elements
                    .iter()
                    .map(|element| element.element_id)
                    .collect(),
                samples,
                pressure_pa,
                diagnostics,
            },
        };
        value.validate_for(workload, mesh)?;
        Ok(value)
    }

    /// Structural, finiteness and acceptance checks of the retained arrays.
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let d = &self.data;
        let nodes = d.ordered_node_ids.len();
        let tolerance = d.solver.relative_tolerance();
        let unique_ids = !d.ordered_node_ids.is_empty()
            && !d.ordered_node_ids.contains(&0)
            && d.ordered_node_ids
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == nodes;
        if d.formulation.family() != PhysicsFamily::Acoustic
            || d.formulation.procedure() != d.solver.procedure()
            || d.formulation.solution_space() != SolutionSpace::Direct
            || d.publication_declaration.kind != d.solver.state_kind()
            || d.publication_declaration.completeness != StateCompleteness::Complete
            || !unique_ids
            || d.ordered_element_ids.is_empty()
            || d.samples.is_empty()
            || d.samples.len() != d.pressure_pa.len()
            || d.pressure_pa
                .iter()
                .any(|values| values.len() != nodes || values.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid(
                "acoustic response formulation, ordering or sample inventory is invalid",
            ));
        }
        super::validate_observations(d.solver, &d.publication_observations)?;
        match (&d.solver, &d.diagnostics) {
            (
                AcousticResponseSolver::Modal { .. },
                AcousticResponseDiagnostics::Modal {
                    frequencies_hz,
                    relative_residuals,
                    mass_orthogonality_error,
                    ..
                },
            ) => {
                if frequencies_hz.len() != d.samples.len()
                    || relative_residuals.len() != d.samples.len()
                    || frequencies_hz
                        .iter()
                        .any(|value| !value.is_finite() || *value < 0.0)
                    || frequencies_hz.windows(2).any(|pair| pair[0] > pair[1])
                    || relative_residuals
                        .iter()
                        .any(|value| !(value.is_finite() && *value >= 0.0 && *value <= tolerance))
                    || !(mass_orthogonality_error.is_finite()
                        && *mass_orthogonality_error <= 1.0e-6)
                    || d.samples.iter().enumerate().any(|(index, sample)| {
                        *sample != RecoveredFieldSample::ModeIndex(index + 1)
                    })
                {
                    return Err(invalid("acoustic cavity-mode evidence is invalid"));
                }
            }
            (
                AcousticResponseSolver::Transient { .. },
                AcousticResponseDiagnostics::Transient {
                    time_points_s,
                    acoustic_energy_j,
                    boundary_absorbed_energy_j,
                    maximum_linear_relative_residual,
                    ..
                },
            ) => {
                let times = d
                    .samples
                    .iter()
                    .map(|sample| match sample {
                        RecoveredFieldSample::TimeSeconds(time) => Some(*time),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>();
                if time_points_s.is_empty()
                    || times.is_none_or(|times| {
                        times.iter().any(|time| !time.is_finite())
                            || times.windows(2).any(|pair| pair[0] >= pair[1])
                            || times.iter().any(|time| {
                                *time < time_points_s[0] || *time > *time_points_s.last().unwrap()
                            })
                    })
                    || time_points_s.windows(2).any(|pair| !(pair[0] < pair[1]))
                    || acoustic_energy_j.len() != time_points_s.len()
                    || boundary_absorbed_energy_j.len() != time_points_s.len()
                    || acoustic_energy_j
                        .iter()
                        .chain(boundary_absorbed_energy_j)
                        .any(|value| !value.is_finite() || *value < -f64::EPSILON)
                    || !(maximum_linear_relative_residual.is_finite()
                        && *maximum_linear_relative_residual <= tolerance)
                {
                    return Err(invalid("acoustic transient evidence is invalid"));
                }
            }
            _ => {
                return Err(invalid(
                    "acoustic response diagnostics differ from their procedure",
                ))
            }
        }
        Ok(())
    }

    /// Bind the retained result to its exact workload, formulation and mesh projection.
    pub fn validate_for(
        &self,
        workload: &AcousticResponseWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let d = &self.data;
        let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
        if d.source_workload != workload.identity()?
            || d.producer != workload.producer_identity()?
            || d.solver != workload.solver
            || d.formulation != workload.formulation(mesh)?
            || &d.publication_declaration != &workload.step()?.state_outputs[0]
            || d.publication_observations.as_slice() != workload.step()?.observations.as_slice()
            || d.mesh_digest != mesh.canonical_digest
            || d.projection_digest != projection.projection_digest
            || !d
                .ordered_node_ids
                .iter()
                .copied()
                .eq(projection.nodes.iter().map(|node| node.node_id))
            || !d
                .ordered_element_ids
                .iter()
                .copied()
                .eq(projection.elements.iter().map(|element| element.element_id))
        {
            return Err(invalid(
                "acoustic response result does not belong to the workload and mesh",
            ));
        }
        if let (
            AcousticResponseSolver::Transient { .. },
            runmat_analysis_core::IndependentDomain::TimeSeconds {
                start,
                end,
                interval_count,
                ..
            },
            AcousticResponseDiagnostics::Transient { time_points_s, .. },
        ) = (
            d.solver,
            &workload.step()?.independent_domain,
            &d.diagnostics,
        ) {
            // Fields are retained exactly at the sampled schedule points the step implies.
            let retention =
                crate::physics::acoustic::TransientRetention::for_step(workload.step()?)
                    .map_err(invalid)?;
            let expected = retention
                .retained_points(time_points_s)
                .map_err(invalid)?
                .into_iter()
                .map(RecoveredFieldSample::TimeSeconds)
                .collect::<Vec<_>>();
            let canonical_grid = time_points_s.iter().enumerate().all(|(index, time)| {
                crate::excitation::ProcedureTimeGrid::uniform_coordinate_at(
                    *start,
                    *end,
                    *interval_count,
                    index,
                )
                .is_ok_and(|expected| *time == expected)
            });
            if interval_count.checked_add(1) != Some(time_points_s.len())
                || time_points_s.first() != Some(start)
                || time_points_s.last() != Some(end)
                || !canonical_grid
                || d.samples != expected
            {
                return Err(invalid(
                    "acoustic transient samples differ from the authored time schedule",
                ));
            }
        }
        if let (
            AcousticResponseSolver::Modal { .. },
            runmat_analysis_core::IndependentDomain::Eigenvalue { requested_count },
        ) = (d.solver, &workload.step()?.independent_domain)
        {
            if d.samples.len() > *requested_count {
                return Err(invalid("acoustic cavity modes exceed the authored count"));
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "acoustic-response-result",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(PREFIX, "acoustic-response-result", bytes, LIMITS)?;
        value.validate()?;
        Ok(value)
    }
    pub fn from_canonical_bytes_for(
        bytes: &[u8],
        workload: &AcousticResponseWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        let value = Self::from_canonical_bytes(bytes)?;
        value.validate_for(workload, mesh)?;
        Ok(value)
    }
    pub fn source_result_identity(&self) -> AnalysisExecutionResult<SolverResultIdentity> {
        SolverResultIdentity::derive(
            "acoustic-response-result",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }
    pub fn source_workload(&self) -> AnalysisWorkloadIdentity {
        self.data.source_workload
    }
    pub fn producer(&self) -> StepIdentity {
        self.data.producer
    }
    pub fn formulation(&self) -> &AnalysisRunIdentity {
        &self.data.formulation
    }
    pub fn solver(&self) -> AcousticResponseSolver {
        self.data.solver
    }
    pub fn mesh_digest(&self) -> runmat_meshing_core::StableDigest {
        self.data.mesh_digest
    }
    pub fn ordered_node_ids(&self) -> &[u64] {
        &self.data.ordered_node_ids
    }
    pub fn ordered_element_ids(&self) -> &[u64] {
        &self.data.ordered_element_ids
    }
    pub fn samples(&self) -> &[RecoveredFieldSample] {
        &self.data.samples
    }
    pub fn pressure_pa(&self) -> &[Vec<f64>] {
        &self.data.pressure_pa
    }
    pub fn diagnostics(&self) -> &AcousticResponseDiagnostics {
        &self.data.diagnostics
    }
    pub fn publication_observations(&self) -> &[runmat_analysis_core::ObservationRequest] {
        &self.data.publication_observations
    }
    pub(super) fn publication_declaration(&self) -> &runmat_analysis_core::StateOutput {
        &self.data.publication_declaration
    }
}
