//! The primary state of a line-parameter step: the per-length matrices and line quantities
//! solved on the section's triangle mesh.
use super::*;
use crate::physics::electrical::cross_section::LineParameters;
use runmat_meshing_core::SolverMeshArtifact;

const LINE_PARAMETERS_PREFIX: &[u8] = b"runmat-analysis-line-parameters/v1\0";
const LINE_PARAMETERS_LIMITS: CanonicalLimits = wire_bounded(CanonicalLimits::new(0, 0, 4_096, 16));

impl LineParameters {
    /// The numerical content digest of a solved section. The step's solver-result identity and
    /// its published state both derive from it.
    pub fn numerical_digest(&self) -> AnalysisExecutionResult<Digest> {
        canonical_contract_digest(
            LINE_PARAMETERS_PREFIX,
            "line-parameters",
            self,
            LINE_PARAMETERS_LIMITS,
        )
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalLineParametersStatePayload {
    pub parameters: LineParameters,
}

impl CanonicalLineParametersStatePayload {
    /// The payload is the solve `source_result` names, produced by `producer`.
    pub fn validate_for(
        &self,
        source_result: SolverResultIdentity,
        producer: StepIdentity,
    ) -> AnalysisExecutionResult<()> {
        let parameters = &self.parameters;
        let n = parameters.conductors.len();
        let square = |matrix: &Vec<Vec<f64>>| {
            matrix.len() == n
                && matrix
                    .iter()
                    .all(|row| row.len() == n && row.iter().all(|v| v.is_finite()))
        };
        if n == 0
            || !square(&parameters.capacitance_f_per_m)
            || !square(&parameters.vacuum_capacitance_f_per_m)
            || !square(&parameters.inductance_h_per_m)
            || !square(&parameters.characteristic_impedance_ohm)
            || parameters.impedance_ohm.len() != n
            || parameters.effective_permittivity.len() != n
            || parameters.delay_s_per_m.len() != n
            || parameters.pair.is_some() != (n == 2)
            || !parameters.relative_residual.is_finite()
        {
            return Err(AnalysisExecutionError::Invalid(
                "line-parameter state inventory does not match its signal conductors".into(),
            ));
        }
        if SolverResultIdentity::from_step_result(producer, parameters.numerical_digest()?)?
            != source_result
        {
            return Err(AnalysisExecutionError::Identity(
                "line-parameter state names another solver result",
            ));
        }
        Ok(())
    }
}

impl CanonicalSemanticState {
    /// The step's one declared state, holding the parameters solved on `mesh`.
    pub fn from_line_parameters(
        producer: StepIdentity,
        declaration: &StateOutput,
        mesh: &SolverMeshArtifact,
        parameters: &LineParameters,
    ) -> AnalysisExecutionResult<Self> {
        let source_result =
            SolverResultIdentity::from_step_result(producer, parameters.numerical_digest()?)?;
        Self::new(CanonicalSemanticStateInput {
            producer,
            declaration: declaration.clone(),
            topology_digest: crate::results::observation::line_parameter_topology_digest(mesh),
            source_result,
            payload: CanonicalSemanticStatePayload::LineParameters {
                state: Box::new(CanonicalLineParametersStatePayload {
                    parameters: parameters.clone(),
                }),
            },
        })
    }
}
