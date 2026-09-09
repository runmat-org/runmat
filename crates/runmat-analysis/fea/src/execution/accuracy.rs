use runmat_analysis_core::{AccuracyPlan, FormulationRequirement};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::results::observation::{ObservationArtifactIdentity, ObservationArtifactManifest};

use super::{
    AccuracyEvaluationRequestIdentity, AnalysisExecutionError, AnalysisExecutionResult,
    ObservationEvaluationRequestIdentity, SolverResultIdentity, StepIdentity,
};

const ACCURACY_EVALUATION_PREFIX: &[u8] = b"runmat-analysis-accuracy-evaluation/v1\0";
const ACCURACY_EVALUATION_LIMITS: CanonicalLimits =
    CanonicalLimits::new(16 * 1024 * 1024, 1_000_000, 65_536, 64);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "AccuracyEvaluationRequestWire")]
pub struct AccuracyEvaluationRequest {
    pub schema_version: u16,
    identity: AccuracyEvaluationRequestIdentity,
    pub producer: StepIdentity,
    pub source_result: SolverResultIdentity,
    pub observation_request: Option<ObservationEvaluationRequestIdentity>,
    pub source_observations: Option<ObservationArtifactIdentity>,
    pub formulation_requirement: FormulationRequirement,
    pub plan: AccuracyPlan,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AccuracyEvaluationRequestWire {
    schema_version: u16,
    identity: AccuracyEvaluationRequestIdentity,
    producer: StepIdentity,
    source_result: SolverResultIdentity,
    observation_request: Option<ObservationEvaluationRequestIdentity>,
    source_observations: Option<ObservationArtifactIdentity>,
    formulation_requirement: FormulationRequirement,
    plan: AccuracyPlan,
}

impl TryFrom<AccuracyEvaluationRequestWire> for AccuracyEvaluationRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: AccuracyEvaluationRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            source_result: wire.source_result,
            observation_request: wire.observation_request,
            source_observations: wire.source_observations,
            formulation_requirement: wire.formulation_requirement,
            plan: wire.plan,
        };
        request.validate()?;
        Ok(request)
    }
}

impl AccuracyEvaluationRequest {
    pub fn new(
        producer: StepIdentity,
        source_result: SolverResultIdentity,
        observations: Option<&ObservationArtifactManifest>,
        formulation_requirement: FormulationRequirement,
        plan: AccuracyPlan,
    ) -> AnalysisExecutionResult<Self> {
        let (observation_request, source_observations) = match observations {
            Some(observations) => {
                observations
                    .validate()
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
                if observations.producer != producer || observations.source_result != source_result
                {
                    return Err(AnalysisExecutionError::Identity(
                        "accuracy observations differ from their solver source",
                    ));
                }
                (
                    Some(observations.evaluation_request),
                    Some(observations.identity),
                )
            }
            None => (None, None),
        };
        let identity = AccuracyEvaluationRequestIdentity::derive(
            "accuracy-evaluation-request",
            &(
                producer,
                source_result,
                observation_request,
                source_observations,
                formulation_requirement,
                &plan,
            ),
        )?;
        let request = Self {
            schema_version: 1,
            identity,
            producer,
            source_result,
            observation_request,
            source_observations,
            formulation_requirement,
            plan,
        };
        request.validate()?;
        Ok(request)
    }

    pub const fn identity(&self) -> AccuracyEvaluationRequestIdentity {
        self.identity
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || self.observation_request.is_some() != self.source_observations.is_some()
        {
            return Err(AnalysisExecutionError::Invalid(
                "accuracy evaluation schema or observation binding is invalid".into(),
            ));
        }
        crate::results::accuracy::evaluate_accuracy(&self.plan, &[])
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let expected = AccuracyEvaluationRequestIdentity::derive(
            "accuracy-evaluation-request",
            &(
                self.producer,
                self.source_result,
                self.observation_request,
                self.source_observations,
                self.formulation_requirement,
                &self.plan,
            ),
        )?;
        if self.identity != expected {
            return Err(AnalysisExecutionError::Identity(
                "accuracy evaluation request identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            ACCURACY_EVALUATION_PREFIX,
            "accuracy-evaluation-request",
            self,
            ACCURACY_EVALUATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let request = decode_contract(
            ACCURACY_EVALUATION_PREFIX,
            "accuracy-evaluation-request",
            bytes,
            ACCURACY_EVALUATION_LIMITS,
        )?;
        Ok(request)
    }
}
