use runmat_analysis_core::AccuracyPlan;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::results::observation::{
    ObservationPublicationCatalog, ObservationPublicationCatalogIdentity,
};

use super::{
    AccuracyEvaluationRequestIdentity, AnalysisExecutionError, AnalysisExecutionResult,
    SolverResultIdentity, StepIdentity,
};

const ACCURACY_EVALUATION_PREFIX: &[u8] = b"runmat-analysis-accuracy-evaluation/v2\0";
const ACCURACY_EVALUATION_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracySourceKind {
    NonlinearSolid,
    FluidSteady,
    MaxwellHarmonic,
    Magnetostatic,
    ElectricalStatic,
    AcousticHarmonic,
    AcousticResponse,
    Thermal,
    StructuralLinear,
    StructuralModalResponseSpectrum,
    StructuralRandomResponse,
}

impl ObservationPublicationCatalog {
    /// Bind a publication catalog to the actual producing field authority, not a
    /// single evaluation request (which cannot represent a multi-sample history).
    pub fn validate_source_result(
        &self,
        producer: StepIdentity,
        kind: AccuracySourceKind,
        result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        self.validate_source_result_binding(producer, kind, result)
    }

    /// Admit and validate metadata once, then bind it to the original producing result.
    /// The ceiling excludes physical source reconstruction and borrowed catalog storage.
    pub fn validate_source_result_admitted(
        &self,
        producer: StepIdentity,
        kind: AccuracySourceKind,
        result: SolverResultIdentity,
        maximum_metadata_bytes: u64,
    ) -> AnalysisExecutionResult<()> {
        self.validate_metadata_admitted(maximum_metadata_bytes)?;
        self.validate_source_result_binding(producer, kind, result)
    }

    fn validate_source_result_binding(
        &self,
        producer: StepIdentity,
        kind: AccuracySourceKind,
        result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<()> {
        if self.producer() != producer {
            return Err(AnalysisExecutionError::Identity(
                "observation publication producer differs",
            ));
        }
        match (kind, self.source().source()) {
            (
                AccuracySourceKind::Magnetostatic | AccuracySourceKind::MaxwellHarmonic,
                crate::results::observation::ObservationFieldCatalogSource::SampledRecovery {
                    recovery_request,
                    ..
                },
            ) if recovery_request.producer == producer
                && recovery_request.source.solver_result() == Some(result) =>
            {
                Ok(())
            }
            (
                AccuracySourceKind::ElectricalStatic | AccuracySourceKind::AcousticHarmonic,
                crate::results::observation::ObservationFieldCatalogSource::VolumeMeshRecovery {
                    recovery_request,
                    manifest,
                    ..
                },
            ) if recovery_request.producer == producer
                && recovery_request.source.solver_result() == Some(result)
                && manifest.source == recovery_request.source =>
            {
                Ok(())
            }
            (
                AccuracySourceKind::Thermal,
                crate::results::observation::ObservationFieldCatalogSource::Thermal { manifest },
            ) if manifest.producer == producer && manifest.identity() == result => Ok(()),
            (
                AccuracySourceKind::StructuralLinear | AccuracySourceKind::NonlinearSolid,
                crate::results::observation::ObservationFieldCatalogSource::Structural {
                    recovery_request,
                    manifest,
                },
            ) if recovery_request.producer == producer
                && recovery_request.source.solver_result() == Some(result)
                && manifest.source == recovery_request.source =>
            {
                Ok(())
            }
            // The family-neutral native mesh catalog serves any family whose step fields were
            // recovered from its own solver result (statistical structural steps among them).
            (
                _,
                crate::results::observation::ObservationFieldCatalogSource::VolumeMeshRecovery {
                    recovery_request,
                    manifest,
                    ..
                },
            ) if recovery_request.producer == producer
                && recovery_request.source.solver_result() == Some(result)
                && manifest.source == recovery_request.source =>
            {
                Ok(())
            }
            _ => Err(AnalysisExecutionError::Identity(
                "observation publication differs from solver result or supported source family",
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "AccuracyEvaluationRequestWire")]
pub struct AccuracyEvaluationRequest {
    pub schema_version: u16,
    identity: AccuracyEvaluationRequestIdentity,
    pub producer: StepIdentity,
    pub source_kind: AccuracySourceKind,
    pub source_result: SolverResultIdentity,
    pub source_observations: Option<ObservationPublicationCatalogIdentity>,
    pub plan: AccuracyPlan,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AccuracyEvaluationRequestWire {
    schema_version: u16,
    identity: AccuracyEvaluationRequestIdentity,
    producer: StepIdentity,
    source_kind: AccuracySourceKind,
    source_result: SolverResultIdentity,
    source_observations: Option<ObservationPublicationCatalogIdentity>,
    plan: AccuracyPlan,
}

impl TryFrom<AccuracyEvaluationRequestWire> for AccuracyEvaluationRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: AccuracyEvaluationRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            source_kind: wire.source_kind,
            source_result: wire.source_result,
            source_observations: wire.source_observations,
            plan: wire.plan,
        };
        request.validate()?;
        Ok(request)
    }
}

impl AccuracyEvaluationRequest {
    pub fn new(
        producer: StepIdentity,
        source_kind: AccuracySourceKind,
        source_result: SolverResultIdentity,
        observations: Option<&ObservationPublicationCatalog>,
        plan: AccuracyPlan,
    ) -> AnalysisExecutionResult<Self> {
        let source_observations = match observations {
            Some(observations) => {
                observations.validate_source_result_admitted(
                    producer,
                    source_kind,
                    source_result,
                    plan.resources.resident_byte_budget(),
                )?;
                Some(observations.identity())
            }
            None => None,
        };
        let identity = AccuracyEvaluationRequestIdentity::derive(
            "accuracy-evaluation-request",
            &(
                producer,
                source_kind,
                source_result,
                source_observations,
                &plan,
            ),
        )?;
        let request = Self {
            schema_version: 2,
            identity,
            producer,
            source_kind,
            source_result,
            source_observations,
            plan,
        };
        request.validate()?;
        Ok(request)
    }

    pub const fn identity(&self) -> AccuracyEvaluationRequestIdentity {
        self.identity
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 2 {
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
                self.source_kind,
                self.source_result,
                self.source_observations,
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
