//! Study-level numerical accuracy evaluation and publication admission.
//!
//! Physics implementations submit typed evidence. This module is the sole owner of comparing
//! that evidence with authored criteria and deciding whether a result may be published.

use std::collections::BTreeSet;

use runmat_analysis_core::{
    AccuracyLimitAction, AccuracyPlan, AccuracyPlanId, AccuracyTarget, AccuracyTolerance,
    ErrorContributionKind, ObservationQuantity, RequiredAccuracyEvidence,
    UnavailableEvidencePolicy,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::execution::{
    AccuracyEvaluationRequestIdentity, AnalysisResultIdentity, DiscretizationIdentity,
    OperatorIdentity, ProblemIdentity, SolverResultIdentity, StepIdentity,
};

use super::observation::{ObservationPublicationCatalogIdentity, ObservationResultIdentity};

mod resource_profile;
pub use resource_profile::{
    accuracy_evaluation_resource_profile, AccuracyEvaluationResourceProfile,
};

const ACCURACY_REPORT_PREFIX: &[u8] = b"runmat-fea-accuracy-report/v1\0";
const ACCURACY_ARTIFACT_PREFIX: &[u8] = b"runmat-fea-accuracy-artifact/v2\0";
const ACCURACY_REPORT_DOMAIN: &str = "analysis.fea.accuracy-report/v1";
const ACCURACY_EVIDENCE_PREFIX: &[u8] = b"runmat-fea-accuracy-evidence/v1\0";
const ACCURACY_PLAN_PREFIX: &[u8] = b"runmat-fea-accuracy-plan/v1\0";
const ACCURACY_PLAN_DOMAIN: &str = "analysis.fea.accuracy-plan/v1";
const ACCURACY_EVIDENCE_DOMAIN: &str = "analysis.fea.accuracy-evidence/v1";
const ACCURACY_REPORT_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(4096, 32);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracyMeasure {
    Relative(f64),
    Absolute {
        quantity: ObservationQuantity,
        canonical_value: f64,
    },
}

#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracyEvidenceScope {
    Step(StepIdentity),
    Problem(ProblemIdentity),
    Discretization(DiscretizationIdentity),
    Operator(OperatorIdentity),
    SolverResult(SolverResultIdentity),
    ObservationResult(ObservationResultIdentity),
    AnalysisResult(AnalysisResultIdentity),
}

#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AccuracyEvidenceIdentity(Digest);

impl AccuracyEvidenceIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "AccuracyEvidenceWire")]
pub struct AccuracyEvidence {
    identity: AccuracyEvidenceIdentity,
    pub target: AccuracyTarget,
    pub contribution: ErrorContributionKind,
    pub kind: RequiredAccuracyEvidence,
    pub measured: AccuracyMeasure,
    pub assumptions: Vec<String>,
    pub scope: AccuracyEvidenceScope,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AccuracyEvidenceWire {
    identity: AccuracyEvidenceIdentity,
    target: AccuracyTarget,
    contribution: ErrorContributionKind,
    kind: RequiredAccuracyEvidence,
    measured: AccuracyMeasure,
    assumptions: Vec<String>,
    scope: AccuracyEvidenceScope,
}

impl TryFrom<AccuracyEvidenceWire> for AccuracyEvidence {
    type Error = AccuracyEvaluationError;

    fn try_from(wire: AccuracyEvidenceWire) -> Result<Self, Self::Error> {
        let evidence = Self {
            identity: wire.identity,
            target: wire.target,
            contribution: wire.contribution,
            kind: wire.kind,
            measured: wire.measured,
            assumptions: wire.assumptions,
            scope: wire.scope,
        };
        evidence.validate()?;
        Ok(evidence)
    }
}

impl AccuracyEvidence {
    pub fn new(
        target: AccuracyTarget,
        contribution: ErrorContributionKind,
        kind: RequiredAccuracyEvidence,
        measured: AccuracyMeasure,
        mut assumptions: Vec<String>,
        scope: AccuracyEvidenceScope,
    ) -> Result<Self, AccuracyEvaluationError> {
        assumptions.sort();
        assumptions.dedup();
        let mut evidence = Self {
            identity: AccuracyEvidenceIdentity(Digest::sha256([])),
            target,
            contribution,
            kind,
            measured,
            assumptions,
            scope,
        };
        evidence.identity = evidence.derive_identity()?;
        evidence.validate()?;
        Ok(evidence)
    }

    pub const fn identity(&self) -> AccuracyEvidenceIdentity {
        self.identity
    }

    fn derive_identity(&self) -> Result<AccuracyEvidenceIdentity, AccuracyEvaluationError> {
        encode_contract(
            ACCURACY_EVIDENCE_PREFIX,
            ACCURACY_EVIDENCE_DOMAIN,
            &(
                &self.target,
                self.contribution,
                self.kind,
                self.measured,
                &self.assumptions,
                self.scope,
            ),
            ACCURACY_REPORT_LIMITS,
        )
        .map(|encoded| AccuracyEvidenceIdentity(Digest::sha256(encoded)))
        .map_err(|error| AccuracyEvaluationError::IdentityEncoding(error.to_string()))
    }

    pub fn validate(&self) -> Result<(), AccuracyEvaluationError> {
        let measure_is_valid = match self.measured {
            AccuracyMeasure::Relative(value)
            | AccuracyMeasure::Absolute {
                canonical_value: value,
                ..
            } => value.is_finite() && value >= 0.0,
        };
        if !measure_is_valid
            || self.assumptions.iter().any(|item| item.trim().is_empty())
            || !self.assumptions.windows(2).all(|pair| pair[0] < pair[1])
            || self.identity != self.derive_identity()?
        {
            return Err(AccuracyEvaluationError::InvalidEvidence(
                "evidence must have a valid measure, canonical assumptions, and convergent identity"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracyStatus {
    Satisfied,
    NotSatisfied,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracyContribution {
    pub criterion_id: String,
    pub contribution: ErrorContributionKind,
    pub required_evidence: RequiredAccuracyEvidence,
    pub mandatory: bool,
    pub status: AccuracyStatus,
    pub measured: Option<AccuracyMeasure>,
    pub normalized_error: Option<f64>,
    pub assumptions: Vec<String>,
    pub scopes: Vec<AccuracyEvidenceScope>,
    pub evidence_identities: Vec<AccuracyEvidenceIdentity>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracyReport {
    pub accuracy_plan_id: AccuracyPlanId,
    pub accuracy_plan_digest: Digest,
    pub status: AccuracyStatus,
    pub contributions: Vec<AccuracyContribution>,
    pub governing_criterion_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AccuracyReportIdentity(Digest);

impl AccuracyReportIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "AccuracyEvaluationWire")]
pub struct AccuracyEvaluation {
    identity: AccuracyReportIdentity,
    report: AccuracyReport,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AccuracyEvaluationWire {
    identity: AccuracyReportIdentity,
    report: AccuracyReport,
}

impl TryFrom<AccuracyEvaluationWire> for AccuracyEvaluation {
    type Error = AccuracyEvaluationError;

    fn try_from(wire: AccuracyEvaluationWire) -> Result<Self, Self::Error> {
        let evaluation = Self {
            identity: wire.identity,
            report: wire.report,
        };
        evaluation.validate_identity()?;
        Ok(evaluation)
    }
}

impl AccuracyEvaluation {
    pub const fn identity(&self) -> AccuracyReportIdentity {
        self.identity
    }

    pub const fn report(&self) -> &AccuracyReport {
        &self.report
    }

    pub fn validate_identity(&self) -> Result<(), AccuracyEvaluationError> {
        if self.identity != report_identity(&self.report)? {
            return Err(AccuracyEvaluationError::InvalidEvidence(
                "accuracy report identity does not match its content".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationDecision {
    Allow,
    Block,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracyPublicationGate {
    pub decision: PublicationDecision,
    pub status: AccuracyStatus,
    pub governing_criterion_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AccuracyArtifactIdentity(Digest);

impl AccuracyArtifactIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "AccuracyArtifactWire")]
pub struct AccuracyArtifact {
    pub schema_version: u16,
    identity: AccuracyArtifactIdentity,
    pub evaluation_request: AccuracyEvaluationRequestIdentity,
    pub producer: StepIdentity,
    pub source_kind: crate::execution::AccuracySourceKind,
    pub source_result: SolverResultIdentity,
    pub source_observations: Option<ObservationPublicationCatalogIdentity>,
    evidence: Vec<AccuracyEvidence>,
    evaluation: AccuracyEvaluation,
    gate: AccuracyPublicationGate,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AccuracyArtifactWire {
    schema_version: u16,
    identity: AccuracyArtifactIdentity,
    evaluation_request: AccuracyEvaluationRequestIdentity,
    producer: StepIdentity,
    source_kind: crate::execution::AccuracySourceKind,
    source_result: SolverResultIdentity,
    source_observations: Option<ObservationPublicationCatalogIdentity>,
    evidence: Vec<AccuracyEvidence>,
    evaluation: AccuracyEvaluation,
    gate: AccuracyPublicationGate,
}

impl TryFrom<AccuracyArtifactWire> for AccuracyArtifact {
    type Error = AccuracyEvaluationError;

    fn try_from(wire: AccuracyArtifactWire) -> Result<Self, Self::Error> {
        let artifact = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            evaluation_request: wire.evaluation_request,
            producer: wire.producer,
            source_kind: wire.source_kind,
            source_result: wire.source_result,
            source_observations: wire.source_observations,
            evidence: wire.evidence,
            evaluation: wire.evaluation,
            gate: wire.gate,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

pub struct AccuracyArtifactInput {
    pub evaluation_request: AccuracyEvaluationRequestIdentity,
    pub producer: StepIdentity,
    pub source_kind: crate::execution::AccuracySourceKind,
    pub source_result: SolverResultIdentity,
    pub source_observations: Option<ObservationPublicationCatalogIdentity>,
    pub evidence: Vec<AccuracyEvidence>,
    pub evaluation: AccuracyEvaluation,
    pub gate: AccuracyPublicationGate,
}

impl AccuracyArtifact {
    pub fn new(input: AccuracyArtifactInput) -> Result<Self, AccuracyEvaluationError> {
        let mut artifact = Self {
            schema_version: 2,
            identity: AccuracyArtifactIdentity(Digest::sha256([])),
            evaluation_request: input.evaluation_request,
            producer: input.producer,
            source_kind: input.source_kind,
            source_result: input.source_result,
            source_observations: input.source_observations,
            evidence: input.evidence,
            evaluation: input.evaluation,
            gate: input.gate,
        };
        artifact.identity = artifact.derive_identity()?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub const fn identity(&self) -> AccuracyArtifactIdentity {
        self.identity
    }

    pub fn evidence(&self) -> &[AccuracyEvidence] {
        &self.evidence
    }

    pub const fn evaluation(&self) -> &AccuracyEvaluation {
        &self.evaluation
    }

    pub const fn gate(&self) -> &AccuracyPublicationGate {
        &self.gate
    }

    pub fn validate(&self) -> Result<(), AccuracyEvaluationError> {
        if self.schema_version != 2 {
            return Err(AccuracyEvaluationError::InvalidEvidence(
                "accuracy artifact schema is unsupported".into(),
            ));
        }
        validate_evidence_boundary(&self.evidence)?;
        self.evaluation.validate_identity()?;
        if self.gate.status != self.evaluation.report().status
            || self.gate.governing_criterion_id != self.evaluation.report().governing_criterion_id
        {
            return Err(AccuracyEvaluationError::InvalidEvidence(
                "accuracy publication gate differs from its report".into(),
            ));
        }
        if self.identity != self.derive_identity()? {
            return Err(AccuracyEvaluationError::InvalidEvidence(
                "accuracy artifact identity does not converge".into(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AccuracyEvaluationError> {
        self.validate()?;
        encode_contract(
            ACCURACY_ARTIFACT_PREFIX,
            "analysis.fea.accuracy-artifact/v2",
            self,
            ACCURACY_REPORT_LIMITS,
        )
        .map_err(|error| AccuracyEvaluationError::IdentityEncoding(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, AccuracyEvaluationError> {
        decode_contract(
            ACCURACY_ARTIFACT_PREFIX,
            "analysis.fea.accuracy-artifact/v2",
            bytes,
            ACCURACY_REPORT_LIMITS,
        )
        .map_err(|error| AccuracyEvaluationError::IdentityEncoding(error.to_string()))
    }

    fn derive_identity(&self) -> Result<AccuracyArtifactIdentity, AccuracyEvaluationError> {
        encode_contract(
            ACCURACY_ARTIFACT_PREFIX,
            "analysis.fea.accuracy-artifact-identity/v2",
            &(
                self.schema_version,
                self.evaluation_request,
                self.producer,
                self.source_kind,
                self.source_result,
                self.source_observations,
                &self.evidence,
                &self.evaluation,
                &self.gate,
            ),
            ACCURACY_REPORT_LIMITS,
        )
        .map(Digest::sha256)
        .map(AccuracyArtifactIdentity)
        .map_err(|error| AccuracyEvaluationError::IdentityEncoding(error.to_string()))
    }
}

/// Evaluate one admitted request and package its exact evidence into the canonical artifact.
///
/// Family implementations own evidence construction; this function remains the single authority
/// for plan evaluation and publication admission.
pub fn evaluate_accuracy_artifact(
    request: &crate::execution::AccuracyEvaluationRequest,
    evidence: Vec<AccuracyEvidence>,
) -> Result<AccuracyArtifact, AccuracyEvaluationError> {
    let evaluation = evaluate_accuracy(&request.plan, &evidence)?;
    let gate = publication_gate(&request.plan, &evaluation)?;
    AccuracyArtifact::new(AccuracyArtifactInput {
        evaluation_request: request.identity(),
        producer: request.producer,
        source_kind: request.source_kind,
        source_result: request.source_result,
        source_observations: request.source_observations,
        evidence,
        evaluation,
        gate,
    })
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum AccuracyEvaluationError {
    #[error("accuracy plan is invalid: {0}")]
    InvalidPlan(String),
    #[error("accuracy evidence is invalid: {0}")]
    InvalidEvidence(String),
    #[error("accuracy report identity encoding failed: {0}")]
    IdentityEncoding(String),
}

mod evaluation;
pub use evaluation::evaluate_accuracy;

pub fn publication_gate(
    plan: &AccuracyPlan,
    evaluation: &AccuracyEvaluation,
) -> Result<AccuracyPublicationGate, AccuracyEvaluationError> {
    evaluation.validate_identity()?;
    if evaluation.report.accuracy_plan_id != plan.accuracy_plan_id
        || evaluation.report.accuracy_plan_digest != plan_digest(plan)?
    {
        return Err(AccuracyEvaluationError::InvalidEvidence(
            "accuracy report identity does not match the authored plan and report content".into(),
        ));
    }
    let decision = match evaluation.report.status {
        AccuracyStatus::Satisfied => PublicationDecision::Allow,
        AccuracyStatus::Indeterminate
            if plan.unavailable_evidence == UnavailableEvidencePolicy::MarkIndeterminate =>
        {
            PublicationDecision::Allow
        }
        _ => PublicationDecision::Block,
    };
    Ok(AccuracyPublicationGate {
        decision,
        status: evaluation.report.status,
        governing_criterion_id: evaluation.report.governing_criterion_id.clone(),
    })
}

fn plan_digest(plan: &AccuracyPlan) -> Result<Digest, AccuracyEvaluationError> {
    encode_contract(
        ACCURACY_PLAN_PREFIX,
        ACCURACY_PLAN_DOMAIN,
        plan,
        ACCURACY_REPORT_LIMITS,
    )
    .map(Digest::sha256)
    .map_err(|error| AccuracyEvaluationError::IdentityEncoding(error.to_string()))
}

fn report_identity(
    report: &AccuracyReport,
) -> Result<AccuracyReportIdentity, AccuracyEvaluationError> {
    encode_contract(
        ACCURACY_REPORT_PREFIX,
        ACCURACY_REPORT_DOMAIN,
        report,
        ACCURACY_REPORT_LIMITS,
    )
    .map(|encoded| AccuracyReportIdentity(Digest::sha256(encoded)))
    .map_err(|error| AccuracyEvaluationError::IdentityEncoding(error.to_string()))
}

fn normalized_error(
    measured: AccuracyMeasure,
    tolerance: AccuracyTolerance,
) -> Result<f64, AccuracyEvaluationError> {
    let (value, limit) = match (measured, tolerance) {
        (AccuracyMeasure::Relative(value), AccuracyTolerance::Relative(limit)) => (value, limit),
        (
            AccuracyMeasure::Absolute {
                quantity,
                canonical_value,
            },
            AccuracyTolerance::Absolute {
                quantity: expected,
                canonical_value: limit,
            },
        ) if quantity == expected => (canonical_value, limit),
        _ => {
            return Err(AccuracyEvaluationError::InvalidEvidence(
                "accuracy measure does not match the criterion tolerance".into(),
            ))
        }
    };
    if !value.is_finite() || value < 0.0 || !limit.is_finite() || limit <= 0.0 {
        return Err(AccuracyEvaluationError::InvalidEvidence(
            "accuracy measures must be finite and non-negative and tolerances positive".into(),
        ));
    }
    Ok(value / limit)
}

fn validate_plan_boundary(plan: &AccuracyPlan) -> Result<(), AccuracyEvaluationError> {
    if plan.accuracy_plan_id.0.trim().is_empty()
        || plan.criteria.is_empty()
        || !plan.criteria.iter().any(|criterion| criterion.mandatory)
    {
        return Err(AccuracyEvaluationError::InvalidPlan(
            "plan requires an id, criteria, and at least one mandatory criterion".into(),
        ));
    }
    let mut ids = BTreeSet::new();
    for criterion in &plan.criteria {
        let target_is_valid = !matches!(
            &criterion.target,
            AccuracyTarget::Observation(observation_id) if observation_id.0.trim().is_empty()
        );
        let tolerance_is_valid = match criterion.tolerance {
            AccuracyTolerance::Relative(value)
            | AccuracyTolerance::Absolute {
                canonical_value: value,
                ..
            } => value.is_finite() && value > 0.0,
        };
        if criterion.criterion_id.trim().is_empty()
            || !ids.insert(&criterion.criterion_id)
            || !target_is_valid
            || !tolerance_is_valid
        {
            return Err(AccuracyEvaluationError::InvalidPlan(
                "criterion ids, targets, and positive finite tolerances must be valid".into(),
            ));
        }
    }
    let equivalence_is_valid = match plan.equivalence {
        runmat_analysis_core::NumericalEquivalencePolicy::Deterministic => true,
        runmat_analysis_core::NumericalEquivalencePolicy::RelativeTolerance(value)
        | runmat_analysis_core::NumericalEquivalencePolicy::AbsoluteTolerance(value) => {
            value.is_finite() && value > 0.0
        }
    };
    let resources_are_valid = plan.resources.maximum_samples > 0
        && plan.resources.maximum_wall_time_milliseconds > 0
        && plan.resources.resident_limit() != Some(0)
        && match plan.exceeded_limit {
            AccuracyLimitAction::Refine => plan.resources.maximum_refinement_cycles > 0,
            AccuracyLimitAction::TightenSolver => plan.resources.maximum_solver_tightenings > 0,
            AccuracyLimitAction::ExtendSampling => plan.resources.maximum_samples > 1,
            AccuracyLimitAction::Fail => true,
        };
    if !equivalence_is_valid || !resources_are_valid {
        return Err(AccuracyEvaluationError::InvalidPlan(
            "equivalence tolerances and resource ceilings must be finite and actionable".into(),
        ));
    }
    Ok(())
}

fn validate_evidence_boundary(
    evidence: &[AccuracyEvidence],
) -> Result<(), AccuracyEvaluationError> {
    let mut identities = BTreeSet::new();
    for item in evidence {
        item.validate()?;
        if !identities.insert(item.identity()) {
            return Err(AccuracyEvaluationError::InvalidEvidence(
                "evidence identities must be unique".into(),
            ));
        }
    }
    Ok(())
}

fn governing_criterion(
    contributions: &[AccuracyContribution],
    status: AccuracyStatus,
) -> Option<String> {
    contributions
        .iter()
        .filter(|item| item.mandatory && item.status == status)
        .max_by(|left, right| {
            left.normalized_error
                .unwrap_or(f64::INFINITY)
                .total_cmp(&right.normalized_error.unwrap_or(f64::INFINITY))
        })
        .map(|item| item.criterion_id.clone())
}

#[cfg(test)]
mod tests;
