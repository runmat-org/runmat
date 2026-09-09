//! Study-level numerical accuracy evaluation and publication admission.
//!
//! Physics implementations submit typed evidence. This module is the sole owner of comparing
//! that evidence with authored criteria and deciding whether a result may be published.

use std::collections::BTreeSet;

use runmat_analysis_core::{
    AccuracyLimitAction, AccuracyPlan, AccuracyPlanId, AccuracyTarget, AccuracyTolerance,
    ErrorContributionKind, FormulationRequirement, ObservationQuantity, RequiredAccuracyEvidence,
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

use super::observation::{ObservationArtifactIdentity, ObservationResultIdentity};

const ACCURACY_REPORT_PREFIX: &[u8] = b"runmat-fea-accuracy-report/v1\0";
const ACCURACY_REPORT_DOMAIN: &str = "analysis.fea.accuracy-report/v1";
const ACCURACY_EVIDENCE_PREFIX: &[u8] = b"runmat-fea-accuracy-evidence/v1\0";
const ACCURACY_PLAN_PREFIX: &[u8] = b"runmat-fea-accuracy-plan/v1\0";
const ACCURACY_PLAN_DOMAIN: &str = "analysis.fea.accuracy-plan/v1";
const ACCURACY_EVIDENCE_DOMAIN: &str = "analysis.fea.accuracy-evidence/v1";
const ACCURACY_REPORT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(16 * 1024 * 1024, 1_000_000, 4096, 32);

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
    pub source_result: SolverResultIdentity,
    pub source_observations: Option<ObservationArtifactIdentity>,
    pub formulation_requirement: FormulationRequirement,
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
    source_result: SolverResultIdentity,
    source_observations: Option<ObservationArtifactIdentity>,
    formulation_requirement: FormulationRequirement,
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
            source_result: wire.source_result,
            source_observations: wire.source_observations,
            formulation_requirement: wire.formulation_requirement,
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
    pub source_result: SolverResultIdentity,
    pub source_observations: Option<ObservationArtifactIdentity>,
    pub formulation_requirement: FormulationRequirement,
    pub evidence: Vec<AccuracyEvidence>,
    pub evaluation: AccuracyEvaluation,
    pub gate: AccuracyPublicationGate,
}

impl AccuracyArtifact {
    pub fn new(input: AccuracyArtifactInput) -> Result<Self, AccuracyEvaluationError> {
        let mut artifact = Self {
            schema_version: 1,
            identity: AccuracyArtifactIdentity(Digest::sha256([])),
            evaluation_request: input.evaluation_request,
            producer: input.producer,
            source_result: input.source_result,
            source_observations: input.source_observations,
            formulation_requirement: input.formulation_requirement,
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
        if self.schema_version != 1 {
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
            ACCURACY_REPORT_PREFIX,
            "analysis.fea.accuracy-artifact/v1",
            self,
            ACCURACY_REPORT_LIMITS,
        )
        .map_err(|error| AccuracyEvaluationError::IdentityEncoding(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, AccuracyEvaluationError> {
        decode_contract(
            ACCURACY_REPORT_PREFIX,
            "analysis.fea.accuracy-artifact/v1",
            bytes,
            ACCURACY_REPORT_LIMITS,
        )
        .map_err(|error| AccuracyEvaluationError::IdentityEncoding(error.to_string()))
    }

    fn derive_identity(&self) -> Result<AccuracyArtifactIdentity, AccuracyEvaluationError> {
        encode_contract(
            ACCURACY_REPORT_PREFIX,
            "analysis.fea.accuracy-artifact-identity/v1",
            &(
                self.schema_version,
                self.evaluation_request,
                self.producer,
                self.source_result,
                self.source_observations,
                self.formulation_requirement,
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

#[derive(Debug, Error, Clone, PartialEq)]
pub enum AccuracyEvaluationError {
    #[error("accuracy plan is invalid: {0}")]
    InvalidPlan(String),
    #[error("accuracy evidence is invalid: {0}")]
    InvalidEvidence(String),
    #[error("accuracy report identity encoding failed: {0}")]
    IdentityEncoding(String),
}

pub fn evaluate_accuracy(
    plan: &AccuracyPlan,
    evidence: &[AccuracyEvidence],
) -> Result<AccuracyEvaluation, AccuracyEvaluationError> {
    validate_plan_boundary(plan)?;
    validate_evidence_boundary(evidence)?;
    if evidence.iter().any(|item| {
        !plan.criteria.iter().any(|criterion| {
            item.target == criterion.target
                && item.contribution == criterion.contribution
                && item.kind == criterion.evidence
        })
    }) {
        return Err(AccuracyEvaluationError::InvalidEvidence(
            "every submitted evidence item must satisfy a declared accuracy criterion".into(),
        ));
    }
    let accuracy_plan_digest = plan_digest(plan)?;

    let mut contributions = Vec::with_capacity(plan.criteria.len());
    for criterion in &plan.criteria {
        let mut matching = evidence
            .iter()
            .filter(|item| {
                item.target == criterion.target
                    && item.contribution == criterion.contribution
                    && item.kind == criterion.evidence
            })
            .collect::<Vec<_>>();
        matching.sort_by_key(|item| item.identity());

        if matching.is_empty() {
            contributions.push(AccuracyContribution {
                criterion_id: criterion.criterion_id.clone(),
                contribution: criterion.contribution,
                required_evidence: criterion.evidence,
                mandatory: criterion.mandatory,
                status: AccuracyStatus::Indeterminate,
                measured: None,
                normalized_error: None,
                assumptions: Vec::new(),
                scopes: Vec::new(),
                evidence_identities: Vec::new(),
            });
            continue;
        }

        let mut selected = None;
        for item in &matching {
            let normalized = normalized_error(item.measured, criterion.tolerance)?;
            if selected
                .as_ref()
                .is_none_or(|(_, current): &(&AccuracyEvidence, f64)| normalized > *current)
            {
                selected = Some((*item, normalized));
            }
        }
        let (selected, normalized) = selected.ok_or_else(|| {
            AccuracyEvaluationError::InvalidEvidence(
                "matching evidence could not be reduced to a finite measure".into(),
            )
        })?;
        let mut assumptions = matching
            .iter()
            .flat_map(|item| item.assumptions.iter().cloned())
            .collect::<Vec<_>>();
        assumptions.sort();
        assumptions.dedup();
        let mut scopes = matching.iter().map(|item| item.scope).collect::<Vec<_>>();
        scopes.sort_unstable();
        scopes.dedup();
        contributions.push(AccuracyContribution {
            criterion_id: criterion.criterion_id.clone(),
            contribution: criterion.contribution,
            required_evidence: criterion.evidence,
            mandatory: criterion.mandatory,
            status: if normalized <= 1.0 {
                AccuracyStatus::Satisfied
            } else {
                AccuracyStatus::NotSatisfied
            },
            measured: Some(selected.measured),
            normalized_error: Some(normalized),
            assumptions,
            scopes,
            evidence_identities: matching.iter().map(|item| item.identity()).collect(),
        });
    }

    let status = if contributions
        .iter()
        .any(|item| item.mandatory && item.status == AccuracyStatus::NotSatisfied)
    {
        AccuracyStatus::NotSatisfied
    } else if contributions
        .iter()
        .any(|item| item.mandatory && item.status == AccuracyStatus::Indeterminate)
    {
        AccuracyStatus::Indeterminate
    } else {
        AccuracyStatus::Satisfied
    };
    let governing_criterion_id = governing_criterion(&contributions, status);
    let report = AccuracyReport {
        accuracy_plan_id: plan.accuracy_plan_id.clone(),
        accuracy_plan_digest,
        status,
        contributions,
        governing_criterion_id,
    };
    Ok(AccuracyEvaluation {
        identity: report_identity(&report)?,
        report,
    })
}

pub fn publication_gate(
    plan: &AccuracyPlan,
    requirement: FormulationRequirement,
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
    let decision = match (requirement, evaluation.report.status) {
        (_, AccuracyStatus::Satisfied) => PublicationDecision::Allow,
        (FormulationRequirement::Production, _) => PublicationDecision::Block,
        (FormulationRequirement::Experimental, AccuracyStatus::Indeterminate)
            if plan.unavailable_evidence
                == UnavailableEvidencePolicy::MarkIndeterminateExperimental =>
        {
            PublicationDecision::Allow
        }
        (FormulationRequirement::Experimental, AccuracyStatus::NotSatisfied)
            if plan.exceeded_limit == AccuracyLimitAction::WarnExperimental =>
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
        && plan.resources.maximum_resident_bytes > 0
        && match plan.exceeded_limit {
            AccuracyLimitAction::Refine => plan.resources.maximum_refinement_cycles > 0,
            AccuracyLimitAction::TightenSolver => plan.resources.maximum_solver_tightenings > 0,
            AccuracyLimitAction::ExtendSampling => plan.resources.maximum_samples > 1,
            AccuracyLimitAction::Fail | AccuracyLimitAction::WarnExperimental => true,
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
mod tests {
    use super::*;

    fn scope() -> AccuracyEvidenceScope {
        let sequence = crate::execution::SequenceIdentity::new(Digest::sha256("sequence")).unwrap();
        AccuracyEvidenceScope::Step(StepIdentity::new(sequence, Digest::sha256("step")).unwrap())
    }

    fn step_identity() -> StepIdentity {
        let AccuracyEvidenceScope::Step(step) = scope() else {
            unreachable!()
        };
        step
    }

    fn evidence(value: f64) -> AccuracyEvidence {
        AccuracyEvidence::new(
            AccuracyTarget::GlobalEquilibrium,
            ErrorContributionKind::Algebraic,
            RequiredAccuracyEvidence::ResidualBound,
            AccuracyMeasure::Relative(value),
            vec!["canonical residual norm".into()],
            scope(),
        )
        .unwrap()
    }

    #[test]
    fn blocked_accuracy_cannot_be_repackaged_as_a_published_result() {
        let plan = AccuracyPlan::experimental_baseline("accuracy");
        let evaluation = evaluate_accuracy(&plan, &[evidence(1.0e-6)]).unwrap();
        let gate =
            publication_gate(&plan, FormulationRequirement::Production, &evaluation).unwrap();
        assert_eq!(gate.decision, PublicationDecision::Block);
        let producer = step_identity();
        let source_result =
            SolverResultIdentity::from_step_result(producer, Digest::sha256("solver-result"))
                .unwrap();
        let request = crate::execution::AccuracyEvaluationRequest::new(
            producer,
            source_result,
            None,
            FormulationRequirement::Production,
            plan,
        )
        .unwrap();
        let artifact = AccuracyArtifact::new(AccuracyArtifactInput {
            evaluation_request: request.identity(),
            producer,
            source_result,
            source_observations: None,
            formulation_requirement: FormulationRequirement::Production,
            evidence: vec![evidence(1.0e-6)],
            evaluation,
            gate,
        })
        .unwrap();
        let publication = crate::execution::ResultPublicationRequest::new(&artifact, None).unwrap();
        assert!(
            crate::results::publication::PublishedAnalysisResult::new(&publication, &artifact,)
                .is_err()
        );
    }

    #[test]
    fn mandatory_evidence_governs_status_and_production_publication() {
        let plan = AccuracyPlan::experimental_baseline("accuracy");
        let satisfied = evaluate_accuracy(&plan, &[evidence(1.0e-9)]).unwrap();
        assert_eq!(satisfied.report.status, AccuracyStatus::Satisfied);
        assert_eq!(
            publication_gate(&plan, FormulationRequirement::Production, &satisfied)
                .unwrap()
                .decision,
            PublicationDecision::Allow
        );

        let failed = evaluate_accuracy(&plan, &[evidence(1.0e-6)]).unwrap();
        assert_eq!(failed.report.status, AccuracyStatus::NotSatisfied);
        assert_eq!(
            publication_gate(&plan, FormulationRequirement::Production, &failed)
                .unwrap()
                .decision,
            PublicationDecision::Block
        );
    }

    #[test]
    fn missing_mandatory_evidence_is_indeterminate_and_blocks_production() {
        let plan = AccuracyPlan::experimental_baseline("accuracy");
        let evaluation = evaluate_accuracy(&plan, &[]).unwrap();
        assert_eq!(evaluation.report.status, AccuracyStatus::Indeterminate);
        assert_eq!(
            publication_gate(&plan, FormulationRequirement::Production, &evaluation)
                .unwrap()
                .decision,
            PublicationDecision::Block
        );
        assert_eq!(
            publication_gate(&plan, FormulationRequirement::Experimental, &evaluation)
                .unwrap()
                .decision,
            PublicationDecision::Allow
        );
    }

    #[test]
    fn report_identity_is_independent_of_evidence_submission_order() {
        let mut plan = AccuracyPlan::experimental_baseline("accuracy");
        plan.criteria.push(runmat_analysis_core::AccuracyCriterion {
            criterion_id: "precision".into(),
            target: AccuracyTarget::ExecutionEquivalence,
            contribution: ErrorContributionKind::ExecutionEquivalence,
            tolerance: AccuracyTolerance::Relative(1.0e-10),
            evidence: RequiredAccuracyEvidence::ResidualBound,
            mandatory: false,
        });
        let second = AccuracyEvidence::new(
            AccuracyTarget::ExecutionEquivalence,
            ErrorContributionKind::ExecutionEquivalence,
            RequiredAccuracyEvidence::ResidualBound,
            AccuracyMeasure::Relative(1.0e-12),
            vec!["same canonical operator".into()],
            scope(),
        )
        .unwrap();
        let first = evidence(1.0e-9);
        let forward = evaluate_accuracy(&plan, &[first.clone(), second.clone()]).unwrap();
        let reverse = evaluate_accuracy(&plan, &[second, first]).unwrap();
        assert_eq!(forward.identity, reverse.identity);
        assert_eq!(forward.report, reverse.report);
    }

    #[test]
    fn publication_rejects_a_changed_plan_with_the_same_authored_id() {
        let plan = AccuracyPlan::experimental_baseline("accuracy");
        let evaluation = evaluate_accuracy(&plan, &[evidence(1.0e-9)]).unwrap();
        let mut changed = plan.clone();
        changed.criteria[0].tolerance = AccuracyTolerance::Relative(1.0e-4);

        assert!(
            publication_gate(&changed, FormulationRequirement::Production, &evaluation).is_err()
        );
    }

    #[test]
    fn publication_rejects_mutated_report_content() {
        let plan = AccuracyPlan::experimental_baseline("accuracy");
        let mut evaluation = evaluate_accuracy(&plan, &[evidence(1.0e-9)]).unwrap();
        evaluation.report.status = AccuracyStatus::NotSatisfied;

        assert!(publication_gate(&plan, FormulationRequirement::Production, &evaluation).is_err());
    }

    #[test]
    fn deserialization_rejects_a_nonconvergent_report_identity() {
        let plan = AccuracyPlan::experimental_baseline("accuracy");
        let evaluation = evaluate_accuracy(&plan, &[evidence(1.0e-9)]).unwrap();
        let mut encoded = serde_json::to_value(evaluation).unwrap();
        encoded["report"]["status"] = serde_json::json!("not_satisfied");

        assert!(serde_json::from_value::<AccuracyEvaluation>(encoded).is_err());
    }

    #[test]
    fn evidence_identity_is_derived_from_typed_scope_and_content() {
        let baseline = evidence(1.0e-9);
        let mut encoded = serde_json::to_value(&baseline).unwrap();
        encoded["measured"]["relative"] = serde_json::json!(1.0e-3);

        assert!(serde_json::from_value::<AccuracyEvidence>(encoded).is_err());

        let different_scope = AccuracyEvidence::new(
            AccuracyTarget::GlobalEquilibrium,
            ErrorContributionKind::Algebraic,
            RequiredAccuracyEvidence::ResidualBound,
            AccuracyMeasure::Relative(1.0e-9),
            vec!["canonical residual norm".into()],
            AccuracyEvidenceScope::Problem(
                ProblemIdentity::new(
                    match scope() {
                        AccuracyEvidenceScope::Step(step) => step,
                        _ => unreachable!(),
                    },
                    Digest::sha256("model"),
                    None,
                    crate::execution::ConstraintPlanIdentity::new(
                        match scope() {
                            AccuracyEvidenceScope::Step(step) => step,
                            _ => unreachable!(),
                        },
                        Digest::sha256("mesh"),
                        Digest::sha256("constraints"),
                    )
                    .unwrap(),
                    Vec::new(),
                )
                .unwrap(),
            ),
        )
        .unwrap();
        assert_ne!(baseline.identity(), different_scope.identity());
    }
}
