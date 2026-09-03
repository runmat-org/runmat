use serde::{Deserialize, Serialize};

use super::output::{ObservationId, ObservationQuantity};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AccuracyPlanId(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracyPlan {
    pub accuracy_plan_id: AccuracyPlanId,
    pub criteria: Vec<AccuracyCriterion>,
    pub unavailable_evidence: UnavailableEvidencePolicy,
    pub exceeded_limit: AccuracyLimitAction,
    pub equivalence: NumericalEquivalencePolicy,
    pub resources: AccuracyResourceCeiling,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracyCriterion {
    pub criterion_id: String,
    pub target: AccuracyTarget,
    pub contribution: ErrorContributionKind,
    pub tolerance: AccuracyTolerance,
    pub evidence: RequiredAccuracyEvidence,
    pub mandatory: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracyTarget {
    Observation(ObservationId),
    Quantity(ObservationQuantity),
    GlobalEquilibrium,
    InterfaceClosure,
    ExecutionEquivalence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorContributionKind {
    GeometryAdmission,
    BoundaryApproximation,
    SpatialDiscretization,
    Algebraic,
    Nonlinear,
    Temporal,
    FrequencySampling,
    ModalTruncation,
    Stochastic,
    Coupling,
    InterfaceProjection,
    FieldTransfer,
    SignalInterpolation,
    Recovery,
    ObservationSampling,
    Precision,
    ExecutionEquivalence,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracyTolerance {
    Relative(f64),
    Absolute {
        quantity: ObservationQuantity,
        canonical_value: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequiredAccuracyEvidence {
    APosterioriEstimator,
    ConvergenceStudy,
    ConservationBalance,
    ResidualBound,
    ReferenceComparison,
    StatisticalConfidence,
    TransferBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnavailableEvidencePolicy {
    Fail,
    MarkIndeterminateExperimental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracyLimitAction {
    Refine,
    TightenSolver,
    ExtendSampling,
    Fail,
    WarnExperimental,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericalEquivalencePolicy {
    Deterministic,
    RelativeTolerance(f64),
    AbsoluteTolerance(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracyResourceCeiling {
    pub maximum_refinement_cycles: usize,
    pub maximum_solver_tightenings: usize,
    pub maximum_samples: usize,
    pub maximum_wall_time_milliseconds: u64,
    pub maximum_resident_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracyReport {
    pub accuracy_plan_id: AccuracyPlanId,
    pub status: AccuracyStatus,
    pub contributions: Vec<AccuracyContribution>,
    pub governing_criterion_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracyStatus {
    Satisfied,
    NotSatisfied,
    Indeterminate,
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracyContribution {
    pub criterion_id: String,
    pub contribution: ErrorContributionKind,
    pub status: AccuracyStatus,
    pub measured: Option<AccuracyMeasure>,
    pub estimator_assumptions: Vec<String>,
    pub scope_digest: String,
    pub evidence_digest: String,
}

impl AccuracyPlan {
    /// Explicit baseline for formulations that have not yet earned production accuracy evidence.
    pub fn experimental_baseline(id: impl Into<String>) -> Self {
        Self {
            accuracy_plan_id: AccuracyPlanId(id.into()),
            criteria: vec![AccuracyCriterion {
                criterion_id: "algebraic_residual".to_string(),
                target: AccuracyTarget::GlobalEquilibrium,
                contribution: ErrorContributionKind::Algebraic,
                tolerance: AccuracyTolerance::Relative(1.0e-8),
                evidence: RequiredAccuracyEvidence::ResidualBound,
                mandatory: true,
            }],
            unavailable_evidence: UnavailableEvidencePolicy::MarkIndeterminateExperimental,
            exceeded_limit: AccuracyLimitAction::Fail,
            equivalence: NumericalEquivalencePolicy::RelativeTolerance(1.0e-10),
            resources: AccuracyResourceCeiling {
                maximum_refinement_cycles: 0,
                maximum_solver_tightenings: 0,
                maximum_samples: 1_000_000,
                maximum_wall_time_milliseconds: 3_600_000,
                maximum_resident_bytes: 8 * 1024 * 1024 * 1024,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracyMeasure {
    Relative(f64),
    Absolute {
        quantity: ObservationQuantity,
        canonical_value: f64,
    },
}
