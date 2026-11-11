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
    NumericalResult,
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
    /// A relative residual of the solved equations, normalized as each producer documents.
    ResidualBound,
    /// The normwise backward error of the computed solution in the ∞-norm:
    /// `‖b − A x‖ / (‖A‖ ‖x‖ + ‖b‖)` for a linear system and
    /// `‖K x − θ M x‖ / ((‖K‖ + |θ| ‖M‖) ‖x‖)` for an eigenpair. See
    /// [`AlgebraicAccuracyMeasure::BackwardError`].
    BackwardErrorBound,
    ReferenceComparison,
    StatisticalConfidence,
    TransferBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnavailableEvidencePolicy {
    Fail,
    MarkIndeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracyLimitAction {
    Refine,
    TightenSolver,
    ExtendSampling,
    Fail,
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
    #[serde(with = "runmat_types::wire_integer")]
    pub maximum_refinement_cycles: usize,
    #[serde(with = "runmat_types::wire_integer")]
    pub maximum_solver_tightenings: usize,
    #[serde(with = "runmat_types::wire_integer")]
    pub maximum_samples: usize,
    #[serde(with = "runmat_types::wire_integer")]
    pub maximum_wall_time_milliseconds: u64,
    /// Authored resident-memory ceiling; `None` (`null` on the wire, never omitted) means no
    /// authored limit, and placement then limits residency by the machine-capacity authority.
    /// Read it only through [`AccuracyResourceCeiling::resident_limit`],
    /// [`AccuracyResourceCeiling::admits_resident_bytes`] or
    /// [`AccuracyResourceCeiling::resident_byte_budget`].
    #[serde(with = "runmat_types::wire_integer::option")]
    pub maximum_resident_bytes: Option<u64>,
}

impl AccuracyResourceCeiling {
    /// The authored resident-memory ceiling, if one was authored.
    pub fn resident_limit(&self) -> Option<u64> {
        self.maximum_resident_bytes
    }

    /// Whether `bytes` of residency fits the authored ceiling; always true without one.
    pub fn admits_resident_bytes(&self, bytes: u64) -> bool {
        self.maximum_resident_bytes
            .is_none_or(|limit| bytes <= limit)
    }

    /// The authored ceiling as a saturating byte budget for planning arithmetic: `u64::MAX`
    /// without an authored limit, so the only binding limit is the placement's machine capacity.
    pub fn resident_byte_budget(&self) -> u64 {
        self.maximum_resident_bytes.unwrap_or(u64::MAX)
    }

    /// Stores a derived saturating budget as the authored ceiling: the inverse of
    /// `resident_byte_budget`. A budget above the wire bound (2^53 - 1 bytes) cannot bind any
    /// placement, so it is stored as no cap instead of an unrepresentable integer.
    pub fn set_resident_byte_budget(&mut self, budget: u64) {
        self.maximum_resident_bytes = runmat_types::wire_integer::bounded(budget);
    }
}

/// The measure the mandatory algebraic criterion of a step applies its tolerance to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgebraicAccuracyMeasure {
    /// Normwise backward error in the ∞-norm (Rigal–Gaches for linear systems; Tisseur 2000
    /// for eigenpairs): the smallest relative change of the operators (and load) for which the
    /// computed answer is exact. It is invariant to scaling and does not grow with the
    /// conditioning of the mesh, so one tolerance means the same on every model. The default
    /// tolerance, [`AlgebraicAccuracyMeasure::BACKWARD_ERROR_TOLERANCE`], is 1e-12: backward-
    /// stable direct and converged iterative solvers return about the unit roundoff
    /// (u ≈ 1.1e-16) in this measure, and the evaluation of `A x` itself is uncertain by up to
    /// `m · u` for rows of `m` entries (a few 1e-14 for quadratic solid elements). 1e-12 sits
    /// above that floor and keeps the qualified pair exact for operators within 1e-12 of the
    /// assembled ones, far below the uncertainty of any material or geometric input.
    BackwardError,
    /// Euclidean residual relative to the terms that form it, `‖r‖ / ‖b‖` or
    /// `‖K x − θ M x‖ / (‖K x‖ + |θ| ‖M x‖)`, with tolerance 1e-8 by default. Its floor in
    /// double precision is about `u · κ`, so it can refuse a correct answer on a fine mesh.
    RelativeResidual,
}

impl AlgebraicAccuracyMeasure {
    pub const BACKWARD_ERROR_TOLERANCE: f64 = 1.0e-12;
    pub const RELATIVE_RESIDUAL_TOLERANCE: f64 = 1.0e-8;

    /// The evidence an algebraic criterion in this measure requires.
    pub const fn evidence(self) -> RequiredAccuracyEvidence {
        match self {
            Self::BackwardError => RequiredAccuracyEvidence::BackwardErrorBound,
            Self::RelativeResidual => RequiredAccuracyEvidence::ResidualBound,
        }
    }

    pub const fn from_evidence(evidence: RequiredAccuracyEvidence) -> Option<Self> {
        match evidence {
            RequiredAccuracyEvidence::BackwardErrorBound => Some(Self::BackwardError),
            RequiredAccuracyEvidence::ResidualBound => Some(Self::RelativeResidual),
            _ => None,
        }
    }

    pub const fn default_tolerance(self) -> f64 {
        match self {
            Self::BackwardError => Self::BACKWARD_ERROR_TOLERANCE,
            Self::RelativeResidual => Self::RELATIVE_RESIDUAL_TOLERANCE,
        }
    }

    /// The `.fea` spelling.
    pub const fn label(self) -> &'static str {
        match self {
            Self::BackwardError => "backward_error",
            Self::RelativeResidual => "relative_residual",
        }
    }
}

impl AccuracyPlan {
    /// The measure of the plan's mandatory algebraic residual criteria, when they share one.
    pub fn algebraic_measure(&self) -> Option<AlgebraicAccuracyMeasure> {
        let mut measures = self
            .criteria
            .iter()
            .filter(|criterion| criterion.contribution == ErrorContributionKind::Algebraic)
            .filter_map(|criterion| AlgebraicAccuracyMeasure::from_evidence(criterion.evidence));
        let first = measures.next()?;
        measures.all(|measure| measure == first).then_some(first)
    }

    /// Moves every algebraic residual criterion to `measure` with that measure's default
    /// tolerance. Returns false, changing nothing, when the plan has no such criterion.
    pub fn set_algebraic_measure(&mut self, measure: AlgebraicAccuracyMeasure) -> bool {
        let mut changed = false;
        for criterion in self.criteria.iter_mut().filter(|criterion| {
            criterion.contribution == ErrorContributionKind::Algebraic
                && AlgebraicAccuracyMeasure::from_evidence(criterion.evidence).is_some()
        }) {
            criterion.evidence = measure.evidence();
            criterion.tolerance = AccuracyTolerance::Relative(measure.default_tolerance());
            changed = true;
        }
        changed
    }

    /// Baseline plan: a mandatory relative algebraic residual bound of 1e-8. Every solver produces
    /// that evidence, so a result without it is indeterminate and is not published; a study may
    /// opt into publishing such results, labeled indeterminate, with `MarkIndeterminate`.
    pub fn baseline(id: impl Into<String>) -> Self {
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
            unavailable_evidence: UnavailableEvidencePolicy::Fail,
            exceeded_limit: AccuracyLimitAction::Fail,
            equivalence: NumericalEquivalencePolicy::RelativeTolerance(1.0e-10),
            resources: AccuracyResourceCeiling {
                maximum_refinement_cycles: 0,
                maximum_solver_tightenings: 0,
                maximum_samples: 1_000_000,
                maximum_wall_time_milliseconds: 3_600_000,
                // No authored limit by default: resident memory is execution policy, limited at
                // placement by the machine-capacity authority. An authored ceiling stays in the definition.
                maximum_resident_bytes: None,
            },
        }
    }

    pub fn modal_response_spectrum(id: impl Into<String>) -> Self {
        let mut plan = Self::baseline(id);
        plan.criteria = vec![AccuracyCriterion {
            criterion_id: "modal_mass_capture".into(),
            target: AccuracyTarget::NumericalResult,
            contribution: ErrorContributionKind::ModalTruncation,
            tolerance: AccuracyTolerance::Relative(0.1),
            evidence: RequiredAccuracyEvidence::ConvergenceStudy,
            mandatory: true,
        }];
        plan
    }

    pub fn random_response(id: impl Into<String>) -> Self {
        let mut plan = Self::baseline(id);
        plan.criteria = vec![AccuracyCriterion {
            criterion_id: "frequency_quadrature".into(),
            target: AccuracyTarget::NumericalResult,
            contribution: ErrorContributionKind::FrequencySampling,
            tolerance: AccuracyTolerance::Relative(1.0e-5),
            evidence: RequiredAccuracyEvidence::ConvergenceStudy,
            mandatory: true,
        }];
        plan.resources.maximum_refinement_cycles = 14;
        plan
    }
}
