use std::sync::Once;

use runmat_analysis_core::{
    AccuracyLimitAction, ErrorContributionKind, RequiredAccuracyEvidence, UnavailableEvidencePolicy,
};

use super::super::*;
use crate::builtins::fea::study_values::{
    from_typed_value, register_immutable_runtime_classes, register_immutable_schemas,
    to_typed_value, ImmutableStudyClass, VALUE_FIELD,
};
use runmat_builtins::fea_contract::policy_types::classes;

pub(super) const CONTRIBUTION_CLASS: &str = classes::ERROR_CONTRIBUTION.display_name();
pub(super) const EVIDENCE_CLASS: &str = classes::REQUIRED_ACCURACY_EVIDENCE.display_name();
pub(super) const UNAVAILABLE_CLASS: &str = classes::UNAVAILABLE_EVIDENCE_POLICY.display_name();
pub(super) const LIMIT_ACTION_CLASS: &str = classes::ACCURACY_LIMIT_ACTION.display_name();

macro_rules! choice {
    ($function:ident, $builtin:literal, $class:expr, $value:expr $(,)?) => {
        #[runtime_builtin(
                                    name = $builtin,
                                    binding_variant = "default",
                                    builtin_path = "crate::builtins::fea::study_policies::accuracy"
                                )]
        pub async fn $function() -> BuiltinResult<Value> {
            to_typed_value($builtin, $class, $value)
        }
    };
}

macro_rules! choices {
    ($class:expr, $(($function:ident, $builtin:literal, $value:expr)),+ $(,)?) => {
        $(choice!($function, $builtin, $class, $value);)+
    };
}

choices!(
    CONTRIBUTION_CLASS,
    (
        contribution_geometry,
        "fea.ErrorContribution.geometryAdmission",
        ErrorContributionKind::GeometryAdmission
    ),
    (
        contribution_boundary,
        "fea.ErrorContribution.boundaryApproximation",
        ErrorContributionKind::BoundaryApproximation
    ),
    (
        contribution_spatial,
        "fea.ErrorContribution.spatialDiscretization",
        ErrorContributionKind::SpatialDiscretization
    ),
    (
        contribution_algebraic,
        "fea.ErrorContribution.algebraic",
        ErrorContributionKind::Algebraic
    ),
    (
        contribution_nonlinear,
        "fea.ErrorContribution.nonlinear",
        ErrorContributionKind::Nonlinear
    ),
    (
        contribution_temporal,
        "fea.ErrorContribution.temporal",
        ErrorContributionKind::Temporal
    ),
    (
        contribution_frequency,
        "fea.ErrorContribution.frequencySampling",
        ErrorContributionKind::FrequencySampling
    ),
    (
        contribution_modal,
        "fea.ErrorContribution.modalTruncation",
        ErrorContributionKind::ModalTruncation
    ),
    (
        contribution_stochastic,
        "fea.ErrorContribution.stochastic",
        ErrorContributionKind::Stochastic
    ),
    (
        contribution_coupling,
        "fea.ErrorContribution.coupling",
        ErrorContributionKind::Coupling
    ),
    (
        contribution_interface,
        "fea.ErrorContribution.interfaceProjection",
        ErrorContributionKind::InterfaceProjection
    ),
    (
        contribution_transfer,
        "fea.ErrorContribution.fieldTransfer",
        ErrorContributionKind::FieldTransfer
    ),
    (
        contribution_signal,
        "fea.ErrorContribution.signalInterpolation",
        ErrorContributionKind::SignalInterpolation
    ),
    (
        contribution_recovery,
        "fea.ErrorContribution.recovery",
        ErrorContributionKind::Recovery
    ),
    (
        contribution_observation,
        "fea.ErrorContribution.observationSampling",
        ErrorContributionKind::ObservationSampling
    ),
    (
        contribution_precision,
        "fea.ErrorContribution.precision",
        ErrorContributionKind::Precision
    ),
    (
        contribution_equivalence,
        "fea.ErrorContribution.executionEquivalence",
        ErrorContributionKind::ExecutionEquivalence
    ),
);
choices!(
    EVIDENCE_CLASS,
    (
        evidence_estimator,
        "fea.RequiredAccuracyEvidence.aPosterioriEstimator",
        RequiredAccuracyEvidence::APosterioriEstimator
    ),
    (
        evidence_convergence,
        "fea.RequiredAccuracyEvidence.convergenceStudy",
        RequiredAccuracyEvidence::ConvergenceStudy
    ),
    (
        evidence_conservation,
        "fea.RequiredAccuracyEvidence.conservationBalance",
        RequiredAccuracyEvidence::ConservationBalance
    ),
    (
        evidence_residual,
        "fea.RequiredAccuracyEvidence.residualBound",
        RequiredAccuracyEvidence::ResidualBound
    ),
    (
        evidence_reference,
        "fea.RequiredAccuracyEvidence.referenceComparison",
        RequiredAccuracyEvidence::ReferenceComparison
    ),
    (
        evidence_statistical,
        "fea.RequiredAccuracyEvidence.statisticalConfidence",
        RequiredAccuracyEvidence::StatisticalConfidence
    ),
    (
        evidence_transfer,
        "fea.RequiredAccuracyEvidence.transferBound",
        RequiredAccuracyEvidence::TransferBound
    ),
);
choices!(
    UNAVAILABLE_CLASS,
    (
        unavailable_fail,
        "fea.UnavailableEvidencePolicy.fail",
        UnavailableEvidencePolicy::Fail
    ),
    (
        unavailable_mark_indeterminate,
        "fea.UnavailableEvidencePolicy.markIndeterminate",
        UnavailableEvidencePolicy::MarkIndeterminate
    ),
);
choices!(
    LIMIT_ACTION_CLASS,
    (
        limit_refine,
        "fea.AccuracyLimitAction.refine",
        AccuracyLimitAction::Refine
    ),
    (
        limit_tighten,
        "fea.AccuracyLimitAction.tightenSolver",
        AccuracyLimitAction::TightenSolver
    ),
    (
        limit_extend,
        "fea.AccuracyLimitAction.extendSampling",
        AccuracyLimitAction::ExtendSampling
    ),
    (
        limit_fail,
        "fea.AccuracyLimitAction.fail",
        AccuracyLimitAction::Fail
    ),
);

pub(crate) fn contribution(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<ErrorContributionKind> {
    from_typed_value(builtin, CONTRIBUTION_CLASS, value)
}
pub(crate) fn evidence(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<RequiredAccuracyEvidence> {
    from_typed_value(builtin, EVIDENCE_CLASS, value)
}
pub(crate) fn unavailable(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<UnavailableEvidencePolicy> {
    from_typed_value(builtin, UNAVAILABLE_CLASS, value)
}
pub(crate) fn limit_action(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<AccuracyLimitAction> {
    from_typed_value(builtin, LIMIT_ACTION_CLASS, value)
}

const ACCURACY_POLICY_CLASSES: [ImmutableStudyClass; 4] = [
    ImmutableStudyClass {
        class_name: CONTRIBUTION_CLASS,
        type_identity: "runmat.fea.error-contribution",
        fields: VALUE_FIELD,
        methods: &CONTRIBUTION_METHODS,
    },
    ImmutableStudyClass {
        class_name: EVIDENCE_CLASS,
        type_identity: "runmat.fea.required-accuracy-evidence",
        fields: VALUE_FIELD,
        methods: &EVIDENCE_METHODS,
    },
    ImmutableStudyClass {
        class_name: UNAVAILABLE_CLASS,
        type_identity: "runmat.fea.unavailable-evidence-policy",
        fields: VALUE_FIELD,
        methods: &[
            ("fail", "fea.UnavailableEvidencePolicy.fail"),
            (
                "markIndeterminate",
                "fea.UnavailableEvidencePolicy.markIndeterminate",
            ),
        ],
    },
    ImmutableStudyClass {
        class_name: LIMIT_ACTION_CLASS,
        type_identity: "runmat.fea.accuracy-limit-action",
        fields: VALUE_FIELD,
        methods: &[
            ("refine", "fea.AccuracyLimitAction.refine"),
            ("tightenSolver", "fea.AccuracyLimitAction.tightenSolver"),
            ("extendSampling", "fea.AccuracyLimitAction.extendSampling"),
            ("fail", "fea.AccuracyLimitAction.fail"),
        ],
    },
];

pub(super) fn ensure_classes_registered() {
    static SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            CONTRIBUTION_CLASS,
        ));
    SCHEMAS.call_once(|| {
        static FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        register_immutable_schemas(&FIELDS, &ACCURACY_POLICY_CLASSES);
    });
    RUNTIME_CLASSES.ensure(|| register_immutable_runtime_classes(&ACCURACY_POLICY_CLASSES));
}

const CONTRIBUTION_METHODS: [(&str, &str); 17] = [
    (
        "geometryAdmission",
        "fea.ErrorContribution.geometryAdmission",
    ),
    (
        "boundaryApproximation",
        "fea.ErrorContribution.boundaryApproximation",
    ),
    (
        "spatialDiscretization",
        "fea.ErrorContribution.spatialDiscretization",
    ),
    ("algebraic", "fea.ErrorContribution.algebraic"),
    ("nonlinear", "fea.ErrorContribution.nonlinear"),
    ("temporal", "fea.ErrorContribution.temporal"),
    (
        "frequencySampling",
        "fea.ErrorContribution.frequencySampling",
    ),
    ("modalTruncation", "fea.ErrorContribution.modalTruncation"),
    ("stochastic", "fea.ErrorContribution.stochastic"),
    ("coupling", "fea.ErrorContribution.coupling"),
    (
        "interfaceProjection",
        "fea.ErrorContribution.interfaceProjection",
    ),
    ("fieldTransfer", "fea.ErrorContribution.fieldTransfer"),
    (
        "signalInterpolation",
        "fea.ErrorContribution.signalInterpolation",
    ),
    ("recovery", "fea.ErrorContribution.recovery"),
    (
        "observationSampling",
        "fea.ErrorContribution.observationSampling",
    ),
    ("precision", "fea.ErrorContribution.precision"),
    (
        "executionEquivalence",
        "fea.ErrorContribution.executionEquivalence",
    ),
];
const EVIDENCE_METHODS: [(&str, &str); 7] = [
    (
        "aPosterioriEstimator",
        "fea.RequiredAccuracyEvidence.aPosterioriEstimator",
    ),
    (
        "convergenceStudy",
        "fea.RequiredAccuracyEvidence.convergenceStudy",
    ),
    (
        "conservationBalance",
        "fea.RequiredAccuracyEvidence.conservationBalance",
    ),
    (
        "residualBound",
        "fea.RequiredAccuracyEvidence.residualBound",
    ),
    (
        "referenceComparison",
        "fea.RequiredAccuracyEvidence.referenceComparison",
    ),
    (
        "statisticalConfidence",
        "fea.RequiredAccuracyEvidence.statisticalConfidence",
    ),
    (
        "transferBound",
        "fea.RequiredAccuracyEvidence.transferBound",
    ),
];

#[cfg(test)]
mod tests;
