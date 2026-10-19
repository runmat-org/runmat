use std::sync::Once;

use runmat_analysis_core::{
    AccuracyLimitAction, ErrorContributionKind, RequiredAccuracyEvidence, UnavailableEvidencePolicy,
};

use super::super::*;
use crate::builtins::fea::study_values::{
    from_typed_value, register_immutable_class, to_typed_value, VALUE_FIELD,
};

pub(super) const CONTRIBUTION_CLASS: &str = "fea.ErrorContribution";
pub(super) const EVIDENCE_CLASS: &str = "fea.RequiredAccuracyEvidence";
pub(super) const UNAVAILABLE_CLASS: &str = "fea.UnavailableEvidencePolicy";
pub(super) const LIMIT_ACTION_CLASS: &str = "fea.AccuracyLimitAction";

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const CHOICE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Closed FEA accuracy-policy choices have no numeric inputs.",
};

macro_rules! choice {
    ($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $class:expr, $value:expr) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: concat!("value = ", $builtin, "()"),
            inputs: &NO_INPUTS,
            outputs: &OUT_ANY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
        #[runtime_builtin(
                                    name = $builtin,
                                    category = "fea",
                                    summary = "Select a typed FEA accuracy policy.",
                                    keywords = "fea,accuracy,evidence,acceptance,typed",
                                    descriptor($descriptor),
                                    integer_audit(CHOICE_INTEGER_AUDIT),
                                    builtin_path = "crate::builtins::fea::study_policies::accuracy"
                                )]
        pub async fn $function() -> BuiltinResult<Value> {
            to_typed_value($builtin, $class, $value)
        }
    };
}

macro_rules! choices {
    ($class:expr, $(($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $value:expr)),+ $(,)?) => {
        $(choice!($function, $descriptor, $signatures, $builtin, $class, $value);)+
    };
}

choices!(
    CONTRIBUTION_CLASS,
    (
        contribution_geometry,
        CONTRIBUTION_GEOMETRY_DESCRIPTOR,
        CONTRIBUTION_GEOMETRY_SIGNATURES,
        "fea.ErrorContribution.geometryAdmission",
        ErrorContributionKind::GeometryAdmission
    ),
    (
        contribution_boundary,
        CONTRIBUTION_BOUNDARY_DESCRIPTOR,
        CONTRIBUTION_BOUNDARY_SIGNATURES,
        "fea.ErrorContribution.boundaryApproximation",
        ErrorContributionKind::BoundaryApproximation
    ),
    (
        contribution_spatial,
        CONTRIBUTION_SPATIAL_DESCRIPTOR,
        CONTRIBUTION_SPATIAL_SIGNATURES,
        "fea.ErrorContribution.spatialDiscretization",
        ErrorContributionKind::SpatialDiscretization
    ),
    (
        contribution_algebraic,
        CONTRIBUTION_ALGEBRAIC_DESCRIPTOR,
        CONTRIBUTION_ALGEBRAIC_SIGNATURES,
        "fea.ErrorContribution.algebraic",
        ErrorContributionKind::Algebraic
    ),
    (
        contribution_nonlinear,
        CONTRIBUTION_NONLINEAR_DESCRIPTOR,
        CONTRIBUTION_NONLINEAR_SIGNATURES,
        "fea.ErrorContribution.nonlinear",
        ErrorContributionKind::Nonlinear
    ),
    (
        contribution_temporal,
        CONTRIBUTION_TEMPORAL_DESCRIPTOR,
        CONTRIBUTION_TEMPORAL_SIGNATURES,
        "fea.ErrorContribution.temporal",
        ErrorContributionKind::Temporal
    ),
    (
        contribution_frequency,
        CONTRIBUTION_FREQUENCY_DESCRIPTOR,
        CONTRIBUTION_FREQUENCY_SIGNATURES,
        "fea.ErrorContribution.frequencySampling",
        ErrorContributionKind::FrequencySampling
    ),
    (
        contribution_modal,
        CONTRIBUTION_MODAL_DESCRIPTOR,
        CONTRIBUTION_MODAL_SIGNATURES,
        "fea.ErrorContribution.modalTruncation",
        ErrorContributionKind::ModalTruncation
    ),
    (
        contribution_stochastic,
        CONTRIBUTION_STOCHASTIC_DESCRIPTOR,
        CONTRIBUTION_STOCHASTIC_SIGNATURES,
        "fea.ErrorContribution.stochastic",
        ErrorContributionKind::Stochastic
    ),
    (
        contribution_coupling,
        CONTRIBUTION_COUPLING_DESCRIPTOR,
        CONTRIBUTION_COUPLING_SIGNATURES,
        "fea.ErrorContribution.coupling",
        ErrorContributionKind::Coupling
    ),
    (
        contribution_interface,
        CONTRIBUTION_INTERFACE_DESCRIPTOR,
        CONTRIBUTION_INTERFACE_SIGNATURES,
        "fea.ErrorContribution.interfaceProjection",
        ErrorContributionKind::InterfaceProjection
    ),
    (
        contribution_transfer,
        CONTRIBUTION_TRANSFER_DESCRIPTOR,
        CONTRIBUTION_TRANSFER_SIGNATURES,
        "fea.ErrorContribution.fieldTransfer",
        ErrorContributionKind::FieldTransfer
    ),
    (
        contribution_signal,
        CONTRIBUTION_SIGNAL_DESCRIPTOR,
        CONTRIBUTION_SIGNAL_SIGNATURES,
        "fea.ErrorContribution.signalInterpolation",
        ErrorContributionKind::SignalInterpolation
    ),
    (
        contribution_recovery,
        CONTRIBUTION_RECOVERY_DESCRIPTOR,
        CONTRIBUTION_RECOVERY_SIGNATURES,
        "fea.ErrorContribution.recovery",
        ErrorContributionKind::Recovery
    ),
    (
        contribution_observation,
        CONTRIBUTION_OBSERVATION_DESCRIPTOR,
        CONTRIBUTION_OBSERVATION_SIGNATURES,
        "fea.ErrorContribution.observationSampling",
        ErrorContributionKind::ObservationSampling
    ),
    (
        contribution_precision,
        CONTRIBUTION_PRECISION_DESCRIPTOR,
        CONTRIBUTION_PRECISION_SIGNATURES,
        "fea.ErrorContribution.precision",
        ErrorContributionKind::Precision
    ),
    (
        contribution_equivalence,
        CONTRIBUTION_EQUIVALENCE_DESCRIPTOR,
        CONTRIBUTION_EQUIVALENCE_SIGNATURES,
        "fea.ErrorContribution.executionEquivalence",
        ErrorContributionKind::ExecutionEquivalence
    )
);
choices!(
    EVIDENCE_CLASS,
    (
        evidence_estimator,
        EVIDENCE_ESTIMATOR_DESCRIPTOR,
        EVIDENCE_ESTIMATOR_SIGNATURES,
        "fea.RequiredAccuracyEvidence.aPosterioriEstimator",
        RequiredAccuracyEvidence::APosterioriEstimator
    ),
    (
        evidence_convergence,
        EVIDENCE_CONVERGENCE_DESCRIPTOR,
        EVIDENCE_CONVERGENCE_SIGNATURES,
        "fea.RequiredAccuracyEvidence.convergenceStudy",
        RequiredAccuracyEvidence::ConvergenceStudy
    ),
    (
        evidence_conservation,
        EVIDENCE_CONSERVATION_DESCRIPTOR,
        EVIDENCE_CONSERVATION_SIGNATURES,
        "fea.RequiredAccuracyEvidence.conservationBalance",
        RequiredAccuracyEvidence::ConservationBalance
    ),
    (
        evidence_residual,
        EVIDENCE_RESIDUAL_DESCRIPTOR,
        EVIDENCE_RESIDUAL_SIGNATURES,
        "fea.RequiredAccuracyEvidence.residualBound",
        RequiredAccuracyEvidence::ResidualBound
    ),
    (
        evidence_reference,
        EVIDENCE_REFERENCE_DESCRIPTOR,
        EVIDENCE_REFERENCE_SIGNATURES,
        "fea.RequiredAccuracyEvidence.referenceComparison",
        RequiredAccuracyEvidence::ReferenceComparison
    ),
    (
        evidence_statistical,
        EVIDENCE_STATISTICAL_DESCRIPTOR,
        EVIDENCE_STATISTICAL_SIGNATURES,
        "fea.RequiredAccuracyEvidence.statisticalConfidence",
        RequiredAccuracyEvidence::StatisticalConfidence
    ),
    (
        evidence_transfer,
        EVIDENCE_TRANSFER_DESCRIPTOR,
        EVIDENCE_TRANSFER_SIGNATURES,
        "fea.RequiredAccuracyEvidence.transferBound",
        RequiredAccuracyEvidence::TransferBound
    )
);
choices!(
    UNAVAILABLE_CLASS,
    (
        unavailable_fail,
        UNAVAILABLE_FAIL_DESCRIPTOR,
        UNAVAILABLE_FAIL_SIGNATURES,
        "fea.UnavailableEvidencePolicy.fail",
        UnavailableEvidencePolicy::Fail
    ),
    (
        unavailable_experimental,
        UNAVAILABLE_EXPERIMENTAL_DESCRIPTOR,
        UNAVAILABLE_EXPERIMENTAL_SIGNATURES,
        "fea.UnavailableEvidencePolicy.markIndeterminateExperimental",
        UnavailableEvidencePolicy::MarkIndeterminateExperimental
    )
);
choices!(
    LIMIT_ACTION_CLASS,
    (
        limit_refine,
        LIMIT_REFINE_DESCRIPTOR,
        LIMIT_REFINE_SIGNATURES,
        "fea.AccuracyLimitAction.refine",
        AccuracyLimitAction::Refine
    ),
    (
        limit_tighten,
        LIMIT_TIGHTEN_DESCRIPTOR,
        LIMIT_TIGHTEN_SIGNATURES,
        "fea.AccuracyLimitAction.tightenSolver",
        AccuracyLimitAction::TightenSolver
    ),
    (
        limit_extend,
        LIMIT_EXTEND_DESCRIPTOR,
        LIMIT_EXTEND_SIGNATURES,
        "fea.AccuracyLimitAction.extendSampling",
        AccuracyLimitAction::ExtendSampling
    ),
    (
        limit_fail,
        LIMIT_FAIL_DESCRIPTOR,
        LIMIT_FAIL_SIGNATURES,
        "fea.AccuracyLimitAction.fail",
        AccuracyLimitAction::Fail
    ),
    (
        limit_warn,
        LIMIT_WARN_DESCRIPTOR,
        LIMIT_WARN_SIGNATURES,
        "fea.AccuracyLimitAction.warnExperimental",
        AccuracyLimitAction::WarnExperimental
    )
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

pub(super) fn ensure_classes_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        register_immutable_class(
            CONTRIBUTION_CLASS,
            "runmat.fea.error-contribution",
            VALUE_FIELD,
            &CONTRIBUTION_METHODS,
        );
        register_immutable_class(
            EVIDENCE_CLASS,
            "runmat.fea.required-accuracy-evidence",
            VALUE_FIELD,
            &EVIDENCE_METHODS,
        );
        register_immutable_class(
            UNAVAILABLE_CLASS,
            "runmat.fea.unavailable-evidence-policy",
            VALUE_FIELD,
            &[
                ("fail", "fea.UnavailableEvidencePolicy.fail"),
                (
                    "markIndeterminateExperimental",
                    "fea.UnavailableEvidencePolicy.markIndeterminateExperimental",
                ),
            ],
        );
        register_immutable_class(
            LIMIT_ACTION_CLASS,
            "runmat.fea.accuracy-limit-action",
            VALUE_FIELD,
            &[
                ("refine", "fea.AccuracyLimitAction.refine"),
                ("tightenSolver", "fea.AccuracyLimitAction.tightenSolver"),
                ("extendSampling", "fea.AccuracyLimitAction.extendSampling"),
                ("fail", "fea.AccuracyLimitAction.fail"),
                (
                    "warnExperimental",
                    "fea.AccuracyLimitAction.warnExperimental",
                ),
            ],
        );
    });
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
