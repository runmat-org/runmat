use runmat_analysis_fea::contracts::{
    structural_linear_quantity_for_selector, RandomResponseFieldSelector,
    ResponseSpectrumFieldSelector,
};
use runmat_analysis_fea::solve::dynamics::ResponseSpectrumNodalQuantity;

use super::*;
use crate::builtins::fea::study_values::{from_typed_value, to_typed_value};

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Statistical field selectors have no integer inputs.",
};

macro_rules! response_spectrum_nodal_selector {
    ($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $variant:ident) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: concat!("field = ", $builtin, "()"),
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
                            summary = "Select a typed response-spectrum nodal envelope.",
                            keywords = "fea,structural,response spectrum,field,envelope,typed",
                            descriptor($descriptor),
                            integer_audit(INTEGER_AUDIT),
                            builtin_path = "crate::builtins::fea::statistical_field_selectors"
                        )]
        pub async fn $function() -> BuiltinResult<Value> {
            to_typed_value(
                $builtin,
                field_selectors::RESPONSE_SPECTRUM_FIELD_CLASS,
                ResponseSpectrumFieldSelector::Nodal(ResponseSpectrumNodalQuantity::$variant),
            )
        }
    };
}

response_spectrum_nodal_selector!(
    response_spectrum_relative_displacement,
    RESPONSE_SPECTRUM_RELATIVE_DISPLACEMENT_DESCRIPTOR,
    RESPONSE_SPECTRUM_RELATIVE_DISPLACEMENT_SIGNATURES,
    "fea.ResponseSpectrumField.relativeDisplacement",
    RelativeDisplacement
);
response_spectrum_nodal_selector!(
    response_spectrum_rotation,
    RESPONSE_SPECTRUM_ROTATION_DESCRIPTOR,
    RESPONSE_SPECTRUM_ROTATION_SIGNATURES,
    "fea.ResponseSpectrumField.rotation",
    Rotation
);
response_spectrum_nodal_selector!(
    response_spectrum_relative_velocity,
    RESPONSE_SPECTRUM_RELATIVE_VELOCITY_DESCRIPTOR,
    RESPONSE_SPECTRUM_RELATIVE_VELOCITY_SIGNATURES,
    "fea.ResponseSpectrumField.relativeVelocity",
    RelativeVelocity
);
response_spectrum_nodal_selector!(
    response_spectrum_relative_angular_velocity,
    RESPONSE_SPECTRUM_RELATIVE_ANGULAR_VELOCITY_DESCRIPTOR,
    RESPONSE_SPECTRUM_RELATIVE_ANGULAR_VELOCITY_SIGNATURES,
    "fea.ResponseSpectrumField.relativeAngularVelocity",
    RelativeAngularVelocity
);
response_spectrum_nodal_selector!(
    response_spectrum_relative_acceleration,
    RESPONSE_SPECTRUM_RELATIVE_ACCELERATION_DESCRIPTOR,
    RESPONSE_SPECTRUM_RELATIVE_ACCELERATION_SIGNATURES,
    "fea.ResponseSpectrumField.relativeAcceleration",
    RelativeAcceleration
);
response_spectrum_nodal_selector!(
    response_spectrum_relative_angular_acceleration,
    RESPONSE_SPECTRUM_RELATIVE_ANGULAR_ACCELERATION_DESCRIPTOR,
    RESPONSE_SPECTRUM_RELATIVE_ANGULAR_ACCELERATION_SIGNATURES,
    "fea.ResponseSpectrumField.relativeAngularAcceleration",
    RelativeAngularAcceleration
);
response_spectrum_nodal_selector!(
    response_spectrum_absolute_acceleration,
    RESPONSE_SPECTRUM_ABSOLUTE_ACCELERATION_DESCRIPTOR,
    RESPONSE_SPECTRUM_ABSOLUTE_ACCELERATION_SIGNATURES,
    "fea.ResponseSpectrumField.absoluteAcceleration",
    AbsoluteAcceleration
);
response_spectrum_nodal_selector!(
    response_spectrum_absolute_angular_acceleration,
    RESPONSE_SPECTRUM_ABSOLUTE_ANGULAR_ACCELERATION_DESCRIPTOR,
    RESPONSE_SPECTRUM_ABSOLUTE_ANGULAR_ACCELERATION_SIGNATURES,
    "fea.ResponseSpectrumField.absoluteAngularAcceleration",
    AbsoluteAngularAcceleration
);
response_spectrum_nodal_selector!(
    response_spectrum_pseudo_velocity,
    RESPONSE_SPECTRUM_PSEUDO_VELOCITY_DESCRIPTOR,
    RESPONSE_SPECTRUM_PSEUDO_VELOCITY_SIGNATURES,
    "fea.ResponseSpectrumField.pseudoVelocity",
    PseudoVelocity
);
response_spectrum_nodal_selector!(
    response_spectrum_pseudo_angular_velocity,
    RESPONSE_SPECTRUM_PSEUDO_ANGULAR_VELOCITY_DESCRIPTOR,
    RESPONSE_SPECTRUM_PSEUDO_ANGULAR_VELOCITY_SIGNATURES,
    "fea.ResponseSpectrumField.pseudoAngularVelocity",
    PseudoAngularVelocity
);
response_spectrum_nodal_selector!(
    response_spectrum_pseudo_acceleration,
    RESPONSE_SPECTRUM_PSEUDO_ACCELERATION_DESCRIPTOR,
    RESPONSE_SPECTRUM_PSEUDO_ACCELERATION_SIGNATURES,
    "fea.ResponseSpectrumField.pseudoAcceleration",
    PseudoAcceleration
);
response_spectrum_nodal_selector!(
    response_spectrum_pseudo_angular_acceleration,
    RESPONSE_SPECTRUM_PSEUDO_ANGULAR_ACCELERATION_DESCRIPTOR,
    RESPONSE_SPECTRUM_PSEUDO_ANGULAR_ACCELERATION_SIGNATURES,
    "fea.ResponseSpectrumField.pseudoAngularAcceleration",
    PseudoAngularAcceleration
);

const STRUCTURAL_SELECTOR_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "quantity",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A linear fea.StructuralField quantity.",
}];

fn linear_quantity(
    builtin: &'static str,
    selector: Value,
) -> BuiltinResult<runmat_analysis_fea::post::fields::StructuralLinearFieldQuantity> {
    let selector = from_typed_value(builtin, field_selectors::STRUCTURAL_FIELD_CLASS, &selector)?;
    structural_linear_quantity_for_selector(selector).ok_or_else(|| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            "statistical recovery requires displacement, rotation, strain, stress, or a linear shell-resultant selector",
        )
    })
}

const RESPONSE_SPECTRUM_ENVELOPE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "field = fea.ResponseSpectrumField.envelope(quantity)",
        inputs: &STRUCTURAL_SELECTOR_INPUT,
        outputs: &OUT_ANY,
    }];
pub(super) const RESPONSE_SPECTRUM_ENVELOPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RESPONSE_SPECTRUM_ENVELOPE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
#[runtime_builtin(
    name = "fea.ResponseSpectrumField.envelope",
    category = "fea",
    summary = "Select a typed response-spectrum deformation envelope.",
    keywords = "fea,structural,response spectrum,field,envelope,typed",
    descriptor(RESPONSE_SPECTRUM_ENVELOPE_DESCRIPTOR),
    integer_audit(INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::statistical_field_selectors"
)]
pub async fn response_spectrum_envelope(quantity: Value) -> BuiltinResult<Value> {
    let quantity = linear_quantity("fea.ResponseSpectrumField.envelope", quantity)?;
    to_typed_value(
        "fea.ResponseSpectrumField.envelope",
        field_selectors::RESPONSE_SPECTRUM_FIELD_CLASS,
        ResponseSpectrumFieldSelector::Deformation(quantity),
    )
}

macro_rules! random_response_selector {
    ($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $constructor:ident) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: concat!("field = ", $builtin, "(quantity)"),
            inputs: &STRUCTURAL_SELECTOR_INPUT,
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
                            summary = "Select a typed random-response statistical field.",
                            keywords = "fea,structural,random response,field,statistic,typed",
                            descriptor($descriptor),
                            integer_audit(INTEGER_AUDIT),
                            builtin_path = "crate::builtins::fea::statistical_field_selectors"
                        )]
        pub async fn $function(quantity: Value) -> BuiltinResult<Value> {
            let quantity = linear_quantity($builtin, quantity)?;
            to_typed_value(
                $builtin,
                field_selectors::RANDOM_RESPONSE_FIELD_CLASS,
                RandomResponseFieldSelector::$constructor(quantity),
            )
        }
    };
}

random_response_selector!(
    random_response_rms,
    RANDOM_RESPONSE_RMS_DESCRIPTOR,
    RANDOM_RESPONSE_RMS_SIGNATURES,
    "fea.RandomResponseField.rms",
    root_mean_square
);
random_response_selector!(
    random_response_variance,
    RANDOM_RESPONSE_VARIANCE_DESCRIPTOR,
    RANDOM_RESPONSE_VARIANCE_SIGNATURES,
    "fea.RandomResponseField.variance",
    variance
);
