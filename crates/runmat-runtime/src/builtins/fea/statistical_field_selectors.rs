use runmat_analysis_fea::contracts::{
    structural_linear_quantity_for_selector, RandomResponseFieldSelector,
    ResponseSpectrumFieldSelector,
};
use runmat_analysis_fea::solve::dynamics::ResponseSpectrumNodalQuantity;

use super::*;
use crate::builtins::fea::study_values::{from_typed_value, to_typed_value};

macro_rules! response_spectrum_nodal_selector {
    ($function:ident, $builtin:literal, $variant:ident) => {
        #[runtime_builtin(
                            name = $builtin,
                            binding_variant = "default",
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
    "fea.ResponseSpectrumField.relativeDisplacement",
    RelativeDisplacement
);
response_spectrum_nodal_selector!(
    response_spectrum_rotation,
    "fea.ResponseSpectrumField.rotation",
    Rotation
);
response_spectrum_nodal_selector!(
    response_spectrum_relative_velocity,
    "fea.ResponseSpectrumField.relativeVelocity",
    RelativeVelocity
);
response_spectrum_nodal_selector!(
    response_spectrum_relative_angular_velocity,
    "fea.ResponseSpectrumField.relativeAngularVelocity",
    RelativeAngularVelocity
);
response_spectrum_nodal_selector!(
    response_spectrum_relative_acceleration,
    "fea.ResponseSpectrumField.relativeAcceleration",
    RelativeAcceleration
);
response_spectrum_nodal_selector!(
    response_spectrum_relative_angular_acceleration,
    "fea.ResponseSpectrumField.relativeAngularAcceleration",
    RelativeAngularAcceleration
);
response_spectrum_nodal_selector!(
    response_spectrum_absolute_acceleration,
    "fea.ResponseSpectrumField.absoluteAcceleration",
    AbsoluteAcceleration
);
response_spectrum_nodal_selector!(
    response_spectrum_absolute_angular_acceleration,
    "fea.ResponseSpectrumField.absoluteAngularAcceleration",
    AbsoluteAngularAcceleration
);
response_spectrum_nodal_selector!(
    response_spectrum_pseudo_velocity,
    "fea.ResponseSpectrumField.pseudoVelocity",
    PseudoVelocity
);
response_spectrum_nodal_selector!(
    response_spectrum_pseudo_angular_velocity,
    "fea.ResponseSpectrumField.pseudoAngularVelocity",
    PseudoAngularVelocity
);
response_spectrum_nodal_selector!(
    response_spectrum_pseudo_acceleration,
    "fea.ResponseSpectrumField.pseudoAcceleration",
    PseudoAcceleration
);
response_spectrum_nodal_selector!(
    response_spectrum_pseudo_angular_acceleration,
    "fea.ResponseSpectrumField.pseudoAngularAcceleration",
    PseudoAngularAcceleration
);

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

#[runtime_builtin(
    name = "fea.ResponseSpectrumField.envelope",
    binding_variant = "default",
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
    ($function:ident, $builtin:literal, $constructor:ident) => {
        #[runtime_builtin(
                            name = $builtin,
                            binding_variant = "default",
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
    "fea.RandomResponseField.rms",
    root_mean_square
);
random_response_selector!(
    random_response_variance,
    "fea.RandomResponseField.variance",
    variance
);
