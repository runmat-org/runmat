use runmat_analysis_fea::contracts::{
    RandomResponseFieldSelector, ResponseSpectrumFieldSelector, StructuralFieldSelector,
};

use super::*;
use crate::builtins::fea::study_values::{from_typed_value, to_typed_value};
use runmat_builtins::fea_contract::identities;

pub(super) const STRUCTURAL_FIELD_CLASS: &str =
    identities::FEA_STRUCTURAL_FIELD_CLASS.display_name();
pub(super) const RESPONSE_SPECTRUM_FIELD_CLASS: &str =
    identities::FEA_RESPONSE_SPECTRUM_FIELD_CLASS.display_name();
pub(super) const RANDOM_RESPONSE_FIELD_CLASS: &str =
    identities::FEA_RANDOM_RESPONSE_FIELD_CLASS.display_name();
pub(super) const OUTPUT_FIELD_CLASS: &str = identities::FEA_OUTPUT_FIELD_CLASS.display_name();

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub(super) struct OutputFieldSelector(String);

#[derive(Debug, PartialEq)]
pub(super) enum FieldSelector {
    Structural(StructuralFieldSelector),
    ResponseSpectrum(ResponseSpectrumFieldSelector),
    RandomResponse(RandomResponseFieldSelector),
    Output(OutputFieldSelector),
}

impl FieldSelector {
    pub(super) fn field_id(&self) -> String {
        match self {
            Self::Structural(selector) => selector.field_id().to_string(),
            Self::ResponseSpectrum(selector) => selector.field_id(),
            Self::RandomResponse(selector) => selector.field_id(),
            Self::Output(selector) => selector.0.clone(),
        }
    }
}

#[runtime_builtin(
    name = "fea.OutputField.named",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::field_selectors"
)]
pub async fn output_named(name: Value) -> BuiltinResult<Value> {
    const BUILTIN: &str = "fea.OutputField.named";
    let name = scalar_string(&name, BUILTIN, &ERROR_INPUT)?;
    let mut characters = name.chars();
    let valid = matches!(characters.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric());
    if !valid || name.len() > 512 {
        return Err(builtin_error(
            BUILTIN,
            &ERROR_INPUT,
            "authored output name must be a valid FEA identifier",
        ));
    }
    to_typed_value(BUILTIN, OUTPUT_FIELD_CLASS, OutputFieldSelector(name))
}

macro_rules! selector {
    ($function:ident, $builtin:literal, $variant:ident) => {
        #[runtime_builtin(
                                            name = $builtin,
                                            binding_variant = "default",
                                            builtin_path = "crate::builtins::fea::field_selectors"
                                        )]
        pub async fn $function() -> BuiltinResult<Value> {
            to_typed_value(
                $builtin,
                STRUCTURAL_FIELD_CLASS,
                StructuralFieldSelector::$variant,
            )
        }
    };
}

selector!(
    displacement,
    "fea.StructuralField.displacement",
    Displacement
);
selector!(rotation, "fea.StructuralField.rotation", Rotation);
selector!(von_mises, "fea.StructuralField.vonMises", VonMises);
selector!(
    nodal_von_mises,
    "fea.StructuralField.nodalVonMises",
    NodalVonMises
);
selector!(strain, "fea.StructuralField.strain", Strain);
selector!(stress, "fea.StructuralField.stress", Stress);
selector!(
    integration_point_strain,
    "fea.StructuralField.integrationPointStrain",
    IntegrationPointStrain
);
selector!(
    integration_point_stress,
    "fea.StructuralField.integrationPointStress",
    IntegrationPointStress
);
selector!(
    reaction_force,
    "fea.StructuralField.reactionForce",
    ReactionForce
);
selector!(
    reaction_moment,
    "fea.StructuralField.reactionMoment",
    ReactionMoment
);
selector!(
    constraint_reaction_force,
    "fea.StructuralField.constraintReactionForce",
    ConstraintReactionForce
);
selector!(
    constraint_reaction_moment,
    "fea.StructuralField.constraintReactionMomentAboutGlobalOrigin",
    ConstraintReactionMomentAboutGlobalOrigin
);
selector!(
    strain_energy_density,
    "fea.StructuralField.strainEnergyDensity",
    StrainEnergyDensity
);
selector!(
    total_strain_energy,
    "fea.StructuralField.totalStrainEnergy",
    TotalStrainEnergy
);
selector!(
    residual_norm,
    "fea.StructuralField.residualNorm",
    ResidualNorm
);
selector!(
    equation_scale,
    "fea.StructuralField.equationScale",
    EquationScale
);
selector!(
    beam_axial_force,
    "fea.StructuralField.beamAxialForce",
    BeamAxialForce
);
selector!(
    beam_shear_force,
    "fea.StructuralField.beamShearForce",
    BeamShearForce
);
selector!(
    beam_torsion_moment,
    "fea.StructuralField.beamTorsionMoment",
    BeamTorsionMoment
);
selector!(
    beam_bending_moment,
    "fea.StructuralField.beamBendingMoment",
    BeamBendingMoment
);
selector!(
    beam_bending_stress,
    "fea.StructuralField.beamBendingStress",
    BeamBendingStress
);
selector!(
    beam_torsion_stress,
    "fea.StructuralField.beamTorsionStress",
    BeamTorsionStress
);
selector!(
    shell_membrane_force,
    "fea.StructuralField.shellMembraneForce",
    ShellMembraneForce
);
selector!(
    shell_bending_moment,
    "fea.StructuralField.shellBendingMoment",
    ShellBendingMoment
);
selector!(
    shell_transverse_shear,
    "fea.StructuralField.shellTransverseShear",
    ShellTransverseShear
);
selector!(
    shell_von_mises,
    "fea.StructuralField.shellVonMises",
    ShellVonMises
);

pub(super) fn from_value(builtin: &'static str, value: &Value) -> BuiltinResult<FieldSelector> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "field selector must be a typed FEA field selector",
        ));
    };
    match object.class_name().display_name() {
        STRUCTURAL_FIELD_CLASS => from_typed_value(builtin, STRUCTURAL_FIELD_CLASS, value)
            .map(FieldSelector::Structural),
        RESPONSE_SPECTRUM_FIELD_CLASS => {
            from_typed_value(builtin, RESPONSE_SPECTRUM_FIELD_CLASS, value)
                .map(FieldSelector::ResponseSpectrum)
        }
        RANDOM_RESPONSE_FIELD_CLASS => {
            from_typed_value(builtin, RANDOM_RESPONSE_FIELD_CLASS, value)
                .map(FieldSelector::RandomResponse)
        }
        OUTPUT_FIELD_CLASS => from_typed_value(builtin, OUTPUT_FIELD_CLASS, value)
            .map(FieldSelector::Output),
        _ => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "field selector must be a fea.StructuralField, fea.ResponseSpectrumField, fea.RandomResponseField, or fea.OutputField value",
        )),
    }
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;

    use super::*;

    #[test]
    fn authored_output_field_selectors_are_typed_named_data() {
        super::super::ensure_value_classes_registered();
        let selector = block_on(output_named(Value::String("peak_stress".into())))
            .expect("typed authored output field");
        let Value::Object(object) = &selector else {
            panic!("expected authored output field selector object");
        };
        assert_eq!(object.class_name().display_name(), OUTPUT_FIELD_CLASS);
        assert_eq!(
            from_value("test", &selector).unwrap().field_id(),
            "peak_stress"
        );
        let encoded = crate::execution::value_codec::encode_inline_value(&selector)
            .expect("authored output selector should be portable");
        assert_eq!(
            crate::execution::value_codec::decode_inline_value(&encoded)
                .expect("portable authored output selector should decode"),
            selector
        );
        assert!(block_on(output_named(Value::String("not valid".into()))).is_err());
    }
}
