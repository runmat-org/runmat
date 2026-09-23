use runmat_analysis_fea::contracts::{
    RandomResponseFieldSelector, ResponseSpectrumFieldSelector, StructuralFieldSelector,
};

use super::*;
use crate::builtins::fea::study_values::{from_typed_value, to_typed_value};

pub(super) const STRUCTURAL_FIELD_CLASS: &str = "fea.StructuralField";
pub(super) const RESPONSE_SPECTRUM_FIELD_CLASS: &str = "fea.ResponseSpectrumField";
pub(super) const RANDOM_RESPONSE_FIELD_CLASS: &str = "fea.RandomResponseField";
pub(super) const OUTPUT_FIELD_CLASS: &str = "fea.OutputField";

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

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Structural field selectors have no integer inputs.",
};

const OUTPUT_NAME_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Authored output identifier declared by the FEA study.",
}];
const OUTPUT_NAMED_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.OutputField.named(name)",
    inputs: &OUTPUT_NAME_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const OUTPUT_NAMED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OUTPUT_NAMED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.OutputField.named",
    category = "fea",
    summary = "Select an authored FEA output by its declared identifier.",
    keywords = "fea,output,field,result,typed",
    descriptor(OUTPUT_NAMED_DESCRIPTOR),
    integer_audit(INTEGER_AUDIT),
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
                                            summary = "Select a typed structural result field.",
                                            keywords = "fea,structural,field,result,typed",
                                            descriptor($descriptor),
                                            integer_audit(INTEGER_AUDIT),
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
    DISPLACEMENT_DESCRIPTOR,
    DISPLACEMENT_SIGNATURES,
    "fea.StructuralField.displacement",
    Displacement
);
selector!(
    rotation,
    ROTATION_DESCRIPTOR,
    ROTATION_SIGNATURES,
    "fea.StructuralField.rotation",
    Rotation
);
selector!(
    von_mises,
    VON_MISES_DESCRIPTOR,
    VON_MISES_SIGNATURES,
    "fea.StructuralField.vonMises",
    VonMises
);
selector!(
    nodal_von_mises,
    NODAL_VON_MISES_DESCRIPTOR,
    NODAL_VON_MISES_SIGNATURES,
    "fea.StructuralField.nodalVonMises",
    NodalVonMises
);
selector!(
    strain,
    STRAIN_DESCRIPTOR,
    STRAIN_SIGNATURES,
    "fea.StructuralField.strain",
    Strain
);
selector!(
    stress,
    STRESS_DESCRIPTOR,
    STRESS_SIGNATURES,
    "fea.StructuralField.stress",
    Stress
);
selector!(
    integration_point_strain,
    INTEGRATION_POINT_STRAIN_DESCRIPTOR,
    INTEGRATION_POINT_STRAIN_SIGNATURES,
    "fea.StructuralField.integrationPointStrain",
    IntegrationPointStrain
);
selector!(
    integration_point_stress,
    INTEGRATION_POINT_STRESS_DESCRIPTOR,
    INTEGRATION_POINT_STRESS_SIGNATURES,
    "fea.StructuralField.integrationPointStress",
    IntegrationPointStress
);
selector!(
    reaction_force,
    REACTION_FORCE_DESCRIPTOR,
    REACTION_FORCE_SIGNATURES,
    "fea.StructuralField.reactionForce",
    ReactionForce
);
selector!(
    reaction_moment,
    REACTION_MOMENT_DESCRIPTOR,
    REACTION_MOMENT_SIGNATURES,
    "fea.StructuralField.reactionMoment",
    ReactionMoment
);
selector!(
    constraint_reaction_force,
    CONSTRAINT_REACTION_FORCE_DESCRIPTOR,
    CONSTRAINT_REACTION_FORCE_SIGNATURES,
    "fea.StructuralField.constraintReactionForce",
    ConstraintReactionForce
);
selector!(
    constraint_reaction_moment,
    CONSTRAINT_REACTION_MOMENT_DESCRIPTOR,
    CONSTRAINT_REACTION_MOMENT_SIGNATURES,
    "fea.StructuralField.constraintReactionMomentAboutGlobalOrigin",
    ConstraintReactionMomentAboutGlobalOrigin
);
selector!(
    strain_energy_density,
    STRAIN_ENERGY_DENSITY_DESCRIPTOR,
    STRAIN_ENERGY_DENSITY_SIGNATURES,
    "fea.StructuralField.strainEnergyDensity",
    StrainEnergyDensity
);
selector!(
    total_strain_energy,
    TOTAL_STRAIN_ENERGY_DESCRIPTOR,
    TOTAL_STRAIN_ENERGY_SIGNATURES,
    "fea.StructuralField.totalStrainEnergy",
    TotalStrainEnergy
);
selector!(
    residual_norm,
    RESIDUAL_NORM_DESCRIPTOR,
    RESIDUAL_NORM_SIGNATURES,
    "fea.StructuralField.residualNorm",
    ResidualNorm
);
selector!(
    equation_scale,
    EQUATION_SCALE_DESCRIPTOR,
    EQUATION_SCALE_SIGNATURES,
    "fea.StructuralField.equationScale",
    EquationScale
);
selector!(
    beam_axial_force,
    BEAM_AXIAL_FORCE_DESCRIPTOR,
    BEAM_AXIAL_FORCE_SIGNATURES,
    "fea.StructuralField.beamAxialForce",
    BeamAxialForce
);
selector!(
    beam_shear_force,
    BEAM_SHEAR_FORCE_DESCRIPTOR,
    BEAM_SHEAR_FORCE_SIGNATURES,
    "fea.StructuralField.beamShearForce",
    BeamShearForce
);
selector!(
    beam_torsion_moment,
    BEAM_TORSION_MOMENT_DESCRIPTOR,
    BEAM_TORSION_MOMENT_SIGNATURES,
    "fea.StructuralField.beamTorsionMoment",
    BeamTorsionMoment
);
selector!(
    beam_bending_moment,
    BEAM_BENDING_MOMENT_DESCRIPTOR,
    BEAM_BENDING_MOMENT_SIGNATURES,
    "fea.StructuralField.beamBendingMoment",
    BeamBendingMoment
);
selector!(
    beam_bending_stress,
    BEAM_BENDING_STRESS_DESCRIPTOR,
    BEAM_BENDING_STRESS_SIGNATURES,
    "fea.StructuralField.beamBendingStress",
    BeamBendingStress
);
selector!(
    beam_torsion_stress,
    BEAM_TORSION_STRESS_DESCRIPTOR,
    BEAM_TORSION_STRESS_SIGNATURES,
    "fea.StructuralField.beamTorsionStress",
    BeamTorsionStress
);
selector!(
    shell_membrane_force,
    SHELL_MEMBRANE_FORCE_DESCRIPTOR,
    SHELL_MEMBRANE_FORCE_SIGNATURES,
    "fea.StructuralField.shellMembraneForce",
    ShellMembraneForce
);
selector!(
    shell_bending_moment,
    SHELL_BENDING_MOMENT_DESCRIPTOR,
    SHELL_BENDING_MOMENT_SIGNATURES,
    "fea.StructuralField.shellBendingMoment",
    ShellBendingMoment
);
selector!(
    shell_transverse_shear,
    SHELL_TRANSVERSE_SHEAR_DESCRIPTOR,
    SHELL_TRANSVERSE_SHEAR_SIGNATURES,
    "fea.StructuralField.shellTransverseShear",
    ShellTransverseShear
);
selector!(
    shell_von_mises,
    SHELL_VON_MISES_DESCRIPTOR,
    SHELL_VON_MISES_SIGNATURES,
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
    match object.class_name.as_str() {
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
        assert_eq!(object.class_name, OUTPUT_FIELD_CLASS);
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
