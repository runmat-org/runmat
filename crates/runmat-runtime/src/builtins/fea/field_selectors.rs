use runmat_analysis_fea::contracts::StructuralFieldSelector;

use super::*;
use crate::builtins::fea::study_values::{from_typed_value, to_typed_value};

pub(super) const STRUCTURAL_FIELD_CLASS: &str = "fea.StructuralField";

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Structural field selectors have no integer inputs.",
};

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

pub(super) fn from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<StructuralFieldSelector> {
    from_typed_value(builtin, STRUCTURAL_FIELD_CLASS, value)
}
