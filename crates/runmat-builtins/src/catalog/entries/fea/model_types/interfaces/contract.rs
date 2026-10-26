//! Unchanged descriptor/integer metadata lifted from Runtime interfaces.rs.
use crate::fea_contract::{descriptors::ERRORS, parameters::OUT_ANY};
use crate::*;

pub const TYPED_INTERFACE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Typed FEA interface constructors accept identifiers, immutable geometry.Region values, and typed interface parameters; they have no direct integer input role.",
    };

const REGION_PAIR_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Interface identifier.",
    ),
    required("primary", BuiltinParamType::Any, "Primary geometry.Region."),
    required(
        "secondary",
        BuiltinParamType::Any,
        "Secondary geometry.Region.",
    ),
];
const FLUID_PAIR_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Interface identifier.",
    ),
    required(
        "fluid_region",
        BuiltinParamType::Any,
        "Fluid-side geometry.Region.",
    ),
    required(
        "structure_region",
        BuiltinParamType::Any,
        "Structure-side geometry.Region.",
    ),
];
const CONTACT_FRICTION_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Interface identifier.",
    ),
    required("primary", BuiltinParamType::Any, "Primary geometry.Region."),
    required(
        "secondary",
        BuiltinParamType::Any,
        "Secondary geometry.Region.",
    ),
    required(
        "friction",
        BuiltinParamType::Any,
        "Typed fea.FrictionCoefficient.",
    ),
];
const CHT_VALUE_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Interface identifier.",
    ),
    required(
        "fluid_region",
        BuiltinParamType::Any,
        "Fluid-side geometry.Region.",
    ),
    required(
        "structure_region",
        BuiltinParamType::Any,
        "Structure-side geometry.Region.",
    ),
    required(
        "transfer_value",
        BuiltinParamType::Any,
        "Typed conductance or thermal contact resistance.",
    ),
];

const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

macro_rules! descriptor {
    ($descriptor:ident, $signatures:ident, $label:literal, $inputs:ident) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &$inputs,
            outputs: &OUT_ANY,
        }];
        pub const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

descriptor!(
    CONTACT_FRICTIONLESS_DESCRIPTOR,
    CONTACT_FRICTIONLESS_SIGNATURES,
    "interface = fea.ContactInterface.frictionless(id, primary, secondary)",
    REGION_PAIR_INPUTS
);
descriptor!(
    CONTACT_COULOMB_DESCRIPTOR,
    CONTACT_COULOMB_SIGNATURES,
    "interface = fea.ContactInterface.coulomb(id, primary, secondary, friction)",
    CONTACT_FRICTION_INPUTS
);
descriptor!(
    FLUID_STRUCTURE_INTERFACE_DESCRIPTOR,
    FLUID_STRUCTURE_INTERFACE_SIGNATURES,
    "interface = fea.FluidStructureInterface.noSlip(id, fluidRegion, structureRegion)",
    FLUID_PAIR_INPUTS
);
descriptor!(
    CHT_PERFECT_DESCRIPTOR,
    CHT_PERFECT_SIGNATURES,
    "interface = fea.ConjugateHeatTransferInterface.perfect(id, fluidRegion, solidRegion)",
    FLUID_PAIR_INPUTS
);
descriptor!(
    CHT_CONDUCTANCE_DESCRIPTOR,
    CHT_CONDUCTANCE_SIGNATURES,
    "interface = fea.ConjugateHeatTransferInterface.conductance(id, fluidRegion, solidRegion, conductance)",
    CHT_VALUE_INPUTS
);
descriptor!(
    CHT_RESISTANCE_DESCRIPTOR,
    CHT_RESISTANCE_SIGNATURES,
    "interface = fea.ConjugateHeatTransferInterface.resistance(id, fluidRegion, solidRegion, resistance)",
    CHT_VALUE_INPUTS
);
