use crate::*;

pub const GEOMETRY_REGION_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("geometry.Region");

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "asset",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "geometry.Asset returned by geometry.load.",
    },
    BuiltinParamDescriptor {
        name: "selector",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Exact region ID, name, or tag; optionally prefixed with id:, region:, name:, or tag:.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "region",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Immutable geometry.Region bound to the asset's identity and revision.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "region = geometry.region(asset, selector)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.REGION.INVALID",
    identifier: Some("RunMat:geometry:region:InvalidRegion"),
    when: "The selector is empty, uses an unsupported prefix, matches no region, or matches more than one region.",
    message: "geometry.region: invalid region selection",
};
pub const ERROR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GEOMETRY.REGION.INVALID_INPUT",
    identifier: Some("RunMat:geometry:region:InvalidInput"),
    when: "The call does not pass exactly a geometry.Asset and a scalar text selector.",
    message: "geometry.region: invalid input",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID, ERROR_INPUT];

pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "geometry.region resolves text metadata from a geometry.Asset; its geometry_revision property is a native uint32 value and it has no integer input role.",
};
