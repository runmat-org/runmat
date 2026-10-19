use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};

const MESH_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mesh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Validated meshing.Mesh object.",
}];
const VALUE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operation result.",
}];
const GENERATE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor { name: "geometry", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Imported geometry.Asset." },
    BuiltinParamDescriptor { name: "Name, Value", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Variadic, default: None, description: "TargetSize, MaximumDeviation, MaximumGradingRatio, ElementOrder, MaximumElements, or Seed." },
];
const LOAD_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "meshPath",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Canonical solver-mesh artifact path.",
    },
    BuiltinParamDescriptor {
        name: "evidencePath",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Matching canonical meshing-evidence path.",
    },
];
const SAVE_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "mesh",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Validated meshing.Mesh object.",
    },
    BuiltinParamDescriptor {
        name: "meshPath",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Destination solver-mesh artifact path.",
    },
    BuiltinParamDescriptor {
        name: "evidencePath",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Destination meshing-evidence path.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional Force logical replacement policy.",
    },
];
const MESH_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mesh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Validated meshing.Mesh object.",
}];
const DISPLAY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "mesh",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Validated meshing.Mesh object.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Mesh display options: Edges, Regions, and Section.",
    },
];
const GENERATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "mesh = meshing.generate(geometry, Name, Value, ...)",
    inputs: &GENERATE_INPUTS,
    outputs: &MESH_OUTPUT,
}];
const LOAD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "mesh = meshing.load(meshPath, evidencePath)",
    inputs: &LOAD_INPUTS,
    outputs: &MESH_OUTPUT,
}];
const SAVE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "saved = meshing.save(mesh, meshPath, evidencePath, Name, Value, ...)",
    inputs: &SAVE_INPUTS,
    outputs: &VALUE_OUTPUT,
}];
const NODES_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [inspect_signature("nodes = meshing.nodes(mesh)")];
const ELEMENTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [inspect_signature("elements = meshing.elements(mesh)")];
const BOUNDARY_FACES_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [inspect_signature("faces = meshing.boundaryFaces(mesh)")];
const REGIONS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [inspect_signature("regions = meshing.regions(mesh)")];
const QUALITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [inspect_signature("quality = meshing.quality(mesh)")];
const EVIDENCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [inspect_signature("evidence = meshing.evidence(mesh)")];
const DISPLAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "figure = mesh(mesh, Name, Value, ...)",
    inputs: &DISPLAY_INPUTS,
    outputs: &VALUE_OUTPUT,
}];
const ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MESHING.INVALID",
    identifier: Some("RunMat:meshing:InvalidInput"),
    when: "A meshing object, path, option, artifact, or evidence contract is invalid.",
    message: "meshing operation failed",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR];

pub const GENERATE_DESCRIPTOR: BuiltinDescriptor = public_descriptor(&GENERATE_SIGNATURES);
pub const LOAD_DESCRIPTOR: BuiltinDescriptor = public_descriptor(&LOAD_SIGNATURES);
pub const SAVE_DESCRIPTOR: BuiltinDescriptor = public_descriptor(&SAVE_SIGNATURES);
pub const NODES_DESCRIPTOR: BuiltinDescriptor = public_descriptor(&NODES_SIGNATURES);
pub const ELEMENTS_DESCRIPTOR: BuiltinDescriptor = public_descriptor(&ELEMENTS_SIGNATURES);
pub const BOUNDARY_FACES_DESCRIPTOR: BuiltinDescriptor =
    public_descriptor(&BOUNDARY_FACES_SIGNATURES);
pub const REGIONS_DESCRIPTOR: BuiltinDescriptor = public_descriptor(&REGIONS_SIGNATURES);
pub const QUALITY_DESCRIPTOR: BuiltinDescriptor = public_descriptor(&QUALITY_SIGNATURES);
pub const EVIDENCE_DESCRIPTOR: BuiltinDescriptor = public_descriptor(&EVIDENCE_SIGNATURES);
pub const METHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DISPLAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &ERRORS,
};

const fn public_descriptor(signatures: &'static [BuiltinSignatureDescriptor]) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    }
}

const fn inspect_signature(label: &'static str) -> BuiltinSignatureDescriptor {
    BuiltinSignatureDescriptor {
        label,
        inputs: &MESH_INPUT,
        outputs: &VALUE_OUTPUT,
    }
}

const GENERATE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "MaximumElements", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Accepted exactly as a positive u64 resource ceiling; zero, negative, and fractional values are rejected." },
    BuiltinIntegerInputCapability { name: "Seed", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Accepted exactly as a nonnegative u64 deterministic seed; negative and fractional values are rejected." },
];
pub const GENERATE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "MaximumElements and Seed",
    inputs: &GENERATE_INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::StructuralParameter,
    notes: "Integer controls remain exact through provider request construction; physical sizing controls enter the floating-point geometry domain.",
}];
const METHOD_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Section",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A four-element integer plane vector crosses once into finite binary64 visualization coordinates.",
    }];
pub const METHOD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Section plane",
        inputs: &METHOD_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Section is a physical visualization plane; typed integer coordinates are converted explicitly at that boundary.",
    }];
pub const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "These operations accept typed meshing objects or paths and have no integer input role.",
};
