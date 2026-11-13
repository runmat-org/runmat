use crate::*;

pub const PATCH_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "patch-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "patch with a typed-integer numeric axes alias is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PatchIntegerAxesHandleExtension"),
};

pub const PATCH_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [PATCH_INTEGER_AXES_EXTENSION];

macro_rules! patch_integer_input {
    ($name:literal, $availability:expr, $notes:literal) => {
        [BuiltinIntegerInputCapability {
            name: $name,
            classes: &crate::ALL_INTEGER_CLASSES,
            availability: $availability,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: $notes,
        }]
    };
}

const INTEGER_X: [BuiltinIntegerInputCapability; 1] = patch_integer_input!(
    "X",
    BuiltinIntegerInputAvailability::Documented,
    "The compatibility target explicitly lists every built-in integer class for X coordinates."
);
const INTEGER_Y: [BuiltinIntegerInputCapability; 1] = patch_integer_input!(
    "Y",
    BuiltinIntegerInputAvailability::Documented,
    "The compatibility target explicitly lists every built-in integer class for Y coordinates."
);
const INTEGER_Z: [BuiltinIntegerInputCapability; 1] = patch_integer_input!(
    "Z",
    BuiltinIntegerInputAvailability::Documented,
    "The compatibility target explicitly lists every built-in integer class for Z coordinates."
);
const INTEGER_C: [BuiltinIntegerInputCapability; 1] = patch_integer_input!(
    "CData",
    BuiltinIntegerInputAvailability::Documented,
    "Integer color indices remain authoritative as patch CData before renderer colormap conversion."
);
const INTEGER_FACES: [BuiltinIntegerInputCapability; 1] = patch_integer_input!(
    "Faces",
    BuiltinIntegerInputAvailability::Documented,
    "The compatibility target explicitly lists every built-in integer class for one-based face definitions."
);
const INTEGER_VERTICES: [BuiltinIntegerInputCapability; 1] = patch_integer_input!(
    "Vertices",
    BuiltinIntegerInputAvailability::Documented,
    "The compatibility target explicitly lists every built-in integer class for vertex coordinates."
);
const INTEGER_AX: [BuiltinIntegerInputCapability; 1] = patch_integer_input!(
    "ax",
    BuiltinIntegerInputAvailability::RunMatOnly,
    "MATLAB targets a graphics object; typed-integer numeric axes aliases are a gated RunMat representation extension."
);

const fn patch_data_capability(
    form: &'static str,
    inputs: &'static [BuiltinIntegerInputCapability],
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native storage and shape remain authoritative on patch properties and in scene persistence; host triangulation and rendering are explicit floating boundaries, and documented gpuArray input executes through client gather.",
    }
}

pub const PATCH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 7] = [
    patch_data_capability("p = patch(integer_X, Y, C, ...)", &INTEGER_X),
    patch_data_capability("p = patch(X, integer_Y, C, ...)", &INTEGER_Y),
    patch_data_capability("p = patch(X, Y, integer_Z, C, ...)", &INTEGER_Z),
    patch_data_capability("p = patch(..., integer_CData, ...)", &INTEGER_C),
    patch_data_capability("p = patch('Faces', integer_F, 'Vertices', V, ...)", &INTEGER_FACES),
    patch_data_capability("p = patch('Faces', F, 'Vertices', integer_V, ...)", &INTEGER_VERTICES),
    BuiltinIntegerCapabilityDescriptor {
        form: "p = patch(integer_ax, ...)",
        inputs: &INTEGER_AX,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility gate runs before axes selection, input gathering, triangulation, or rendering.",
    },
];

const PATCH_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created patch object.",
}];

const PATCH_INPUTS_XY_C: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Patch X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Patch Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Face color specification.",
    },
];

const PATCH_INPUTS_XYZ_C: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Patch X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Patch Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Patch Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Face color specification.",
    },
];

const PATCH_INPUTS_STRUCT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Struct containing patch properties.",
}];

const PATCH_INPUTS_FACEVERT_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "facevert",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Faces/Vertices property/value pairs.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional patch property/value pairs.",
    },
];

const PATCH_INPUTS_AX_DATA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Patch positional/property arguments.",
    },
];

const PATCH_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "h = patch(X, Y, C)",
        inputs: &PATCH_INPUTS_XY_C,
        outputs: &PATCH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = patch(X, Y, Z, C)",
        inputs: &PATCH_INPUTS_XYZ_C,
        outputs: &PATCH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = patch(S)",
        inputs: &PATCH_INPUTS_STRUCT,
        outputs: &PATCH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = patch('Faces', F, 'Vertices', V, ...)",
        inputs: &PATCH_INPUTS_FACEVERT_PROPS,
        outputs: &PATCH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = patch(ax, ...)",
        inputs: &PATCH_INPUTS_AX_DATA,
        outputs: &PATCH_OUTPUT_HANDLE,
    },
];

pub const PATCH_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PATCH.INVALID_ARGUMENT",
    identifier: Some("RunMat:patch:InvalidArgument"),
    when: "Patch coordinates, faces/vertices, or property/value arguments are malformed.",
    message: "patch: invalid argument",
};

pub const PATCH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PATCH.INTERNAL",
    identifier: Some("RunMat:patch:Internal"),
    when: "Internal patch triangulation/render setup fails unexpectedly.",
    message: "patch: internal operation failed",
};

const PATCH_ERRORS: [BuiltinErrorDescriptor; 2] =
    [PATCH_ERROR_INVALID_ARGUMENT, PATCH_ERROR_INTERNAL];

pub const PATCH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PATCH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PATCH_ERRORS,
};
