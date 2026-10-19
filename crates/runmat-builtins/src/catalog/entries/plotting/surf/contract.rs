use crate::*;

const SURF_INTEGER_X: [BuiltinIntegerInputCapability; 1] = [surface_integer_input("X")];
const SURF_INTEGER_Y: [BuiltinIntegerInputCapability; 1] = [surface_integer_input("Y")];
const SURF_INTEGER_Z: [BuiltinIntegerInputCapability; 1] = [surface_integer_input("Z")];
const SURF_INTEGER_C: [BuiltinIntegerInputCapability; 1] = [surface_integer_input("C")];
const fn surface_integer_input(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "The compatibility target explicitly lists every native integer class for this surface coordinate role.",
    }
}
const fn surface_integer_capability(
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
        notes: "Native class, exact values, and shape remain authoritative in scene XData/YData/ZData; rendering and contour/color preparation are explicit floating boundaries.",
    }
}
pub const SURF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    surface_integer_capability("h = surf(integer_X, Y, Z, ...)", &SURF_INTEGER_X),
    surface_integer_capability("h = surf(X, integer_Y, Z, ...)", &SURF_INTEGER_Y),
    surface_integer_capability("h = surf(X, Y, integer_Z, ...)", &SURF_INTEGER_Z),
    surface_integer_capability("h = surf(X, Y, Z, integer_C, ...)", &SURF_INTEGER_C),
];

const SURF_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered surface plot.",
}];

const SURF_INPUTS_Z: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Surface height grid.",
}];

const SURF_INPUTS_Z_C: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Surface height grid.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Independent indexed m-by-n or truecolor m-by-n-by-3 surface color data.",
    },
];

const SURF_INPUTS_X_Y_Z: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X axis vector or full coordinate matrix matching Z.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y axis vector or full coordinate matrix matching Z.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Surface height grid.",
    },
];

const SURF_INPUTS_X_Y_Z_C: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X axis vector or full coordinate matrix matching Z.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y axis vector or full coordinate matrix matching Z.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Surface height grid.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Independent indexed m-by-n or truecolor m-by-n-by-3 color data.",
    },
];

const SURF_INPUTS_Z_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Surface height grid.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value surface style options.",
    },
];

const SURF_INPUTS_X_Y_Z_PROPS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X axis vector or full coordinate matrix matching Z.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y axis vector or full coordinate matrix matching Z.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Surface height grid.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value surface style options.",
    },
];

const SURF_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "h = surf(Z)",
        inputs: &SURF_INPUTS_Z,
        outputs: &SURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surf(X, Y, Z)",
        inputs: &SURF_INPUTS_X_Y_Z,
        outputs: &SURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surf(Z, C)",
        inputs: &SURF_INPUTS_Z_C,
        outputs: &SURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surf(X, Y, Z, C)",
        inputs: &SURF_INPUTS_X_Y_Z_C,
        outputs: &SURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surf(Z, Name, Value, ...)",
        inputs: &SURF_INPUTS_Z_PROPS,
        outputs: &SURF_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = surf(X, Y, Z, Name, Value, ...)",
        inputs: &SURF_INPUTS_X_Y_Z_PROPS,
        outputs: &SURF_OUTPUT_HANDLE,
    },
];

pub const SURF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SURF.INVALID_ARGUMENT",
    identifier: Some("RunMat:surf:InvalidArgument"),
    when: "Surface/grid/axis inputs or style name/value arguments are invalid.",
    message: "surf: invalid argument",
};

pub const SURF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SURF.INTERNAL",
    identifier: Some("RunMat:surf:Internal"),
    when: "Internal surface construction or render preparation fails unexpectedly.",
    message: "surf: internal operation failed",
};

const SURF_ERRORS: [BuiltinErrorDescriptor; 2] = [SURF_ERROR_INVALID_ARGUMENT, SURF_ERROR_INTERNAL];

pub const SURF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SURF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SURF_ERRORS,
};
