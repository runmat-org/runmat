use crate::*;

const OUTPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cartesian x coordinate.",
};
const OUTPUT_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cartesian y coordinate.",
};
const OUTPUT_Z: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cartesian z coordinate.",
};
const OUTPUT_XY: [BuiltinParamDescriptor; 2] = [OUTPUT_X, OUTPUT_Y];
const OUTPUT_XYZ: [BuiltinParamDescriptor; 3] = [OUTPUT_X, OUTPUT_Y, OUTPUT_Z];

const INPUT_THETA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "theta",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Polar or cylindrical angular coordinate in radians.",
};
const INPUT_RHO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "rho",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Polar or cylindrical radial coordinate.",
};
const INPUT_Z: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "z",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cylindrical elevation coordinate.",
};
const INPUTS_POLAR: [BuiltinParamDescriptor; 2] = [INPUT_THETA, INPUT_RHO];
const INPUTS_CYLINDRICAL: [BuiltinParamDescriptor; 3] = [INPUT_THETA, INPUT_RHO, INPUT_Z];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[x, y] = pol2cart(theta, rho)",
        inputs: &INPUTS_POLAR,
        outputs: &OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[x, y, z] = pol2cart(theta, rho, z)",
        inputs: &INPUTS_CYLINDRICAL,
        outputs: &OUTPUT_XYZ,
    },
];

pub const POL2CART_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POL2CART.INVALID_INPUT",
    identifier: Some("RunMat:pol2cart:InvalidInput"),
    when: "An input is not a supported real floating-point scalar or array.",
    message: "pol2cart: invalid input",
};
pub const POL2CART_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POL2CART.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:pol2cart:ComplexUnsupported"),
    when: "At least one input is complex.",
    message: "pol2cart: complex inputs are not supported",
};
pub const POL2CART_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POL2CART.SIZE_MISMATCH",
    identifier: Some("RunMat:pol2cart:SizeMismatch"),
    when: "Input arrays do not have compatible sizes for implicit expansion.",
    message: "pol2cart: size mismatch",
};
pub const POL2CART_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POL2CART.OUTPUT_COUNT",
    identifier: Some("RunMat:pol2cart:OutputCount"),
    when: "More outputs are requested than the selected syntax supports.",
    message: "pol2cart: too many output arguments",
};
pub const POL2CART_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POL2CART.INTERNAL",
    identifier: Some("RunMat:pol2cart:Internal"),
    when: "Internal gather/conversion/allocation flow failed.",
    message: "pol2cart: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    POL2CART_ERROR_INVALID_INPUT,
    POL2CART_ERROR_COMPLEX_UNSUPPORTED,
    POL2CART_ERROR_SIZE_MISMATCH,
    POL2CART_ERROR_OUTPUT_COUNT,
    POL2CART_ERROR_INTERNAL,
];

pub const POL2CART_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const POL2CART_INTEGER_THETA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pol2cart-integer-theta",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pol2cart accepts typed-integer angular coordinates as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Pol2cartIntegerThetaExtension"),
    };
pub const POL2CART_INTEGER_RHO_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pol2cart-integer-rho",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pol2cart accepts typed-integer radial coordinates as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Pol2cartIntegerRhoExtension"),
};
pub const POL2CART_INTEGER_Z_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pol2cart-integer-z",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pol2cart accepts typed-integer elevation coordinates as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Pol2cartIntegerZExtension"),
};
pub const POL2CART_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    POL2CART_INTEGER_THETA_EXTENSION,
    POL2CART_INTEGER_RHO_EXTENSION,
    POL2CART_INTEGER_Z_EXTENSION,
];

const POL2CART_INTEGER_THETA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "theta",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents real single and double angular coordinates; RunMat admits typed integers only after exact binary64 conversion is proved.",
    }];
const POL2CART_INTEGER_RHO_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "rho",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents real single and double radial coordinates; RunMat admits typed integers only at its checked floating transform boundary.",
    }];
const POL2CART_INTEGER_Z_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "z",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents real single and double elevation coordinates; RunMat admits typed integers only when exact binary64 conversion is possible.",
    }];
pub const POL2CART_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[x,y] = pol2cart(integer_theta,rho) or [x,y,z] = pol2cart(integer_theta,rho,z)",
        inputs: &POL2CART_INTEGER_THETA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "The angular role is independently gated before provider dispatch and crosses once into the trigonometric domain.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[x,y] = pol2cart(theta,integer_rho) or [x,y,z] = pol2cart(theta,integer_rho,z)",
        inputs: &POL2CART_INTEGER_RHO_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "The radial role is independently gated before provider dispatch and floating multiplication.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[x,y,z] = pol2cart(theta,rho,integer_z)",
        inputs: &POL2CART_INTEGER_Z_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Elevation is independently gated and broadcast only after its exact floating boundary is proved.",
    },
];
