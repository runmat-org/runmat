//! Canonical meshgrid signatures, errors, extensions, and integer forms.
use crate::*;

pub const MESHGRID_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "meshgrid-like",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "the meshgrid \"like\" prototype selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MeshgridLikeExtension"),
};

pub const MESHGRID_COMPLEX_AXES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "meshgrid-complex-axes",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "complex meshgrid axes are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:MeshgridComplexAxesExtension"),
    };

pub const MESHGRID_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [MESHGRID_LIKE_EXTENSION, MESHGRID_COMPLEX_AXES_EXTENSION];

const MESHGRID_OUTPUT_XY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid coordinates along X-axis.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid coordinates along Y-axis.",
    },
];

const MESHGRID_OUTPUT_XYZ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid coordinates along X-axis.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid coordinates along Y-axis.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Grid coordinates along Z-axis.",
    },
];

const MESHGRID_SIG_X_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "X-axis vector.",
}];

const MESHGRID_SIG_XY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis vector.",
    },
];

const MESHGRID_SIG_XYZ_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Z-axis vector.",
    },
];

const MESHGRID_SIG_X_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype controlling class/device residency.",
    },
];

const MESHGRID_SIG_XY_LIKE_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype controlling class/device residency.",
    },
];

const MESHGRID_SIG_XYZ_LIKE_INPUTS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Z-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype controlling class/device residency.",
    },
];

const MESHGRID_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "[X,Y] = meshgrid(x)",
        inputs: &MESHGRID_SIG_X_INPUTS,
        outputs: &MESHGRID_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = meshgrid(x, y)",
        inputs: &MESHGRID_SIG_XY_INPUTS,
        outputs: &MESHGRID_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,Z] = meshgrid(x, y, z)",
        inputs: &MESHGRID_SIG_XYZ_INPUTS,
        outputs: &MESHGRID_OUTPUT_XYZ,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = meshgrid(x, \"like\", prototype)",
        inputs: &MESHGRID_SIG_X_LIKE_INPUTS,
        outputs: &MESHGRID_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = meshgrid(x, y, \"like\", prototype)",
        inputs: &MESHGRID_SIG_XY_LIKE_INPUTS,
        outputs: &MESHGRID_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,Z] = meshgrid(x, y, z, \"like\", prototype)",
        inputs: &MESHGRID_SIG_XYZ_LIKE_INPUTS,
        outputs: &MESHGRID_OUTPUT_XYZ,
    },
];

const MESHGRID_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x, y, z",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every documented integer axis class is replicated without conversion and each corresponding grid preserves that axis class.",
    }];

pub const MESHGRID_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[X,Y,Z] = meshgrid(integer_x, integer_y, integer_z)",
        inputs: &MESHGRID_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host axes preserve exact native storage independently. MATLAB documents gpuArray axes only for single and double; explicit integer gpuArray axes reject, while automatic integer residency gathers transparently and returns exact host grids.",
    }];

const MESHGRID_ERRORS: [BuiltinErrorDescriptor; 11] = [
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.MISSING_AXIS",
        identifier: None,
        when: "No axis vectors are provided.",
        message: "meshgrid: at least one input vector is required",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.TOO_MANY_AXES",
        identifier: None,
        when: "More than three axis vectors are provided.",
        message: "meshgrid: expected at most three input vectors",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "meshgrid: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.MULTIPLE_LIKE",
        identifier: None,
        when: "The 'like' keyword is provided multiple times.",
        message: "meshgrid: multiple 'like' specifications are not supported",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.LIKE_POSITION",
        identifier: None,
        when: "The 'like' keyword is in an invalid position or not final.",
        message: "meshgrid: 'like' must be the final argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not recognized.",
        message: "meshgrid: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.INVALID_AXIS_INPUT",
        identifier: None,
        when: "Axis inputs are non-numeric or non-vector shapes.",
        message: "meshgrid: input argument must be numeric vector data",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.INVALID_PROTOTYPE",
        identifier: None,
        when: "The 'like' prototype is unsupported.",
        message: "meshgrid: prototypes must be numeric arrays",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.OUTPUT_COUNT_EXCEEDED",
        identifier: None,
        when: "Requested outputs exceed available outputs for provided axes.",
        message:
            "meshgrid: supports at most two outputs for 2-axis inputs and three for 3-axis inputs",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.THIRD_OUTPUT_UNAVAILABLE",
        identifier: None,
        when: "A third output is requested without supplying a Z-axis vector.",
        message: "meshgrid: third output requested but no Z vector was supplied",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.COMPLEX_REAL_CONVERSION",
        identifier: None,
        when: "Complex axis values cannot be represented in requested real output class.",
        message: "meshgrid: cannot represent complex values in a real output",
    },
];

pub const MESHGRID_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MESHGRID_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MESHGRID_ERRORS,
};
