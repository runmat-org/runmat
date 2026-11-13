use crate::*;

const INPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real radius profile; a scalar repeats at the base and top.",
}];
const INPUT_R_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "r",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real radius profile; a scalar repeats at the base and top.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive number of equal circumference divisions.",
    },
];
const OUTPUT_XYZ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cylinder x coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cylinder y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Equally spaced heights from zero to one.",
    },
];
const OUTPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cylinder x coordinates.",
}];
const OUTPUT_XY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cylinder x coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cylinder y coordinates.",
    },
];
const INPUT_OPTIONS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Name,Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Supported surface property name/value pairs for statement-form plotting.",
}];
const INPUT_R_N_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "r",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real radius profile.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive circumference division count.",
    },
    BuiltinParamDescriptor {
        name: "Name,Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Supported surface property name/value pairs.",
    },
];
const INPUT_AX_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "arguments",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Radius, resolution, and supported surface properties.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 15] = [
    BuiltinSignatureDescriptor {
        label: "X = cylinder",
        inputs: &[],
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "X = cylinder(r)",
        inputs: &INPUT_R,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "X = cylinder(r,n)",
        inputs: &INPUT_R_N,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = cylinder",
        inputs: &[],
        outputs: &OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = cylinder(r)",
        inputs: &INPUT_R,
        outputs: &OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = cylinder(r,n)",
        inputs: &INPUT_R_N,
        outputs: &OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,Z] = cylinder",
        inputs: &[],
        outputs: &OUTPUT_XYZ,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,Z] = cylinder(r)",
        inputs: &INPUT_R,
        outputs: &OUTPUT_XYZ,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,Z] = cylinder(r,n)",
        inputs: &INPUT_R_N,
        outputs: &OUTPUT_XYZ,
    },
    BuiltinSignatureDescriptor {
        label: "cylinder",
        inputs: &[],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "cylinder(r)",
        inputs: &INPUT_R,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "cylinder(r,n)",
        inputs: &INPUT_R_N,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "cylinder(r,n,Name,Value,...)",
        inputs: &INPUT_R_N_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "cylinder(Name,Value,...)",
        inputs: &INPUT_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "cylinder(ax,...)",
        inputs: &INPUT_AX_OPTIONS,
        outputs: &[],
    },
];
pub const CYLINDER_ERRORS: [BuiltinErrorDescriptor; 5] = [
    BuiltinErrorDescriptor {
        code: "RM.CYLINDER.OUTPUT_COUNT",
        identifier: Some("RunMat:cylinder:OutputCount"),
        when: "More than three outputs are requested.",
        message: "cylinder returns at most three coordinate arrays",
    },
    BuiltinErrorDescriptor {
        code: "RM.CYLINDER.INVALID_ARGUMENT",
        identifier: Some("RunMat:cylinder:InvalidArgument"),
        when: "Radius, resolution, axes, or plot options are invalid.",
        message: "cylinder input is invalid",
    },
    BuiltinErrorDescriptor {
        code: "RM.CYLINDER.OUT_OF_MEMORY",
        identifier: Some("RunMat:cylinder:OutOfMemory"),
        when: "The coordinate geometry overflows or the exact host allocation cannot be admitted.",
        message: "cylinder grid is too large",
    },
    BuiltinErrorDescriptor {
        code: "RM.CYLINDER.INTERNAL",
        identifier: Some("RunMat:cylinder:Internal"),
        when: "Coordinate publication or plotting fails.",
        message: "cylinder could not publish the surface",
    },
    BuiltinErrorDescriptor {
        code: "RM.CYLINDER.CANCELLED",
        identifier: Some("RunMat:cylinder:Cancelled"),
        when: "Managed host allocation is cancelled; CPU work interruptions retain the canonical execution-service cause.",
        message: "cylinder coordinate allocation was cancelled",
    },
];
pub const CYLINDER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CYLINDER_ERRORS,
};

pub const CYLINDER_TYPED_N_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cylinder-typed-resolution",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "A typed-integer cylinder circumference count is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CylinderTypedResolutionExtension"),
};
pub const CYLINDER_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [CYLINDER_TYPED_N_EXTENSION];
const INTEGER_N_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "n",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The positive whole count is decoded exactly before checking the output extent; typed storage is a gated RunMat extension.",
}];
pub const CYLINDER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[X,Y,Z] = cylinder(r,integer_n)",
        inputs: &INTEGER_N_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes:
            "Typed integer n only sets the checked output geometry; coordinates remain host Double.",
    },
];
