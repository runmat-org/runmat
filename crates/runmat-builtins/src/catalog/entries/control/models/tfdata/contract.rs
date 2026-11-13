use crate::*;

const INPUT_SYS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar tf or ss model, or a homogeneous tf/ss model array.",
}];
const INPUT_V: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "sys",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "SISO tf or ss model.",
    },
    BuiltinParamDescriptor {
        name: "v",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector output selector.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "num",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient cell or row vector.",
    },
    BuiltinParamDescriptor {
        name: "den",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Denominator coefficient cell or row vector.",
    },
    BuiltinParamDescriptor {
        name: "ts",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Sample time.",
    },
    BuiltinParamDescriptor {
        name: "sdnum",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Empty uncertainty for a deterministic model.",
    },
    BuiltinParamDescriptor {
        name: "sdden",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Empty uncertainty for a deterministic model.",
    },
];
const INPUT_INDEXED: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "sys",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Homogeneous tf or ss model array.",
    },
    BuiltinParamDescriptor {
        name: "J1,...,JN",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Positive scalar model-array indices in column-major order.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "[num,den,ts,sdnum,sdden] = tfdata(sys)",
        inputs: &INPUT_SYS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[num,den] = tfdata(sys,'v')",
        inputs: &INPUT_V,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[num,den] = tfdata(sys,J1,...,JN)",
        inputs: &INPUT_INDEXED,
        outputs: &OUTPUTS,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.TFDATA.INVALID_ARGUMENT",
        identifier: Some("RunMat:tfdata:InvalidArgument"),
        when: "Input selector or requested output count is invalid.",
        message: "tfdata: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.TFDATA.INVALID_MODEL",
        identifier: Some("RunMat:tfdata:InvalidModel"),
        when: "Consumed model properties are malformed.",
        message: "tfdata: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.TFDATA.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:tfdata:UnsupportedModel"),
        when: "Model representation is not supported.",
        message: "tfdata: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.TFDATA.INTERNAL",
        identifier: Some("RunMat:tfdata:Internal"),
        when: "Coefficient output construction fails.",
        message: "tfdata: internal error",
    },
];
pub const TFDATA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
