use crate::*;
const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exact table object.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonempty scalar text path on the selected filesystem provider.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Delimiter and WriteVariableNames name-value pairs.",
    },
];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bytes",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Exact double byte count; requesting this output requires RunMat extension mode.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "writetable(T, filename, Name=Value)",
        inputs: &INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "bytes = writetable(T, filename, Name=Value)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
];
pub const WRITETABLE_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITETABLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:writetable:InvalidArgument"),
    when: "Input class, path, option, or output count is invalid.",
    message: "writetable: invalid argument",
};
pub const WRITETABLE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITETABLE.IO",
    identifier: Some("RunMat:writetable:IOError"),
    when: "The captured filesystem provider cannot stage, write, or publish the output.",
    message: "writetable: file write failed",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [WRITETABLE_ERROR_ARGUMENT, WRITETABLE_ERROR_IO];
pub const WRITETABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
