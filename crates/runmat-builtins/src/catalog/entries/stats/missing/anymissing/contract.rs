use crate::*;

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, text, cell, structure, table, datetime, or duration input.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "TF",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when at least one input element is missing.",
}];

pub const ANYMISSING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "TF = anymissing(A)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        BuiltinErrorDescriptor {
            code: "RM.MISSING.INVALID_ARGUMENT",
            identifier: Some("RunMat:missing:InvalidArgument"),
            when: "Input count or a nested missing-value form is invalid, or a resident input cannot be gathered.",
            message: "missing-value builtin: invalid argument",
        },
        BuiltinErrorDescriptor {
            code: "RM.MISSING.UNSUPPORTED_TYPE",
            identifier: Some("RunMat:missing:UnsupportedType"),
            when: "The input has no supported missing-value representation.",
            message: "missing-value builtin: unsupported input type",
        },
        BuiltinErrorDescriptor {
            code: "RM.MISSING.INTERNAL",
            identifier: Some("RunMat:missing:InternalError"),
            when: "Internal table or aggregate materialization fails.",
            message: "missing-value builtin: internal error",
        },
    ],
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Fixed-width integer values cannot represent the standard missing sentinel.",
}];
pub const ANYMISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "TF = anymissing(integer_A)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer scalars and arrays return false without floating conversion. Resident values currently gather through their owner before recursive inspection.",
    }];
