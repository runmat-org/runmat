//! Canonical public GPU-stream state controls. The current runtime implements
//! Threefry; alternate documented generators remain an explicit contract gap.
use crate::*;

const STATE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Previous GPU stream snapshot with Type, Seed, and State fields.",
}];
const NONE: [BuiltinParamDescriptor; 0] = [];
const ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "seed, generator, option, or state",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonnegative seed, generator, default/shuffle option, or prior snapshot.",
}];
const TWO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "seed",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonnegative integer seed below 2^32.",
    },
    BuiltinParamDescriptor {
        name: "generator",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "GPU random generator algorithm.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "s = gpurng()",
        inputs: &NONE,
        outputs: &STATE,
    },
    BuiltinSignatureDescriptor {
        label: "s = gpurng(seed|generator|option|state)",
        inputs: &ONE,
        outputs: &STATE,
    },
    BuiltinSignatureDescriptor {
        label: "s = gpurng(seed,generator)",
        inputs: &TWO,
        outputs: &STATE,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.GPURNG.INVALID_ARGUMENT",
        identifier: Some("RunMat:gpurng:InvalidArgument"),
        when: "A seed or opaque snapshot is malformed.",
        message: "gpurng: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.GPURNG.GENERATOR_UNSUPPORTED",
        identifier: Some("RunMat:gpurng:GeneratorUnsupported"),
        when: "A documented generator has no implemented request-bound route.",
        message: "gpurng: generator is not supported by this runtime",
    },
];
pub const GPURNG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
