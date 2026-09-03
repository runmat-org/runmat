use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
use documentation::PRIMES_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Same-class row of prime numbers less than or equal to n.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real integer scalar upper bound.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = primes(n)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const PRIMES_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRIMES.INVALID_INPUT",
    identifier: Some("RunMat:primes:InvalidInput"),
    when: "Input is not a finite real integer scalar.",
    message: "primes: invalid input",
};
pub const PRIMES_ERROR_LIMIT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRIMES.LIMIT",
    identifier: Some("RunMat:primes:Limit"),
    when: "The requested sieve exceeds RunMat's bounded allocation limit.",
    message: "primes: requested limit is too large",
};
pub const PRIMES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRIMES.INTERNAL",
    identifier: Some("RunMat:primes:Internal"),
    when: "GPU gathering or result construction fails.",
    message: "primes: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    PRIMES_ERROR_INVALID_INPUT,
    PRIMES_ERROR_LIMIT,
    PRIMES_ERROR_INTERNAL,
];
pub const PRIMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "n",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The result keeps the exact fixed-width integer class of n.",
}];
pub const PRIMES_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "p = primes(n)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "All eight integer classes produce an exact same-class host row; values below two produce an empty row.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const PRIMES_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "primes" },
    category: "math/discrete",
    documentation: PRIMES_DOCUMENTATION,
    descriptor: &PRIMES_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Discrete(
            DiscreteInferenceRule::Primes,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Never,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &[],
    integer_capabilities: &PRIMES_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
