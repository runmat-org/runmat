use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
use documentation::GCD_DOCUMENTATION;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real integer-valued scalar, vector, or array.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real integer-valued scalar, vector, or array.",
    },
];
const DIVISOR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "G",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonnegative greatest common divisors of A and B.",
}];
const EXTENDED_OUTPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "G",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonnegative greatest common divisors of A and B.",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "First Bezout coefficient satisfying A.*U + B.*V = G.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Second Bezout coefficient satisfying A.*U + B.*V = G.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "G = gcd(A, B)",
        inputs: &INPUTS,
        outputs: &DIVISOR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[G, U, V] = gcd(A, B)",
        inputs: &INPUTS,
        outputs: &EXTENDED_OUTPUTS,
    },
];

pub const GCD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GCD.INVALID_INPUT",
    identifier: Some("RunMat:gcd:InvalidInput"),
    when: "Inputs are not real integer-valued numeric values, or their classes cannot be combined.",
    message: "gcd: invalid input",
};
pub const GCD_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GCD.SIZE_MISMATCH",
    identifier: Some("RunMat:gcd:SizeMismatch"),
    when: "Inputs do not have the same size and neither input is scalar.",
    message: "gcd: input sizes are not compatible",
};
pub const GCD_ERROR_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GCD.OVERFLOW",
    identifier: Some("RunMat:gcd:Overflow"),
    when: "A divisor or requested Bezout coefficient cannot be represented in the output class.",
    message: "gcd: output overflows numeric class",
};
pub const GCD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GCD.INTERNAL",
    identifier: Some("RunMat:gcd:Internal"),
    when: "GPU gathering or result construction fails.",
    message: "gcd: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    GCD_ERROR_INVALID_INPUT,
    GCD_ERROR_SIZE_MISMATCH,
    GCD_ERROR_OVERFLOW,
    GCD_ERROR_INTERNAL,
];
pub const GCD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "B must use the same integer class or be a scalar double.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A must use the same integer class or be a scalar double.",
    },
];
const EXTENDED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &SIGNED_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Extended outputs require a signed integer, single, or double result class.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &SIGNED_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Extended outputs require a signed integer, single, or double result class.",
    },
];
pub const GCD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "G = gcd(A, B)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Signed and zero values are accepted; G is always nonnegative.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[G, U, V] = gcd(A, B)",
        inputs: &EXTENDED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Unsigned inputs are rejected because Bezout coefficients require a signed class.",
    },
];

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const GCD_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "gcd" },
    category: "math/discrete",
    documentation: GCD_DOCUMENTATION,
    descriptor: &GCD_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Discrete(
            DiscreteInferenceRule::Binary(BinaryNumberTheoryRule::Gcd),
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
    integer_capabilities: &GCD_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
