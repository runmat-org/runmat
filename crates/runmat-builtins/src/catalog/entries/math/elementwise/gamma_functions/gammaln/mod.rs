use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor,
    BuiltinDistributedPolicy, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, GammaFunctionInferenceRule, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
use documentation::GAMMALN_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real log-gamma values with the input shape and floating result class.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real, nonnegative single- or double-precision input; RunMat mode also admits exact integer, logical, and character values.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = gammaln(A)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const GAMMALN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMMALN.INVALID_INPUT",
    identifier: Some("RunMat:gammaln:InvalidInput"),
    when: "Input is not a supported dense real representation, or an integer cannot cross the exact binary64 boundary.",
    message: "gammaln: invalid input",
};
pub const GAMMALN_ERROR_DOMAIN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMMALN.DOMAIN",
    identifier: Some("RunMat:gammaln:Domain"),
    when: "At least one real input value is negative.",
    message: "gammaln: input must be nonnegative",
};
pub const GAMMALN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMMALN.INTERNAL",
    identifier: Some("RunMat:gammaln:Internal"),
    when: "Internal allocation, provider proof, execution, gather, or residency restoration fails.",
    message: "gammaln: internal error",
};
pub const GAMMALN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMMALN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:gammaln:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "gammaln: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    GAMMALN_ERROR_INVALID_INPUT,
    GAMMALN_ERROR_DOMAIN,
    GAMMALN_ERROR_INTERNAL,
    GAMMALN_ERROR_TOO_MANY_OUTPUTS,
];
pub const GAMMALN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const GAMMALN_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gammaln-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gammaln with fixed-width integer input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GammalnIntegerInputExtension"),
    };
pub const GAMMALN_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gammaln-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gammaln with logical input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GammalnLogicalInputExtension"),
    };
pub const GAMMALN_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gammaln-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gammaln with character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GammalnCharacterInputExtension"),
    };
const EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    GAMMALN_INTEGER_INPUT_EXTENSION,
    GAMMALN_LOGICAL_INPUT_EXTENSION,
    GAMMALN_CHARACTER_INPUT_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every integer must be exactly representable as binary64 before the floating log-gamma boundary.",
}];
pub const GAMMALN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = gammaln(integer_A)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Integer admission is validated before floating conversion; resident integer input gathers through its exact owner and returns resident double output.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const GAMMALN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "gammaln" },
    category: "math/elementwise",
    documentation: GAMMALN_DOCUMENTATION,
    descriptor: &GAMMALN_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::GammaFunction(
            GammaFunctionInferenceRule::LogGamma,
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
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Never,
        distributed: BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &EXTENSIONS,
    integer_capabilities: &GAMMALN_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
