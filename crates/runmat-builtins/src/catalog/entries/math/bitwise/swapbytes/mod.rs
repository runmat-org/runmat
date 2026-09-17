mod documentation;

use crate::{
    BitwiseInferenceRule, BuiltinAcceleratorPolicy, BuiltinAsyncBehavior,
    BuiltinBindingDeclaration, BuiltinCatalogEntry, BuiltinCatalogIdentity, BuiltinCompatibility,
    BuiltinCompletionPolicy, BuiltinContractDeclaration, BuiltinContractMaturity,
    BuiltinDescriptor, BuiltinDistributedPolicy, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use documentation::SWAPBYTES_DOCUMENTATION;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric result with each element's byte order reversed.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric scalar or dense array.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = swapbytes(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const SWAPBYTES_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BITWISE.INVALID_INPUT",
    identifier: Some("RunMat:bitwise:InvalidInput"),
    when: "The input is not a real numeric scalar or dense numeric array, or provider gathering fails.",
    message: "bitwise operation: invalid input",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [SWAPBYTES_ERROR_INVALID_INPUT];
pub const SWAPBYTES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const SWAPBYTES_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "swapbytes-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Allow host fallback for explicit gpuArray input to swapbytes",
        error_identifier: Some("RunMat:compatibility:SwapbytesExplicitGpuInputExtension"),
    };
pub const SWAPBYTES_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [SWAPBYTES_EXPLICIT_GPU_EXTENSION];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every native integer class is byte-swapped directly in authoritative storage while preserving class and shape.",
}];
pub const SWAPBYTES_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = swapbytes(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Each element's native byte sequence is reversed directly; 8-bit classes are unchanged. Automatic residency gathers transparently, while explicit gpuArray fallback is independently gated.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const SWAPBYTES_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "swapbytes" },
    category: "math/bitwise",
    documentation: SWAPBYTES_DOCUMENTATION,
    descriptor: &SWAPBYTES_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Bitwise(
            BitwiseInferenceRule::SwapBytes,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::Elementwise,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &SWAPBYTES_EXTENSIONS,
    integer_capabilities: &SWAPBYTES_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) fn extend_entries(values: &mut Vec<&'static BuiltinCatalogEntry>) {
    values.push(&SWAPBYTES_CATALOG_ENTRY);
}
