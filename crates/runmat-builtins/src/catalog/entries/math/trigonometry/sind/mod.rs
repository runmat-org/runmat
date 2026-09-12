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
    BuiltinSignatureDescriptor, DegreeTrigonometricFunction, MathInferenceRule,
    ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;

use documentation::SIND_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise sine result for degree-valued input.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Single or double real/complex input; integer and logical forms are RunMat-only extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = sind(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const SIND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIND.INVALID_INPUT",
    identifier: Some("RunMat:sind:InvalidInput"),
    when: "Input is not a supported numeric or logical value.",
    message: "sind: invalid input",
};
pub const SIND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIND.INTERNAL",
    identifier: Some("RunMat:sind:Internal"),
    when: "Internal gather, conversion, or allocation failed.",
    message: "sind: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [SIND_ERROR_INVALID_INPUT, SIND_ERROR_INTERNAL];
pub const SIND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const SIND_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sind-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sind with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SindIntegerInputExtension"),
};
pub const SIND_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sind-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sind with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SindLogicalInputExtension"),
};
pub const SIND_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [SIND_INTEGER_INPUT_EXTENSION, SIND_LOGICAL_INPUT_EXTENSION];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "The integer must be exactly representable at the binary64 degree-trigonometric boundary.",
}];
pub const SIND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = sind(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat validates native integer storage before conversion and preserves exact canonical degree results.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const SIND_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "sind" },
    category: "math/trigonometry",
    documentation: SIND_DOCUMENTATION,
    descriptor: &SIND_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::DegreeTrigonometric(
            DegreeTrigonometricFunction::Sin,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
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
    extensions: &SIND_EXTENSIONS,
    integer_capabilities: &SIND_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&SIND_CATALOG_ENTRY];
