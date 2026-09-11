use super::{BITWISE_ERROR_INVALID_INPUT, BITWISE_ERROR_SIZE_MISMATCH};
use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;

use documentation::DOCUMENTATION;

pub const BITSHIFT_SINGLE_VALUE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bitshift-single-value-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bitshift with single-precision A is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BitshiftSingleValueInputExtension"),
    };
pub const BITSHIFT_SINGLE_COUNT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bitshift-single-count-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bitshift with single-precision k is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BitshiftSingleCountInputExtension"),
    };
pub const BITSHIFT_LOGICAL_VALUE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bitshift-logical-value-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bitshift with logical A is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BitshiftLogicalValueInputExtension"),
    };
pub const BITSHIFT_LOGICAL_COUNT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bitshift-logical-count-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bitshift with logical k is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BitshiftLogicalCountInputExtension"),
    };
pub const BITSHIFT_GPU_UNDOCUMENTED_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bitshift-gpu-undocumented-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bitshift with resident input outside the documented non-64-bit integer-array GPU domain is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BitshiftGpuUndocumentedInputExtension"),
    };
pub const BITSHIFT_GPU_ASSUMED_TYPE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "bitshift-gpu-assumedtype",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "bitshift with resident input and assumedtype is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BitshiftGpuAssumedTypeExtension"),
    };
pub const BITSHIFT_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    BITSHIFT_SINGLE_VALUE_EXTENSION,
    BITSHIFT_SINGLE_COUNT_EXTENSION,
    BITSHIFT_LOGICAL_VALUE_EXTENSION,
    BITSHIFT_LOGICAL_COUNT_EXTENSION,
    BITSHIFT_GPU_UNDOCUMENTED_INPUT_EXTENSION,
    BITSHIFT_GPU_ASSUMED_TYPE_EXTENSION,
];

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Shifted values in the class of A.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive counts shift left and negative counts shift right.",
    },
];
const ASSUMED_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive counts shift left and negative counts shift right.",
    },
    BuiltinParamDescriptor {
        name: "assumedtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Integer class used to interpret double input A.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = bitshift(A, k)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = bitshift(A, k, assumedtype)",
        inputs: &ASSUMED_INPUTS,
        outputs: &OUTPUTS,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 2] =
    [BITWISE_ERROR_INVALID_INPUT, BITWISE_ERROR_SIZE_MISMATCH];
pub const BITSHIFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A accepts double or every fixed-width integer class and determines output class.",
    },
    BuiltinIntegerInputCapability {
        name: "k",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer class is accepted for finite integral shift counts; its class does not affect output class.",
    },
];
pub const BITSHIFT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "C = bitshift(A, k)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Left shifts truncate overflow bits; signed right shifts extend the sign. The documented resident domain requires a non-64-bit integer array and excludes signed A.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = bitshift(A, k, assumedtype)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "assumedtype selects the width for double A and must match typed A. Resident assumedtype calls are a gated RunMat extension.",
    },
];
const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const BITSHIFT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "bitshift" },
    category: "math/bitwise",
    documentation: DOCUMENTATION,
    descriptor: &BITSHIFT_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Bitwise(
            BitwiseInferenceRule::Shift,
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
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &BITSHIFT_EXTENSIONS,
    integer_capabilities: &BITSHIFT_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
