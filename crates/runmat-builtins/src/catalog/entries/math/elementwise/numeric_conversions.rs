use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinFusionPolicy, BuiltinInferenceRule, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement, NumericClass};

mod documentation;

use documentation::{
    INT16_DOCUMENTATION, INT32_DOCUMENTATION, INT64_DOCUMENTATION, INT8_DOCUMENTATION,
    UINT16_DOCUMENTATION, UINT32_DOCUMENTATION, UINT64_DOCUMENTATION, UINT8_DOCUMENTATION,
};

macro_rules! define_integer_conversion {
    ($module:ident, $name:literal, $upper:literal, $class:expr, $documentation:expr) => {
        mod $module {
            use super::*;

            const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
                name: "Y",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: concat!($name, "-converted output value."),
            }];
            const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
                name: "X",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Input scalar or array value to convert.",
            }];
            const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
                label: concat!("Y = ", $name, "(X)"),
                inputs: &INPUTS,
                outputs: &OUTPUTS,
            }];

            pub const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
                code: concat!("RM.", $upper, ".INVALID_ARGUMENT"),
                identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
                when: "The call has an unsupported number of arguments.",
                message: concat!($name, ": invalid argument"),
            };
            pub const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
                code: concat!("RM.", $upper, ".INVALID_INPUT"),
                identifier: Some(concat!("RunMat:", $name, ":InvalidInput")),
                when: concat!("Input cannot be converted to ", $name, "."),
                message: concat!($name, ": invalid input"),
            };
            pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
                code: concat!("RM.", $upper, ".INTERNAL"),
                identifier: Some(concat!("RunMat:", $name, ":Internal")),
                when: "Internal conversion, gather, or provider upload fails.",
                message: concat!($name, ": internal error"),
            };
            const ERRORS: [BuiltinErrorDescriptor; 3] = [
                ERROR_INVALID_ARGUMENT,
                ERROR_INVALID_INPUT,
                ERROR_INTERNAL,
            ];
            pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
                signatures: &SIGNATURES,
                output_mode: BuiltinOutputMode::Fixed,
                completion_policy: BuiltinCompletionPolicy::Public,
                errors: &ERRORS,
            };

            const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
                [BuiltinIntegerInputCapability {
                    name: "X",
                    classes: &ALL_INTEGER_CLASSES,
                    availability: BuiltinIntegerInputAvailability::Documented,
                    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
                    notes: concat!(
                        "Every native integer class converts directly to authoritative ",
                        $name,
                        " storage without a floating intermediate."
                    ),
                }];
            pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
                [BuiltinIntegerCapabilityDescriptor {
                    form: concat!("Y = ", $name, "(integer_X)"),
                    inputs: &INTEGER_INPUTS,
                    computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
                    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
                    overflow: BuiltinIntegerOverflowRule::Saturate,
                    backend: BuiltinIntegerBackendRule::HostAndGpu,
                    overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
                    notes: concat!(
                        "Host and resident conversion is exact and saturating. Real and paired-complex gpuArray inputs preserve native ",
                        $name,
                        " device storage, owner, and residency."
                    ),
                }];

            const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
            const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];

            pub const CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
                identity: BuiltinCatalogIdentity { name: $name },
                category: "math/elementwise",
                documentation: $documentation,
                descriptor: &DESCRIPTOR,
                contract: BuiltinContractDeclaration {
                    maturity: BuiltinContractMaturity::Complete,
                    inference_rule: BuiltinInferenceRule::Math(
                        MathInferenceRule::NumericConversion($class),
                    ),
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
                    residency: BuiltinResidencyPolicy::PreserveInputs,
                    fusion: BuiltinFusionPolicy::Candidate,
                    distributed: crate::BuiltinDistributedPolicy::MapUnary,
                },
                link: BuiltinLinkContract {
                    reachability: BuiltinReachability::Always,
                    policy: BuiltinLinkPolicy::PortableRuntime,
                    execution_stack: ExecutionStackRequirement::Any,
                    artifact_dependencies: &[],
                },
                bindings: &BINDINGS,
                extensions: &[],
                integer_capabilities: &INTEGER_CAPABILITIES,
                integer_audit: None,
                suppress_auto_output: false,
            };
        }
    };
}

define_integer_conversion!(int8, "int8", "INT8", NumericClass::Int8, INT8_DOCUMENTATION);
define_integer_conversion!(
    int16,
    "int16",
    "INT16",
    NumericClass::Int16,
    INT16_DOCUMENTATION
);
define_integer_conversion!(
    int32,
    "int32",
    "INT32",
    NumericClass::Int32,
    INT32_DOCUMENTATION
);
define_integer_conversion!(
    int64,
    "int64",
    "INT64",
    NumericClass::Int64,
    INT64_DOCUMENTATION
);
define_integer_conversion!(
    uint8,
    "uint8",
    "UINT8",
    NumericClass::UInt8,
    UINT8_DOCUMENTATION
);
define_integer_conversion!(
    uint16,
    "uint16",
    "UINT16",
    NumericClass::UInt16,
    UINT16_DOCUMENTATION
);
define_integer_conversion!(
    uint32,
    "uint32",
    "UINT32",
    NumericClass::UInt32,
    UINT32_DOCUMENTATION
);
define_integer_conversion!(
    uint64,
    "uint64",
    "UINT64",
    NumericClass::UInt64,
    UINT64_DOCUMENTATION
);

pub use int16::{
    CATALOG_ENTRY as INT16_CATALOG_ENTRY, DESCRIPTOR as INT16_DESCRIPTOR,
    ERROR_INTERNAL as INT16_ERROR_INTERNAL, ERROR_INVALID_ARGUMENT as INT16_ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_INPUT as INT16_ERROR_INVALID_INPUT,
    INTEGER_CAPABILITIES as INT16_INTEGER_CAPABILITIES,
};
pub use int32::{
    CATALOG_ENTRY as INT32_CATALOG_ENTRY, DESCRIPTOR as INT32_DESCRIPTOR,
    ERROR_INTERNAL as INT32_ERROR_INTERNAL, ERROR_INVALID_ARGUMENT as INT32_ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_INPUT as INT32_ERROR_INVALID_INPUT,
    INTEGER_CAPABILITIES as INT32_INTEGER_CAPABILITIES,
};
pub use int64::{
    CATALOG_ENTRY as INT64_CATALOG_ENTRY, DESCRIPTOR as INT64_DESCRIPTOR,
    ERROR_INTERNAL as INT64_ERROR_INTERNAL, ERROR_INVALID_ARGUMENT as INT64_ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_INPUT as INT64_ERROR_INVALID_INPUT,
    INTEGER_CAPABILITIES as INT64_INTEGER_CAPABILITIES,
};
pub use int8::{
    CATALOG_ENTRY as INT8_CATALOG_ENTRY, DESCRIPTOR as INT8_DESCRIPTOR,
    ERROR_INTERNAL as INT8_ERROR_INTERNAL, ERROR_INVALID_ARGUMENT as INT8_ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_INPUT as INT8_ERROR_INVALID_INPUT,
    INTEGER_CAPABILITIES as INT8_INTEGER_CAPABILITIES,
};
pub use uint16::{
    CATALOG_ENTRY as UINT16_CATALOG_ENTRY, DESCRIPTOR as UINT16_DESCRIPTOR,
    ERROR_INTERNAL as UINT16_ERROR_INTERNAL,
    ERROR_INVALID_ARGUMENT as UINT16_ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_INPUT as UINT16_ERROR_INVALID_INPUT,
    INTEGER_CAPABILITIES as UINT16_INTEGER_CAPABILITIES,
};
pub use uint32::{
    CATALOG_ENTRY as UINT32_CATALOG_ENTRY, DESCRIPTOR as UINT32_DESCRIPTOR,
    ERROR_INTERNAL as UINT32_ERROR_INTERNAL,
    ERROR_INVALID_ARGUMENT as UINT32_ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_INPUT as UINT32_ERROR_INVALID_INPUT,
    INTEGER_CAPABILITIES as UINT32_INTEGER_CAPABILITIES,
};
pub use uint64::{
    CATALOG_ENTRY as UINT64_CATALOG_ENTRY, DESCRIPTOR as UINT64_DESCRIPTOR,
    ERROR_INTERNAL as UINT64_ERROR_INTERNAL,
    ERROR_INVALID_ARGUMENT as UINT64_ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_INPUT as UINT64_ERROR_INVALID_INPUT,
    INTEGER_CAPABILITIES as UINT64_INTEGER_CAPABILITIES,
};
pub use uint8::{
    CATALOG_ENTRY as UINT8_CATALOG_ENTRY, DESCRIPTOR as UINT8_DESCRIPTOR,
    ERROR_INTERNAL as UINT8_ERROR_INTERNAL, ERROR_INVALID_ARGUMENT as UINT8_ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_INPUT as UINT8_ERROR_INVALID_INPUT,
    INTEGER_CAPABILITIES as UINT8_INTEGER_CAPABILITIES,
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &INT8_CATALOG_ENTRY,
    &INT16_CATALOG_ENTRY,
    &INT32_CATALOG_ENTRY,
    &INT64_CATALOG_ENTRY,
    &UINT8_CATALOG_ENTRY,
    &UINT16_CATALOG_ENTRY,
    &UINT32_CATALOG_ENTRY,
    &UINT64_CATALOG_ENTRY,
];
