pub(super) use crate::{
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
pub(super) use runmat_types::{EffectKind, ExecutionStackRequirement, NumericClass};

macro_rules! define_integer_conversion_contract {
    ($name:literal, $upper:literal, $class:expr, $documentation:expr) => {
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
            when: "The call has an unsupported number of input arguments or requests more than one output.",
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
        const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];

        pub const CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
            audience: crate::BuiltinCatalogAudience::Public,
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "math/elementwise",
            documentation: $documentation,
            descriptor: &DESCRIPTOR,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Complete,
                inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::NumericConversion(
                    $class,
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
                preparation: crate::BuiltinOperandPreparation::Forms(&[crate::BuiltinPreparationForm {
                    id: crate::BuiltinPreparationFormId(0),
                    binding_variant: "default",
                    signatures: &[0],
                    data_arguments: &[0],
                    operation: crate::BuiltinPreparationOperation::Unary,
                    provider_requirement: crate::BuiltinPreparationProviderRequirement::InputSupported,
                }]),
                portability: BuiltinPortability::NativeAndWasm,
                accelerator: BuiltinAcceleratorPolicy::Optional,
                residency: BuiltinResidencyPolicy::PreserveInputs,
                // Native provider casts are standalone operations, not an
                // admitted scalar/structural fusion lowering.
                fusion: BuiltinFusionPolicy::Boundary,
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
    };
}

pub(super) use define_integer_conversion_contract;
