pub(super) use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, MathInferenceRule, ALL_INTEGER_CLASSES,
};
pub(super) use runmat_types::{EffectKind, ExecutionStackRequirement, NumericClass};

macro_rules! define_floating_conversion_contract {
    (
        $name:literal,
        $upper:literal,
        $class:expr,
        $documentation:expr,
        $output_description:literal,
        $gpu_storage_description:literal,
        $extension_id:literal,
        $extension_description:literal,
        $extension_error:literal,
        $integer_backend:expr,
        $integer_output:expr,
        $integer_notes:literal,
        $backend_notes:literal
    ) => {
        const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
            name: "Y",
            ty: BuiltinParamType::NumericArray,
            arity: BuiltinParamArity::Required,
            default: None,
            description: $output_description,
        }];
        const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
            name: "X",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Input scalar or array value to convert.",
        }];
        const LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
            BuiltinParamDescriptor {
                name: "X",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Input scalar or array value to convert.",
            },
            BuiltinParamDescriptor {
                name: "like",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Literal string \"like\".",
            },
            BuiltinParamDescriptor {
                name: "prototype",
                ty: BuiltinParamType::LikePrototype,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Output residency prototype; the conversion class remains fixed.",
            },
        ];
        const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
            BuiltinSignatureDescriptor {
                label: concat!("Y = ", $name, "(X)"),
                inputs: &INPUT,
                outputs: &OUTPUTS,
            },
            BuiltinSignatureDescriptor {
                label: concat!("Y = ", $name, "(X, \"like\", prototype)"),
                inputs: &LIKE_INPUTS,
                outputs: &OUTPUTS,
            },
        ];

        pub const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INVALID_ARGUMENT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
            when: "Optional arguments are malformed or unsupported.",
            message: concat!($name, ": invalid argument"),
        };
        pub const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INVALID_INPUT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidInput")),
            when: concat!(
                "Input value or prototype cannot be converted to ",
                $name,
                "."
            ),
            message: concat!($name, ": invalid input"),
        };
        pub const ERROR_GPU_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".GPU_UNSUPPORTED"),
            identifier: Some(concat!("RunMat:", $name, ":GpuUnsupported")),
            when: concat!(
                "GPU output via \"like\" is requested but no compatible ",
                $gpu_storage_description,
                " provider is active."
            ),
            message: concat!($name, ": gpu output not supported"),
        };
        pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INTERNAL"),
            identifier: Some(concat!("RunMat:", $name, ":Internal")),
            when: "Internal conversion, gather, or provider upload failed.",
            message: concat!($name, ": internal error"),
        };
        const ERRORS: [BuiltinErrorDescriptor; 4] = [
            ERROR_INVALID_ARGUMENT,
            ERROR_INVALID_INPUT,
            ERROR_GPU_UNSUPPORTED,
            ERROR_INTERNAL,
        ];
        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };

        pub const LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $extension_id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $extension_description,
            error_identifier: Some($extension_error),
        };
        pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [LIKE_EXTENSION];

        const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
            [BuiltinIntegerInputCapability {
                name: "X",
                classes: &ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::Documented,
                scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: $integer_notes,
            }];
        pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
            [BuiltinIntegerCapabilityDescriptor {
                form: concat!("Y = ", $name, "(integer_X)"),
                inputs: &INTEGER_INPUTS,
                computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
                output_class: $integer_output,
                overflow: BuiltinIntegerOverflowRule::NotApplicable,
                backend: $integer_backend,
                overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
                notes: $backend_notes,
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
                    MathInferenceRule::NumericConversionWithLike($class),
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
                residency: BuiltinResidencyPolicy::Dynamic,
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
            extensions: &EXTENSIONS,
            integer_capabilities: &INTEGER_CAPABILITIES,
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}
