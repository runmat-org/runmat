//! Shared metadata shape; every invocation remains an identity-owned catalog entry.
macro_rules! define_degree_inverse_contract {
    ($name:literal, $upper:literal, $function:expr) => {
        use crate::*;
        use runmat_types::{EffectKind, ExecutionStackRequirement};

        mod documentation;

        const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
            name: "X",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Real or complex single/double input; integer, logical and character forms are RunMat extensions.",
        }];
        const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
            name: "Y",
            ty: BuiltinParamType::NumericArray,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Principal inverse-trigonometric angle in degrees, preserving the input shape.",
        }];
        const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: concat!("Y = ", $name, "(X)"),
            inputs: &INPUTS,
            outputs: &OUTPUTS,
        }];

        pub const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INVALID_INPUT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidInput")),
            when: "Input is not supported numeric, logical or character data.",
            message: concat!($name, ": invalid input"),
        };
        pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INTERNAL"),
            identifier: Some(concat!("RunMat:", $name, ":Internal")),
            when: "Gather, allocation, computation or owner-preserving restoration fails.",
            message: concat!($name, ": internal error"),
        };
        pub const ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".TOO_MANY_OUTPUTS"),
            identifier: Some(concat!("RunMat:", $name, ":TooManyOutputs")),
            when: "More than one output is requested.",
            message: concat!($name, ": too many output arguments"),
        };
        const ERRORS: [BuiltinErrorDescriptor; 3] = [
            ERROR_INVALID_INPUT,
            ERROR_INTERNAL,
            ERROR_TOO_MANY_OUTPUTS,
        ];
        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };

        pub const INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
            BuiltinExtensionDescriptor {
                id: concat!($name, "-integer-input"),
                mode: BuiltinExtensionMode::RunMatOnly,
                description: concat!($name, " with integer input is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $upper, "IntegerInputExtension")),
            };
        pub const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
            BuiltinExtensionDescriptor {
                id: concat!($name, "-logical-input"),
                mode: BuiltinExtensionMode::RunMatOnly,
                description: concat!($name, " with logical input is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $upper, "LogicalInputExtension")),
            };
        pub const CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
            BuiltinExtensionDescriptor {
                id: concat!($name, "-character-input"),
                mode: BuiltinExtensionMode::RunMatOnly,
                description: concat!($name, " with character input is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $upper, "CharacterInputExtension")),
            };
        pub const GPU_REAL_COMPLEX_EXTENSION: BuiltinExtensionDescriptor =
            BuiltinExtensionDescriptor {
                id: concat!($name, "-gpu-real-complex-promotion"),
                mode: BuiltinExtensionMode::RunMatOnly,
                description: concat!($name, " explicit real gpuArray requiring complex output is a RunMat extension"),
                error_identifier: Some(concat!("RunMat:compatibility:", $upper, "GpuRealComplexPromotionExtension")),
            };
        const EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
            INTEGER_INPUT_EXTENSION,
            LOGICAL_INPUT_EXTENSION,
            CHARACTER_INPUT_EXTENSION,
            GPU_REAL_COMPLEX_EXTENSION,
        ];
        const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
            [BuiltinIntegerInputCapability {
                name: "X",
                classes: &ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "Every integer value must be exactly representable at the binary64 computation boundary.",
            }];
        const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
            [BuiltinIntegerCapabilityDescriptor {
                form: concat!("Y = ", $name, "(integer_X)"),
                inputs: &INTEGER_INPUTS,
                computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
                output_class: BuiltinIntegerOutputClassRule::Double,
                overflow: BuiltinIntegerOverflowRule::Error,
                backend: BuiltinIntegerBackendRule::GatherFallback,
                overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
                notes: "Integer storage remains authoritative until the explicit binary64 inverse-degree computation and resident output is restored through the input owner.",
            }];
        const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];

        pub const CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
            audience: BuiltinCatalogAudience::Public,
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "math/trigonometry",
            documentation: documentation::DOCUMENTATION,
            descriptor: &DESCRIPTOR,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Incomplete,
                inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
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
                preparation: BuiltinOperandPreparation::RuntimeOwned,
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
            bindings: &REQUIRED_DEFAULT_BINDING,
            extensions: &EXTENSIONS,
            integer_capabilities: &INTEGER_CAPABILITIES,
            integer_audit: None,
            suppress_auto_output: false,
        };

        fn infer(request: &runmat_types::CallRequest, entry: &BuiltinCatalogEntry) -> runmat_types::CallInference {
            super::inference::infer(request, entry, $function)
        }
    };
}

pub(super) use define_degree_inverse_contract;
