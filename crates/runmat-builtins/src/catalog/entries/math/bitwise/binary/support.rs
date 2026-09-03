macro_rules! define_binary_bitwise_entry {
    (
        entry: $entry:ident,
        descriptor: $descriptor:ident,
        extensions: $extensions:ident,
        single_extension: $single_extension:ident,
        gpu_domain_extension: $gpu_domain_extension:ident,
        gpu_assumed_extension: $gpu_assumed_extension:ident,
        integer_capabilities: $integer_capabilities:ident,
        name: $name:literal,
        upper: $upper:literal,
        operator: $operator:expr,
        documentation: $documentation:expr
    ) => {
        const ERRORS: [crate::BuiltinErrorDescriptor; 2] = [
            crate::BITWISE_ERROR_INVALID_INPUT,
            crate::BITWISE_ERROR_SIZE_MISMATCH,
        ];
        const OUTPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
            name: "C",
            ty: crate::BuiltinParamType::NumericArray,
            arity: crate::BuiltinParamArity::Required,
            default: None,
            description: "Bitwise result with the compatible expanded shape.",
        }];
        const INPUTS: [crate::BuiltinParamDescriptor; 2] = [
            crate::BuiltinParamDescriptor { name: "A", ty: crate::BuiltinParamType::NumericArray, arity: crate::BuiltinParamArity::Required, default: None, description: "Left integer-valued operand." },
            crate::BuiltinParamDescriptor { name: "B", ty: crate::BuiltinParamType::NumericArray, arity: crate::BuiltinParamArity::Required, default: None, description: "Right integer-valued operand." },
        ];
        const ASSUMED_INPUTS: [crate::BuiltinParamDescriptor; 3] = [
            crate::BuiltinParamDescriptor { name: "A", ty: crate::BuiltinParamType::NumericArray, arity: crate::BuiltinParamArity::Required, default: None, description: "Left integer-valued operand." },
            crate::BuiltinParamDescriptor { name: "B", ty: crate::BuiltinParamType::NumericArray, arity: crate::BuiltinParamArity::Required, default: None, description: "Right integer-valued operand." },
            crate::BuiltinParamDescriptor { name: "assumedtype", ty: crate::BuiltinParamType::StringScalar, arity: crate::BuiltinParamArity::Optional, default: None, description: "Integer class used to interpret double inputs." },
        ];
        const SIGNATURES: [crate::BuiltinSignatureDescriptor; 2] = [
            crate::BuiltinSignatureDescriptor { label: concat!("C = ", $name, "(A, B)"), inputs: &INPUTS, outputs: &OUTPUTS },
            crate::BuiltinSignatureDescriptor { label: concat!("C = ", $name, "(A, B, assumedtype)"), inputs: &ASSUMED_INPUTS, outputs: &OUTPUTS },
        ];
        pub const $descriptor: crate::BuiltinDescriptor = crate::BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: crate::BuiltinOutputMode::Fixed,
            completion_policy: crate::BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };

        pub const $single_extension: crate::BuiltinExtensionDescriptor = crate::BuiltinExtensionDescriptor {
            id: concat!($name, "-single-input"),
            mode: crate::BuiltinExtensionMode::RunMatOnly,
            description: concat!($name, " with single-precision input is a RunMat extension"),
            error_identifier: Some(concat!("RunMat:compatibility:", $upper, "SingleInputExtension")),
        };
        pub const $gpu_domain_extension: crate::BuiltinExtensionDescriptor = crate::BuiltinExtensionDescriptor {
            id: concat!($name, "-gpu-undocumented-input"),
            mode: crate::BuiltinExtensionMode::RunMatOnly,
            description: concat!($name, " with resident input outside the documented uint8/uint16/uint32 GPU domain is a RunMat extension"),
            error_identifier: Some(concat!("RunMat:compatibility:", $upper, "GpuUndocumentedInputExtension")),
        };
        pub const $gpu_assumed_extension: crate::BuiltinExtensionDescriptor = crate::BuiltinExtensionDescriptor {
            id: concat!($name, "-gpu-assumedtype"),
            mode: crate::BuiltinExtensionMode::RunMatOnly,
            description: concat!($name, " with resident input and assumedtype is a RunMat extension"),
            error_identifier: Some(concat!("RunMat:compatibility:", $upper, "GpuAssumedTypeExtension")),
        };
        pub const $extensions: [crate::BuiltinExtensionDescriptor; 3] = [
            $single_extension,
            $gpu_domain_extension,
            $gpu_assumed_extension,
        ];

        const INTEGER_INPUTS: [crate::BuiltinIntegerInputCapability; 2] = [
            crate::BuiltinIntegerInputCapability { name: "A", classes: &crate::ALL_INTEGER_CLASSES, availability: crate::BuiltinIntegerInputAvailability::Documented, scalar_double: crate::BuiltinIntegerScalarDoubleRule::Allowed, notes: "All fixed-width integer classes are accepted; a typed integer array may be paired with a scalar double." },
            crate::BuiltinIntegerInputCapability { name: "B", classes: &crate::ALL_INTEGER_CLASSES, availability: crate::BuiltinIntegerInputAvailability::Documented, scalar_double: crate::BuiltinIntegerScalarDoubleRule::Allowed, notes: "All fixed-width integer classes are accepted; a typed integer array may be paired with a scalar double." },
        ];
        pub const $integer_capabilities: [crate::BuiltinIntegerCapabilityDescriptor; 2] = [
            crate::BuiltinIntegerCapabilityDescriptor {
                form: concat!("C = ", $name, "(A, B)"),
                inputs: &INTEGER_INPUTS,
                computation_domain: crate::BuiltinIntegerComputationDomain::ExactInteger,
                output_class: crate::BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
                overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
                backend: crate::BuiltinIntegerBackendRule::GatherFallback,
                overload: crate::BuiltinIntegerOverloadKind::BroadcastCompatible,
                notes: "Same-class integer inputs retain their class and exact two's-complement representation. Documented resident execution accepts uint8, uint16, and uint32 and restores the result to the owning provider.",
            },
            crate::BuiltinIntegerCapabilityDescriptor {
                form: concat!("C = ", $name, "(A, B, assumedtype)"),
                inputs: &INTEGER_INPUTS,
                computation_domain: crate::BuiltinIntegerComputationDomain::ExactInteger,
                output_class: crate::BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
                overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
                backend: crate::BuiltinIntegerBackendRule::HostOnly,
                overload: crate::BuiltinIntegerOverloadKind::BroadcastCompatible,
                notes: "assumedtype selects the signed or unsigned bit width for double inputs and must match typed integer inputs. Resident assumedtype calls are a separately gated RunMat extension.",
            },
        ];
        const BINDINGS: [crate::BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
        const EFFECTS: [runmat_types::EffectKind; 2] = [runmat_types::EffectKind::MaySuspend, runmat_types::EffectKind::MayThrow];
        pub const $entry: crate::BuiltinCatalogEntry = crate::BuiltinCatalogEntry {
            identity: crate::BuiltinCatalogIdentity { name: $name },
            category: "math/bitwise",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: crate::BuiltinContractDeclaration {
                maturity: crate::BuiltinContractMaturity::Complete,
                inference_rule: crate::BuiltinInferenceRule::Math(crate::MathInferenceRule::Bitwise(crate::BitwiseInferenceRule::Binary($operator))),
                compatibility: crate::BuiltinCompatibility::Matlab,
                async_behavior: crate::BuiltinAsyncBehavior::MaySuspend,
                purity: crate::BuiltinPurity::Pure,
                semantic_kind: crate::BuiltinSemanticKind::Elementwise,
                workspace_effect: None,
                environment_effect: None,
                effects: &EFFECTS,
                capabilities: &[],
            },
            placement: crate::BuiltinPlacementContract {
                portability: crate::BuiltinPortability::NativeAndWasm,
                accelerator: crate::BuiltinAcceleratorPolicy::Optional,
                residency: crate::BuiltinResidencyPolicy::PreserveInputs,
                fusion: crate::BuiltinFusionPolicy::Boundary,
                distributed: crate::BuiltinDistributedPolicy::MaterializeArguments,
            },
            link: crate::BuiltinLinkContract {
                reachability: crate::BuiltinReachability::Always,
                policy: crate::BuiltinLinkPolicy::PortableRuntime,
                execution_stack: runmat_types::ExecutionStackRequirement::Any,
                artifact_dependencies: &[],
            },
            bindings: &BINDINGS,
            extensions: &$extensions,
            integer_capabilities: &$integer_capabilities,
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

pub(super) use define_binary_bitwise_entry;
