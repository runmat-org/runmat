macro_rules! define_component_projection {
    (
        $name:literal,
        $upper:literal,
        $rule:expr,
        $documentation:expr,
        $output_description:literal,
        $integer_notes:literal,
        $capability_notes:literal
    ) => {
        const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
            name: "Y",
            ty: BuiltinParamType::NumericArray,
            arity: BuiltinParamArity::Required,
            default: None,
            description: $output_description,
        }];
        const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
            name: "X",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Numeric, logical, character, or complex input.",
        }];
        const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: concat!("Y = ", $name, "(X)"),
            inputs: &INPUTS,
            outputs: &OUTPUTS,
        }];

        pub const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INVALID_INPUT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidInput")),
            when: "Input cannot be interpreted as numeric, logical, character, or complex data.",
            message: concat!($name, ": invalid input"),
        };
        pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INTERNAL"),
            identifier: Some(concat!("RunMat:", $name, ":Internal")),
            when: "Internal tensor conversion, allocation, or provider interaction fails.",
            message: concat!($name, ": internal error"),
        };
        const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];
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
                notes: $integer_notes,
            }];
        pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
            [BuiltinIntegerCapabilityDescriptor {
                form: concat!("Y = ", $name, "(integer_X)"),
                inputs: &INTEGER_INPUTS,
                computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
                output_class: BuiltinIntegerOutputClassRule::PreserveInput,
                overflow: BuiltinIntegerOverflowRule::NotApplicable,
                backend: BuiltinIntegerBackendRule::HostAndGpu,
                overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
                notes: $capability_notes,
            }];

        const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
        pub const CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
            identity: BuiltinCatalogIdentity { name: $name },
            category: "math/elementwise",
            documentation: $documentation,
            descriptor: &DESCRIPTOR,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Complete,
                inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::NumericComponent(
                    $rule,
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
    };
}

pub(super) use define_component_projection;
