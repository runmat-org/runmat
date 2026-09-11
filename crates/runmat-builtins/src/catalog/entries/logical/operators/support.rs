macro_rules! define_binary_logical_entry {
    (
        entry: $entry:ident,
        descriptor: $descriptor:ident,
        invalid: $invalid:ident,
        mismatch: $mismatch:ident,
        integer_capabilities: $integer_capabilities:ident,
        name: $name:literal,
        upper: $upper:literal,
        operator: $operator:expr,
        extensions: $extensions:expr,
        documentation: $documentation:expr,
        input_description: $input_description:literal
    ) => {
        pub const $invalid: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INVALID_INPUT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidInput")),
            when: "An operand is not supported logical, numeric, character, complex, table, timetable, or resident numeric data.",
            message: concat!($name, ": unsupported input type"),
        };
        pub const $mismatch: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".SIZE_MISMATCH"),
            identifier: Some(concat!("RunMat:", $name, ":SizeMismatch")),
            when: "Operand shapes are not compatible for implicit expansion, or tabular operands cannot be aligned.",
            message: concat!($name, ": input sizes are not compatible"),
        };
        const ERRORS: [crate::BuiltinErrorDescriptor; 2] = [$invalid, $mismatch];
        const OUTPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
            name: "tf",
            ty: crate::BuiltinParamType::Any,
            arity: crate::BuiltinParamArity::Required,
            default: None,
            description: "Logical result with the expanded array shape or the input tabular container.",
        }];
        const INPUTS: [crate::BuiltinParamDescriptor; 2] = [
            crate::BuiltinParamDescriptor { name: "A", ty: crate::BuiltinParamType::Any, arity: crate::BuiltinParamArity::Required, default: None, description: $input_description },
            crate::BuiltinParamDescriptor { name: "B", ty: crate::BuiltinParamType::Any, arity: crate::BuiltinParamArity::Required, default: None, description: $input_description },
        ];
        const SIGNATURES: [crate::BuiltinSignatureDescriptor; 1] = [crate::BuiltinSignatureDescriptor {
            label: concat!("tf = ", $name, "(A, B)"),
            inputs: &INPUTS,
            outputs: &OUTPUTS,
        }];
        pub const $descriptor: crate::BuiltinDescriptor = crate::BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: crate::BuiltinOutputMode::Fixed,
            completion_policy: crate::BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
        const INTEGER_INPUTS: [crate::BuiltinIntegerInputCapability; 2] = [
            crate::BuiltinIntegerInputCapability { name: "A", classes: &crate::ALL_INTEGER_CLASSES, availability: crate::BuiltinIntegerInputAvailability::Documented, scalar_double: crate::BuiltinIntegerScalarDoubleRule::Allowed, notes: "Zero is false and every nonzero fixed-width integer is true without binary64 conversion." },
            crate::BuiltinIntegerInputCapability { name: "B", classes: &crate::ALL_INTEGER_CLASSES, availability: crate::BuiltinIntegerInputAvailability::Documented, scalar_double: crate::BuiltinIntegerScalarDoubleRule::Allowed, notes: "Zero is false and every nonzero fixed-width integer is true without binary64 conversion." },
        ];
        pub const $integer_capabilities: [crate::BuiltinIntegerCapabilityDescriptor; 1] =
            [crate::BuiltinIntegerCapabilityDescriptor {
                form: concat!("tf = ", $name, "(integer_A, integer_B)"),
                inputs: &INTEGER_INPUTS,
                computation_domain: crate::BuiltinIntegerComputationDomain::Predicate,
                output_class: crate::BuiltinIntegerOutputClassRule::Logical,
                overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
                backend: crate::BuiltinIntegerBackendRule::GatherFallback,
                overload: crate::BuiltinIntegerOverloadKind::BroadcastCompatible,
                notes: "Integer truth is evaluated from authoritative typed storage; explicitly resident fallback results return to the owning provider.",
            }];
        const BINDINGS: [crate::BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
        const EFFECTS: [runmat_types::EffectKind; 2] = [
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ];
        pub const $entry: crate::BuiltinCatalogEntry = crate::BuiltinCatalogEntry {
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: crate::BuiltinCatalogIdentity { name: $name },
            category: "logical/bit",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: crate::BuiltinContractDeclaration {
                maturity: crate::BuiltinContractMaturity::Complete,
                inference_rule: crate::BuiltinInferenceRule::Logical(
                    crate::LogicalInferenceRule::Elementwise(
                        crate::LogicalElementwiseRule::Binary($operator),
                    ),
                ),
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
                fusion: crate::BuiltinFusionPolicy::Candidate,
                distributed: crate::BuiltinDistributedPolicy::MaterializeArguments,
            },
            link: crate::BuiltinLinkContract {
                reachability: crate::BuiltinReachability::Always,
                policy: crate::BuiltinLinkPolicy::PortableRuntime,
                execution_stack: runmat_types::ExecutionStackRequirement::Any,
                artifact_dependencies: &[],
            },
            bindings: &BINDINGS,
            extensions: $extensions,
            integer_capabilities: &$integer_capabilities,
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

pub(super) use define_binary_logical_entry;
