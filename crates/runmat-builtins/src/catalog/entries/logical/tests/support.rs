macro_rules! define_numeric_classification_entry {
    (
        entry: $entry:ident,
        descriptor: $descriptor:ident,
        invalid_error: $invalid_error:ident,
        internal_error: $internal_error:ident,
        output_error: $output_error:ident,
        name: $name:literal,
        upper: $upper:literal,
        predicate: $predicate:expr,
        documentation: $documentation:expr,
        input_description: $input_description:literal,
        output_description: $output_description:literal,
        integer_notes: $integer_notes:literal
    ) => {
        const OUTPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
            name: "tf",
            ty: crate::BuiltinParamType::LogicalArray,
            arity: crate::BuiltinParamArity::Required,
            default: None,
            description: $output_description,
        }];
        const INPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
            name: "A",
            ty: crate::BuiltinParamType::Any,
            arity: crate::BuiltinParamArity::Required,
            default: None,
            description: $input_description,
        }];
        const SIGNATURES: [crate::BuiltinSignatureDescriptor; 1] =
            [crate::BuiltinSignatureDescriptor {
                label: concat!("tf = ", $name, "(A)"),
                inputs: &INPUTS,
                outputs: &OUTPUTS,
            }];

        pub const $invalid_error: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INVALID_INPUT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidInput")),
            when: "Input is sparse or is not numeric, logical, character, or string data.",
            message: concat!($name, ": invalid input"),
        };
        pub const $internal_error: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INTERNAL"),
            identifier: Some(concat!("RunMat:", $name, ":InternalError")),
            when: "Internal mask construction, provider execution, or residency restoration fails.",
            message: concat!($name, ": internal error"),
        };
        pub const $output_error: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".TOO_MANY_OUTPUTS"),
            identifier: Some(concat!("RunMat:", $name, ":TooManyOutputs")),
            when: "More than one output is requested.",
            message: concat!($name, ": too many output arguments"),
        };
        const ERRORS: [crate::BuiltinErrorDescriptor; 3] =
            [$invalid_error, $internal_error, $output_error];
        pub const $descriptor: crate::BuiltinDescriptor = crate::BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: crate::BuiltinOutputMode::Fixed,
            completion_policy: crate::BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };

        const INTEGER_INPUTS: [crate::BuiltinIntegerInputCapability; 1] =
            [crate::BuiltinIntegerInputCapability {
                name: "A",
                classes: &crate::ALL_INTEGER_CLASSES,
                availability: crate::BuiltinIntegerInputAvailability::Documented,
                scalar_double: crate::BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: $integer_notes,
            }];
        const INTEGER_CAPABILITIES: [crate::BuiltinIntegerCapabilityDescriptor; 1] =
            [crate::BuiltinIntegerCapabilityDescriptor {
                form: concat!("tf = ", $name, "(integer_A)"),
                inputs: &INTEGER_INPUTS,
                computation_domain: crate::BuiltinIntegerComputationDomain::Predicate,
                output_class: crate::BuiltinIntegerOutputClassRule::Logical,
                overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
                backend: crate::BuiltinIntegerBackendRule::HostAndGpu,
                overload: crate::BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
                notes: $integer_notes,
            }];

        const BINDINGS: [crate::BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
        const EFFECTS: [runmat_types::EffectKind; 2] = [
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ];
        pub const $entry: crate::BuiltinCatalogEntry = crate::BuiltinCatalogEntry {
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: crate::BuiltinCatalogIdentity { name: $name },
            category: "logical/tests",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: crate::BuiltinContractDeclaration {
                maturity: crate::BuiltinContractMaturity::Complete,
                inference_rule: crate::BuiltinInferenceRule::Logical(
                    crate::LogicalInferenceRule::NumericClassification($predicate),
                ),
                compatibility: crate::BuiltinCompatibility::Matlab,
                async_behavior: crate::BuiltinAsyncBehavior::MaySuspend,
                purity: crate::BuiltinPurity::Pure,
                semantic_kind: crate::BuiltinSemanticKind::General,
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
                distributed: crate::BuiltinDistributedPolicy::MapUnary,
            },
            link: crate::BuiltinLinkContract {
                reachability: crate::BuiltinReachability::Always,
                policy: crate::BuiltinLinkPolicy::PortableRuntime,
                execution_stack: runmat_types::ExecutionStackRequirement::Any,
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

pub(super) use define_numeric_classification_entry;

macro_rules! define_metadata_predicate_entry {
    (
        entry: $entry:ident,
        descriptor: $descriptor:ident,
        internal_error: $internal_error:ident,
        output_error: $output_error:ident,
        name: $name:literal,
        upper: $upper:literal,
        predicate: $predicate:expr,
        documentation: $documentation:expr,
        input_description: $input_description:literal,
        output_description: $output_description:literal,
        distributed: $distributed:expr,
        integer_capabilities: $integer_capabilities:expr,
        integer_audit: $integer_audit:expr
    ) => {
        const OUTPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
            name: "tf",
            ty: crate::BuiltinParamType::LogicalArray,
            arity: crate::BuiltinParamArity::Required,
            default: None,
            description: $output_description,
        }];
        const INPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
            name: "A",
            ty: crate::BuiltinParamType::Any,
            arity: crate::BuiltinParamArity::Required,
            default: None,
            description: $input_description,
        }];
        const SIGNATURES: [crate::BuiltinSignatureDescriptor; 1] =
            [crate::BuiltinSignatureDescriptor {
                label: concat!("tf = ", $name, "(A)"),
                inputs: &INPUTS,
                outputs: &OUTPUTS,
            }];

        pub const $internal_error: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INTERNAL"),
            identifier: Some(concat!("RunMat:", $name, ":InternalError")),
            when: "Internal value or resident-handle metadata is contradictory.",
            message: concat!($name, ": internal error"),
        };
        pub const $output_error: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".TOO_MANY_OUTPUTS"),
            identifier: Some(concat!("RunMat:", $name, ":TooManyOutputs")),
            when: "More than one output is requested.",
            message: concat!($name, ": too many output arguments"),
        };
        const ERRORS: [crate::BuiltinErrorDescriptor; 2] = [$internal_error, $output_error];
        pub const $descriptor: crate::BuiltinDescriptor = crate::BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: crate::BuiltinOutputMode::Fixed,
            completion_policy: crate::BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };

        const BINDINGS: [crate::BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
        const EFFECTS: [runmat_types::EffectKind; 1] = [runmat_types::EffectKind::MayThrow];
        pub const $entry: crate::BuiltinCatalogEntry = crate::BuiltinCatalogEntry {
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: crate::BuiltinCatalogIdentity { name: $name },
            category: "logical/tests",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: crate::BuiltinContractDeclaration {
                maturity: crate::BuiltinContractMaturity::Complete,
                inference_rule: crate::BuiltinInferenceRule::Logical(
                    crate::LogicalInferenceRule::MetadataPredicate($predicate),
                ),
                compatibility: crate::BuiltinCompatibility::Matlab,
                async_behavior: crate::BuiltinAsyncBehavior::NeverSuspends,
                purity: crate::BuiltinPurity::Pure,
                semantic_kind: crate::BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &EFFECTS,
                capabilities: &[],
            },
            placement: crate::BuiltinPlacementContract {
                portability: crate::BuiltinPortability::NativeAndWasm,
                accelerator: crate::BuiltinAcceleratorPolicy::Optional,
                residency: crate::BuiltinResidencyPolicy::Host,
                fusion: crate::BuiltinFusionPolicy::Boundary,
                distributed: $distributed,
            },
            link: crate::BuiltinLinkContract {
                reachability: crate::BuiltinReachability::Always,
                policy: crate::BuiltinLinkPolicy::PortableRuntime,
                execution_stack: runmat_types::ExecutionStackRequirement::Any,
                artifact_dependencies: &[],
            },
            bindings: &BINDINGS,
            extensions: &[],
            integer_capabilities: $integer_capabilities,
            integer_audit: $integer_audit,
            suppress_auto_output: false,
        };
    };
}

pub(super) use define_metadata_predicate_entry;
