macro_rules! define_shape_predicate_entry {
    (
        entry: $entry:ident,
        descriptor: $descriptor:ident,
        internal_error: $internal_error:ident,
        output_error: $output_error:ident,
        integer_audit: $integer_audit:ident,
        name: $name:literal,
        upper: $upper:literal,
        predicate: $predicate:expr,
        documentation: $documentation:expr,
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
            description: "Value whose MATLAB-visible dimensions are inspected.",
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
            when: "Value or distributed shape metadata is invalid.",
            message: concat!($name, ": invalid shape metadata"),
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
        pub const $integer_audit: crate::BuiltinIntegerAuditDescriptor =
            crate::BuiltinIntegerAuditDescriptor {
                kind: crate::BuiltinIntegerAuditKind::NotApplicable,
                canonical_builtin: None,
                notes: $integer_notes,
            };

        const BINDINGS: [crate::BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
        const EFFECTS: [runmat_types::EffectKind; 1] = [runmat_types::EffectKind::MayThrow];
        pub const $entry: crate::BuiltinCatalogEntry = crate::BuiltinCatalogEntry {
            identity: crate::BuiltinCatalogIdentity { name: $name },
            category: "array/introspection",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: crate::BuiltinContractDeclaration {
                maturity: crate::BuiltinContractMaturity::Complete,
                inference_rule: crate::BuiltinInferenceRule::Array(
                    crate::ArrayInferenceRule::ShapePredicate($predicate),
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
                distributed: crate::BuiltinDistributedPolicy::InspectHandles,
            },
            link: crate::BuiltinLinkContract {
                reachability: crate::BuiltinReachability::Always,
                policy: crate::BuiltinLinkPolicy::PortableRuntime,
                execution_stack: runmat_types::ExecutionStackRequirement::Any,
                artifact_dependencies: &[],
            },
            bindings: &BINDINGS,
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: Some(&$integer_audit),
            suppress_auto_output: false,
        };
    };
}

pub(super) use define_shape_predicate_entry;

macro_rules! define_shape_scalar_query_entry {
    (
        entry: $entry:ident,
        descriptor: $descriptor:ident,
        name: $name:literal,
        query: $query:expr,
        documentation: $documentation:expr,
        errors: $errors:expr,
        integer_capabilities: $integer_capabilities:expr,
        output_description: $output_description:literal
    ) => {
        const OUTPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
            name: "n",
            ty: crate::BuiltinParamType::NumericScalar,
            arity: crate::BuiltinParamArity::Required,
            default: None,
            description: $output_description,
        }];
        const INPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
            name: "A",
            ty: crate::BuiltinParamType::Any,
            arity: crate::BuiltinParamArity::Required,
            default: None,
            description: "Value whose MATLAB-visible dimensions are inspected.",
        }];
        const SIGNATURES: [crate::BuiltinSignatureDescriptor; 1] =
            [crate::BuiltinSignatureDescriptor {
                label: concat!("n = ", $name, "(A)"),
                inputs: &INPUTS,
                outputs: &OUTPUTS,
            }];
        pub const $descriptor: crate::BuiltinDescriptor = crate::BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: crate::BuiltinOutputMode::Fixed,
            completion_policy: crate::BuiltinCompletionPolicy::Public,
            errors: $errors,
        };
        const BINDINGS: [crate::BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
        const EFFECTS: [runmat_types::EffectKind; 1] = [runmat_types::EffectKind::MayThrow];
        pub const $entry: crate::BuiltinCatalogEntry = crate::BuiltinCatalogEntry {
            identity: crate::BuiltinCatalogIdentity { name: $name },
            category: "array/introspection",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: crate::BuiltinContractDeclaration {
                maturity: crate::BuiltinContractMaturity::Complete,
                inference_rule: crate::BuiltinInferenceRule::Array(
                    crate::ArrayInferenceRule::ShapeScalarQuery($query),
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
                distributed: crate::BuiltinDistributedPolicy::InspectHandles,
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
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

pub(super) use define_shape_scalar_query_entry;
