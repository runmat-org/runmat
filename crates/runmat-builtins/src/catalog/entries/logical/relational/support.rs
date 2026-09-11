macro_rules! define_relational_entry {
    (
        entry: $entry:ident,
        descriptor: $descriptor:ident,
        invalid: $invalid:ident,
        mismatch: $mismatch:ident,
        ownership: $ownership:ident,
        upload: $upload:ident,
        integer_capabilities: $integer_capabilities:ident,
        name: $name:literal,
        upper: $upper:literal,
        operator: $operator:expr,
        documentation: $documentation:expr,
        output_description: $output_description:literal
    ) => {
        pub const $invalid: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INVALID_INPUT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidInput")),
            when: "Operands contain unsupported types or mix numeric and text domains.",
            message: concat!($name, ": mixing numeric and string inputs is not supported"),
        };
        pub const $mismatch: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".SIZE_MISMATCH"),
            identifier: Some(concat!("RunMat:", $name, ":SizeMismatch")),
            when: "Operands are not broadcast-compatible.",
            message: concat!($name, ": array sizes are not compatible for broadcasting"),
        };
        pub const $ownership: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".PROVIDER_OWNERSHIP_MISMATCH"),
            identifier: Some("RunMat:gpu:ProviderOwnershipMismatch"),
            when: "Resident operands do not have one exact owning provider.",
            message: concat!($name, ": resident operands must have one exact owning provider"),
        };
        pub const $upload: crate::BuiltinErrorDescriptor = crate::BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".GPU_UPLOAD_FAILED"),
            identifier: Some(concat!("RunMat:", $name, ":GpuUploadFailed")),
            when: "An explicitly resident logical result cannot be restored to its provider.",
            message: concat!($name, ": failed to preserve explicit gpuArray residency"),
        };
        const ERRORS: &[crate::BuiltinErrorDescriptor] =
            &[$invalid, $mismatch, $ownership, $upload];
        const OUTPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
            name: "tf",
            ty: crate::BuiltinParamType::LogicalArray,
            arity: crate::BuiltinParamArity::Required,
            default: None,
            description: $output_description,
        }];
        const INPUTS: [crate::BuiltinParamDescriptor; 2] = [
            crate::BuiltinParamDescriptor {
                name: "A",
                ty: crate::BuiltinParamType::Any,
                arity: crate::BuiltinParamArity::Required,
                default: None,
                description: "Left operand.",
            },
            crate::BuiltinParamDescriptor {
                name: "B",
                ty: crate::BuiltinParamType::Any,
                arity: crate::BuiltinParamArity::Required,
                default: None,
                description: "Right operand.",
            },
        ];
        const SIGNATURES: [crate::BuiltinSignatureDescriptor; 1] =
            [crate::BuiltinSignatureDescriptor {
                label: concat!("tf = ", $name, "(A, B)"),
                inputs: &INPUTS,
                outputs: &OUTPUTS,
            }];
        pub const $descriptor: crate::BuiltinDescriptor = crate::BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: crate::BuiltinOutputMode::Fixed,
            completion_policy: crate::BuiltinCompletionPolicy::Public,
            errors: ERRORS,
        };
        const INTEGER_INPUTS: [crate::BuiltinIntegerInputCapability; 2] = [
            crate::BuiltinIntegerInputCapability {
                name: "A",
                classes: &crate::ALL_INTEGER_CLASSES,
                availability: crate::BuiltinIntegerInputAvailability::Documented,
                scalar_double: crate::BuiltinIntegerScalarDoubleRule::Allowed,
                notes: "Every fixed-width integer class compares without conversion through binary64.",
            },
            crate::BuiltinIntegerInputCapability {
                name: "B",
                classes: &crate::ALL_INTEGER_CLASSES,
                availability: crate::BuiltinIntegerInputAvailability::Documented,
                scalar_double: crate::BuiltinIntegerScalarDoubleRule::Allowed,
                notes: "Mixed integer and floating operands retain exact relational ordering.",
            },
        ];
        pub const $integer_capabilities: [crate::BuiltinIntegerCapabilityDescriptor; 1] =
            [crate::BuiltinIntegerCapabilityDescriptor {
                form: concat!("tf = ", $name, "(A, B)"),
                inputs: &INTEGER_INPUTS,
                computation_domain: crate::BuiltinIntegerComputationDomain::Predicate,
                output_class: crate::BuiltinIntegerOutputClassRule::Logical,
                overflow: crate::BuiltinIntegerOverflowRule::NotApplicable,
                backend: crate::BuiltinIntegerBackendRule::HostAndGpu,
                overload: crate::BuiltinIntegerOverloadKind::BroadcastCompatible,
                notes: "Signed, unsigned, floating, logical, character, and supported complex values compare without widening authoritative integer storage.",
            }];
        const BINDINGS: [crate::BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
        const EFFECTS: [runmat_types::EffectKind; 2] = [
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ];
        pub const $entry: crate::BuiltinCatalogEntry = crate::BuiltinCatalogEntry {
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: crate::BuiltinCatalogIdentity { name: $name },
            category: "logical/rel",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: crate::BuiltinContractDeclaration {
                maturity: crate::BuiltinContractMaturity::Complete,
                inference_rule: crate::BuiltinInferenceRule::Logical(
                    crate::LogicalInferenceRule::Relational($operator),
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
            extensions: &[],
            integer_capabilities: &$integer_capabilities,
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

pub(super) use define_relational_entry;
