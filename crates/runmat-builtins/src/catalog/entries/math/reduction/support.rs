mod dispersion_inference;
mod dispersion_options;
mod mean_inference;
mod mean_options;
mod median_inference;
mod median_options;
mod min_options;
mod min_inference;
mod max_inference;
mod missing_policy;
mod numeric_inference;
mod product_inference;
mod sum_inference;
mod sum_options;

pub(super) use dispersion_inference::{
    admitted_std_controls, admitted_var_controls, inferred_dispersion_shape,
};
pub use dispersion_options::{DispersionControl, DispersionNormalization};
pub(super) use max_inference::{
    admitted_max_pairwise_options, admitted_max_reduction_controls, inferred_empty_max_shape,
    inferred_max_shape,
};
pub(super) use mean_inference::{admitted_mean_controls, inferred_mean_shape};
pub use mean_options::{MeanControl, MeanOutputClass};
pub(super) use median_inference::{admitted_median_controls, inferred_median_shape};
pub use median_options::MedianControl;
pub(super) use min_inference::{
    admitted_min_pairwise_options, admitted_min_reduction_controls, inferred_min_shape,
};
pub use min_options::{MinComparisonMethod, MinControl};
pub use missing_policy::ReductionNaN;
pub(super) use numeric_inference::LiteralReductionOutput;
pub(super) use numeric_inference::{has_literal_zero_axis, is_empty_placeholder};
pub(super) use product_inference::{admitted_product_controls, inferred_product_shape};
pub(super) use sum_inference::{admitted_sum_controls, inferred_sum_shape};
pub use sum_options::{SumControl, SumOutputClass};

macro_rules! define_logical_reduction_entry {
    (
        entry: $entry:ident,
        descriptor: $descriptor:ident,
        name: $name:literal,
        kind: $kind:expr,
        documentation: $documentation:expr,
        errors: $errors:expr,
        extensions: $extensions:expr,
        integer_capabilities: $integer_capabilities:expr
    ) => {
        const OUTPUTS: [crate::BuiltinParamDescriptor; 1] = [crate::BuiltinParamDescriptor {
            name: "B",
            ty: crate::BuiltinParamType::LogicalArray,
            arity: crate::BuiltinParamArity::Required,
            default: None,
            description: "Logical reduction result with reduced dimensions retained at size one.",
        }];
        const INPUT_A: crate::BuiltinParamDescriptor = crate::BuiltinParamDescriptor {
            name: "A",
            ty: crate::BuiltinParamType::Any,
            arity: crate::BuiltinParamArity::Required,
            default: None,
            description: "Numeric, logical, complex, or character input array.",
        };
        const INPUT_DIM: crate::BuiltinParamDescriptor = crate::BuiltinParamDescriptor {
            name: "dim",
            ty: crate::BuiltinParamType::Any,
            arity: crate::BuiltinParamArity::Required,
            default: None,
            description: "Positive scalar dimension or vector of distinct dimensions.",
        };
        const INPUT_ALL: crate::BuiltinParamDescriptor = crate::BuiltinParamDescriptor {
            name: "all",
            ty: crate::BuiltinParamType::StringScalar,
            arity: crate::BuiltinParamArity::Required,
            default: Some("\"all\""),
            description: "Reduce across every dimension.",
        };
        const INPUT_NANFLAG: crate::BuiltinParamDescriptor = crate::BuiltinParamDescriptor {
            name: "nanflag",
            ty: crate::BuiltinParamType::StringScalar,
            arity: crate::BuiltinParamArity::Required,
            default: None,
            description: "RunMat-only NaN policy: \"omitnan\" or \"includenan\".",
        };
        const INPUTS_CORE: [crate::BuiltinParamDescriptor; 1] = [INPUT_A];
        const INPUTS_DIM: [crate::BuiltinParamDescriptor; 2] = [INPUT_A, INPUT_DIM];
        const INPUTS_ALL: [crate::BuiltinParamDescriptor; 2] = [INPUT_A, INPUT_ALL];
        const INPUTS_NANFLAG: [crate::BuiltinParamDescriptor; 2] = [INPUT_A, INPUT_NANFLAG];
        const INPUTS_DIM_NANFLAG: [crate::BuiltinParamDescriptor; 3] =
            [INPUT_A, INPUT_DIM, INPUT_NANFLAG];
        const INPUTS_NANFLAG_DIM: [crate::BuiltinParamDescriptor; 3] =
            [INPUT_A, INPUT_NANFLAG, INPUT_DIM];
        const INPUTS_OPTIONS: [crate::BuiltinParamDescriptor; 2] = [INPUT_A, crate::BuiltinParamDescriptor {
            name: "options", ty: crate::BuiltinParamType::Any, arity: crate::BuiltinParamArity::Variadic,
            default: None, description: "Existing option grammar: empty dimension placeholders, at most one nonempty dimension selector or all selector, and at most one NaN flag.",
        }];
        const SIGNATURES: [crate::BuiltinSignatureDescriptor; 7] = [
            crate::BuiltinSignatureDescriptor {
                label: concat!("B = ", $name, "(A)"),
                inputs: &INPUTS_CORE,
                outputs: &OUTPUTS,
            },
            crate::BuiltinSignatureDescriptor {
                label: concat!("B = ", $name, "(A, dim)"),
                inputs: &INPUTS_DIM,
                outputs: &OUTPUTS,
            },
            crate::BuiltinSignatureDescriptor {
                label: concat!("B = ", $name, "(A, \"all\")"),
                inputs: &INPUTS_ALL,
                outputs: &OUTPUTS,
            },
            crate::BuiltinSignatureDescriptor {
                label: concat!("B = ", $name, "(A, nanflag)"),
                inputs: &INPUTS_NANFLAG,
                outputs: &OUTPUTS,
            },
            crate::BuiltinSignatureDescriptor {
                label: concat!("B = ", $name, "(A, dim, nanflag)"),
                inputs: &INPUTS_DIM_NANFLAG,
                outputs: &OUTPUTS,
            },
            crate::BuiltinSignatureDescriptor {
                label: concat!("B = ", $name, "(A, nanflag, dim)"),
                inputs: &INPUTS_NANFLAG_DIM,
                outputs: &OUTPUTS,
            },
            crate::BuiltinSignatureDescriptor {
                label: concat!("B = ", $name, "(A, options...)"),
                inputs: &INPUTS_OPTIONS,
                outputs: &OUTPUTS,
            },
        ];
        pub const $descriptor: crate::BuiltinDescriptor = crate::BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: crate::BuiltinOutputMode::Fixed,
            completion_policy: crate::BuiltinCompletionPolicy::Public,
            errors: $errors,
        };
        const BINDINGS: [crate::BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
        const EFFECTS: [runmat_types::EffectKind; 2] = [
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ];
        pub const $entry: crate::BuiltinCatalogEntry = crate::BuiltinCatalogEntry {
            audience: crate::BuiltinCatalogAudience::Public,
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: crate::BuiltinCatalogIdentity { name: $name },
            category: "math/reduction",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: crate::BuiltinContractDeclaration {
                maturity: crate::BuiltinContractMaturity::Complete,
                inference_rule: crate::BuiltinInferenceRule::Math(
                    crate::MathInferenceRule::LogicalReduction($kind),
                ),
                compatibility: crate::BuiltinCompatibility::Matlab,
                async_behavior: crate::BuiltinAsyncBehavior::MaySuspend,
                purity: crate::BuiltinPurity::Pure,
                semantic_kind: crate::BuiltinSemanticKind::Reduction,
                workspace_effect: None,
                environment_effect: None,
                effects: &EFFECTS,
                capabilities: &[],
            },
            placement: crate::BuiltinPlacementContract {
                preparation: crate::BuiltinOperandPreparation::Forms(&[crate::BuiltinPreparationForm {
                    id: crate::BuiltinPreparationFormId(0), binding_variant: "default",
                    signatures: &[0, 1, 2, 3, 4, 5, 6], data_arguments: &[0],
                    operation: crate::BuiltinPreparationOperation::Reduction(match $kind {
                        crate::LogicalReductionKind::All => crate::BuiltinPreparationReduction::All,
                        crate::LogicalReductionKind::Any => crate::BuiltinPreparationReduction::Any,
                    }),
                    provider_requirement: crate::BuiltinPreparationProviderRequirement::InputSupported,
                }]),
                portability: crate::BuiltinPortability::NativeAndWasm,
                accelerator: crate::BuiltinAcceleratorPolicy::Optional,
                residency: crate::BuiltinResidencyPolicy::Host,
                // The legacy shader template is not a typed logical-result
                // fusion lowering. Provider reductions remain available.
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
            extensions: $extensions,
            integer_capabilities: $integer_capabilities,
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

pub(super) use define_logical_reduction_entry;
