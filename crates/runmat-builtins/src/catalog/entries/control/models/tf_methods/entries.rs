use super::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

macro_rules! method_entry {
    ($constant:ident, $name:literal, $documentation:ident, $descriptor:ident, $inference:ident, $extensions:expr, $integers:expr) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            audience: crate::BuiltinCatalogAudience::Public,
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "control",
            documentation: documentation::$documentation,
            descriptor: &contract::$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Incomplete,
                inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
                    inference::$inference
                        as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
                )),
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::MaySuspend,
                purity: BuiltinPurity::Pure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &[EffectKind::MaySuspend, EffectKind::MayThrow],
                capabilities: &[],
            },
            placement: BuiltinPlacementContract {
                preparation: BuiltinOperandPreparation::RuntimeOwned,
                portability: BuiltinPortability::NativeAndWasm,
                accelerator: BuiltinAcceleratorPolicy::Forbidden,
                residency: BuiltinResidencyPolicy::GatherToHost,
                fusion: BuiltinFusionPolicy::Boundary,
                distributed: BuiltinDistributedPolicy::Unsupported,
            },
            link: BuiltinLinkContract {
                reachability: BuiltinReachability::Always,
                policy: BuiltinLinkPolicy::PortableRuntime,
                execution_stack: ExecutionStackRequirement::Any,
                artifact_dependencies: &[],
            },
            bindings: &REQUIRED_DEFAULT_BINDING,
            extensions: $extensions,
            integer_capabilities: $integers,
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

method_entry!(
    TF_PLUS_CATALOG_ENTRY,
    "tf.plus",
    PLUS,
    PLUS_DESCRIPTOR,
    binary,
    &contract::CONSUMED_MODEL_EXTENSIONS,
    &contract::INTEGER_GAIN_CAPABILITIES
);
method_entry!(
    TF_MINUS_CATALOG_ENTRY,
    "tf.minus",
    MINUS,
    MINUS_DESCRIPTOR,
    binary,
    &contract::CONSUMED_MODEL_EXTENSIONS,
    &contract::INTEGER_GAIN_CAPABILITIES
);
method_entry!(
    TF_UPLUS_CATALOG_ENTRY,
    "tf.uplus",
    UPLUS,
    UPLUS_DESCRIPTOR,
    unary,
    &contract::CONSUMED_MODEL_EXTENSIONS,
    &[]
);
method_entry!(
    TF_UMINUS_CATALOG_ENTRY,
    "tf.uminus",
    UMINUS,
    UMINUS_DESCRIPTOR,
    unary,
    &contract::CONSUMED_MODEL_EXTENSIONS,
    &[]
);
method_entry!(
    TF_TIMES_CATALOG_ENTRY,
    "tf.times",
    TIMES,
    TIMES_DESCRIPTOR,
    binary,
    &contract::CONSUMED_MODEL_EXTENSIONS,
    &contract::INTEGER_GAIN_CAPABILITIES
);
method_entry!(
    TF_MTIMES_CATALOG_ENTRY,
    "tf.mtimes",
    MTIMES,
    MTIMES_DESCRIPTOR,
    binary,
    &contract::CONSUMED_MODEL_EXTENSIONS,
    &contract::INTEGER_GAIN_CAPABILITIES
);
method_entry!(
    TF_RDIVIDE_CATALOG_ENTRY,
    "tf.rdivide",
    RDIVIDE,
    RDIVIDE_DESCRIPTOR,
    binary,
    &contract::CONSUMED_MODEL_EXTENSIONS,
    &contract::INTEGER_GAIN_CAPABILITIES
);
method_entry!(
    TF_MRDIVIDE_CATALOG_ENTRY,
    "tf.mrdivide",
    MRDIVIDE,
    MRDIVIDE_DESCRIPTOR,
    binary,
    &contract::CONSUMED_MODEL_EXTENSIONS,
    &contract::INTEGER_GAIN_CAPABILITIES
);
method_entry!(
    TF_LDIVIDE_CATALOG_ENTRY,
    "tf.ldivide",
    LDIVIDE,
    LDIVIDE_DESCRIPTOR,
    binary,
    &contract::CONSUMED_MODEL_EXTENSIONS,
    &contract::INTEGER_GAIN_CAPABILITIES
);
method_entry!(
    TF_MLDIVIDE_CATALOG_ENTRY,
    "tf.mldivide",
    MLDIVIDE,
    MLDIVIDE_DESCRIPTOR,
    binary,
    &contract::CONSUMED_MODEL_EXTENSIONS,
    &contract::INTEGER_GAIN_CAPABILITIES
);
method_entry!(
    TF_POWER_CATALOG_ENTRY,
    "tf.power",
    POWER,
    POWER_DESCRIPTOR,
    power,
    &contract::CONSUMED_MODEL_EXTENSIONS,
    &contract::INTEGER_EXPONENT_CAPABILITIES
);
method_entry!(
    TF_MPOWER_CATALOG_ENTRY,
    "tf.mpower",
    MPOWER,
    MPOWER_DESCRIPTOR,
    power,
    &contract::CONSUMED_MODEL_EXTENSIONS,
    &contract::INTEGER_EXPONENT_CAPABILITIES
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &TF_PLUS_CATALOG_ENTRY,
    &TF_MINUS_CATALOG_ENTRY,
    &TF_UPLUS_CATALOG_ENTRY,
    &TF_UMINUS_CATALOG_ENTRY,
    &TF_TIMES_CATALOG_ENTRY,
    &TF_MTIMES_CATALOG_ENTRY,
    &TF_RDIVIDE_CATALOG_ENTRY,
    &TF_MRDIVIDE_CATALOG_ENTRY,
    &TF_LDIVIDE_CATALOG_ENTRY,
    &TF_MLDIVIDE_CATALOG_ENTRY,
    &TF_POWER_CATALOG_ENTRY,
    &TF_MPOWER_CATALOG_ENTRY,
];
