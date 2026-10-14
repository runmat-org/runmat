use super::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

macro_rules! layer_entry {
    ($constant:ident, $name:literal, $documentation:ident, $descriptor:ident, $inference:ident, $extensions:expr, $integer_capabilities:expr) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            audience: crate::BuiltinCatalogAudience::Public,
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "deep_learning",
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
            integer_capabilities: $integer_capabilities,
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

layer_entry!(
    FULLY_CONNECTED_CATALOG_ENTRY,
    "fullyConnectedLayer",
    FULLY_CONNECTED,
    FULLY_CONNECTED_DESCRIPTOR,
    fully_connected,
    &[],
    &contract::FC_INTEGER_CAPABILITIES
);
layer_entry!(
    FEATURE_INPUT_CATALOG_ENTRY,
    "featureInputLayer",
    FEATURE_INPUT,
    FEATURE_INPUT_DESCRIPTOR,
    feature_input,
    &contract::FEATURE_INPUT_EXTENSIONS,
    &contract::FEATURE_INTEGER_CAPABILITIES
);
layer_entry!(
    SEQUENCE_INPUT_CATALOG_ENTRY,
    "sequenceInputLayer",
    SEQUENCE_INPUT,
    SEQUENCE_INPUT_DESCRIPTOR,
    sequence_input,
    &[],
    &contract::SEQUENCE_INTEGER_CAPABILITIES
);
layer_entry!(
    RELU_CATALOG_ENTRY,
    "reluLayer",
    RELU,
    RELU_DESCRIPTOR,
    relu,
    &[],
    &[]
);
layer_entry!(
    ELU_CATALOG_ENTRY,
    "eluLayer",
    ELU,
    ELU_DESCRIPTOR,
    elu,
    &[],
    &[]
);
layer_entry!(
    SOFTMAX_CATALOG_ENTRY,
    "softmaxLayer",
    SOFTMAX,
    SOFTMAX_DESCRIPTOR,
    softmax,
    &[],
    &[]
);
layer_entry!(
    CLASSIFICATION_CATALOG_ENTRY,
    "classificationLayer",
    CLASSIFICATION,
    CLASSIFICATION_DESCRIPTOR,
    classification,
    &[],
    &[]
);
layer_entry!(
    REGRESSION_CATALOG_ENTRY,
    "regressionLayer",
    REGRESSION,
    REGRESSION_DESCRIPTOR,
    regression,
    &[],
    &[]
);
layer_entry!(
    LAYER_NORM_CATALOG_ENTRY,
    "layerNormalizationLayer",
    LAYER_NORM,
    LAYER_NORM_DESCRIPTOR,
    layer_norm,
    &[],
    &[]
);
layer_entry!(
    GLOBAL_POOL_CATALOG_ENTRY,
    "globalAveragePooling1dLayer",
    GLOBAL_POOL,
    GLOBAL_POOL_DESCRIPTOR,
    global_pool,
    &[],
    &[]
);
layer_entry!(
    LSTM_CATALOG_ENTRY,
    "lstmLayer",
    LSTM,
    LSTM_DESCRIPTOR,
    lstm,
    &[],
    &contract::LSTM_INTEGER_CAPABILITIES
);
layer_entry!(
    BILSTM_CATALOG_ENTRY,
    "bilstmLayer",
    BILSTM,
    BILSTM_DESCRIPTOR,
    bilstm,
    &[],
    &contract::BILSTM_INTEGER_CAPABILITIES
);
layer_entry!(
    CONVOLUTION_CATALOG_ENTRY,
    "convolution1dLayer",
    CONVOLUTION,
    CONVOLUTION_DESCRIPTOR,
    convolution,
    &[],
    &contract::CONV_INTEGER_CAPABILITIES
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &FULLY_CONNECTED_CATALOG_ENTRY,
    &FEATURE_INPUT_CATALOG_ENTRY,
    &SEQUENCE_INPUT_CATALOG_ENTRY,
    &RELU_CATALOG_ENTRY,
    &ELU_CATALOG_ENTRY,
    &SOFTMAX_CATALOG_ENTRY,
    &CLASSIFICATION_CATALOG_ENTRY,
    &REGRESSION_CATALOG_ENTRY,
    &LAYER_NORM_CATALOG_ENTRY,
    &GLOBAL_POOL_CATALOG_ENTRY,
    &LSTM_CATALOG_ENTRY,
    &BILSTM_CATALOG_ENTRY,
    &CONVOLUTION_CATALOG_ENTRY,
];
