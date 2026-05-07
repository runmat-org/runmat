use crate::{AccelTag, BuiltinFunction};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuiltinSemantics {
    pub compatibility: BuiltinCompatibility,
    pub async_behavior: BuiltinAsyncBehavior,
    pub effects: BuiltinEffects,
    pub workspace_effect: Option<BuiltinWorkspaceEffect>,
    pub environment_effect: Option<BuiltinEnvironmentEffect>,
    pub purity: BuiltinPurity,
    pub semantic_kind: BuiltinSemanticKind,
}

impl BuiltinSemantics {
    pub const fn unknown() -> Self {
        Self {
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            effects: BuiltinEffects::unknown(),
            workspace_effect: None,
            environment_effect: None,
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::General,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinCompatibility {
    Matlab,
    RunMatExtended,
    InteractiveOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinAsyncBehavior {
    NeverSuspends,
    MaySuspend,
    RequiresAsyncRuntime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinPurity {
    Pure,
    DeterministicReadOnly,
    Impure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinWorkspaceEffect {
    ReadsWorkspace,
    CreatesBinding,
    ClearsBinding,
    ClearsFunctionCache,
    LoadsExternalBindings,
    DynamicEval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinEnvironmentEffect {
    PathMutation,
    WorkingDirectoryMutation,
    FunctionCacheInvalidation,
    DynamicLookupInvalidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuiltinEffects {
    pub workspace: bool,
    pub environment: bool,
    pub filesystem: bool,
    pub network: bool,
    pub ui: bool,
    pub random: bool,
    pub time: bool,
    pub host_callback: bool,
    pub unknown: bool,
}

impl BuiltinEffects {
    pub const fn none() -> Self {
        Self {
            workspace: false,
            environment: false,
            filesystem: false,
            network: false,
            ui: false,
            random: false,
            time: false,
            host_callback: false,
            unknown: false,
        }
    }

    pub const fn unknown() -> Self {
        Self {
            unknown: true,
            ..Self::none()
        }
    }

    pub const fn with_workspace(mut self) -> Self {
        self.workspace = true;
        self
    }

    pub const fn with_environment(mut self) -> Self {
        self.environment = true;
        self
    }

    pub const fn with_filesystem(mut self) -> Self {
        self.filesystem = true;
        self
    }

    pub const fn with_network(mut self) -> Self {
        self.network = true;
        self
    }

    pub const fn with_ui(mut self) -> Self {
        self.ui = true;
        self
    }

    pub const fn with_random(mut self) -> Self {
        self.random = true;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinSemanticKind {
    General,
    Elementwise,
    ArrayConstructor,
    ShapeTransform(ShapeTransformKind),
    Reduction,
    LinearAlgebra,
    DataApi(DataApiOp),
    Plotting,
    Filesystem,
    Network,
    Workspace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeTransformKind {
    Reshape,
    Permute,
    Repmat,
    Dot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataApiOp {
    Namespace,
    Open,
    Array,
    Read,
    Write,
    BeginTransaction,
    Commit,
}

pub fn builtin_semantics_for(function: &BuiltinFunction) -> BuiltinSemantics {
    builtin_semantics_for_name(function.name).unwrap_or_else(|| derive_semantics(function))
}

pub fn builtin_semantics_for_name(name: &str) -> Option<BuiltinSemantics> {
    Some(match name {
        "data" => data_api(DataApiOp::Namespace),
        "data.open" => data_api(DataApiOp::Open),

        "ones" | "zeros" => pure(BuiltinSemanticKind::ArrayConstructor),
        "rand" | "randn" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_random(),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::ArrayConstructor,
            ..pure(BuiltinSemanticKind::ArrayConstructor)
        },
        "reshape" => pure(BuiltinSemanticKind::ShapeTransform(
            ShapeTransformKind::Reshape,
        )),
        "permute" => pure(BuiltinSemanticKind::ShapeTransform(
            ShapeTransformKind::Permute,
        )),
        "repmat" => pure(BuiltinSemanticKind::ShapeTransform(
            ShapeTransformKind::Repmat,
        )),
        "dot" => pure(BuiltinSemanticKind::ShapeTransform(ShapeTransformKind::Dot)),
        "sum" | "mean" | "max" | "min" => pure(BuiltinSemanticKind::Reduction),

        "load" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_filesystem().with_workspace(),
            workspace_effect: Some(BuiltinWorkspaceEffect::LoadsExternalBindings),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Filesystem,
            ..BuiltinSemantics::unknown()
        },
        "save" | "fopen" | "fclose" | "fread" | "fwrite" | "fprintf" | "fileread" | "filewrite"
        | "copyfile" | "movefile" | "delete" | "mkdir" | "rmdir" | "dir" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_filesystem(),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Filesystem,
            ..BuiltinSemantics::unknown()
        },
        "webread" | "webwrite" | "tcpclient" | "tcpserver" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_network(),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Network,
            ..BuiltinSemantics::unknown()
        },
        "path" | "addpath" | "rmpath" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_environment(),
            environment_effect: Some(BuiltinEnvironmentEffect::PathMutation),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..BuiltinSemantics::unknown()
        },
        "cd" | "chdir" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_environment(),
            environment_effect: Some(BuiltinEnvironmentEffect::WorkingDirectoryMutation),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..BuiltinSemantics::unknown()
        },
        "rehash" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_environment(),
            environment_effect: Some(BuiltinEnvironmentEffect::FunctionCacheInvalidation),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..BuiltinSemantics::unknown()
        },
        "eval" | "evalin" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_workspace(),
            workspace_effect: Some(BuiltinWorkspaceEffect::DynamicEval),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..BuiltinSemantics::unknown()
        },
        "assignin" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_workspace(),
            workspace_effect: Some(BuiltinWorkspaceEffect::CreatesBinding),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..BuiltinSemantics::unknown()
        },
        "clear" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_workspace(),
            workspace_effect: Some(BuiltinWorkspaceEffect::ClearsBinding),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..BuiltinSemantics::unknown()
        },
        _ => return None,
    })
}

pub fn data_api_method_op_for_name(name: &str) -> Option<DataApiOp> {
    match name {
        "open" => Some(DataApiOp::Open),
        "array" => Some(DataApiOp::Array),
        "read" => Some(DataApiOp::Read),
        "write" => Some(DataApiOp::Write),
        "begin" => Some(DataApiOp::BeginTransaction),
        "commit" => Some(DataApiOp::Commit),
        _ => None,
    }
}

pub fn is_data_namespace_symbol(name: &str) -> bool {
    name == "data"
}

pub fn is_data_open_name(name: &str) -> bool {
    name == "data.open"
}

fn derive_semantics(function: &BuiltinFunction) -> BuiltinSemantics {
    let mut semantics = BuiltinSemantics {
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        effects: BuiltinEffects::none(),
        workspace_effect: None,
        environment_effect: None,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
    };

    if function.is_sink {
        semantics.purity = BuiltinPurity::Impure;
    }

    let category = function.category.to_ascii_lowercase();
    if category.contains("plot") {
        semantics.semantic_kind = BuiltinSemanticKind::Plotting;
        semantics.effects = semantics.effects.with_ui();
        semantics.async_behavior = BuiltinAsyncBehavior::MaySuspend;
        semantics.purity = BuiltinPurity::Impure;
    } else if category.contains("io") || category.contains("file") {
        semantics.semantic_kind = BuiltinSemanticKind::Filesystem;
        semantics.effects = semantics.effects.with_filesystem();
        semantics.async_behavior = BuiltinAsyncBehavior::MaySuspend;
        semantics.purity = BuiltinPurity::Impure;
    } else if category.contains("linalg") || category.contains("linear") {
        semantics.semantic_kind = BuiltinSemanticKind::LinearAlgebra;
    } else if category.contains("reduction") {
        semantics.semantic_kind = BuiltinSemanticKind::Reduction;
    } else if function
        .accel_tags
        .iter()
        .any(|tag| matches!(tag, AccelTag::Elementwise | AccelTag::Unary))
    {
        semantics.semantic_kind = BuiltinSemanticKind::Elementwise;
    } else if function
        .accel_tags
        .iter()
        .any(|tag| matches!(tag, AccelTag::ArrayConstruct))
    {
        semantics.semantic_kind = BuiltinSemanticKind::ArrayConstructor;
    }

    semantics
}

fn pure(semantic_kind: BuiltinSemanticKind) -> BuiltinSemantics {
    BuiltinSemantics {
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        effects: BuiltinEffects::none(),
        workspace_effect: None,
        environment_effect: None,
        purity: BuiltinPurity::Pure,
        semantic_kind,
    }
}

fn data_api(op: DataApiOp) -> BuiltinSemantics {
    BuiltinSemantics {
        compatibility: BuiltinCompatibility::RunMatExtended,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        effects: BuiltinEffects::none().with_filesystem(),
        workspace_effect: None,
        environment_effect: None,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::DataApi(op),
    }
}
