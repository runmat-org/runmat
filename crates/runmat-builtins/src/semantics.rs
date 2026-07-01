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

    pub const fn with_time(mut self) -> Self {
        self.time = true;
        self
    }

    pub const fn with_host_callback(mut self) -> Self {
        self.host_callback = true;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinSemanticKind {
    General,
    Elementwise,
    ArrayConstructor,
    ParameterizedArrayConstructor,
    PermutationConstructor,
    RangeConstructor,
    EmptyConstructor,
    ShapeTransform(ShapeTransformKind),
    Reduction,
    LinearAlgebra,
    Plotting,
    Filesystem,
    Network,
    Workspace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeTransformKind {
    General,
    Reshape,
    Permute,
    Repmat,
    Dot,
    Transpose,
    Concatenate(ConcatKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConcatKind {
    Dimension,
    Horizontal,
    Vertical,
}

pub fn builtin_semantics_for(function: &BuiltinFunction) -> BuiltinSemantics {
    builtin_semantics_for_name(function.name).unwrap_or_else(|| derive_semantics(function))
}

pub fn builtin_semantics_for_name(name: &str) -> Option<BuiltinSemantics> {
    Some(match name {
        "ones" | "zeros" => pure(BuiltinSemanticKind::ArrayConstructor),
        "empty" => pure(BuiltinSemanticKind::EmptyConstructor),
        "range" | "colon" | "linspace" => pure(BuiltinSemanticKind::RangeConstructor),
        "rand" | "randn" | "unifrnd" | "normrnd" | "exprnd" => {
            random(BuiltinSemanticKind::ArrayConstructor)
        }
        "randi" => random(BuiltinSemanticKind::ParameterizedArrayConstructor),
        "randperm" => random(BuiltinSemanticKind::PermutationConstructor),
        "rng" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_random(),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::General,
            ..pure(BuiltinSemanticKind::General)
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
        "ipermute" => pure(BuiltinSemanticKind::ShapeTransform(
            ShapeTransformKind::Permute,
        )),
        "transpose" | "ctranspose" => pure(BuiltinSemanticKind::ShapeTransform(
            ShapeTransformKind::Transpose,
        )),
        "squeeze" | "flip" | "fliplr" | "flipud" | "rot90" | "circshift" | "tril" | "triu"
        | "diag" | "kron" => pure(BuiltinSemanticKind::ShapeTransform(
            ShapeTransformKind::General,
        )),
        "cat" => pure(BuiltinSemanticKind::ShapeTransform(
            ShapeTransformKind::Concatenate(ConcatKind::Dimension),
        )),
        "horzcat" => pure(BuiltinSemanticKind::ShapeTransform(
            ShapeTransformKind::Concatenate(ConcatKind::Horizontal),
        )),
        "vertcat" => pure(BuiltinSemanticKind::ShapeTransform(
            ShapeTransformKind::Concatenate(ConcatKind::Vertical),
        )),
        "dot" => pure(BuiltinSemanticKind::ShapeTransform(ShapeTransformKind::Dot)),
        "sum" | "mean" | "max" | "min" => pure(BuiltinSemanticKind::Reduction),

        "tic" | "toc" | "datetime" => time_effect(BuiltinAsyncBehavior::NeverSuspends),
        "pause" => time_effect(BuiltinAsyncBehavior::MaySuspend),
        "timeit" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_time().with_host_callback(),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::General,
            ..pure(BuiltinSemanticKind::General)
        },

        "input" => BuiltinSemantics {
            compatibility: BuiltinCompatibility::InteractiveOnly,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            effects: BuiltinEffects::none().with_ui().with_host_callback(),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::General,
            ..BuiltinSemantics::unknown()
        },
        "disp" | "clc" => console_io(),
        "format" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_environment().with_ui(),
            environment_effect: Some(BuiltinEnvironmentEffect::DynamicLookupInvalidation),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..pure(BuiltinSemanticKind::Workspace)
        },

        "who" | "whos" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_workspace(),
            workspace_effect: Some(BuiltinWorkspaceEffect::ReadsWorkspace),
            purity: BuiltinPurity::DeterministicReadOnly,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..pure(BuiltinSemanticKind::Workspace)
        },
        "which" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_environment().with_workspace(),
            workspace_effect: Some(BuiltinWorkspaceEffect::ReadsWorkspace),
            environment_effect: Some(BuiltinEnvironmentEffect::DynamicLookupInvalidation),
            purity: BuiltinPurity::DeterministicReadOnly,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..pure(BuiltinSemanticKind::Workspace)
        },

        "jsondecode" | "jsonencode" | "fullfile" => pure(BuiltinSemanticKind::General),
        "pwd" | "getenv" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_environment(),
            purity: BuiltinPurity::DeterministicReadOnly,
            semantic_kind: BuiltinSemanticKind::General,
            ..pure(BuiltinSemanticKind::General)
        },
        "setenv" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_environment(),
            environment_effect: Some(BuiltinEnvironmentEffect::DynamicLookupInvalidation),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::General,
            ..pure(BuiltinSemanticKind::General)
        },
        "tempname" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_filesystem().with_random(),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Filesystem,
            ..pure(BuiltinSemanticKind::Filesystem)
        },

        "feval" | "call_method" | "subsref" | "subsasgn" | "notify" | "fzero" | "fsolve"
        | "ode45" | "ode23" | "ode15s" => host_callback(),
        "addlistener" | "new_handle_object" => BuiltinSemantics {
            effects: BuiltinEffects::unknown(),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::General,
            ..BuiltinSemantics::unknown()
        },

        "load" | "run" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_filesystem().with_workspace(),
            workspace_effect: Some(BuiltinWorkspaceEffect::LoadsExternalBindings),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Filesystem,
            ..BuiltinSemantics::unknown()
        },
        "save" | "fopen" | "fclose" | "fread" | "fwrite" | "fileread" | "filewrite"
        | "copyfile" | "movefile" | "delete" | "mkdir" | "rmdir" | "dir" | "readmatrix"
        | "csvwrite" | "csvread" | "dlmread" | "dlmwrite" | "writematrix" | "ls" | "genpath" => {
            BuiltinSemantics {
                effects: BuiltinEffects::none().with_filesystem(),
                purity: BuiltinPurity::Impure,
                semantic_kind: BuiltinSemanticKind::Filesystem,
                ..BuiltinSemantics::unknown()
            }
        }
        "fprintf" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_filesystem().with_ui(),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Filesystem,
            ..BuiltinSemantics::unknown()
        },
        "uigetfile" | "uiputfile" => BuiltinSemantics {
            compatibility: BuiltinCompatibility::InteractiveOnly,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            effects: BuiltinEffects::none()
                .with_filesystem()
                .with_ui()
                .with_host_callback(),
            workspace_effect: None,
            environment_effect: None,
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Filesystem,
        },
        "webread" | "webwrite" | "tcpclient" | "tcpserver" | "accept" | "read" | "readline"
        | "write" => network_io(),
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
        "savepath" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_environment().with_filesystem(),
            environment_effect: Some(BuiltinEnvironmentEffect::FunctionCacheInvalidation),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..BuiltinSemantics::unknown()
        },
        "exist" => BuiltinSemantics {
            effects: BuiltinEffects::none()
                .with_filesystem()
                .with_workspace()
                .with_environment(),
            workspace_effect: Some(BuiltinWorkspaceEffect::ReadsWorkspace),
            environment_effect: Some(BuiltinEnvironmentEffect::DynamicLookupInvalidation),
            purity: BuiltinPurity::DeterministicReadOnly,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..BuiltinSemantics::unknown()
        },
        "tempdir" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_environment(),
            purity: BuiltinPurity::DeterministicReadOnly,
            semantic_kind: BuiltinSemanticKind::General,
            ..BuiltinSemantics::unknown()
        },
        "eval" | "evalin" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_workspace(),
            workspace_effect: Some(BuiltinWorkspaceEffect::DynamicEval),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..BuiltinSemantics::unknown()
        },
        "assignin" | "syms" => BuiltinSemantics {
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

fn derive_semantics(function: &BuiltinFunction) -> BuiltinSemantics {
    let mut semantics = BuiltinSemantics {
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        effects: BuiltinEffects::unknown(),
        workspace_effect: None,
        environment_effect: None,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
    };

    let category = function.category.to_ascii_lowercase();
    if category.contains("plot") {
        semantics.semantic_kind = BuiltinSemanticKind::Plotting;
        semantics.effects = BuiltinEffects::none().with_ui();
        semantics.async_behavior = BuiltinAsyncBehavior::MaySuspend;
        semantics.purity = BuiltinPurity::Impure;
    } else if category.contains("file") {
        semantics.semantic_kind = BuiltinSemanticKind::Filesystem;
        semantics.effects = BuiltinEffects::none().with_filesystem();
        semantics.async_behavior = BuiltinAsyncBehavior::MaySuspend;
        semantics.purity = BuiltinPurity::Impure;
    } else if category.contains("linalg") || category.contains("linear") {
        semantics.async_behavior = BuiltinAsyncBehavior::NeverSuspends;
        semantics.effects = BuiltinEffects::none();
        semantics.purity = BuiltinPurity::Pure;
        semantics.semantic_kind = BuiltinSemanticKind::LinearAlgebra;
    } else if category.contains("reduction") {
        semantics.async_behavior = BuiltinAsyncBehavior::NeverSuspends;
        semantics.effects = BuiltinEffects::none();
        semantics.purity = BuiltinPurity::Pure;
        semantics.semantic_kind = BuiltinSemanticKind::Reduction;
    } else if category.contains("array/creation") {
        semantics.async_behavior = BuiltinAsyncBehavior::NeverSuspends;
        semantics.effects = BuiltinEffects::none();
        semantics.purity = BuiltinPurity::Pure;
        semantics.semantic_kind = BuiltinSemanticKind::ArrayConstructor;
    } else if category.contains("array/shape") {
        semantics.async_behavior = BuiltinAsyncBehavior::NeverSuspends;
        semantics.effects = BuiltinEffects::none();
        semantics.purity = BuiltinPurity::Pure;
        semantics.semantic_kind = BuiltinSemanticKind::ShapeTransform(ShapeTransformKind::General);
    } else if function
        .accel_tags
        .iter()
        .any(|tag| matches!(tag, AccelTag::Elementwise | AccelTag::Unary))
    {
        semantics.async_behavior = BuiltinAsyncBehavior::NeverSuspends;
        semantics.effects = BuiltinEffects::none();
        semantics.purity = BuiltinPurity::Pure;
        semantics.semantic_kind = BuiltinSemanticKind::Elementwise;
    } else if function
        .accel_tags
        .iter()
        .any(|tag| matches!(tag, AccelTag::ArrayConstruct))
    {
        semantics.async_behavior = BuiltinAsyncBehavior::NeverSuspends;
        semantics.effects = BuiltinEffects::none();
        semantics.purity = BuiltinPurity::Pure;
        semantics.semantic_kind = BuiltinSemanticKind::ArrayConstructor;
    }

    if function.is_sink {
        semantics.purity = BuiltinPurity::Impure;
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

fn random(semantic_kind: BuiltinSemanticKind) -> BuiltinSemantics {
    BuiltinSemantics {
        effects: BuiltinEffects::none().with_random(),
        purity: BuiltinPurity::Impure,
        semantic_kind,
        ..pure(semantic_kind)
    }
}

fn time_effect(async_behavior: BuiltinAsyncBehavior) -> BuiltinSemantics {
    BuiltinSemantics {
        async_behavior,
        effects: BuiltinEffects::none().with_time(),
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        ..pure(BuiltinSemanticKind::General)
    }
}

fn console_io() -> BuiltinSemantics {
    BuiltinSemantics {
        effects: BuiltinEffects::none().with_ui().with_host_callback(),
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        ..pure(BuiltinSemanticKind::General)
    }
}

fn network_io() -> BuiltinSemantics {
    BuiltinSemantics {
        async_behavior: BuiltinAsyncBehavior::RequiresAsyncRuntime,
        effects: BuiltinEffects::none().with_network(),
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Network,
        ..pure(BuiltinSemanticKind::Network)
    }
}

fn host_callback() -> BuiltinSemantics {
    BuiltinSemantics {
        effects: BuiltinEffects::none().with_host_callback(),
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        ..BuiltinSemantics::unknown()
    }
}
