use crate::{AccelTag, BuiltinFunction};
use serde::{Deserialize, Serialize};

pub mod executor;

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
    /// RunMat-native identity with no claim of MATLAB API parity.
    RunMat,
    InteractiveOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinAsyncBehavior {
    NeverSuspends,
    MaySuspend,
    /// Deterministic work on original arguments may yield cooperatively or
    /// await their owned numerical provider completion. No ambient host work.
    MaySuspendLocally,
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
    DynamicEvalCurrentWorkspace,
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
    ExecutorIntrinsic(executor::BuiltinExecutorIntrinsic),
    General,
    /// Sealed handle/listener validity query: receiver state is data, not a method lookup.
    HandleValidityQuery,
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
    Flip,
    Rotate90,
    CircularShift,
    ShiftDimensions,
    Diagonal,
    Kronecker,
    Triangular(TriangularKind),
    Reshape,
    Permute,
    Repmat,
    Dot,
    Transpose,
    Concatenate(ConcatKind),
    BlockDiagonal,
    RepeatElements,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConcatKind {
    Dimension,
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriangularKind {
    Lower,
    Upper,
}

pub fn builtin_semantics_for(function: &BuiltinFunction) -> BuiltinSemantics {
    crate::builtin_catalog_entry_by_name(function.name)
        .map(crate::BuiltinCatalogEntry::legacy_semantics)
        .or_else(|| builtin_semantics_for_name(function.name))
        .unwrap_or_else(|| derive_semantics(function))
}

const GENERATED_GEOMETRY_BUILDERS: &[&str] = &[
    "geometry.box",
    "geometry.cylinder",
    "geometry.sphere",
    "geometry.transform",
    "geometry.instance",
    "geometry.extrude",
    "geometry.revolve",
    "geometry.hole",
    "geometry.pocket",
    "geometry.fillet",
    "geometry.chamfer",
    "geometry.union",
    "geometry.intersect",
    "geometry.subtract",
    "geometry.finish",
    "geometry.solid",
    "geometry.faces",
    "geometry.edges",
    "geometry.facesIntroducedBy",
    "geometry.edgesIntroducedBy",
    "geometry.withRegion",
];

pub fn builtin_semantics_for_name(name: &str) -> Option<BuiltinSemantics> {
    if GENERATED_GEOMETRY_BUILDERS.contains(&name) {
        return Some(pure(BuiltinSemanticKind::General));
    }
    Some(match name {
        // All supported values, including GPU handles, expose element counts in
        // resident metadata. This reads no workspace or device data and never suspends.
        "numel" => pure(BuiltinSemanticKind::General),
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
        "transpose" | "ctranspose" => pure(BuiltinSemanticKind::ShapeTransform(
            ShapeTransformKind::Transpose,
        )),
        "tril" | "triu" | "diag" | "kron" => pure(BuiltinSemanticKind::ShapeTransform(
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
        "disp" => console_io(),
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
            purity: BuiltinPurity::DeterministicReadOnly,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..pure(BuiltinSemanticKind::Workspace)
        },

        "jsondecode" | "jsonencode" => pure(BuiltinSemanticKind::General),
        "feval" | "call_method" | "subsref" | "subsasgn" | "notify" | "fzero" | "fsolve"
        | "ode45" | "ode23" | "ode15s" => host_callback(),
        "addlistener" | "new_handle_object" => BuiltinSemantics {
            effects: BuiltinEffects::unknown(),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::General,
            ..BuiltinSemantics::unknown()
        },

        "load" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_filesystem().with_workspace(),
            workspace_effect: Some(BuiltinWorkspaceEffect::LoadsExternalBindings),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Filesystem,
            ..BuiltinSemantics::unknown()
        },
        "save" | "fopen" | "fclose" | "fread" | "fwrite" | "fileread" | "filewrite" | "delete"
        | "readmatrix" | "csvwrite" | "csvread" | "dlmread" | "dlmwrite" | "writematrix"
        | "writecell" | "xlsread" | "xlswrite" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_filesystem(),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Filesystem,
            ..BuiltinSemantics::unknown()
        },
        "fprintf" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_filesystem().with_ui(),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Filesystem,
            ..BuiltinSemantics::unknown()
        },
        "webread" | "webwrite" | "tcpclient" | "tcpserver" | "accept" | "read" | "readline"
        | "write" => network_io(),
        "rehash" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_environment(),
            environment_effect: Some(BuiltinEnvironmentEffect::FunctionCacheInvalidation),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..BuiltinSemantics::unknown()
        },
        "eval" | "evalc" => BuiltinSemantics {
            effects: BuiltinEffects::none().with_workspace(),
            workspace_effect: Some(BuiltinWorkspaceEffect::DynamicEvalCurrentWorkspace),
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::Workspace,
            ..BuiltinSemantics::unknown()
        },
        "evalin" => BuiltinSemantics {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_count_is_pure_metadata_without_authorizing_unknown_calls() {
        let semantics = builtin_semantics_for_name("numel").unwrap();
        assert_eq!(semantics.purity, BuiltinPurity::Pure);
        assert_eq!(
            semantics.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        );
        assert_eq!(semantics.effects, BuiltinEffects::none());
        assert_eq!(semantics.workspace_effect, None);
        assert_eq!(semantics.environment_effect, None);
        assert!(builtin_semantics_for_name("unresolved_callable").is_none());
    }

    #[test]
    fn generated_geometry_builders_are_pure_and_non_suspending() {
        for name in GENERATED_GEOMETRY_BUILDERS {
            let semantics = builtin_semantics_for_name(name).unwrap();
            assert_eq!(semantics.purity, BuiltinPurity::Pure, "{name}");
            assert_eq!(
                semantics.async_behavior,
                BuiltinAsyncBehavior::NeverSuspends,
                "{name}"
            );
            assert_eq!(semantics.effects, BuiltinEffects::none(), "{name}");
        }
    }
}
