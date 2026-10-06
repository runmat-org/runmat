//! Shared RunMat figure archive, argument, scene, and error implementation.

use std::collections::HashSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[cfg(test)]
use runmat_builtins::BuiltinIntegerAuditKind;
use runmat_builtins::{
    FIGURE_PERSISTENCE_ERROR_IMPORT as ERROR_IMPORT,
    FIGURE_PERSISTENCE_ERROR_INTERNAL as ERROR_INTERNAL,
    FIGURE_PERSISTENCE_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    FIGURE_PERSISTENCE_ERROR_IO as ERROR_IO, FIGURE_PERSISTENCE_ERROR_RENDER as ERROR_RENDER,
    FIGURE_PERSISTENCE_ERROR_UNSUPPORTED_FORMAT as ERROR_UNSUPPORTED_FORMAT,
};
pub use runmat_builtins::{
    HGLOAD_DESCRIPTOR, HGLOAD_INTEGER_AUDIT, HGSAVE_DESCRIPTOR, HGSAVE_INTEGER_AUDIT,
    OPENFIG_DESCRIPTOR, OPENFIG_INTEGER_AUDIT, SAVEAS_DESCRIPTOR, SAVEAS_INTEGER_AUDIT,
    SAVEFIG_DESCRIPTOR, SAVEFIG_INTEGER_AUDIT,
};
use runmat_value::{CellArray, StructValue, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::replay::limits::ReplayLimits;
use crate::{BuiltinResult, RuntimeError};

use super::properties::{get_properties, resolve_plot_handle, set_properties, PlotHandle};
use super::state::{current_figure_handle, set_figure_visible, FigureHandle};
#[cfg(feature = "plot-core")]
use super::{export_figure_scene, import_figure_scene_async, render_figure_snapshot};

pub(super) const SAVEAS: &str = "saveas";
pub(super) const SAVEFIG: &str = "savefig";
pub(super) const OPENFIG: &str = "openfig";
pub(super) const HGSAVE: &str = "hgsave";
pub(super) const HGLOAD: &str = "hgload";

const DEFAULT_WIDTH: u32 = 800;
const DEFAULT_HEIGHT: u32 = 600;
const FIG_MAGIC: &[u8] = b"RUNMAT-FIGURE-BUNDLE-V1\n";
const MAX_FIGURE_BUNDLE_COUNT: usize = 4096;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::figure_persistence")]
pub const SAVEAS_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "saveas",
    op_kind: GpuOpKind::Custom("figure-export"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "saveas gathers only handle/path/format arguments. PNG output uses the shared plotting renderer; FIG output serializes the figure scene replay payload.",
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::figure_persistence")]
pub const SAVEFIG_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "savefig",
    op_kind: GpuOpKind::Custom("figure-scene-export"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "savefig is an I/O sink and serializes RunMat figure scene payloads without forcing numeric plot data through scalar CPU loops.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "figure_persistence",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Figure persistence performs plotting I/O and terminates fusion graphs.",
};

mod hgload;
mod hgsave;
mod openfig;
mod saveas;
mod savefig;

pub use hgload::hgload_builtin;
pub use hgsave::hgsave_builtin;
pub use openfig::openfig_builtin;
pub use saveas::saveas_builtin;
pub use savefig::savefig_builtin;

#[cfg(target_arch = "wasm32")]
pub(crate) use hgload::__runmat_wasm_register_builtin_hgload_builtin_registered as __runmat_wasm_register_builtin_hgload_builtin;
#[cfg(target_arch = "wasm32")]
pub(crate) use hgsave::__runmat_wasm_register_builtin_hgsave_builtin_registered as __runmat_wasm_register_builtin_hgsave_builtin;
#[cfg(target_arch = "wasm32")]
pub(crate) use openfig::__runmat_wasm_register_builtin_openfig_builtin_registered as __runmat_wasm_register_builtin_openfig_builtin;
#[cfg(target_arch = "wasm32")]
pub(crate) use saveas::__runmat_wasm_register_builtin_saveas_builtin_registered as __runmat_wasm_register_builtin_saveas_builtin;
#[cfg(target_arch = "wasm32")]
pub(crate) use savefig::__runmat_wasm_register_builtin_savefig_builtin_registered as __runmat_wasm_register_builtin_savefig_builtin;

mod arguments;
mod archive;
mod errors;
mod filesystem;
mod handles;
mod operations;
mod overrides;
mod paths;
mod publication;
mod request_types;
mod scene;
#[cfg(test)]
mod tests;

use archive::*;
use arguments::*;
use errors::*;
use filesystem::*;
use handles::*;
use operations::*;
use overrides::*;
use paths::*;
use publication::*;
use request_types::*;
use scene::*;
