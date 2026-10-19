//! Shared RunMat figure archive, argument, scene, and error implementation.

use std::collections::HashSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use runmat_builtins::{
    FIGURE_PERSISTENCE_ERROR_IMPORT as ERROR_IMPORT,
    FIGURE_PERSISTENCE_ERROR_INTERNAL as ERROR_INTERNAL,
    FIGURE_PERSISTENCE_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    FIGURE_PERSISTENCE_ERROR_IO as ERROR_IO, FIGURE_PERSISTENCE_ERROR_RENDER as ERROR_RENDER,
    FIGURE_PERSISTENCE_ERROR_UNSUPPORTED_FORMAT as ERROR_UNSUPPORTED_FORMAT,
};
use runmat_value::{CellArray, StructValue, Tensor, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::replay::limits::ReplayLimits;
use crate::{BuiltinResult, RuntimeError};

use super::properties::{get_properties, resolve_plot_handle, set_properties, PlotHandle};
#[cfg(test)]
use super::state::clone_figure;
use super::state::{
    capture_figure_persistence_snapshot, current_figure_handle, set_figure_visible, FigureHandle,
};

pub(super) const SAVEAS: &str = "saveas";
pub(super) const SAVEFIG: &str = "savefig";
pub(super) const OPENFIG: &str = "openfig";
pub(super) const HGSAVE: &str = "hgsave";
pub(super) const HGLOAD: &str = "hgload";

const DEFAULT_WIDTH: u32 = 800;
const DEFAULT_HEIGHT: u32 = 600;
const FIG_V1_MAGIC: &[u8] = b"RUNMAT-FIGURE-BUNDLE-V1\n";
const FIG_V2_MAGIC: &[u8] = b"RUNMAT-FIGURE-ARCHIVE-V2\n";
const MAX_FIGURE_BUNDLE_COUNT: usize = 4096;
const MAX_FIGURE_ARCHIVE_BYTES: usize = 256 * 1024 * 1024;

mod hgload;
mod hgsave;
mod openfig;
mod saveas;
mod savefig;

#[cfg(test)]
use hgload::hgload_builtin;
#[cfg(test)]
use hgsave::hgsave_builtin;
#[cfg(test)]
use openfig::openfig_builtin;
#[cfg(test)]
use saveas::saveas_builtin;
#[cfg(test)]
use savefig::savefig_builtin;

#[cfg(target_arch = "wasm32")]
pub(crate) use hgload::__runmat_wasm_register_builtin_hgload_builtin_registered;
#[cfg(target_arch = "wasm32")]
pub(crate) use hgsave::__runmat_wasm_register_builtin_hgsave_builtin_registered;
#[cfg(target_arch = "wasm32")]
pub(crate) use openfig::__runmat_wasm_register_builtin_openfig_builtin_registered;
#[cfg(target_arch = "wasm32")]
pub(crate) use saveas::__runmat_wasm_register_builtin_saveas_builtin_registered;
#[cfg(target_arch = "wasm32")]
pub(crate) use savefig::__runmat_wasm_register_builtin_savefig_builtin_registered;

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
mod reuse;
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
use reuse::*;
use scene::*;
