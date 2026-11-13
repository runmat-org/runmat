//! MATLAB-compatible triangulation objects and helpers.
use runmat_types::MemberAccess;

use std::collections::HashMap;
use std::sync::OnceLock;

use runmat_builtins::{BuiltinErrorDescriptor, BuiltinExtensionDescriptor};
use runmat_geometry_ops::{boundary_edges, delaunay_2d, nearest_neighbor_indices, point_locations};
use runmat_macros::runtime_builtin;
use runmat_value::{IntegerStorage, NumericScalar, ObjectInstance, Tensor, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, current_requested_outputs, BuiltinResult};

mod contract;
mod entrypoints;
mod registration;
mod constructor;
mod queries;
mod coordinates;
mod preflight;
mod results;
mod errors;
#[cfg(test)]
mod tests;

use constructor::*;
pub use contract::*;
use coordinates::*;
pub use entrypoints::*;
use errors::*;
use preflight::*;
use queries::*;
use registration::ensure_registered;
pub(crate) use registration::register_delaunaytri_class;
use results::*;
