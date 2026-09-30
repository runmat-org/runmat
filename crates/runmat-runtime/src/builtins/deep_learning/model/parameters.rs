//! One canonical network parameter authority in `Layers`.
//!
//! The public Learnables table is a derived projection. Typed keys, borrowed
//! admission, presentation, and atomic replacement live in bounded modules.
use super::*;

mod projection;
mod schema;
mod update;
mod validation;
mod view;

use schema::{ParameterKey, ParameterRole, ParameterShape, ParameterView};
use validation::{empty_parameter, parameter_data, validate_parameter};
use view::view;

pub(in crate::builtins) use projection::project;
pub(in crate::builtins) use update::apply;
pub(in crate::builtins) use view::validate_matching_networks;
