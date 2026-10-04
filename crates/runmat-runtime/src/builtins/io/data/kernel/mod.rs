//! Shared data-family payload, shape, owner, and manifest kernels.

use super::*;

mod manifest;
mod owner;
mod slice;
mod write;

pub(super) use manifest::*;
pub(super) use owner::*;
pub(super) use slice::*;
pub(super) use write::*;
