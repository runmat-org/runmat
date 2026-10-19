//! Shared exact group-key representation and ordering-independent indexing.

mod atom;
mod index;

pub(crate) use atom::group_key_work_units;
pub(crate) use atom::{format_integer, KeyAtom};
#[cfg(feature = "plot-core")]
pub(crate) use index::GroupIndexBuildError;
pub(crate) use index::{GroupIndex, KeyOrder};
