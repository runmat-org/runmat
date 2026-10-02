//! Shared exact group-key representation and ordering-independent indexing.

mod atom;
mod index;

pub(crate) use atom::{format_integer, group_key_work_units, KeyAtom};
pub(crate) use index::{GroupIndex, GroupIndexBuildError, KeyOrder};
