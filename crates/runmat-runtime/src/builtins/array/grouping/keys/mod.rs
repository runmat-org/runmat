//! Exact group-key representation and ordering-independent indexing.

mod atom;
mod index;

pub(crate) use atom::{format_integer, KeyAtom};
pub(crate) use index::{GroupIndex, KeyOrder};
