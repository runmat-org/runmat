//! Borrowed-CSC sparse LU and tall QR for explicitly unbounded host routes.
//!
//! faer 0.24.4 privately allocates symbolic and data-dependent numeric fill.
//! Its published scratch requirements do not cover those retained factors, so
//! this owner must never be treated as a Managed allocation receipt. A future
//! allocator-aware factor backend can implement the same typed entry contract.

mod faer_adapter;
mod lu;
mod qr;

pub(in crate::builtins::math::linalg) use lu::{SparseLuError, SparseLuFactor};
pub(in crate::builtins::math::linalg) use qr::{SparseQrError, SparseQrFactor};

#[cfg(test)]
mod tests;
