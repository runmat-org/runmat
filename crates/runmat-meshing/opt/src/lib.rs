//! Smoothing, untangling, exact-quality repair, and sliver recovery stages.

pub const CRATE_PURPOSE: &str =
    "post-recovery smoothing, untangling, exact-quality repair, and sliver removal";

pub mod exact_quality;
pub mod sliver;
pub mod smooth;
pub mod untangle;
