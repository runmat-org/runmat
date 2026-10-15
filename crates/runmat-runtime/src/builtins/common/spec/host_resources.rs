//! Optional physical allocation contract of one selected implementation.
//! This grants neither semantic admission nor authority to run a host branch.
use runmat_types::{NumericClass, NumericDomain};

/// Physical output alternatives, not numerical permission to execute a provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum FusionHostOutputDomain {
    Exact(NumericDomain),
    RealOrComplex,
}

#[derive(Clone, Copy, Debug)]
pub struct FusionHostResourceInput<'a> {
    pub shape: &'a [usize],
    pub class: NumericClass,
}

pub struct FusionHostResourceQuery<'a> {
    pub inputs: &'a [FusionHostResourceInput<'a>],
    pub output: FusionHostResourceInput<'a>,
    pub output_domain: FusionHostOutputDomain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusionHostResourceBound {
    /// Operation-created retained allocations, not a standalone physical size
    /// certificate for an aliased result. A reused input allocation remains
    /// charged by its original owner. The fusion caller conservatively keeps
    /// every preceding producer's retained demand through the entire slice.
    pub retained_bytes: u64,
    /// Temporary allocations excluding separately accounted retained output,
    /// source mirrors, and provider readback workspace.
    pub scratch_bytes: u64,
}

pub type FusionHostResourceCallback =
    for<'a> fn(&FusionHostResourceQuery<'a>) -> Option<FusionHostResourceBound>;
