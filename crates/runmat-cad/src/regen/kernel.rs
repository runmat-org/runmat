//! The kernel seam: canonical operations in, opaque bodies plus history out.
//!
//! Defined once, kernel-neutral, in [`runmat_geometry_kernel::op`]: the OCCT backend in
//! `runmat-geometry-io` and this crate's test kernel implement the same trait. Re-exported here
//! under the design layer's path; `CadKernel` is that trait.

pub use runmat_geometry_kernel::op::*;
