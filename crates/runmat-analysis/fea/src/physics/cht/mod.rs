//! Conjugate heat-transfer coupling built on thermal and CFD results.

mod diagnostics;
mod interface;
mod recovery;

pub use recovery::{couple_thermal_flow, ChtCouplingOptions, ChtCouplingResult};
