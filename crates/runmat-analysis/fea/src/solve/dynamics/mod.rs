//! Structural response solvers that operate on the canonical second-order system.

mod dense;
mod direct_harmonic;
mod harmonic_balance;
mod modal_harmonic;
mod modal_projection;
mod modal_transient;

pub use direct_harmonic::{
    solve_direct_harmonic, DirectHarmonicError, DirectHarmonicPolicy, DirectHarmonicResponse,
};
pub use harmonic_balance::{
    evaluate_harmonic_physical_state, HarmonicBalanceError, HarmonicPhysicalState,
};
pub use modal_harmonic::{
    solve_modal_harmonic, ModalHarmonicError, ModalHarmonicLoad, ModalHarmonicResponse,
};
pub use modal_projection::{ModalProjectionError, ModalProjectionEvidence, ModalReducedSystem};
pub use modal_transient::{
    solve_modal_transient, ModalTransientError, ModalTransientHistory, ModalTransientInitialState,
};
