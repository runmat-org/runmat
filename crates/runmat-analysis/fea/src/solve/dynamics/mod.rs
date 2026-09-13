//! Structural response solvers that operate on the canonical second-order system.

mod dense;
mod modal_harmonic;
mod modal_projection;
mod modal_transient;

pub use modal_harmonic::{
    solve_modal_harmonic, ModalHarmonicError, ModalHarmonicLoad, ModalHarmonicResponse,
};
pub use modal_projection::{ModalProjectionError, ModalProjectionEvidence, ModalReducedSystem};
pub use modal_transient::{
    solve_modal_transient, ModalTransientError, ModalTransientHistory, ModalTransientInitialState,
};
