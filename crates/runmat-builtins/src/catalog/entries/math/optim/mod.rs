//! Scalar optimization contracts and shared option identities.
mod fminbnd;
mod coneprog;
mod linprog;
mod optimoptions;
mod secondordercone;
mod program_facts;
mod fzero;
mod integral;
mod options;
mod quad;
mod scalar_facts;
#[cfg(test)]
mod options_tests;

pub use coneprog::*;
pub use fminbnd::*;
pub use fzero::*;
pub use integral::*;
pub use linprog::*;
pub use optimoptions::*;
pub use options::{OptimAlgorithm, OptimDisplay, OptimOptionField, OptimSolver, OptimSwitch};
pub use program_facts::OptimNumericInput;
pub use quad::*;
pub use secondordercone::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(fminbnd::ENTRIES.iter().copied());
    values.extend(coneprog::ENTRIES.iter().copied());
    values.extend(linprog::ENTRIES.iter().copied());
    values.extend(optimoptions::ENTRIES.iter().copied());
    values.extend(secondordercone::ENTRIES.iter().copied());
    values.extend(fzero::ENTRIES.iter().copied());
    values.extend(integral::ENTRIES.iter().copied());
    values.extend(quad::ENTRIES.iter().copied());
}
