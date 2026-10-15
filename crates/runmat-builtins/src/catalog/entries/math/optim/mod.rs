//! Scalar optimization contracts and shared option identities.
mod fminbnd;
mod fminunc;
mod fsolve;
mod lsqcurvefit;
mod lsqnonlin;
mod nonlinear_facts;
#[cfg(test)]
mod nonlinear_tests;
mod coneprog;
mod linprog;
mod optimoptions;
mod optimset;
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
pub use fminunc::*;
pub use fsolve::*;
pub use fzero::*;
pub use integral::*;
pub use linprog::*;
pub use lsqcurvefit::*;
pub use lsqnonlin::*;
pub use nonlinear_facts::nonlinear_real_numeric_metadata;
pub use optimoptions::*;
pub use optimset::*;
pub use options::{OptimAlgorithm, OptimDisplay, OptimOptionField, OptimSolver, OptimSwitch};
pub use program_facts::OptimNumericInput;
pub use quad::*;
pub use secondordercone::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(fminbnd::ENTRIES.iter().copied());
    values.extend(fminunc::ENTRIES.iter().copied());
    values.extend(fsolve::ENTRIES.iter().copied());
    values.extend(lsqcurvefit::ENTRIES.iter().copied());
    values.extend(lsqnonlin::ENTRIES.iter().copied());
    values.extend(coneprog::ENTRIES.iter().copied());
    values.extend(linprog::ENTRIES.iter().copied());
    values.extend(optimoptions::ENTRIES.iter().copied());
    values.extend(optimset::ENTRIES.iter().copied());
    values.extend(secondordercone::ENTRIES.iter().copied());
    values.extend(fzero::ENTRIES.iter().copied());
    values.extend(integral::ENTRIES.iter().copied());
    values.extend(quad::ENTRIES.iter().copied());
}
