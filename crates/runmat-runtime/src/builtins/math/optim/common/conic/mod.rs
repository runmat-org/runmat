//! Shared sparse conic mechanics; public solver row/dual mapping stays local.
use clarabel::{algebra::CscMatrix, solver::SupportedConeT};
use runmat_builtins::BuiltinCatalogIdentity;

pub(crate) mod backend;
pub(crate) mod convexity;
pub(crate) mod matrix;
pub(crate) mod terminal;
pub(crate) mod solve;
pub(crate) mod linear;
pub(crate) use recovery::certify as propose_exact_candidates;
mod recovery;
mod role;
pub(crate) use role::LinearRowRole;

/// Borrow the original model, never a copied kernel workspace or modified data.
pub(crate) struct Model<'a> {
    /// Original symmetric quadratic term, stored in upper-triangular CSC.
    /// None is the exact zero objective used by the linear/cone adapters.
    pub p: Option<&'a CscMatrix<f64>>,
    pub a: &'a CscMatrix<f64>,
    pub b: &'a [f64],
    pub f: &'a [f64],
    pub cones: &'a [SupportedConeT<f64>],
    pub identity: BuiltinCatalogIdentity,
    pub objective_scale: f64,
}

pub(crate) fn dimension(cone: &SupportedConeT<f64>) -> Option<usize> {
    match cone {
        SupportedConeT::ZeroConeT(n) | SupportedConeT::NonnegativeConeT(n) => Some(*n),
        SupportedConeT::SecondOrderConeT(n) if *n > 0 => Some(*n),
        _ => None,
    }
}

impl Model<'_> {
    pub fn valid_layout(&self) -> bool {
        let mut rows = 0usize;
        for cone in self.cones {
            let Some(size) = dimension(cone) else {
                return false;
            };
            let Some(next) = rows.checked_add(size) else {
                return false;
            };
            rows = next;
        }
        rows == self.b.len()
            && self.a.m == rows
            && self.a.n == self.f.len()
            && self.p.is_none_or(|p| {
                p.m == self.f.len()
                    && p.n == self.f.len()
                    && (0..p.n).all(|col| {
                        (p.colptr[col]..p.colptr[col + 1]).all(|index| p.rowval[index] <= col)
                    })
            })
    }
    pub fn work_units(&self) -> u64 {
        self.a
            .nzval
            .len()
            .saturating_add(self.b.len())
            .saturating_add(self.f.len())
            .saturating_add(self.p.map_or(0, |p| p.nzval.len())) as u64
    }
}
