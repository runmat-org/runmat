//! Measured simulation work, in the units of `runmat-simulation-usage`.
//!
//! The engine counts the real floating-point operations of its reference algorithms and reports
//! them in the simulation usage categories:
//!
//! - `assembly`: device evaluation (every arithmetic operation on a value or a partial derivative
//!   in the dual-number models, with `exp`, `ln`, `sqrt` and `powf` counted as one operation each)
//!   and every matrix and right-hand-side accumulation;
//! - `factorization`: the sparse LU (each multiply-subtract counts 2, each division 1; complex
//!   multiply-subtract 8, complex division 11);
//! - `solve`: forward and back substitution under the same rules;
//! - `other`: Newton convergence tests, truncation-error estimates and measurements.
//!
//! The counts are exact for the algorithm as written. They do not depend on threads, vector width
//! or wall time.

use runmat_simulation_usage::{CountOverflow, OperationCount, RealArithmetic};

/// The accounting version the counts follow.
pub const MEASUREMENT_VERSION: &str = "rsua-1";

/// Operation counts per usage category.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub assembly: OperationCount,
    pub factorization: OperationCount,
    pub solve: OperationCount,
    pub other: OperationCount,
}

impl Usage {
    pub fn total(&self) -> Result<OperationCount, CountOverflow> {
        self.assembly
            .checked_add(self.factorization)?
            .checked_add(self.solve)?
            .checked_add(self.other)
    }
    pub fn checked_add(&self, o: &Usage) -> Result<Usage, CountOverflow> {
        Ok(Usage {
            assembly: self.assembly.checked_add(o.assembly)?,
            factorization: self.factorization.checked_add(o.factorization)?,
            solve: self.solve.checked_add(o.solve)?,
            other: self.other.checked_add(o.other)?,
        })
    }
}

/// Raw counters the kernels bump while they run; converted to [`Usage`] through the shared
/// counting convention ([`RealArithmetic`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct Meter {
    /// Single real operations (add, subtract, multiply, divide) per category.
    pub assembly: u64,
    pub factorization: u64,
    pub solve: u64,
    pub other: u64,
}

impl Meter {
    pub fn usage(&self) -> Result<Usage, CountOverflow> {
        let one = |n: u64| RealArithmetic::Add.count(n);
        Ok(Usage {
            assembly: one(self.assembly)?,
            factorization: one(self.factorization)?,
            solve: one(self.solve)?,
            other: one(self.other)?,
        })
    }
    pub fn absorb(&mut self, o: &Meter) {
        self.assembly = self.assembly.saturating_add(o.assembly);
        self.factorization = self.factorization.saturating_add(o.factorization);
        self.solve = self.solve.saturating_add(o.solve);
        self.other = self.other.saturating_add(o.other);
    }
}
