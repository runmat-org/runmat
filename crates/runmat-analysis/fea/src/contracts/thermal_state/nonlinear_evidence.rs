//! Bounded source-bound nonlinear acceptance, distinct from inner linear solves.
use serde::{Deserialize, Serialize};

pub const THERMAL_MAXIMUM_NEWTON_CORRECTIONS: usize = 32;
pub const THERMAL_MAXIMUM_BACKTRACKING_TRIALS: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalNonlinearSampleEvidence {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub time_s: runmat_types::physical::Time,
    pub iteration_count: usize,
    pub backtracking_count: usize,
    pub relative_equilibrium_residual: f64,
    pub relative_increment: f64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    /// Actual free-coordinate physical power scale; not an absolute-Kelvin floor.
    pub power_scale_w: runmat_types::physical::Power,
}

impl ThermalNonlinearSampleEvidence {
    pub fn satisfies(&self, target: f64) -> bool {
        self.time_s.canonical_value().is_finite()
            && self.time_s.canonical_value().to_bits() != (-0.0_f64).to_bits()
            && self.iteration_count > 0
            && self.iteration_count <= THERMAL_MAXIMUM_NEWTON_CORRECTIONS
            && self.backtracking_count <= self.iteration_count * THERMAL_MAXIMUM_BACKTRACKING_TRIALS
            && target.is_finite()
            && target > 0.0
            && [self.relative_equilibrium_residual, self.relative_increment]
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0 && *value <= target)
            && self.power_scale_w.canonical_value().is_finite()
            && self.power_scale_w.canonical_value() >= 0.0
            && (self.power_scale_w.canonical_value() > 0.0
                || self.relative_equilibrium_residual == 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_equilibrium_and_increment_are_both_mandatory() {
        let accepted = ThermalNonlinearSampleEvidence {
            time_s: runmat_types::physical::Time::new(0.0),
            iteration_count: 3,
            backtracking_count: 2,
            relative_equilibrium_residual: 1e-10,
            relative_increment: 1e-10,
            power_scale_w: runmat_types::physical::Power::new(2.0),
        };
        assert!(accepted.satisfies(1e-8));
        for invalid in [
            ThermalNonlinearSampleEvidence {
                relative_equilibrium_residual: 1e-7,
                ..accepted
            },
            ThermalNonlinearSampleEvidence {
                relative_increment: 1e-7,
                ..accepted
            },
            ThermalNonlinearSampleEvidence {
                power_scale_w: runmat_types::physical::Power::new(0.0),
                ..accepted
            },
            ThermalNonlinearSampleEvidence {
                power_scale_w: runmat_types::physical::Power::new(f64::NAN),
                ..accepted
            },
            ThermalNonlinearSampleEvidence {
                iteration_count: 33,
                ..accepted
            },
            ThermalNonlinearSampleEvidence {
                iteration_count: 0,
                ..accepted
            },
            ThermalNonlinearSampleEvidence {
                backtracking_count: 73,
                ..accepted
            },
            ThermalNonlinearSampleEvidence {
                time_s: runmat_types::physical::Time::new(-0.0),
                ..accepted
            },
        ] {
            assert!(!invalid.satisfies(1e-8));
        }
    }
}
