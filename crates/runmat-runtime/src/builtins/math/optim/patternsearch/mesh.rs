//! Classic GPS mesh transitions. A partial poll is never an unsuccessful poll:
//! exhausting a callback budget cannot authorize contraction or convergence.

pub(super) use crate::builtins::math::optim::common::feasible_cone::Basis;

#[derive(Clone, Copy)]
pub(super) struct Parameters {
    pub initial: f64,
    pub maximum: f64,
    pub expansion: f64,
    pub contraction: f64,
    pub tolerance: f64,
    pub rotate: bool,
    pub accelerate: bool,
}

pub(super) struct Mesh {
    size: f64,
    reversed: bool,
    rotation_pending: bool,
    parameters: Parameters,
    policy: Policy,
}

#[derive(Clone, Copy)]
enum Policy {
    Fixed,
    Mads { dimension: usize, basis: Basis },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Update {
    Continue,
    BelowTolerance,
    Unrepresentable,
    Numerical,
}

impl Mesh {
    pub fn nonuniform(parameters: Parameters) -> Option<Self> {
        Self::new(parameters)
    }
    pub fn reconfigure_nonuniform(&mut self, parameters: Parameters) -> crate::BuiltinResult<()> {
        self.reconfigure(parameters)?;
        self.policy = Policy::Fixed;
        self.rotation_pending = false;
        self.reversed = false;
        Ok(())
    }

    /// NUPS owns its intrinsic transition factors. A half-poll is not a
    /// positive-spanning certificate, so it cannot authorize a finite stop.
    pub fn nonuniform_update(&mut self, factor: f64, covered: bool) -> Update {
        if !factor.is_finite() || factor <= 0.0 {
            return Update::Numerical;
        }
        if factor >= 1.0 {
            return self.expand(self.size, factor, self.parameters.maximum);
        }
        let next = self.size * factor;
        if next == 0.0 || next == self.size {
            return if covered {
                Update::Unrepresentable
            } else {
                Update::Continue
            };
        }
        self.size = next;
        if covered && next < self.parameters.tolerance {
            Update::BelowTolerance
        } else {
            Update::Continue
        }
    }

    pub fn new(parameters: Parameters) -> Option<Self> {
        if !parameters.initial.is_finite()
            || parameters.initial <= 0.0
            || parameters.maximum.is_nan()
            || parameters.maximum <= 0.0
            || !parameters.expansion.is_finite()
            || parameters.expansion <= 1.0
            || !parameters.contraction.is_finite()
            || !(0.0..1.0).contains(&parameters.contraction)
            || !parameters.tolerance.is_finite()
            || parameters.tolerance < 0.0
        {
            return None;
        }
        Some(Self {
            size: parameters.initial.min(parameters.maximum),
            reversed: false,
            rotation_pending: false,
            parameters,
            policy: Policy::Fixed,
        })
    }

    pub fn for_method(
        parameters: Parameters,
        method: runmat_builtins::OptimPatternPoll,
        dimension: usize,
    ) -> Option<Self> {
        let mut mesh = Self::new(parameters)?;
        if method.mesh_adaptive() {
            if dimension == 0 {
                return None;
            }
            mesh.policy = Policy::Mads {
                dimension,
                basis: if method.minimal() {
                    Basis::Minimal
                } else {
                    Basis::Maximal
                },
            };
            mesh.size = parameters.initial.min(1.0);
        }
        Some(mesh)
    }
    pub fn adaptive(&self) -> bool {
        matches!(self.policy, Policy::Mads { .. })
    }
    pub fn step_for(&self, method: runmat_builtins::OptimPatternPoll) -> f64 {
        self.size.min(if method.mesh_adaptive() {
            1.0
        } else {
            self.parameters.maximum
        })
    }
    pub fn poll_measure(&self) -> f64 {
        match self.policy {
            Policy::Fixed => self.size,
            Policy::Mads { dimension, basis } => {
                self.size.sqrt()
                    * if basis == Basis::Minimal {
                        dimension as f64
                    } else {
                        1.0
                    }
            }
        }
    }
    pub fn reconfigure_for(
        &mut self,
        parameters: Parameters,
        method: runmat_builtins::OptimPatternPoll,
        dimension: usize,
    ) -> crate::BuiltinResult<()> {
        let prepared = Self::for_method(parameters, method, dimension)
            .ok_or_else(|| super::invalid("invalid mesh parameters"))?;
        let original = self.size;
        self.reconfigure(parameters)?;
        self.policy = prepared.policy;
        if self.adaptive() {
            self.size = original.min(1.0);
            self.rotation_pending = false;
            self.reversed = false;
        }
        Ok(())
    }

    pub fn size(&self) -> f64 {
        self.size
    }

    pub fn reconfigure(&mut self, parameters: Parameters) -> crate::BuiltinResult<()> {
        Self::new(parameters).ok_or_else(|| super::invalid("invalid mesh parameters"))?;
        if self.parameters.rotate != parameters.rotate {
            self.rotation_pending = false;
        }
        self.parameters = parameters;
        self.size = self.size.min(parameters.maximum);
        Ok(())
    }

    pub fn sign(&self) -> f64 {
        if self.reversed {
            -1.0
        } else {
            1.0
        }
    }

    /// Expansion is saturating only at the user's explicit mesh ceiling.
    /// Overflow without a finite ceiling is numerical difficulty, not Inf
    /// passed to a callback or a successful convergence condition.
    pub fn successful(&mut self) -> Update {
        if self.adaptive() {
            return self.expand(self.size, 4.0, 1.0);
        }
        self.expand(
            self.size,
            self.parameters.expansion,
            self.parameters.maximum,
        )
    }
    pub fn successful_search(&mut self, method: runmat_builtins::OptimPatternPoll) -> Update {
        let (factor, maximum) = if method.mesh_adaptive() {
            (4.0, 1.0)
        } else {
            (self.parameters.expansion, self.parameters.maximum)
        };
        let main_maximum = if self.adaptive() {
            1.0
        } else {
            self.parameters.maximum
        };
        self.expand(self.step_for(method), factor, maximum.min(main_maximum))
    }
    fn expand(&mut self, origin: f64, factor: f64, maximum: f64) -> Update {
        self.rotation_pending = false;
        let next = (origin * factor).min(maximum);
        if !next.is_finite() {
            return Update::Unrepresentable;
        }
        self.size = next;
        Update::Continue
    }

    /// Called only after every required direction has been evaluated or proved
    /// infeasible. The controller owns budget/interruption and does not call
    /// this function for an incomplete opportunistic/complete poll.
    pub fn unsuccessful(&mut self, basis: Basis, rotation_forbidden: bool) -> Update {
        if self.adaptive() {
            let next = self.size * 0.25;
            if next == 0.0 || next == self.size || next * 4.0 != self.size {
                return Update::Numerical;
            }
            self.size = next;
            return if self.poll_measure() <= self.parameters.tolerance {
                Update::BelowTolerance
            } else {
                Update::Continue
            };
        }
        let can_rotate = self.parameters.rotate && basis == Basis::Minimal && !rotation_forbidden;
        if can_rotate && self.size < self.parameters.tolerance * 0.01 {
            if !self.rotation_pending {
                self.reversed = !self.reversed;
                self.rotation_pending = true;
                return Update::Continue;
            }
            return Update::BelowTolerance;
        }
        self.rotation_pending = false;
        let contraction =
            if self.parameters.accelerate && self.size < 100.0 * self.parameters.tolerance {
                0.5 * self.parameters.contraction
            } else {
                self.parameters.contraction
            };
        let next = self.size * contraction;
        if next == 0.0 || next == self.size {
            return Update::Unrepresentable;
        }
        self.size = next;
        if !can_rotate && self.size < self.parameters.tolerance {
            Update::BelowTolerance
        } else {
            Update::Continue
        }
    }
}

/// Unconstrained GPS positive bases are generated one ray at a time. Storage
/// is O(n), not an allocated n-by-2n direction matrix. Constrained geometry
/// supplies its own feasible generating rays rather than discarding these
/// coordinate rays and pretending the remaining subset spans a tangent cone.
pub(super) fn direction(
    basis: Basis,
    dimension: usize,
    index: usize,
    sign: f64,
) -> crate::BuiltinResult<Vec<f64>> {
    let count = match basis {
        Basis::Maximal => dimension.checked_mul(2),
        Basis::Minimal => dimension.checked_add(1),
    }
    .ok_or_else(|| super::invalid("direction count overflow"))?;
    if dimension == 0 || index >= count {
        return Err(super::invalid("direction index out of range"));
    }
    let mut ray = super::buffer(dimension)?;
    ray.resize(dimension, 0.0);
    match basis {
        Basis::Maximal => ray[index % dimension] = if index < dimension { sign } else { -sign },
        Basis::Minimal if index < dimension => ray[index] = sign,
        Basis::Minimal => ray.fill(-sign),
    }
    Ok(ray)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters() -> Parameters {
        Parameters {
            initial: 1.0,
            maximum: f64::INFINITY,
            expansion: 2.0,
            contraction: 0.5,
            tolerance: 1e-6,
            rotate: true,
            accelerate: false,
        }
    }

    #[test]
    fn adaptive_mesh_keeps_initial_scale_and_selected_search_transitions() {
        use runmat_builtins::OptimPatternPoll;
        let mut controls = parameters();
        controls.initial = 0.3;
        controls.maximum = 0.1;
        controls.expansion = 2.0;
        controls.tolerance = 0.2;
        let mut maximal = Mesh::for_method(controls, OptimPatternPoll::MadsMaximal, 3).unwrap();
        assert_eq!(maximal.size(), 0.3);
        assert_eq!(
            maximal.unsuccessful(Basis::Maximal, false),
            Update::Continue
        );
        assert_eq!(maximal.size(), 0.075);
        assert_eq!(maximal.poll_measure(), 0.075_f64.sqrt());
        assert_eq!(maximal.successful(), Update::Continue);
        assert_eq!(maximal.size(), 0.3);

        let mut minimal = Mesh::for_method(controls, OptimPatternPoll::MadsMinimal, 3).unwrap();
        assert_eq!(minimal.poll_measure(), 3.0 * 0.3_f64.sqrt());
        assert_eq!(
            minimal.unsuccessful(Basis::Minimal, false),
            Update::Continue
        );
        assert_eq!(minimal.poll_measure(), 3.0 * 0.075_f64.sqrt());
        assert_eq!(
            minimal.successful_search(OptimPatternPoll::GpsMaximal),
            Update::Continue
        );
        assert_eq!(minimal.size(), 0.1);
        assert_eq!(minimal.successful(), Update::Continue);
        assert_eq!(minimal.size(), 0.4);
        minimal
            .reconfigure_for(controls, OptimPatternPoll::MadsMinimal, 3)
            .unwrap();
        assert_eq!(minimal.size(), 0.4);
        assert_eq!(minimal.sign(), 1.0);

        controls.initial = f64::from_bits(1);
        controls.tolerance = 0.0;
        let mut exhausted = Mesh::for_method(controls, OptimPatternPoll::MadsMaximal, 1).unwrap();
        assert_eq!(
            exhausted.unsuccessful(Basis::Maximal, false),
            Update::Numerical
        );
        assert_eq!(exhausted.size(), f64::from_bits(1));
    }

    #[test]
    fn classic_mesh_and_positive_bases() {
        let mut mesh = Mesh::new(parameters()).unwrap();
        assert_eq!(mesh.successful(), Update::Continue);
        assert_eq!(mesh.size(), 2.0);
        assert_eq!(mesh.unsuccessful(Basis::Maximal, false), Update::Continue);
        assert_eq!(mesh.size(), 1.0);
        assert_eq!(direction(Basis::Maximal, 2, 2, 1.0).unwrap(), [-1.0, 0.0]);
        assert_eq!(direction(Basis::Minimal, 2, 2, -1.0).unwrap(), [1.0, 1.0]);
        assert!(direction(Basis::Maximal, usize::MAX, 0, 1.0).is_err());
        let mut invalid = parameters();
        invalid.contraction = 1.0;
        assert!(Mesh::new(invalid).is_none());
        let mut invalid = parameters();
        invalid.maximum = 0.0;
        assert!(Mesh::new(invalid).is_none());

        let mut rotation = parameters();
        rotation.initial = rotation.tolerance * 1e-3;
        let mut mesh = Mesh::new(rotation).unwrap();
        assert_eq!(mesh.unsuccessful(Basis::Minimal, false), Update::Continue);
        assert_eq!(mesh.sign(), -1.0);
        assert_eq!(
            mesh.unsuccessful(Basis::Minimal, false),
            Update::BelowTolerance
        );
        let mut mesh = Mesh::new(rotation).unwrap();
        assert_eq!(
            mesh.unsuccessful(Basis::Minimal, true),
            Update::BelowTolerance
        );
        assert_eq!(mesh.sign(), 1.0);
    }
}
