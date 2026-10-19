//! One real backward-Euler interval on the canonical finite-volume operator.
//! Study orchestration owns schedules, signal evaluation and durable predecessor
//! lineage. This layer borrows original fields; it creates neither a scheduler
//! nor a trusted checkpoint from serialized claims.
mod resources;

use super::{
    geometry_ops::{scale, sub, V},
    identity, iterate, recovery, CanonicalFluidError as Error, CanonicalFluidProblem,
    CanonicalFluidReservation, CanonicalFluidResidual, CanonicalFluidResult,
};
use runmat_execution::Digest;

/// Original predecessor fields, in the prepared canonical cell/face order.
/// Absolute pressure is a reporting projection and is deliberately absent.
#[derive(Clone, Copy)]
pub struct CanonicalFluidFields<'a> {
    pub velocity_m_per_s: &'a [[f64; 3]],
    pub pressure_reference_pa: &'a [f64],
    pub pressure_fluctuation_pa: &'a [f64],
    pub face_volume_flux_m3_per_s: &'a [f64],
}
impl CanonicalFluidResult {
    pub fn original_fields(&self) -> CanonicalFluidFields<'_> {
        CanonicalFluidFields {
            velocity_m_per_s: &self.velocity_m_per_s,
            pressure_fluctuation_pa: &self.pressure_fluctuation_pa,
            pressure_reference_pa: &self.pressure_reference_pa,
            face_volume_flux_m3_per_s: &self.face_volume_flux_m3_per_s,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct TimeDerivative<'a> {
    pub previous_velocity: &'a [V],
    pub duration_s: f64,
}
impl TimeDerivative<'_> {
    pub fn coefficient(self, p: &CanonicalFluidProblem<'_>, cell: usize) -> Result<f64, Error> {
        let value = (p.density * p.topology.cells[cell].volume_m3) / self.duration_s;
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::Invalid("finite positive transient cell inertia"));
        }
        Ok(value)
    }
    pub fn force(self, p: &CanonicalFluidProblem<'_>, cell: usize, current: V) -> Result<V, Error> {
        let value = scale(
            sub(current, self.previous_velocity[cell]),
            self.coefficient(p, cell)?,
        );
        if value.iter().any(|x| !x.is_finite()) {
            return Err(Error::Invalid("nonfinite original transient inertia"));
        }
        Ok(value)
    }
}

/// Immutable admitted numerical interval. Its identity binds exact predecessor
/// fields and both physical times, in addition to the prepared source authority.
/// It is not a production study execution or a durable continuation capability.
pub struct CanonicalFluidBackwardEulerStep<'p, 'mesh, 'state> {
    problem: &'p CanonicalFluidProblem<'mesh>,
    previous: CanonicalFluidFields<'state>,
    derivative: TimeDerivative<'state>,
    numerical_input_identity: Digest,
    reservation: CanonicalFluidReservation,
}

impl<'mesh> CanonicalFluidProblem<'mesh> {
    /// Inputs are already realized at the receiving time through the shared sampler.
    pub fn prepare_backward_euler<'p, 'state>(
        &'p self,
        start_time_s: f64,
        end_time_s: f64,
        previous: CanonicalFluidFields<'state>,
    ) -> Result<CanonicalFluidBackwardEulerStep<'p, 'mesh, 'state>, Error> {
        if crate::is_fea_cancelled() {
            return Err(Error::Cancelled);
        }
        let duration_s = end_time_s - start_time_s;
        if !start_time_s.is_finite()
            || !end_time_s.is_finite()
            || (start_time_s == 0.0 && start_time_s.is_sign_negative())
            || (end_time_s == 0.0 && end_time_s.is_sign_negative())
            || !duration_s.is_finite()
            || duration_s <= 0.0
        {
            return Err(Error::Invalid("finite increasing physical fluid interval"));
        }
        if let crate::excitation::ProjectionSample::TimeSeconds { value, .. } = self.sample {
            if value.to_bits() != end_time_s.to_bits() {
                return Err(Error::Invalid(
                    "receiving fluid sample differs from interval endpoint",
                ));
            }
        }
        // Admission precedes predecessor traversal, encoding and warm-start copies.
        let reservation = resources::reserve(self)?;
        let n = self.topology.cells.len();
        if previous.pressure_reference_pa.len() != n
            || previous
                .pressure_reference_pa
                .iter()
                .any(|v| !v.is_finite())
            || previous.velocity_m_per_s.len() != n
            || previous.pressure_fluctuation_pa.len() != n
            || previous
                .velocity_m_per_s
                .iter()
                .flatten()
                .chain(previous.pressure_fluctuation_pa)
                .any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("original transient predecessor shape/value"));
        }
        self.pressure_reference_plan
            .validate(self, previous.pressure_fluctuation_pa)?;
        let continuity = super::continuity::incidence(self, previous.face_volume_flux_m3_per_s)?;
        if continuity
            .relative
            .iter()
            .any(|r| !r.is_finite() || *r > self.policy.physical_relative_tolerance)
        {
            return Err(Error::Invalid("initial original incompressible flux"));
        }
        let derivative = TimeDerivative {
            previous_velocity: previous.velocity_m_per_s,
            duration_s,
        };
        for cell in 0..n {
            derivative.coefficient(self, cell)?;
        }
        let numerical_input_identity = identity(
            "fea-fluid-backward-euler-interval/v2",
            &(
                self.numerical_input_identity,
                start_time_s,
                end_time_s,
                previous.velocity_m_per_s,
                previous.pressure_fluctuation_pa,
                previous.pressure_reference_pa,
                previous.face_volume_flux_m3_per_s,
            ),
        )?;
        Ok(CanonicalFluidBackwardEulerStep {
            problem: self,
            previous,
            derivative,
            numerical_input_identity,
            reservation,
        })
    }
}
impl CanonicalFluidBackwardEulerStep<'_, '_, '_> {
    pub fn reservation(&self) -> CanonicalFluidReservation {
        self.reservation
    }
    pub fn solve(&self) -> Result<CanonicalFluidResult, Error> {
        iterate::solve_with_time(
            self.problem,
            Some(self.previous),
            Some(self.derivative),
            self.numerical_input_identity,
        )
    }
    /// Cold original equations are re-evaluated using the exact borrowed
    /// predecessor, not the encoded residual summary or a steady surrogate.
    pub fn recover(
        &self,
        original: &CanonicalFluidResult,
    ) -> Result<CanonicalFluidResidual, Error> {
        if crate::is_fea_cancelled() {
            return Err(Error::Cancelled);
        }
        recovery::recover_with_time(
            self.problem,
            original,
            self.numerical_input_identity,
            Some(self.derivative),
        )
    }
}
