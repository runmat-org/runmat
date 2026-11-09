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
    pub velocity_m_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Velocity, &'a [[f64; 3]]>,
    pub pressure_reference_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, &'a [f64]>,
    pub pressure_fluctuation_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, &'a [f64]>,
    pub face_volume_flux_m3_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::VolumeFlowRate, &'a [f64]>,
}
impl CanonicalFluidResult {
    pub fn original_fields(&self) -> CanonicalFluidFields<'_> {
        CanonicalFluidFields {
            velocity_m_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Velocity,
                _,
            >::new(self.velocity_m_per_s.canonical_values()),
            pressure_fluctuation_pa: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Pressure,
                _,
            >::new(
                self.pressure_fluctuation_pa.canonical_values()
            ),
            pressure_reference_pa: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Pressure,
                _,
            >::new(self.pressure_reference_pa.canonical_values()),
            face_volume_flux_m3_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::VolumeFlowRate,
                _,
            >::new(
                self.face_volume_flux_m3_per_s.canonical_values()
            ),
        }
    }
}

/// Arbitrary Lagrangian–Eulerian interval data: the departure cell volumes and the volume each
/// face sweeps over the interval (divided by its duration, outward from the face owner). The
/// geometric conservation law V_new − V_old = Δt Σ w holds cell by cell.
#[derive(Clone, Copy)]
pub struct CanonicalFluidMeshMotion<'a> {
    pub previous_volume_m3: &'a [f64],
    pub face_mesh_flux_m3_per_s: &'a [f64],
}

#[derive(Clone, Copy)]
pub(super) struct TimeDerivative<'a> {
    pub previous_velocity: &'a [V],
    pub duration_s: f64,
    pub motion: Option<CanonicalFluidMeshMotion<'a>>,
}
impl TimeDerivative<'_> {
    pub fn coefficient(self, p: &CanonicalFluidProblem<'_>, cell: usize) -> Result<f64, Error> {
        let value = (p.density * p.topology.cells[cell].volume_m3) / self.duration_s;
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::Invalid("finite positive transient cell inertia"));
        }
        Ok(value)
    }
    /// Departure inertia ρ V_old / Δt: the arrival volume on a fixed mesh.
    pub fn previous_coefficient(
        self,
        p: &CanonicalFluidProblem<'_>,
        cell: usize,
    ) -> Result<f64, Error> {
        let Some(motion) = self.motion else {
            return self.coefficient(p, cell);
        };
        let value = (p.density * motion.previous_volume_m3[cell]) / self.duration_s;
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::Invalid("finite positive departure cell inertia"));
        }
        Ok(value)
    }
    /// Mesh volume flux of a face; zero on a fixed mesh.
    pub fn mesh_flux(self, face: usize) -> f64 {
        self.motion
            .map_or(0.0, |motion| motion.face_mesh_flux_m3_per_s[face])
    }
    pub fn force(self, p: &CanonicalFluidProblem<'_>, cell: usize, current: V) -> Result<V, Error> {
        let value = sub(
            scale(current, self.coefficient(p, cell)?),
            scale(
                self.previous_velocity[cell],
                self.previous_coefficient(p, cell)?,
            ),
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
        if previous.pressure_reference_pa.canonical_values().len() != n
            || previous
                .pressure_reference_pa
                .canonical_values()
                .iter()
                .any(|v| !v.is_finite())
            || previous.velocity_m_per_s.canonical_values().len() != n
            || previous.pressure_fluctuation_pa.canonical_values().len() != n
            || previous
                .velocity_m_per_s
                .canonical_values()
                .iter()
                .flatten()
                .chain(previous.pressure_fluctuation_pa.canonical_values().iter())
                .any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("original transient predecessor shape/value"));
        }
        self.pressure_reference_plan
            .validate(self, previous.pressure_fluctuation_pa.canonical_values())?;
        let continuity = super::continuity::incidence(
            self,
            previous.face_volume_flux_m3_per_s.canonical_values(),
        )?;
        if continuity
            .relative
            .iter()
            .any(|r| !r.is_finite() || *r > self.policy.physical_relative_tolerance)
        {
            return Err(Error::Invalid("initial original incompressible flux"));
        }
        let derivative = TimeDerivative {
            previous_velocity: previous.velocity_m_per_s.canonical_values(),
            duration_s,
            motion: None,
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
                previous.velocity_m_per_s.canonical_values(),
                previous.pressure_fluctuation_pa.canonical_values(),
                previous.pressure_reference_pa.canonical_values(),
                previous.face_volume_flux_m3_per_s.canonical_values(),
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
impl<'mesh> CanonicalFluidProblem<'mesh> {
    /// One backward-Euler interval on a moving mesh. `self` is prepared on the arrival
    /// configuration; `previous` holds the departure fields in the same cell and face order,
    /// and `motion` the departure volumes and swept face fluxes, which must satisfy the
    /// geometric conservation law in every cell.
    pub fn prepare_backward_euler_moving<'p, 'state>(
        &'p self,
        start_time_s: f64,
        end_time_s: f64,
        previous: CanonicalFluidFields<'state>,
        motion: CanonicalFluidMeshMotion<'state>,
    ) -> Result<CanonicalFluidBackwardEulerStep<'p, 'mesh, 'state>, Error> {
        let n = self.topology.cells.len();
        if motion.previous_volume_m3.len() != n
            || motion.face_mesh_flux_m3_per_s.len() != self.topology.faces.len()
            || motion
                .previous_volume_m3
                .iter()
                .chain(motion.face_mesh_flux_m3_per_s)
                .any(|v| !v.is_finite())
            || motion.previous_volume_m3.iter().any(|v| *v <= 0.0)
        {
            return Err(Error::Invalid("moving-mesh interval inventory"));
        }
        let duration = end_time_s - start_time_s;
        let mut swept = vec![0.0; n];
        for (i, face) in self.topology.faces.iter().enumerate() {
            swept[face.owner] += motion.face_mesh_flux_m3_per_s[i];
            if let Some(neighbor) = face.neighbor {
                swept[neighbor] -= motion.face_mesh_flux_m3_per_s[i];
            }
        }
        for (c, cell) in self.topology.cells.iter().enumerate() {
            let change = cell.volume_m3 - motion.previous_volume_m3[c];
            if (change - duration * swept[c]).abs() > 1e-9 * cell.volume_m3 {
                return Err(Error::Invalid(
                    "moving mesh violates the geometric conservation law",
                ));
            }
        }
        let mut step = self.prepare_backward_euler(start_time_s, end_time_s, previous)?;
        step.derivative.motion = Some(motion);
        step.numerical_input_identity = identity(
            "fea-fluid-moving-mesh-interval/v1",
            &(
                step.numerical_input_identity,
                motion.previous_volume_m3,
                motion.face_mesh_flux_m3_per_s,
            ),
        )?;
        Ok(step)
    }

    /// Wall velocities of a moving structure on faces whose authored law is a wall: the
    /// prescribed value of each listed face becomes the wall's own velocity. Only a
    /// moving-mesh interval admits normal wall motion, since its face fluxes sweep the wall.
    pub fn with_wall_velocities(mut self, walls: &[(usize, [f64; 3])]) -> Result<Self, Error> {
        for &(face, velocity) in walls {
            if velocity.iter().any(|v| !v.is_finite()) {
                return Err(Error::Invalid("wall velocity"));
            }
            match self.boundaries.get_mut(face) {
                Some(super::boundary::FaceBoundary::Velocity { value, .. }) => *value = velocity,
                _ => return Err(Error::Invalid("wall velocity off a velocity wall face")),
            }
        }
        self.numerical_input_identity = identity(
            "fea-fluid-structure-wall-velocity/v1",
            &(self.numerical_input_identity, walls),
        )?;
        Ok(self)
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
    /// The interval's solve started from `initial` (fields of the same cell complex, such as
    /// the previous coupling iterate of a fluid-structure step) instead of the predecessor.
    /// The time history, equations, acceptance and retained identity are unchanged.
    pub fn solve_from(&self, initial: CanonicalFluidFields<'_>) -> Result<CanonicalFluidResult, Error> {
        let n = self.problem.topology.cells.len();
        if initial.velocity_m_per_s.canonical_values().len() != n
            || initial.pressure_fluctuation_pa.canonical_values().len() != n
            || initial.face_volume_flux_m3_per_s.canonical_values().len()
                != self.problem.topology.faces.len()
            || initial
                .velocity_m_per_s
                .canonical_values()
                .iter()
                .flatten()
                .chain(initial.pressure_fluctuation_pa.canonical_values().iter())
                .chain(initial.face_volume_flux_m3_per_s.canonical_values().iter())
                .any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("interval starting iterate"));
        }
        iterate::solve_with_time(
            self.problem,
            Some(initial),
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
