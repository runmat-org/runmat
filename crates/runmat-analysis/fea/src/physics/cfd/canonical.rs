//! Conservative cell-centred steady incompressible Newtonian operator.
//!
//! This kernel has no profile/correlation solver and no production qualification.
//! Original cell velocity, physical pressure and one oriented face flux are retained.
//! Recovery evaluates those original equations; it never invokes the solver.

mod acceleration;
mod boundary;
mod boundary_balances;
mod continuity;
mod coupled;
mod flux_evidence;
mod forcing;
mod geometry_ops;
mod initial;
mod input_authority;
pub(crate) use input_authority::{
    potential_boundary_authorities, supports_action, supports_signal,
};
mod iterate;
pub mod qualification;
pub use initial::CanonicalFluidInitialState;
mod linear;
mod momentum;
mod moving;
pub use moving::swept_face_flux;
mod policy;
mod pressure;
mod pressure_gradient;
mod pressure_reference;
mod reconstruction;
mod recovery;
mod resources;
#[cfg(test)]
mod tests;
mod transient;
mod velocity_gradient;

use crate::topology::{AnalysisProjection, FiniteVolumeTopology};
use runmat_analysis_core::{
    AnalysisBoundary, CfdDomain, ExcitationAction, ExcitationTarget, FluidConstraint,
    FluidConstraintKind, PhysicalSignal,
};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use boundary_balances::CanonicalFluidBoundaryBalance;
pub use flux_evidence::CanonicalFluidFaceForces;
pub use recovery::CanonicalFluidResidual;
pub use resources::CanonicalFluidReservation;
pub use transient::{
    CanonicalFluidBackwardEulerStep, CanonicalFluidFields, CanonicalFluidMeshMotion,
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalFluidPolicy {
    /// How the pressure–velocity fixed point is reached; the discrete equations are the same.
    #[serde(default)]
    pub pressure_velocity_coupling: PressureVelocityCoupling,
    #[serde(deserialize_with = "deserialize_acceleration")]
    pub fixed_point_acceleration: Option<crate::solve::fixed_point::AndersonOnePolicy>,
    pub maximum_outer_iterations: usize,
    pub maximum_nonorthogonal_iterations: usize,
    pub maximum_linear_iterations: usize,
    pub gmres_restart: usize,
    pub momentum_relaxation: f64,
    pub pressure_relaxation: f64,
    pub linear_relative_tolerance: f64,
    pub physical_relative_tolerance: f64,
    pub maximum_resident_bytes: u64,
    pub maximum_operator_work: u64,
    pub maximum_authority_items: usize,
}

/// Solution path of the incompressible pressure–velocity system.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureVelocityCoupling {
    /// Picard steps on the whole (u, p) system by flexible GMRES with a SIMPLE
    /// block preconditioner.
    #[default]
    Coupled,
    /// Relaxed SIMPLE outer iteration (segregated momentum and pressure solves).
    Segregated,
}

impl Default for CanonicalFluidPolicy {
    fn default() -> Self {
        Self {
            pressure_velocity_coupling: PressureVelocityCoupling::Coupled,
            fixed_point_acceleration: Some(crate::solve::fixed_point::AndersonOnePolicy::default()),
            maximum_outer_iterations: 200,
            maximum_nonorthogonal_iterations: 4,
            maximum_linear_iterations: 1000,
            gmres_restart: 32,
            momentum_relaxation: 0.5,
            pressure_relaxation: 0.25,
            linear_relative_tolerance: 1e-8,
            physical_relative_tolerance: 1e-8,
            // No fixed size caps: the step's authored resource budget (accuracy resources)
            // bounds resident memory, and the execution envelope bounds time.
            maximum_resident_bytes: u64::MAX,
            maximum_operator_work: u64::MAX,
            maximum_authority_items: 1_000_000,
        }
    }
}

fn deserialize_acceleration<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<crate::solve::fixed_point::AndersonOnePolicy>, D::Error> {
    Option::deserialize(deserializer)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalFluidResult {
    pub numerical_input_identity: Digest,
    pub authored_authority_identity: Digest,
    /// Metres per second, in source-bound canonical cell order.
    pub velocity_m_per_s: Vec<[f64; 3]>,
    /// Original governing pressure relative to the source-bound component outlet or gauge.
    /// This is retained independently of the rounded absolute-pressure projection.
    pub pressure_fluctuation_pa: Vec<f64>,
    /// Original source-bound pressure coordinate; retained without subtracting
    /// fluctuation from rounded absolute Pa during continuation.
    pub pressure_reference_pa: Vec<f64>,
    /// Physical pressure in pascals, NOT kinematic pressure.
    /// Exact f64 projection reference + original fluctuation; never restart authority.
    pub pressure_pa: Vec<f64>,
    /// Volume flux m^3/s, outward from the canonical face owner.
    pub face_volume_flux_m3_per_s: Vec<f64>,
    pub outer_iterations: usize,
    pub linear_iterations: u64,
    pub residual: CanonicalFluidResidual,
}

#[derive(Debug, Error)]
pub enum CanonicalFluidError {
    #[error("invalid canonical fluid input: {0}")]
    Invalid(&'static str),
    #[error("unsupported canonical fluid envelope: {0}")]
    Unsupported(&'static str),
    #[error("canonical fluid resource admission failed: {0}")]
    Resource(&'static str),
    #[error("canonical fluid input realization failed: {0}")]
    InputRealization(String),
    #[error("canonical fluid sparse operator failed: {0}")]
    Operator(String),
    #[error("canonical fluid execution cancelled")]
    Cancelled,
    #[error("canonical fluid linear correction did not converge: iterations={iterations}, relative_residual={relative_residual:?}, breakdown={breakdown}")]
    LinearNotConverged {
        iterations: u32,
        relative_residual: Option<f64>,
        breakdown: bool,
    },
    #[error("canonical fluid exhausted outer={outer_iterations}, linear={linear_iterations}, momentum={momentum_relative}, continuity={continuity_relative}, flux={flux_constitutive_relative}, global_momentum={global_momentum_relative}, maximum_cell_momentum={maximum_cell_momentum_relative}")]
    IterationExhausted {
        outer_iterations: usize,
        linear_iterations: u64,
        momentum_relative: f64,
        continuity_relative: f64,
        flux_constitutive_relative: f64,
        global_momentum_relative: f64,
        maximum_cell_momentum_relative: f64,
    },
}

/// Prepared inputs are immutable after source/authority validation.
pub struct CanonicalFluidProblem<'a> {
    source: &'a AnalysisProjection,
    topology: &'a FiniteVolumeTopology,
    density: f64,
    viscosity: f64,
    /// Shear-rate dependent viscosity law; `viscosity` is then its low-shear reference.
    rheology: Option<runmat_analysis_core::GeneralizedNewtonianLaw>,
    /// Per cell: the porous-zone linear (μ/K) and quadratic (½ρC₂) resistance coefficients.
    porous: Vec<Option<(f64, f64)>>,
    boundaries: Vec<boundary::FaceBoundary<'a>>,
    boundary_balance_plan: boundary_balances::BoundaryBalancePlan,
    body_force_n_per_m3: Vec<[f64; 3]>,
    policy: CanonicalFluidPolicy,
    reservation: CanonicalFluidReservation,
    numerical_input_identity: Digest,
    authored_authority_identity: Digest,
    sample: crate::excitation::ProjectionSample,
    pressure_reference_pa: Vec<f64>,
    pressure_reference_plan: pressure_reference::PressureReferencePlan,
    pressure_stencils: Vec<pressure_gradient::PressureGradientStencil>,
    velocity_stencils: Vec<velocity_gradient::VelocityGradientStencil>,
    /// Cell-adjacency CSR support shared (by reference count) by every momentum/pressure
    /// operator of this problem; derived once instead of per outer iteration.
    operator_pattern: crate::algebra::SparsePattern,
}

/// Shared authored authorities are borrowed during preparation, never replaced by
/// a callback, a profile correlation or a private fluid signal representation.
pub struct CanonicalFluidInput<'a> {
    pub source: &'a AnalysisProjection,
    pub topology: &'a FiniteVolumeTopology,
    pub material: &'a CfdDomain,
    pub constraints: &'a [FluidConstraint],
    pub boundary_operators: &'a [AnalysisBoundary],
    pub signals: &'a [PhysicalSignal],
    pub excitations: &'a [ExcitationAction],
    pub policy: CanonicalFluidPolicy,
}

impl<'a> CanonicalFluidProblem<'a> {
    pub fn prepare(input: CanonicalFluidInput<'a>) -> Result<Self, CanonicalFluidError> {
        Self::prepare_at(input, crate::excitation::ProjectionSample::Static)
    }

    pub fn prepare_at(
        input: CanonicalFluidInput<'a>,
        sample: crate::excitation::ProjectionSample,
    ) -> Result<Self, CanonicalFluidError> {
        input.policy.validate()?;
        let sample_point = crate::excitation::ExcitationSamplePoint::try_from(&sample)
            .map_err(|e| CanonicalFluidError::InputRealization(e.to_string()))?;
        if !matches!(
            sample_point,
            crate::excitation::ExcitationSamplePoint::Static
                | crate::excitation::ExcitationSamplePoint::TimeSeconds { .. }
        ) {
            return Err(CanonicalFluidError::Unsupported(
                "fluid requires static or physical time input",
            ));
        }
        let CanonicalFluidInput {
            source,
            topology,
            material,
            constraints,
            boundary_operators,
            signals,
            excitations,
            policy,
        } = input;
        let mut items = constraints
            .len()
            .checked_add(boundary_operators.len())
            .and_then(|n| n.checked_add(signals.len()))
            .and_then(|n| n.checked_add(excitations.len()))
            .ok_or(CanonicalFluidError::Resource("authority count overflow"))?;
        // The reservation bounds aggregate resolved-target records, not merely
        // named actions. Count before serialization and target coverage traversal.
        let mut add_entities = |count: usize| -> Result<(), CanonicalFluidError> {
            items = items
                .checked_add(count)
                .ok_or(CanonicalFluidError::Resource("authority count overflow"))?;
            Ok(())
        };
        for constraint in constraints {
            let target = match &constraint.kind {
                FluidConstraintKind::PrescribedVelocity { target, .. }
                | FluidConstraintKind::MovingWall { target, .. }
                | FluidConstraintKind::NoSlipWall { target }
                | FluidConstraintKind::SlipWall { target }
                | FluidConstraintKind::Symmetry { target }
                | FluidConstraintKind::PressureReference { target, .. } => target,
                FluidConstraintKind::Periodic(_) => {
                    return Err(CanonicalFluidError::Unsupported("periodic fluid boundary"));
                }
            };
            add_entities(target.resolved_entities().len())?;
        }
        for boundary in boundary_operators {
            add_entities(boundary.target.resolved_entities().len())?;
        }
        for excitation in excitations {
            if !supports_action(excitation) {
                return Err(CanonicalFluidError::Unsupported(
                    "unsupported fluid excitation",
                ));
            }
            let action = excitation.excitation();
            if let ExcitationTarget::Region(target) = &action.target {
                add_entities(target.resolved_entities().len())?;
            }
        }
        if items > policy.maximum_authority_items {
            return Err(CanonicalFluidError::Resource("authority items"));
        }
        let reservation = resources::reserve(source, topology, policy)?;
        if constraints
            .iter()
            .any(|c| c.constraint_id.is_empty() || c.constraint_id.len() > 256)
            || boundary_operators
                .iter()
                .any(|b| b.boundary_id.is_empty() || b.boundary_id.len() > 256)
            || source
                .boundary_faces
                .iter()
                .any(|f| f.provenance.len() > 64)
        {
            return Err(CanonicalFluidError::Resource("boundary authority metadata"));
        }
        let reservation = resources::admit_authority_profile(
            source,
            constraints,
            boundary_operators,
            signals,
            excitations,
            policy,
            reservation,
        )?;
        runmat_analysis_core::validate_physical_signals(signals)
            .map_err(|e| CanonicalFluidError::InputRealization(e.to_string()))?;
        let authored_authority_identity = identity(
            "fea-fluid-authored-authority/v1",
            &(
                material,
                constraints,
                boundary_operators,
                signals,
                excitations,
            ),
        )?;
        topology
            .validate_for(source)
            .map_err(|_| CanonicalFluidError::Invalid("source-bound geometry"))?;
        let density = material.density.kilograms_per_cubic_meter;
        let viscosity = material.dynamic_viscosity.pascal_seconds;
        if !density.is_finite() || density <= 0.0 || !viscosity.is_finite() || viscosity <= 0.0 {
            return Err(CanonicalFluidError::Invalid(
                "positive finite density and dynamic viscosity required",
            ));
        }
        if material.rheology.is_some_and(|law| !law.is_valid()) {
            return Err(CanonicalFluidError::Invalid("generalized-Newtonian viscosity law"));
        }
        if material
            .porous_zones
            .iter()
            .any(|zone| !zone.law.is_valid() || zone.regions.is_empty())
        {
            return Err(CanonicalFluidError::Invalid("porous zone law or regions"));
        }
        let porous = topology
            .cells
            .iter()
            .map(|cell| {
                material
                    .porous_zones
                    .iter()
                    .find(|zone| zone.regions.binary_search(&cell.region).is_ok())
                    .map(|zone| {
                        (
                            viscosity / zone.law.permeability_m2,
                            0.5 * density * zone.law.inertial_coefficient_per_m,
                        )
                    })
            })
            .collect::<Vec<_>>();
        if material.turbulence_intensity.ratio != 0.0 {
            return Err(CanonicalFluidError::Unsupported("turbulence closure"));
        }
        let body_force_n_per_m3 =
            forcing::compile(source, topology, excitations, signals, sample_point)?;
        let boundaries = boundary::compile(boundary::BoundaryInput {
            source,
            topology,
            constraints,
            operators: boundary_operators,
            signals,
            excitations,
            density,
            sample: sample_point,
        })?;
        let boundary_balance_plan = boundary_balances::BoundaryBalancePlan::derive(&boundaries);
        let (pressure_reference_plan, pressure_reference_pa) =
            pressure_reference::PressureReferencePlan::derive(
                source,
                topology,
                &boundaries,
                constraints,
                signals,
                sample_point,
            )?;
        let pressure_stencils = pressure_gradient::derive(topology, &boundaries)?;
        let velocity_stencils = velocity_gradient::derive(topology, &boundaries)?;
        let numerical_input_identity = identity(
            "fea-fluid-numerical-input/v8",
            &(
                topology.mesh_digest,
                topology.projection_digest,
                topology.geometry_digest,
                density,
                (viscosity, material.rheology, &material.porous_zones),
                &boundaries,
                &body_force_n_per_m3,
                &pressure_reference_pa,
                &pressure_reference_plan,
                &sample,
                policy,
            ),
        )?;
        Ok(Self {
            source,
            topology,
            density,
            viscosity,
            rheology: material.rheology,
            porous,
            boundaries,
            boundary_balance_plan,
            body_force_n_per_m3,
            policy,
            reservation,
            numerical_input_identity,
            authored_authority_identity,
            sample,
            pressure_reference_pa,
            pressure_reference_plan,
            pressure_stencils,
            velocity_stencils,
            operator_pattern: linear::cell_operator_pattern(topology)?,
        })
    }

    /// Borrowed original metadata for exact shared result codec profiling.
    pub(crate) fn boundary_balance_authorities(&self) -> impl Iterator<Item = &str> + Clone {
        self.boundary_balance_plan.authorities(&self.boundaries)
    }

    pub fn reservation(&self) -> CanonicalFluidReservation {
        self.reservation
    }

    /// Source-bound pressure reference of every cell, Pa.
    pub fn pressure_reference(&self) -> &[f64] {
        &self.pressure_reference_pa
    }

    pub fn solve(&self) -> Result<CanonicalFluidResult, CanonicalFluidError> {
        if self.sample != crate::excitation::ProjectionSample::Static {
            return Err(CanonicalFluidError::Unsupported(
                "timed fluid solve requires original predecessor interval",
            ));
        }
        iterate::solve(self)
    }

    /// Steady solve continuing from earlier fields of the same cell complex, such as the
    /// previous iterate of a coupled study. The admitted equations, acceptance and retained
    /// input identity are unchanged: only the starting iterate differs.
    pub fn solve_from(
        &self,
        initial: CanonicalFluidFields<'_>,
    ) -> Result<CanonicalFluidResult, CanonicalFluidError> {
        if self.sample != crate::excitation::ProjectionSample::Static {
            return Err(CanonicalFluidError::Unsupported(
                "timed fluid solve requires original predecessor interval",
            ));
        }
        let n = self.topology.cells.len();
        if initial.velocity_m_per_s.len() != n
            || initial.pressure_fluctuation_pa.len() != n
            || initial.face_volume_flux_m3_per_s.len() != self.topology.faces.len()
            || initial
                .velocity_m_per_s
                .iter()
                .flatten()
                .chain(initial.pressure_fluctuation_pa)
                .chain(initial.face_volume_flux_m3_per_s)
                .any(|v| !v.is_finite())
        {
            return Err(CanonicalFluidError::Invalid("steady starting iterate"));
        }
        iterate::solve_with_time(self, Some(initial), None, self.numerical_input_identity)
    }

    pub fn recover(
        &self,
        original: &CanonicalFluidResult,
    ) -> Result<CanonicalFluidResidual, CanonicalFluidError> {
        if self.sample != crate::excitation::ProjectionSample::Static {
            return Err(CanonicalFluidError::Unsupported(
                "timed fluid recovery requires original predecessor interval",
            ));
        }
        recovery::recover(self, original)
    }
}

const INPUT_IDENTITY_LIMITS: runmat_canonical_codec::CanonicalLimits =
    runmat_canonical_codec::CanonicalLimits::new(1 << 44, 1 << 36, 65536, 32);

fn identity<T: Serialize>(domain: &'static str, value: &T) -> Result<Digest, CanonicalFluidError> {
    runmat_canonical_codec::encode_contract(
        b"runmat-canonical-fluid/v3\0",
        domain,
        value,
        INPUT_IDENTITY_LIMITS,
    )
    .map(Digest::sha256)
    .map_err(|_| CanonicalFluidError::Resource("canonical input identity encoding"))
}
