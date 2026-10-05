//! Conservative cell-centred steady incompressible Newtonian operator.
//!
//! This kernel has no profile/correlation solver and no production qualification.
//! Original cell velocity, physical pressure and one oriented face flux are retained.
//! Recovery evaluates those original equations; it never invokes the solver.

mod acceleration;
mod boundary;
mod flux_evidence;
mod forcing;
mod geometry_ops;
mod iterate;
mod linear;
mod momentum;
mod pressure;
mod pressure_gradient;
mod recovery;
mod resources;
#[cfg(test)]
mod tests;

use crate::topology::{AnalysisProjection, FiniteVolumeTopology};
use runmat_analysis_core::{
    AnalysisBoundary, CfdDomain, ExcitationAction, ExcitationTarget, FluidConstraint,
    FluidConstraintKind, PhysicalSignal,
};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use flux_evidence::CanonicalFluidFaceForces;
pub use recovery::CanonicalFluidResidual;
pub use resources::CanonicalFluidReservation;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalFluidPolicy {
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

impl Default for CanonicalFluidPolicy {
    fn default() -> Self {
        Self {
            fixed_point_acceleration: Some(crate::solve::fixed_point::AndersonOnePolicy::default()),
            maximum_outer_iterations: 200,
            maximum_nonorthogonal_iterations: 4,
            maximum_linear_iterations: 1000,
            gmres_restart: 32,
            momentum_relaxation: 0.5,
            pressure_relaxation: 0.25,
            linear_relative_tolerance: 1e-8,
            physical_relative_tolerance: 1e-8,
            maximum_resident_bytes: 1024 * 1024 * 1024,
            maximum_operator_work: 1_000_000_000_000,
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
    /// Original governing pressure relative to the source-bound component outlet.
    /// This is retained independently of the rounded absolute-pressure projection.
    pub pressure_fluctuation_pa: Vec<f64>,
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
    #[error("canonical fluid sparse operator failed: {0}")]
    Operator(String),
    #[error("canonical fluid iteration did not converge")]
    NotConverged,
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
    topology: &'a FiniteVolumeTopology,
    density: f64,
    viscosity: f64,
    boundaries: Vec<boundary::FaceBoundary>,
    body_force_n_per_m3: Vec<[f64; 3]>,
    policy: CanonicalFluidPolicy,
    reservation: CanonicalFluidReservation,
    numerical_input_identity: Digest,
    authored_authority_identity: Digest,
    pressure_reference_pa: Vec<f64>,
    pressure_stencils: Vec<pressure_gradient::PressureGradientStencil>,
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
            let ExcitationAction::BodyForce(action) = excitation else {
                return Err(CanonicalFluidError::Unsupported(
                    "non-bodyforce fluid excitation",
                ));
            };
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
        if material.turbulence_intensity.ratio != 0.0 {
            return Err(CanonicalFluidError::Unsupported("turbulence closure"));
        }
        let body_force_n_per_m3 = forcing::compile(source, topology, excitations, signals)?;
        let boundaries =
            boundary::compile(source, topology, constraints, boundary_operators, signals)?;
        let pressure_reference_pa = boundary::pressure_references(topology, &boundaries)?;
        let pressure_stencils = pressure_gradient::derive(topology, &boundaries)?;
        let numerical_input_identity = identity(
            "fea-fluid-numerical-input/v3",
            &(
                topology.mesh_digest,
                topology.projection_digest,
                topology.geometry_digest,
                density,
                viscosity,
                &boundaries,
                &body_force_n_per_m3,
                &pressure_reference_pa,
                policy,
            ),
        )?;
        Ok(Self {
            topology,
            density,
            viscosity,
            boundaries,
            body_force_n_per_m3,
            policy,
            reservation,
            numerical_input_identity,
            authored_authority_identity,
            pressure_reference_pa,
            pressure_stencils,
        })
    }

    pub fn reservation(&self) -> CanonicalFluidReservation {
        self.reservation
    }

    pub fn solve(&self) -> Result<CanonicalFluidResult, CanonicalFluidError> {
        iterate::solve(self)
    }

    pub fn recover(
        &self,
        original: &CanonicalFluidResult,
    ) -> Result<CanonicalFluidResidual, CanonicalFluidError> {
        recovery::recover(self, original)
    }
}

fn identity<T: Serialize>(domain: &'static str, value: &T) -> Result<Digest, CanonicalFluidError> {
    runmat_canonical_codec::encode_contract(
        b"runmat-canonical-fluid/v2\0",
        domain,
        value,
        runmat_canonical_codec::CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65536, 32),
    )
    .map(Digest::sha256)
    .map_err(|_| CanonicalFluidError::Resource("canonical input identity encoding"))
}
