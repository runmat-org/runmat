//! Conjugate heat transfer: one nodal energy equation over every fluid and solid element.
//!
//! The fluid side advects with the conservative finite-volume face fluxes of the solved flow,
//! lifted into each tetrahedral cell as its lowest-order Raviart–Thomas velocity. That velocity
//! is exactly divergence-free in every cell and has continuous normal flux between cells, so the
//! summed Galerkin advection rows equal the boundary enthalpy flux exactly: no energy is created
//! or lost by the transport discretization. Streamline-upwind/Petrov–Galerkin terms stabilize
//! convection-dominated cells and sum to zero over the partition of unity.
//!
//! Declared fluid–solid interfaces couple the two sides inside the same linear system. A perfect
//! interface shares the conformal interface nodes. A finite conductance or contact resistance
//! gives the interface nodes a second, side-owned degree of freedom joined by
//! `q = h (T_a − T_b)`, which transfers exactly the heat it removes from the other side.
//! The retained evidence reports the heat crossing each interface as seen from each side,
//! the global energy balance and the algebraic residual; cold recovery re-evaluates them
//! from the retained temperatures without solving again.

mod assembly;
mod evidence;
mod velocity;
#[cfg(test)]
mod tests;

use crate::{
    algebra::{spmv, RealSparseMatrix},
    excitation::ExcitationSamplePoint,
    properties::CompiledPropertyMap,
    solve::shared_sparse::{ambient_plan, control, with_plan, CsrArrays},
    topology::{AnalysisProjection, FiniteVolumeTopology},
};
use runmat_analysis_core::AnalysisModel;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use evidence::{ConjugateEnergyBalance, ConjugateHeatEvidence, ConjugateInterfaceBalance};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConjugateHeatPolicy {
    /// Relative residual the refined direct solve must reach on the original operator.
    pub linear_relative_tolerance: f64,
}

impl Default for ConjugateHeatPolicy {
    fn default() -> Self {
        Self {
            linear_relative_tolerance: 1e-10,
        }
    }
}

impl ConjugateHeatPolicy {
    pub fn validate(&self) -> Result<(), ConjugateHeatError> {
        if !self.linear_relative_tolerance.is_finite()
            || self.linear_relative_tolerance <= 0.0
            || self.linear_relative_tolerance >= 1.0
        {
            return Err(ConjugateHeatError::Invalid("conjugate heat linear policy"));
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ConjugateHeatError {
    #[error("invalid conjugate heat input: {0}")]
    Invalid(&'static str),
    #[error("invalid conjugate heat input: {0}")]
    InvalidDetail(String),
    #[error("unsupported conjugate heat envelope: {0}")]
    Unsupported(&'static str),
    #[error("conjugate heat sparse operator failed: {0}")]
    Operator(String),
    #[error("conjugate heat execution cancelled")]
    Cancelled,
    #[error("conjugate heat linear solve did not converge: iterations={iterations}, relative_residual={relative_residual:?}")]
    LinearNotConverged {
        iterations: u32,
        relative_residual: Option<f64>,
    },
}

/// Borrowed authored, geometric and flow authority. Nothing here is a solver default.
pub struct ConjugateHeatInput<'a> {
    pub model: &'a AnalysisModel,
    pub step: &'a runmat_analysis_core::ProcedureStep,
    /// Whole-mesh projection: every fluid and solid element and node.
    pub projection: &'a AnalysisProjection,
    pub properties: &'a CompiledPropertyMap,
    /// Fluid cell complex (region subset) in the order of the retained face fluxes.
    pub fluid_topology: &'a FiniteVolumeTopology,
    /// Retained conservative flow solution, outward from each face owner, m^3/s.
    pub face_volume_flux_m3_per_s: &'a [f64],
    pub(crate) sample: ExcitationSamplePoint<'a>,
    pub policy: ConjugateHeatPolicy,
}

/// Original retained temperatures and their evidence. Absolute kelvin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConjugateHeatResult {
    /// One value per projection node; the first-side value where an interface carries a jump.
    pub temperature_k: Vec<f64>,
    /// Second-side interface temperatures, keyed by projection node index, in node order.
    pub interface_second_side_temperature_k: Vec<(usize, f64)>,
    pub linear_iterations: u64,
    pub evidence: ConjugateHeatEvidence,
}

/// One accepted backward-Euler interval of a transient conjugate study.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConjugateHeatStep {
    pub time_s: f64,
    pub peak_temperature_k: f64,
    pub stored_heat_rate_w: f64,
    pub relative_imbalance: f64,
}

/// Steady flow, the temperature departing the last interval and the final temperature, with
/// every interval's energy evidence. The final interval recovers cold from its departure.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConjugateHeatTransientResult {
    pub flow: crate::physics::cfd::canonical::CanonicalFluidResult,
    pub departure_time_s: f64,
    pub departure: ConjugateHeatResult,
    pub energy: ConjugateHeatResult,
    pub steps: Vec<ConjugateHeatStep>,
}

/// Prepared, immutable operator. Preparation validates every authored thermal authority.
pub struct ConjugateHeatProblem<'a> {
    input: ConjugateHeatInput<'a>,
    system: assembly::AssembledEnergy,
}

impl<'a> ConjugateHeatProblem<'a> {
    pub fn prepare(input: ConjugateHeatInput<'a>) -> Result<Self, ConjugateHeatError> {
        input.policy.validate()?;
        let system = assembly::assemble(&input)?;
        Ok(Self { input, system })
    }

    pub fn degrees_of_freedom(&self) -> usize {
        self.system.dofs.count()
    }

    /// Maximum-width retained result of this operator: exact node, jump-dof and interface
    /// cardinalities with placeholder values, for wire and residency profiles.
    pub fn result_shape(&self) -> ConjugateHeatResult {
        let nodes = self.input.projection.nodes.len();
        let (temperature_k, second) = self
            .system
            .dofs
            .split(&vec![0.0; self.system.dofs.count()])
            .expect("dof inventory splits its own size");
        debug_assert_eq!(temperature_k.len(), nodes);
        ConjugateHeatResult {
            temperature_k,
            interface_second_side_temperature_k: second,
            linear_iterations: 0,
            evidence: ConjugateHeatEvidence {
                algebraic_relative_residual: 0.0,
                interfaces: self
                    .system
                    .interfaces
                    .iter()
                    .map(|plan| ConjugateInterfaceBalance {
                        interface_id: plan.interface_id.clone(),
                        face_count: 0,
                        area_m2: 0.0,
                        heat_into_second_side_w: 0.0,
                        heat_into_first_side_w: 0.0,
                        relative_imbalance: 0.0,
                        maximum_temperature_jump_k: 0.0,
                        law_relative_residual: 0.0,
                    })
                    .collect(),
                energy: ConjugateEnergyBalance {
                    applied_heat_w: 0.0,
                    convective_loss_w: 0.0,
                    enthalpy_outflow_w: 0.0,
                    prescribed_temperature_heat_w: 0.0,
                    stored_heat_rate_w: 0.0,
                    relative_imbalance: 0.0,
                },
            },
        }
    }

    pub fn solve(&self) -> Result<ConjugateHeatResult, ConjugateHeatError> {
        if crate::is_fea_cancelled() {
            return Err(ConjugateHeatError::Cancelled);
        }
        let (values, iterations) = solve_linear(
            &self.system.constrained,
            &self.system.constrained_rhs,
            &self.system.dirichlet,
            self.input.policy,
        )?;
        let (temperature_k, second) = self
            .system
            .dofs
            .split(&values)
            .map_err(ConjugateHeatError::InvalidDetail)?;
        let evidence = evidence::evaluate(&self.input, &self.system, &values, None)?;
        if !evidence.accepted(self.input.policy.linear_relative_tolerance) {
            return Err(ConjugateHeatError::InvalidDetail(format!(
                "conjugate energy residual {:e} exceeds the linear target",
                evidence.algebraic_relative_residual
            )));
        }
        Ok(ConjugateHeatResult {
            temperature_k,
            interface_second_side_temperature_k: second,
            linear_iterations: iterations,
            evidence,
        })
    }

    /// One backward-Euler interval from `departure`, with this problem's sources and prescribed
    /// temperatures realized at the arrival time: (A + C/Δt) T = b + C T_departure/Δt.
    pub fn step_backward_euler(
        &self,
        departure: &ConjugateHeatResult,
        duration_s: f64,
    ) -> Result<ConjugateHeatResult, ConjugateHeatError> {
        if !duration_s.is_finite() || duration_s <= 0.0 {
            return Err(ConjugateHeatError::Invalid("transient interval duration"));
        }
        let previous = self
            .system
            .dofs
            .join(
                &departure.temperature_k,
                &departure.interface_second_side_temperature_k,
            )
            .map_err(ConjugateHeatError::InvalidDetail)?;
        let effective = RealSparseMatrix::scaled_sum(
            &[
                (1.0, &self.system.full),
                (1.0 / duration_s, &self.system.capacity),
            ],
            crate::algebra::OperatorRole::EffectiveTangent,
            crate::algebra::MatrixProperties::GENERAL,
        )
        .map_err(|e| ConjugateHeatError::Operator(e.to_string()))?;
        let mask = self
            .system
            .dirichlet
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>();
        let constrained = effective
            .with_identity_constraints(&mask)
            .map_err(|e| ConjugateHeatError::Operator(e.to_string()))?;
        let stored = spmv(&self.system.capacity, &previous)
            .map_err(|e| ConjugateHeatError::Operator(e.to_string()))?;
        let prescribed = self
            .system
            .dirichlet
            .iter()
            .map(|v| v.unwrap_or(0.0))
            .collect::<Vec<_>>();
        let lifted = spmv(&effective, &prescribed)
            .map_err(|e| ConjugateHeatError::Operator(e.to_string()))?;
        let rhs = (0..previous.len())
            .map(|d| match self.system.dirichlet[d] {
                Some(value) => value,
                None => self.system.rhs[d] + stored[d] / duration_s - lifted[d],
            })
            .collect::<Vec<_>>();
        let (values, iterations) = solve_linear(
            &constrained,
            &rhs,
            &self.system.dirichlet,
            self.input.policy,
        )?;
        let (temperature_k, second) = self
            .system
            .dofs
            .split(&values)
            .map_err(ConjugateHeatError::InvalidDetail)?;
        let evidence = evidence::evaluate(
            &self.input,
            &self.system,
            &values,
            Some((&previous, duration_s)),
        )?;
        if !evidence.accepted(self.input.policy.linear_relative_tolerance) {
            return Err(ConjugateHeatError::InvalidDetail(format!(
                "conjugate energy interval residual {:e} exceeds the linear target",
                evidence.algebraic_relative_residual
            )));
        }
        Ok(ConjugateHeatResult {
            temperature_k,
            interface_second_side_temperature_k: second,
            linear_iterations: iterations,
            evidence,
        })
    }

    /// Re-evaluate a retained interval against its retained departure, without solving.
    pub fn recover_step(
        &self,
        departure: &ConjugateHeatResult,
        original: &ConjugateHeatResult,
        duration_s: f64,
    ) -> Result<(), ConjugateHeatError> {
        let previous = self
            .system
            .dofs
            .join(
                &departure.temperature_k,
                &departure.interface_second_side_temperature_k,
            )
            .map_err(ConjugateHeatError::InvalidDetail)?;
        let values = self
            .system
            .dofs
            .join(
                &original.temperature_k,
                &original.interface_second_side_temperature_k,
            )
            .map_err(ConjugateHeatError::InvalidDetail)?;
        let evidence = evidence::evaluate(
            &self.input,
            &self.system,
            &values,
            Some((&previous, duration_s)),
        )?;
        if !evidence.accepted(self.input.policy.linear_relative_tolerance)
            || evidence != original.evidence
        {
            return Err(ConjugateHeatError::Invalid(
                "retained interval does not satisfy the conjugate energy equations",
            ));
        }
        Ok(())
    }

    /// A uniform state, such as an authored initial temperature, as a departure result.
    pub fn uniform_state(&self, temperature_k: f64) -> ConjugateHeatResult {
        let mut shape = self.result_shape();
        shape.temperature_k.fill(temperature_k);
        for (_, t) in &mut shape.interface_second_side_temperature_k {
            *t = temperature_k;
        }
        shape
    }

    /// Re-evaluate the original equations and evidence from retained temperatures.
    pub fn recover(&self, original: &ConjugateHeatResult) -> Result<(), ConjugateHeatError> {
        let values = self
            .system
            .dofs
            .join(
                &original.temperature_k,
                &original.interface_second_side_temperature_k,
            )
            .map_err(ConjugateHeatError::InvalidDetail)?;
        if original.linear_iterations > MAXIMUM_REFINEMENT_STEPS {
            return Err(ConjugateHeatError::Invalid(
                "retained linear iteration count",
            ));
        }
        let evidence = evidence::evaluate(&self.input, &self.system, &values, None)?;
        if !evidence.accepted(self.input.policy.linear_relative_tolerance) {
            return Err(ConjugateHeatError::Invalid(
                "retained temperatures do not satisfy the conjugate energy equations",
            ));
        }
        if evidence != original.evidence {
            return Err(ConjugateHeatError::Invalid(
                "retained conjugate heat evidence substitution",
            ));
        }
        Ok(())
    }
}

/// Refinement steps of one direct solve never exceed the shared layer's own bound.
const MAXIMUM_REFINEMENT_STEPS: u64 = 64;

/// One sparse LU of the shared solver layer with iterative refinement; acceptance is the
/// residual on the original operator. Identity rows hold prescribed temperatures exactly: the
/// refined iterate meets them only to rounding, so they are restored to their authored values.
fn solve_linear(
    a: &RealSparseMatrix,
    b: &[f64],
    prescribed: &[Option<f64>],
    policy: ConjugateHeatPolicy,
) -> Result<(Vec<f64>, u64), ConjugateHeatError> {
    if crate::is_fea_cancelled() {
        return Err(ConjugateHeatError::Cancelled);
    }
    let arrays = CsrArrays::of(a);
    let mut x = vec![0.0; b.len()];
    let report = with_plan(ambient_plan(), || {
        let factor = runmat_sparse_solvers::direct::DirectSolver::<f64>::new(
            arrays.view().into(),
            runmat_sparse_solvers::direct::DirectOptions::new(
                runmat_sparse_solvers::direct::FactorKind::Lu,
            ),
            control(),
        )?;
        factor.solve_refined(arrays.view().into(), b, &mut x, 1)
    })
    .map_err(|error| match error {
        runmat_sparse_solvers::SolverError::Cancelled => ConjugateHeatError::Cancelled,
        other => ConjugateHeatError::Operator(other.to_string()),
    })?;
    if !(report.relative_residual <= policy.linear_relative_tolerance) {
        return Err(ConjugateHeatError::LinearNotConverged {
            iterations: u32::try_from(report.refinement_steps).unwrap_or(u32::MAX),
            relative_residual: Some(report.relative_residual),
        });
    }
    for (value, prescribed) in x.iter_mut().zip(prescribed) {
        if let Some(prescribed) = prescribed {
            *value = *prescribed;
        }
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ConjugateHeatError::Invalid(
            "nonfinite temperature solution",
        ));
    }
    Ok((x, report.refinement_steps as u64))
}
