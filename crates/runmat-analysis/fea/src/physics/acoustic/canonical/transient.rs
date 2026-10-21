//! Time-domain pressure acoustics: M p'' + B p' + K p = f(t) on the canonical operators.
//!
//! K, M and the boundary admittance B are the same matrices as the harmonic operator
//! A = K - omega² M + i omega B, so impedance and first-order radiation absorb in time
//! exactly as they do per frequency. Harmonic sources enter as b = i omega g; the
//! time-domain load is therefore f = dg/dt, evaluated as the step difference of g.
//! Prescribed pressure signals are imposed exactly at every step.
//!
//! Integration is Newmark average acceleration (trapezoidal, second order, unconditionally
//! stable, energy conserving for B = 0). The effective operator K + 4/dt² M + 2/dt B is
//! symmetric positive definite and factored once.
use super::{
    assembly::AcousticAssembly,
    real_solve::{dot, SpdOperator},
    AcousticHarmonicError,
};
use crate::{
    algebra::{spmv, MatrixProperties, OperatorRole, RealSparseMatrix},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    topology::AnalysisProjection,
};
use runmat_analysis_core::{
    AcousticConstraintKind, AnalysisConstraint, AnalysisModel, AnalysisProcedure,
    IndependentDomain, PhysicsFamily, ProcedureSettings, ProcedureStep, RestartPolicy,
    SolutionSpace, StepObjectRef,
};
use runmat_meshing_core::SolverMeshArtifact;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticTransientPolicy {
    pub linear_relative_tolerance: f64,
    pub linear_maximum_iterations: usize,
}

impl Default for AcousticTransientPolicy {
    fn default() -> Self {
        Self {
            linear_relative_tolerance: 1.0e-11,
            linear_maximum_iterations: 20_000,
        }
    }
}

impl AcousticTransientPolicy {
    pub fn validate(self) -> Result<(), AcousticHarmonicError> {
        if !(self.linear_relative_tolerance.is_finite()
            && self.linear_relative_tolerance > 0.0
            && self.linear_relative_tolerance < 1.0)
            || self.linear_maximum_iterations == 0
        {
            return Err(AcousticHarmonicError::InvalidInput(
                "transient pressure solve policy".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CanonicalAcousticTransient {
    pub projection: AnalysisProjection,
    /// Start time and every accepted step end, ascending.
    pub time_points_s: Vec<f64>,
    /// Real pressure in Pa at each time point, in projection node order.
    pub pressure_pa: Vec<Vec<f64>>,
    /// Acoustic energy 1/2 p'ᵀ M p' + 1/2 pᵀ K p (J) at each time point.
    pub acoustic_energy_j: Vec<f64>,
    /// Energy absorbed by impedance and radiation boundaries up to each time point (J).
    pub boundary_absorbed_energy_j: Vec<f64>,
    pub linear_iterations: usize,
    /// Largest achieved relative residual of any step's effective-operator solve.
    pub maximum_linear_relative_residual: f64,
}

/// Integrate an authored acoustic transient step from rest.
pub fn solve_canonical_transient(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
    policy: AcousticTransientPolicy,
) -> Result<CanonicalAcousticTransient, AcousticHarmonicError> {
    policy.validate()?;
    let step = transient_step(model, step_id)?;
    let IndependentDomain::TimeSeconds {
        start,
        end,
        interval_count,
        ..
    } = step.independent_domain
    else {
        return Err(AcousticHarmonicError::InvalidInput(
            "acoustic transient requires a time domain".into(),
        ));
    };
    if !(start.is_finite() && end.is_finite() && end > start) || interval_count == 0 {
        return Err(AcousticHarmonicError::InvalidInput(
            "acoustic transient time range".into(),
        ));
    }
    let dt = (end - start) / interval_count as f64;
    let times = (0..=interval_count)
        .map(|index| {
            if index == interval_count {
                end
            } else {
                start + dt * index as f64
            }
        })
        .collect::<Vec<_>>();
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| AcousticHarmonicError::Topology(error.to_string()))?;
    let properties = crate::properties::compile_solid_property_map(model, mesh)
        .map_err(|error| AcousticHarmonicError::Material(error.to_string()))?;
    // Unit angular frequency: every assembled source is then i g(t) with real g.
    let mut assembly = AcousticAssembly::assemble(projection, &properties, 1.0)?;
    let n = assembly.projection.nodes.len();
    let sample = |assembly: AcousticAssembly, time: f64| {
        sample_sources(assembly, model, step, time)
    };
    // Boundary admittance B does not depend on time; it is taken from the first sample.
    let (next, mut prescribed, mut g_previous) = sample(assembly, times[0])?;
    assembly = next;
    let damping = assembly.impedance.clone();
    let stiffness = assembly.stiffness.clone();
    let mass = assembly.mass.clone();
    let constrained = prescribed.iter().map(Option::is_some).collect::<Vec<_>>();
    let free = (0..n).filter(|&i| !constrained[i]).collect::<Vec<_>>();
    let fixed = (0..n).filter(|&i| constrained[i]).collect::<Vec<_>>();
    let a0 = 4.0 / (dt * dt);
    let a1 = 2.0 / dt;
    let effective = RealSparseMatrix::scaled_sum(
        &[(1.0, &stiffness), (a0, &mass), (a1, &damping)],
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
    let effective_free = effective
        .principal_submatrix(&free)
        .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
    let mass_free = mass
        .principal_submatrix(&free)
        .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
    let operator = SpdOperator::factor(
        &effective_free,
        policy.linear_relative_tolerance,
        policy.linear_maximum_iterations,
    )?;
    let mut linear_iterations = 0usize;
    let mut maximum_linear_relative_residual = 0.0_f64;
    let apply = |matrix: &RealSparseMatrix, values: &[f64]| {
        spmv(matrix, values).map_err(|error| AcousticHarmonicError::Operator(error.into()))
    };
    // Initial state from rest, with any prescribed pressure at the start time.
    let mut pressure = prescribed
        .iter()
        .map(|value| value.unwrap_or(0.0))
        .collect::<Vec<_>>();
    let mut velocity = vec![0.0; n];
    let mut previous_fixed_pressure = pressure.clone();
    // Load at the start uses the first step's forward difference of g.
    let (next, prescribed_1, g_1) = sample(assembly, times[1])?;
    assembly = next;
    let mut load = g_1
        .iter()
        .zip(&g_previous)
        .map(|(next, previous)| (next - previous) / dt)
        .collect::<Vec<_>>();
    // M a0 = f0 - K p0 - B v0 on free rows; prescribed accelerations start at zero.
    let mut acceleration = vec![0.0; n];
    {
        let k_p = apply(&stiffness, &pressure)?;
        let rhs = free
            .iter()
            .map(|&row| load[row] - k_p[row])
            .collect::<Vec<_>>();
        if rhs.iter().any(|value| *value != 0.0) {
            let mass_operator = SpdOperator::factor(
                &mass_free,
                policy.linear_relative_tolerance,
                policy.linear_maximum_iterations,
            )?;
            let solved = mass_operator.solve(&rhs, None)?;
            linear_iterations += solved.iterations;
            maximum_linear_relative_residual =
                maximum_linear_relative_residual.max(solved.relative_residual);
            for (reduced, &row) in free.iter().enumerate() {
                acceleration[row] = solved.values[reduced];
            }
        }
    }
    let energy = |pressure: &[f64], velocity: &[f64]| -> Result<f64, AcousticHarmonicError> {
        Ok(0.5 * dot(velocity, &apply(&mass, velocity)?)
            + 0.5 * dot(pressure, &apply(&stiffness, pressure)?))
    };
    let mut snapshots = Vec::with_capacity(times.len());
    let mut energies = Vec::with_capacity(times.len());
    let mut absorbed = Vec::with_capacity(times.len());
    snapshots.push(pressure.clone());
    energies.push(energy(&pressure, &velocity)?);
    absorbed.push(0.0);
    let mut absorbed_total = 0.0;
    let mut previous_dissipation = dot(&velocity, &apply(&damping, &velocity)?);
    let mut pending = Some((prescribed_1, g_1));
    let cancellation = crate::progress::cancellation_token();
    for step_index in 1..times.len() {
        if cancellation.is_cancelled() {
            return Err(AcousticHarmonicError::Cancelled);
        }
        let (next_prescribed, g_next) = match pending.take() {
            Some(values) => values,
            None => {
                let (next, prescribed, g) = sample(assembly, times[step_index])?;
                assembly = next;
                (prescribed, g)
            }
        };
        if next_prescribed
            .iter()
            .map(Option::is_some)
            .ne(constrained.iter().copied())
        {
            return Err(AcousticHarmonicError::Boundary(
                "prescribed-pressure faces changed during the transient".into(),
            ));
        }
        load = g_next
            .iter()
            .zip(&g_previous)
            .map(|(next, previous)| (next - previous) / dt)
            .collect();
        let mut next_pressure = vec![0.0; n];
        for &row in &fixed {
            next_pressure[row] = next_prescribed[row].expect("constrained row");
        }
        // History: M(a0 p + 4/dt v + a) + B(a1 p + v) + f, minus the prescribed coupling.
        let mass_history = (0..n)
            .map(|i| a0 * pressure[i] + (4.0 / dt) * velocity[i] + acceleration[i])
            .collect::<Vec<_>>();
        let damping_history = (0..n)
            .map(|i| a1 * pressure[i] + velocity[i])
            .collect::<Vec<_>>();
        let m_h = apply(&mass, &mass_history)?;
        let b_h = apply(&damping, &damping_history)?;
        let coupling = apply(&effective, &next_pressure)?;
        let rhs = free
            .iter()
            .map(|&row| load[row] + m_h[row] + b_h[row] - coupling[row])
            .collect::<Vec<_>>();
        let guess = free.iter().map(|&row| pressure[row]).collect::<Vec<_>>();
        let solved = operator.solve(&rhs, Some(&guess))?;
        linear_iterations += solved.iterations;
        maximum_linear_relative_residual =
            maximum_linear_relative_residual.max(solved.relative_residual);
        for (reduced, &row) in free.iter().enumerate() {
            next_pressure[row] = solved.values[reduced];
        }
        for &i in &free {
            let increment = next_pressure[i] - pressure[i];
            let next_velocity = a1 * increment - velocity[i];
            let next_acceleration = a0 * increment - (4.0 / dt) * velocity[i] - acceleration[i];
            velocity[i] = next_velocity;
            acceleration[i] = next_acceleration;
        }
        // Prescribed kinematics come from the imposed history itself. A Newmark recurrence
        // on an imposed piecewise-linear signal would ring at every kink indefinitely.
        for &i in &fixed {
            velocity[i] = (next_pressure[i] - pressure[i]) / dt;
            acceleration[i] =
                (next_pressure[i] - 2.0 * pressure[i] + previous_fixed_pressure[i]) / (dt * dt);
            previous_fixed_pressure[i] = pressure[i];
        }
        pressure = next_pressure;
        if pressure.iter().any(|value| !value.is_finite()) {
            return Err(AcousticHarmonicError::Operator(
                "nonfinite transient pressure".into(),
            ));
        }
        let dissipation = dot(&velocity, &apply(&damping, &velocity)?);
        absorbed_total += 0.5 * dt * (previous_dissipation + dissipation);
        previous_dissipation = dissipation;
        snapshots.push(pressure.clone());
        energies.push(energy(&pressure, &velocity)?);
        absorbed.push(absorbed_total);
        prescribed = next_prescribed;
        g_previous = g_next;
    }
    let _ = prescribed;
    Ok(CanonicalAcousticTransient {
        projection: assembly.projection,
        time_points_s: times,
        pressure_pa: snapshots,
        acoustic_energy_j: energies,
        boundary_absorbed_energy_j: absorbed,
        linear_iterations,
        maximum_linear_relative_residual,
    })
}

fn transient_step<'a>(
    model: &'a AnalysisModel,
    step_id: &str,
) -> Result<&'a ProcedureStep, AcousticHarmonicError> {
    let step = model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id.as_str() == step_id)
        .ok_or_else(|| AcousticHarmonicError::InvalidInput("step not found".into()))?;
    if step.family != PhysicsFamily::Acoustic
        || step.procedure != AnalysisProcedure::TransientResponse
        || step.solution_space != SolutionSpace::Direct
        || !step.state_inputs.is_empty()
        || step.settings != ProcedureSettings::None
        || step.restart != RestartPolicy::Fresh
    {
        return Err(AcousticHarmonicError::Unsupported(
            "a pressure transient must be a fresh direct acoustic transient step".into(),
        ));
    }
    if model.structural.is_some() || !model.connectors.is_empty() || !model.interfaces.is_empty() {
        return Err(AcousticHarmonicError::Unsupported(
            "explicit structural or coupled interfaces".into(),
        ));
    }
    if !model.initial_conditions.is_empty() {
        return Err(AcousticHarmonicError::Unsupported(
            "acoustic transients start from rest; drive them with boundary pressure or velocity \
             signals"
                .into(),
        ));
    }
    for constraint in &model.constraints {
        if step.object_is_active(&StepObjectRef::Constraint(
            constraint.constraint_id().to_string(),
        )) && !matches!(
            constraint,
            AnalysisConstraint::Acoustic(constraint)
                if matches!(constraint.kind, AcousticConstraintKind::PrescribedPressure { .. })
        ) {
            return Err(AcousticHarmonicError::Unsupported(
                "periodic or non-acoustic constraints in a pressure transient".into(),
            ));
        }
    }
    Ok(step)
}

/// Prescribed real pressure and the real source potential g(t) (assembled source = i g).
#[allow(clippy::type_complexity)]
fn sample_sources(
    assembly: AcousticAssembly,
    model: &AnalysisModel,
    step: &ProcedureStep,
    time: f64,
) -> Result<(AcousticAssembly, Vec<Option<f64>>, Vec<f64>), AcousticHarmonicError> {
    let sample = ExcitationSamplePoint::time(time);
    let excitations = realize_step_excitations(model, step, sample)
        .map_err(|error| AcousticHarmonicError::Boundary(error.to_string()))?;
    let assembly = assembly
        .with_cleared_sources()
        .with_boundaries(model, step, &excitations, sample)?;
    let prescribed = assembly
        .prescribed
        .iter()
        .map(|value| match value {
            None => Ok(None),
            Some(value) if value.imaginary == 0.0 => Ok(Some(value.real)),
            Some(_) => Err(AcousticHarmonicError::Boundary(
                "a time-domain prescribed pressure must be real".into(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if assembly.source.iter().any(|value| value.real != 0.0) {
        return Err(AcousticHarmonicError::Boundary(
            "a time-domain acoustic source must be real".into(),
        ));
    }
    let g = assembly.source.iter().map(|value| value.imaginary).collect();
    Ok((assembly, prescribed, g))
}

#[cfg(test)]
mod tests;
