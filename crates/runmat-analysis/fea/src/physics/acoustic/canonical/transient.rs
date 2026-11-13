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
}

impl Default for AcousticTransientPolicy {
    fn default() -> Self {
        Self {
            linear_relative_tolerance: 1.0e-11,
        }
    }
}

impl AcousticTransientPolicy {
    pub fn validate(self) -> Result<(), AcousticHarmonicError> {
        if !(self.linear_relative_tolerance.is_finite()
            && self.linear_relative_tolerance > 0.0
            && self.linear_relative_tolerance < 1.0)
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
    /// The time points whose pressure is retained, ascending; a subset of `time_points_s`.
    pub retained_time_points_s: Vec<f64>,
    /// Real pressure in Pa at each retained time point, in projection node order.
    pub pressure_pa: Vec<Vec<f64>>,
    /// Acoustic energy 1/2 p'ᵀ M p' + 1/2 pᵀ K p (J) at each time point.
    pub acoustic_energy_j: Vec<f64>,
    /// Energy absorbed by impedance and radiation boundaries up to each time point (J).
    pub boundary_absorbed_energy_j: Vec<f64>,
    pub linear_iterations: usize,
    /// Largest achieved relative residual of any step's effective-operator solve.
    pub maximum_linear_relative_residual: f64,
}

/// Which accepted time points keep their full pressure field. Scalar energy evidence is kept
/// at every time point; fields are kept only where the study samples them, so memory and the
/// retained result stay bounded by the authored outputs rather than the step count.
#[derive(Clone, Debug, PartialEq)]
pub enum TransientRetention {
    /// Every accepted time point.
    Every,
    /// Every grid point, with explicit authored coordinates naming selected points exactly.
    EveryAnd(Vec<f64>),
    /// The final time point and the listed times, each of which must lie on the time grid.
    FinalAnd(Vec<f64>),
}

impl TransientRetention {
    /// Retention implied by an authored step: every increment when an observation samples every
    /// accepted increment; otherwise the final state plus explicitly sampled times.
    pub fn for_step(
        step: &runmat_analysis_core::ProcedureStep,
    ) -> Result<Self, AcousticHarmonicError> {
        use runmat_analysis_core::SamplingPolicy;
        let mut times = Vec::new();
        let mut every = false;
        for observation in &step.observations {
            let Some(spec) = observation.spec() else {
                continue;
            };
            match &spec.sampling {
                SamplingPolicy::EveryAcceptedIncrement => every = true,
                SamplingPolicy::TimeSeconds(values) => times.extend(values.iter().copied()),
                SamplingPolicy::TimeIntervalSeconds {
                    start,
                    end,
                    interval,
                } if *interval > 0.0 && start.is_finite() && end.is_finite() => {
                    let ratio = (end - start) / interval;
                    let nearest = ratio.round();
                    let count =
                        if (ratio - nearest).abs() <= 16.0 * f64::EPSILON * ratio.abs().max(1.0) {
                            nearest
                        } else {
                            ratio.floor()
                        };
                    if !count.is_finite()
                        || count < 0.0
                        || count >= step.accuracy.resources.maximum_samples as f64
                    {
                        return Err(AcousticHarmonicError::InvalidInput(
                            "authored acoustic history interval exceeds the sample ceiling".into(),
                        ));
                    }
                    let count = count as usize;
                    times.extend((0..=count).map(|k| start + interval * k as f64));
                }
                _ => {}
            }
        }
        times.sort_by(f64::total_cmp);
        times.dedup();
        Ok(if every {
            if times.is_empty() {
                Self::Every
            } else {
                Self::EveryAnd(times)
            }
        } else {
            Self::FinalAnd(times)
        })
    }

    /// Check every authored coordinate against its unique accepted-grid index without building
    /// a grid or allocating numerical state. The returned index names the physical solve point;
    /// the authored coordinate remains the public field/history label.
    pub fn validate_for_domain(
        &self,
        start: f64,
        end: f64,
        interval_count: usize,
    ) -> Result<(), AcousticHarmonicError> {
        if !start.is_finite() || !end.is_finite() || end <= start || interval_count == 0 {
            return Err(AcousticHarmonicError::InvalidInput(
                "invalid acoustic history time grid".into(),
            ));
        }
        // `for_step` sorts the authored labels, and the uniform grid is strictly increasing.
        // An adjacent collision is therefore sufficient; no index map allocation is needed.
        let mut previous = None;
        for &time in self.explicit_points() {
            let index = grid_index(start, end, interval_count, time)?;
            if previous
                .is_some_and(|(prior_index, prior_time)| prior_index == index && prior_time != time)
            {
                return Err(AcousticHarmonicError::InvalidInput(
                    "distinct authored acoustic times name one solver time point".into(),
                ));
            }
            previous = Some((index, time));
        }
        Ok(())
    }

    fn explicit_points(&self) -> &[f64] {
        match self {
            Self::Every => &[],
            Self::EveryAnd(points) | Self::FinalAnd(points) => points,
        }
    }

    /// The retained subset of an accepted time grid, ascending.
    pub fn retained_points(&self, times: &[f64]) -> Result<Vec<f64>, AcousticHarmonicError> {
        Ok(self.selected_points(times)?.into_iter().flatten().collect())
    }

    fn mask(&self, times: &[f64]) -> Result<Vec<bool>, AcousticHarmonicError> {
        Ok(self
            .selected_points(times)?
            .iter()
            .map(Option::is_some)
            .collect())
    }

    fn selected_points(&self, times: &[f64]) -> Result<Vec<Option<f64>>, AcousticHarmonicError> {
        let Some((&start, rest)) = times.split_first() else {
            return Err(AcousticHarmonicError::InvalidInput(
                "empty acoustic history time grid".into(),
            ));
        };
        let end = *rest.last().ok_or_else(|| {
            AcousticHarmonicError::InvalidInput(
                "acoustic history time grid needs two points".into(),
            )
        })?;
        if times.iter().any(|time| !time.is_finite())
            || times.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(AcousticHarmonicError::InvalidInput(
                "invalid acoustic history time grid".into(),
            ));
        }
        self.validate_for_domain(start, end, times.len() - 1)?;
        let mut selected = if matches!(self, Self::Every | Self::EveryAnd(_)) {
            times.iter().copied().map(Some).collect::<Vec<_>>()
        } else {
            vec![None; times.len()]
        };
        *selected.last_mut().expect("nonempty time grid") = Some(end);
        for &time in self.explicit_points() {
            let index = grid_index(start, end, times.len() - 1, time)?;
            selected[index] = Some(time);
        }
        Ok(selected)
    }
}

fn grid_index(
    start: f64,
    end: f64,
    interval_count: usize,
    time: f64,
) -> Result<usize, AcousticHarmonicError> {
    let dt = (end - start) / interval_count as f64;
    if !time.is_finite() || !dt.is_finite() || dt <= 0.0 {
        return Err(AcousticHarmonicError::InvalidInput(
            "invalid acoustic sampled time".into(),
        ));
    }
    let position = ((time - start) / dt).round();
    if !position.is_finite() || position < 0.0 || position > interval_count as f64 {
        return Err(AcousticHarmonicError::InvalidInput(format!(
            "sampled time {time} s is not an accepted transient time point",
        )));
    }
    let index = position as usize;
    let grid = crate::excitation::ProcedureTimeGrid::uniform_coordinate_at(
        start,
        end,
        interval_count,
        index,
    )
    .map_err(AcousticHarmonicError::InvalidInput)?;
    if (grid - time).abs() > 1e-9 * dt.max(f64::MIN_POSITIVE) {
        return Err(AcousticHarmonicError::InvalidInput(format!(
            "sampled time {time} s is not an accepted transient time point",
        )));
    }
    Ok(index)
}

/// Integrate an authored acoustic transient step from rest.
pub fn solve_canonical_transient(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
    policy: AcousticTransientPolicy,
    retention: &TransientRetention,
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
    let times = crate::excitation::ProcedureTimeGrid::from_interval_count(step, interval_count)
        .map_err(AcousticHarmonicError::InvalidInput)?
        .into_points_s();
    let retained = retention.mask(&times)?;
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| AcousticHarmonicError::Topology(error.to_string()))?;
    let properties = crate::properties::compile_solid_property_map(model, mesh)
        .map_err(|error| AcousticHarmonicError::Material(error.to_string()))?;
    // Unit angular frequency: every assembled source is then i g(t) with real g.
    let mut assembly = AcousticAssembly::assemble(
        projection,
        &properties,
        runmat_types::physical::AngularFrequency::new(1.0),
    )?;
    if assembly.lossy {
        return Err(AcousticHarmonicError::Unsupported(
            "frequency-domain loss factor in a time-domain acoustic solve".into(),
        ));
    }
    if model.boundaries.iter().any(|boundary| {
        matches!(
            boundary.operator,
            runmat_analysis_core::BoundaryOperator::AcousticImpedanceLaw { .. }
        )
    }) {
        return Err(AcousticHarmonicError::Unsupported(
            "frequency-dependent impedance law in a time-domain acoustic solve".into(),
        ));
    }
    let n = assembly.projection.nodes.len();
    let sample =
        |assembly: AcousticAssembly, time: f64| sample_sources(assembly, model, step, time);
    // Boundary admittance B does not depend on time; it is taken from the first sample.
    let (next, mut prescribed, mut g_previous) = sample(assembly, times[0])?;
    assembly = next;
    // Numerical boundary: the Newmark recurrence runs on the canonical operator storage.
    let damping = assembly
        .impedance
        .with_canonical_storage(RealSparseMatrix::clone);
    let stiffness = assembly
        .stiffness
        .with_canonical_storage(RealSparseMatrix::clone);
    let mass = assembly
        .mass
        .with_canonical_storage(RealSparseMatrix::clone);
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
    let operator = SpdOperator::factor(&effective_free, policy.linear_relative_tolerance)?;
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
            let mass_operator = SpdOperator::factor(&mass_free, policy.linear_relative_tolerance)?;
            let solved = mass_operator.solve(&rhs)?;
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
    let mut snapshots = Vec::with_capacity(retained.iter().filter(|keep| **keep).count());
    let mut energies = Vec::with_capacity(times.len());
    let mut absorbed = Vec::with_capacity(times.len());
    if retained[0] {
        snapshots.push(pressure.clone());
    }
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
        let solved = operator.solve(&rhs)?;
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
        if retained[step_index] {
            snapshots.push(pressure.clone());
        }
        energies.push(energy(&pressure, &velocity)?);
        absorbed.push(absorbed_total);
        prescribed = next_prescribed;
        g_previous = g_next;
    }
    let _ = prescribed;
    let retained_time_points_s = retention.retained_points(&times)?;
    Ok(CanonicalAcousticTransient {
        projection: assembly.projection,
        time_points_s: times,
        retained_time_points_s,
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
            crate::admission::ACOUSTIC_TRANSIENT_INITIAL_CONDITION
                .message
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
                crate::admission::ACOUSTIC_TRANSIENT_CONSTRAINT
                    .message
                    .into(),
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
    let assembly =
        assembly
            .with_cleared_sources()
            .with_boundaries(model, step, &excitations, sample)?;
    let prescribed = assembly
        .prescribed
        .canonical_values()
        .iter()
        .map(|value| match value {
            None => Ok(None),
            Some(value) if value.imaginary == 0.0 => Ok(Some(value.real)),
            Some(_) => Err(AcousticHarmonicError::Boundary(
                "a time-domain prescribed pressure must be real".into(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if assembly
        .source
        .canonical_values()
        .iter()
        .any(|value| value.real != 0.0)
    {
        return Err(AcousticHarmonicError::Boundary(
            "a time-domain acoustic source must be real".into(),
        ));
    }
    let g = assembly
        .source
        .canonical_values()
        .iter()
        .map(|value| value.imaginary)
        .collect();
    Ok((assembly, prescribed, g))
}

#[cfg(test)]
mod tests;
