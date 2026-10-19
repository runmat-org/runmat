//! Backward Euler in retained reference-temperature coordinates.
use super::*;

pub(super) struct TransientSolveInput<'a> {
    pub(super) properties: &'a crate::properties::CompiledPropertyMap,
    pub(super) model: &'a AnalysisModel,
    pub(super) mesh: &'a runmat_meshing_core::SolverMeshArtifact,
    pub(super) step: &'a runmat_analysis_core::ProcedureStep,
    pub(super) initial_assembly: &'a crate::physics::thermal::assembly::ThermalOperatorAssembly,
    pub(super) initial_temperature_k: &'a [f64],
    pub(super) schedule: &'a crate::execution::ThermalSolveSchedule,
    pub(super) facts: &'a crate::planning::resource::ThermalPreassemblyResourceFacts,
    pub(super) requested_solver_backend: LinearAlgebraBackendKind,
    pub(super) algebra: &'a dyn LinearAlgebraBackend,
    pub(super) residual_target: f64,
}

pub(super) fn solve_transient(
    input: TransientSolveInput<'_>,
) -> Result<ProcedureOutcome, FeaRunError> {
    let operators = &input.facts.operators;
    let solve_times = input.schedule.points_s();
    let mut time_points_s = Vec::with_capacity(solve_times.len());
    let mut recovered = RecoveredSnapshots::with_capacity(solve_times.len());
    let mut residual_norms = Vec::with_capacity(solve_times.len() - 1);
    let mut energy_balances = Vec::with_capacity(solve_times.len() - 1);
    let mut nonlinear_evidence = Vec::new();
    if nonlinear::has_radiation(input.model) {
        nonlinear_evidence
            .try_reserve_exact(solve_times.len() - 1)
            .map_err(|error| {
                crate::planning::resource::ResourcePlanningError::Invalid(format!(
                    "cannot reserve admitted thermal nonlinear evidence: {error}"
                ))
            })?;
    }
    let mut source_evidence = input.initial_assembly.source_evidence.clone();
    // Retain departures throughout integration. Reconstructing absolute Kelvin between
    // steps would lose the small increments used for physical storage/energy recovery.
    let reference = input
        .model
        .thermal
        .as_ref()
        .expect("validated thermal domain")
        .reference_temperature
        .kelvin;
    let reference_temperature_k = vec![reference; input.initial_temperature_k.len()];
    let mut previous = input
        .initial_temperature_k
        .iter()
        .map(|temperature| temperature - reference)
        .collect::<Vec<_>>();
    push_recovered_snapshot(
        input.initial_assembly,
        &previous,
        None,
        solve_times[1] - solve_times[0],
        reference,
        &mut recovered,
    )?;
    time_points_s.push(solve_times[0]);
    let mut last_solve = None;
    let mut host_sync_count = 0_u32;
    let mut device_apply_count = 0_u32;
    let mut device_apply_attempt_count = 0_u32;
    let mut accelerator_evidence = Vec::new();
    for step_index in 1..solve_times.len() {
        check_cancelled("fea.run_thermal")?;
        let previous_time_s = solve_times[step_index - 1];
        let time_s = solve_times[step_index];
        let time_step_s = time_s - previous_time_s;
        emit_phase(
            "fea.run_thermal",
            FeaProgressPhase::Solve,
            FeaProgressStatus::Advanced,
            format!("solving physical thermal step {}", step_index + 1),
            Some(step_index as u64),
            Some(solve_times.len() as u64),
        );
        let approach = input.schedule.approach(step_index);
        let sample = sample_point(input.step, time_s, approach)?;
        let excitations = realize_excitations_for_step(input.model, input.step, sample)?;
        let assembly = assemble_thermal_operator(
            input.model,
            input.mesh,
            input.properties,
            operators.support(),
            &excitations,
            sample,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        drop(excitations);
        operators.verify_matrix(&assembly.conductivity)?;
        operators.verify_matrix(&assembly.capacity)?;
        source_evidence.extend(assembly.source_evidence.clone());
        if nonlinear::has_radiation(input.model) {
            let accepted = nonlinear::solve(nonlinear::NonlinearSolveInput {
                assembly: &assembly,
                model: input.model,
                sample,
                reference_temperature_k: &reference_temperature_k,
                initial_departure_k: &previous,
                previous_departure_k: Some(&previous),
                time_step_s,
                time_s,
                algebra: input.algebra,
                facts: input.facts,
                residual_target: input.residual_target,
            })?;
            nonlinear::add_count(&mut host_sync_count, accepted.host_sync_count)?;
            nonlinear::add_count(&mut device_apply_count, accepted.device_apply_count)?;
            nonlinear::add_count(
                &mut device_apply_attempt_count,
                accepted.device_apply_attempt_count,
            )?;
            accelerator_evidence.extend(accepted.accelerator_evidence);
            residual_norms.push(accepted.relative_linear_residual);
            energy_balances.push(accepted.energy_balance);
            nonlinear_evidence.push(accepted.evidence);
            push_recovered_snapshot(
                &assembly,
                &accepted.solve.solution,
                Some(&previous),
                time_step_s,
                reference,
                &mut recovered,
            )?;
            previous = accepted.solve.solution.clone();
            time_points_s.push(time_s);
            last_solve = Some(accepted.solve);
            continue;
        }
        let operator = assembly
            .transient_operator(
                input.model,
                sample,
                time_step_s,
                &previous,
                &reference_temperature_k,
            )
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        let mut solve = solve_checked(
            &operator,
            input.requested_solver_backend,
            input.algebra,
            Some(&previous),
            operators,
            input.residual_target,
        )?;
        host_sync_count = host_sync_count.saturating_add(solve.host_sync_count);
        device_apply_count = device_apply_count.saturating_add(solve.device_apply_k_count);
        device_apply_attempt_count =
            device_apply_attempt_count.saturating_add(solve.device_apply_k_attempt_count);
        backend_evidence::retain(
            &mut accelerator_evidence,
            &mut solve.accelerator_evidence,
            &input.facts.metadata.backend_evidence,
        )?;
        let recover_balance = |temperature: &[f64]| {
            assembly
                .energy_balance(
                    input.model,
                    sample,
                    &previous,
                    temperature,
                    time_step_s,
                    reference,
                )
                .map_err(|error| FeaRunError::Assembly(error.to_string()))
        };
        let mut balance = recover_balance(&solve.solution)?;
        if balance.relative_residual > input.residual_target {
            // |sum r_i| <= sqrt(N) ||r||_2. Translate the measured physical power
            // scale to the shared solver's RHS-relative criterion, reserving half
            // the power budget for assembly/recovery roundoff. One bounded refinement;
            // retained physical acceptance is still authoritative, never rewritten.
            // Shared preparation owns prescribed-temperature lifting. Match PCG's
            // normalization (initial residual when the lifted RHS is exactly zero),
            // without preparing a second constrained matrix/coercivity certificate.
            let lifted_rhs = operator.linear_rhs();
            let lifted_norm = input.algebra.dot(&lifted_rhs, &lifted_rhs).sqrt();
            drop(lifted_rhs);
            let reference_norm = if lifted_norm > 0.0 {
                lifted_norm
            } else {
                solve.residual_norm
            };
            let target = (0.5 * input.residual_target * balance.scale.maximum()
                / ((operator.dof_count() as f64).sqrt() * reference_norm))
                .min(input.residual_target);
            let mut refined = solve_checked(
                &operator,
                input.requested_solver_backend,
                input.algebra,
                Some(&solve.solution),
                operators,
                target,
            )?;
            host_sync_count = host_sync_count.saturating_add(refined.host_sync_count);
            device_apply_count = device_apply_count.saturating_add(refined.device_apply_k_count);
            device_apply_attempt_count =
                device_apply_attempt_count.saturating_add(refined.device_apply_k_attempt_count);
            backend_evidence::retain(
                &mut accelerator_evidence,
                &mut refined.accelerator_evidence,
                &input.facts.metadata.backend_evidence,
            )?;
            balance = recover_balance(&refined.solution)?;
            solve = refined;
        }
        residual_norms.push(linear_residual(&solve)?);
        energy_balances.push(balance);
        push_recovered_snapshot(
            &assembly,
            &solve.solution,
            Some(&previous),
            time_step_s,
            reference,
            &mut recovered,
        )?;
        previous = solve.solution.clone();
        time_points_s.push(time_s);
        last_solve = Some(solve);
    }
    Ok(ProcedureOutcome {
        time_points_s,
        recovered,
        residual_norms,
        energy_balances,
        nonlinear_evidence,
        source_evidence,
        last_solve: last_solve.expect("validated transient controls perform at least one solve"),
        host_sync_count,
        device_apply_count,
        device_apply_attempt_count,
        accelerator_evidence,
    })
}
