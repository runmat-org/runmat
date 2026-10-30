//! One bounded piece of the original load controller, with no schedule or history reset.
use super::super::{
    solid::SolidNewtonProblem, LoadIncrementSchedule, NonlinearSolveOptions, NonlinearSolveResult,
};
use super::{
    admission,
    continuation::{add, digest},
    map_assembly_error, map_error, output, NonlinearExecution, NonlinearLoadContinuation,
    NonlinearLoadWork,
};
use crate::{
    assembly::{nonlinear_solid::CanonicalNonlinearSolidBasis, AssemblySummary},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    ComputeBackend, FeaRunError,
};
use runmat_analysis_core::AnalysisModel;
use runmat_execution::Digest;

pub(crate) enum NonlinearLoadBlockStart<'a> {
    Fresh,
    Continue {
        continuation: &'a NonlinearLoadContinuation,
        expected_digest: Digest,
    },
}

pub(crate) struct NonlinearLoadBlockResult {
    pub result: NonlinearSolveResult,
    pub continuation: NonlinearLoadContinuation,
    pub complete: bool,
}

pub(crate) struct NonlinearLoadBlockRequest<'a> {
    pub model: &'a AnalysisModel,
    pub summary: &'a AssemblySummary,
    pub options: &'a NonlinearSolveOptions,
    pub execution: NonlinearExecution<'a>,
    pub start: f64,
    pub end: f64,
    pub schedule: &'a LoadIncrementSchedule,
    pub input: NonlinearLoadBlockStart<'a>,
    pub accepted_block_limit: usize,
}

pub(crate) fn run_nonlinear_load_block(
    request: NonlinearLoadBlockRequest<'_>,
) -> Result<NonlinearLoadBlockResult, FeaRunError> {
    let NonlinearLoadBlockRequest {
        model,
        summary,
        options,
        execution,
        start,
        end,
        schedule,
        input,
        accepted_block_limit,
    } = request;
    if execution.backend != ComputeBackend::Cpu {
        return Err(FeaRunError::InvalidModel(
            "canonical nonlinear solids run on the CPU backend".into(),
        ));
    }
    if execution.contact.is_none() && !model.interfaces.is_empty() {
        return Err(FeaRunError::InvalidModel(
            "authored contact interfaces need a prepared contact operator".into(),
        ));
    }
    if !start.is_finite() || start != 0.0 || !end.is_finite() || end <= start {
        return Err(FeaRunError::InvalidModel(
            "J2 original load-factor domain must increase from zero".into(),
        ));
    }
    if options.increment_count == 0
        || options.max_accepted_increments == 0
        || accepted_block_limit == 0
    {
        return Err(FeaRunError::InvalidModel(
            "J2 increment inventories must be positive".into(),
        ));
    }
    super::super::NonlinearWorkLimits::derive(options)?;
    schedule
        .validate_for(start, end)
        .map_err(|e| FeaRunError::InvalidModel(e.to_string()))?;
    // Authored restart requests still cannot bypass the shared execution integration.
    admission::require_fixed_structural_nonlinear(model)?;
    runmat_analysis_core::validate_model(model)
        .map_err(|e| FeaRunError::InvalidModel(e.to_string()))?;
    let mesh = options.solver_mesh.as_ref().ok_or_else(|| {
        FeaRunError::InvalidModel("J2 requires its canonical solid solver mesh".into())
    })?;
    crate::progress::check_cancelled("structural_nonlinear")?;
    if let Some(contact) = execution.contact {
        contact.admit_outputs(accepted_block_limit)?;
    }
    let coupling = execution.contact.map(|c| c.coupling);
    let basis = CanonicalNonlinearSolidBasis::prepare_composed(
        model,
        mesh,
        summary,
        coupling.map(|c| c.operator),
    )
    .map_err(map_assembly_error)?;
    // Follower loads are configuration dependent: they are realized from the same authored
    // step, kept out of the dead-load vector, and integrated on the deformed boundary.
    let excitations = crate::excitation::realize_step_excitations(
        model,
        &model.sequence.steps[0],
        crate::excitation::ExcitationSamplePoint::Static,
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let follower = basis
        .prepare_follower_loads(mesh, &excitations)
        .map_err(map_assembly_error)?;
    let mut policy = options.newton_policy();
    let problem = SolidNewtonProblem::prepare_with_contact(&basis, summary, policy, coupling)
        .map_err(map_error)?
        .with_follower_loads(follower.as_ref())
        .map_err(map_error)?
        .with_solver_admission(execution.solver_admission);
    let nominal = (end - start) / options.increment_count as f64;
    if !nominal.is_finite() || nominal <= 0.0 {
        return Err(FeaRunError::InvalidModel(
            "nominal J2 load increment is not representable".into(),
        ));
    }
    // Original model/options and numerical scope are content, not placement. Solver mesh
    // authority is already bound by the physical basis; do not encode its full body twice.
    let numerics = (
        options.increment_count,
        options.max_newton_iters,
        options.tolerance,
        options.linear_tolerance,
        options.max_load_cutbacks,
        options.minimum_load_increment,
        options.max_accepted_increments,
        options.line_search,
        options.max_line_search_backtracks,
        options.line_search_reduction,
    );
    let mut scope = digest(
        b"runmat-j2-load-controller-scope/v3\0",
        &(
            model,
            problem.scope_identity(),
            numerics,
            execution.backend,
            start,
            end,
            schedule,
        ),
    )?;
    if let Some(contact) = execution.contact {
        scope = contact.controller_scope(scope)?;
    }
    let mut prefix_work = super::super::solid::SolidAttemptWork::default();
    let mut cursor = match input {
        NonlinearLoadBlockStart::Fresh => NonlinearLoadContinuation {
            schema_version: 4,
            controller_scope: scope,
            content_digest: Digest::sha256([]),
            original_start: start,
            original_end: end,
            schedule: schedule.clone(),
            nominal_increment: nominal,
            scheduled_origin: start,
            pending_target: None,
            accepted_state: if coupling.is_some() {
                problem.initial_state_record(&mut prefix_work)
            } else {
                problem.initial_state()
            }
            .map_err(map_error)?,
            work: NonlinearLoadWork::default(),
            maximum_relative_residual: 0.0,
            maximum_relative_increment: 0.0,
            maximum_linear_relative_residual: 0.0,
        },
        NonlinearLoadBlockStart::Continue {
            continuation,
            expected_digest,
        } => {
            continuation.validate_for(expected_digest, scope, start, end, schedule, options)?;
            if continuation.complete() {
                return Err(FeaRunError::InvalidModel(
                    "completed J2 controller cannot be resumed".into(),
                ));
            }
            problem
                .validate_accepted_record(continuation.accepted_state(), &mut prefix_work)
                .map_err(map_error)?;
            continuation.clone()
        }
    };
    cursor.work.linear_solves = add(cursor.work.linear_solves, prefix_work.linear_solves)?;
    cursor.maximum_linear_relative_residual = cursor
        .maximum_linear_relative_residual
        .max(prefix_work.maximum_linear_relative_residual);
    // Fresh evaluates zero state above; Continue additionally revalidates its physical
    // prefix. Output initialization always uses original accepted point history.
    cursor.work.controller_evaluations = add(cursor.work.controller_evaluations, 2)?;
    let initial = basis
        .evaluate(
            &cursor.accepted_state.displacement,
            &cursor.accepted_state.constitutive,
        )
        .map_err(|e| FeaRunError::Solve(e.to_string()))?;
    let mut result = output::initial_result(
        &cursor.accepted_state,
        &initial,
        coupling.map(|c| c.operator),
        accepted_block_limit,
    )?;
    drop(initial);
    while !cursor.complete() && result.load_factors.len() < accepted_block_limit {
        if crate::is_fea_cancelled() {
            return Err(FeaRunError::Cancelled);
        }
        let remaining = options
            .max_accepted_increments
            .checked_sub(cursor.work.accepted_increments)
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                FeaRunError::Solve("global accepted J2 increment cap exhausted".into())
            })?;
        let target = *cursor
            .pending_target
            .get_or_insert_with(|| schedule.next_factor(cursor.scheduled_origin, nominal, end));
        if !target.is_finite() || target <= cursor.accepted_state.load_factor() || target > end {
            return Err(FeaRunError::Solve(
                "scheduled J2 load increment does not advance finitely".into(),
            ));
        }
        policy.max_accepted_subincrements = remaining;
        let problem = SolidNewtonProblem::prepare_with_contact(&basis, summary, policy, coupling)
            .map_err(map_error)?
            .with_solver_admission(execution.solver_admission);
        let increment = problem
            .advance_next(&cursor.accepted_state, target)
            .map_err(map_error)?;
        cursor.work.accepted_increments = add(cursor.work.accepted_increments, 1)?;
        cursor.work.rejected_increments = add(cursor.work.rejected_increments, increment.cutbacks)?;
        cursor.work.newton_iterations = add(
            cursor.work.newton_iterations,
            increment.attempted_work.iterations,
        )?;
        cursor.work.backtracks = add(cursor.work.backtracks, increment.attempted_work.backtracks)?;
        cursor.work.linear_solves = add(
            cursor.work.linear_solves,
            increment.attempted_work.linear_solves,
        )?;
        cursor.work.tangent_evaluations = add(
            cursor.work.tangent_evaluations,
            increment.attempted_work.tangent_evaluations,
        )?;
        cursor.work.controller_evaluations = add(cursor.work.controller_evaluations, 1)?;
        cursor
            .work
            .validate_with_contact(options, coupling.is_some())?;
        cursor.maximum_relative_residual = cursor
            .maximum_relative_residual
            .max(increment.relative_residual);
        cursor.maximum_relative_increment = cursor
            .maximum_relative_increment
            .max(increment.relative_increment);
        cursor.maximum_linear_relative_residual = cursor
            .maximum_linear_relative_residual
            .max(increment.attempted_work.maximum_linear_relative_residual);
        output::append_increment(&mut result, &increment, coupling.map(|c| c.operator))?;
        cursor.accepted_state = increment.state;
        if cursor.accepted_state.load_factor().to_bits() == target.to_bits() {
            cursor.pending_target = None;
            cursor.scheduled_origin = target;
        }
    }
    let mut terminal_work = super::super::solid::SolidAttemptWork::default();
    problem
        .validate_accepted_record(&cursor.accepted_state, &mut terminal_work)
        .map_err(map_error)?;
    cursor.work.linear_solves = add(cursor.work.linear_solves, terminal_work.linear_solves)?;
    cursor.maximum_linear_relative_residual = cursor
        .maximum_linear_relative_residual
        .max(terminal_work.maximum_linear_relative_residual);
    cursor.work.controller_evaluations = add(cursor.work.controller_evaluations, 1)?;
    cursor
        .work
        .validate_with_contact(options, coupling.is_some())?;
    cursor.content_digest = cursor.derive_digest()?;
    result.total_increments = result.load_factors.len();
    result.converged_increments = result.total_increments;
    result.accepted_state = cursor.accepted_state.clone();
    result.diagnostics.push(FeaDiagnostic {
        code: "FEA_NONLINEAR_CONVERGENCE".into(), severity: FeaDiagnosticSeverity::Info,
        message: format!("physical_quasistatic_j2 block_increments={} cumulative_accepted={} cumulative_cutbacks={} accumulated_plastic_dissipation_j={}", result.total_increments, cursor.work.accepted_increments, cursor.work.rejected_increments, cursor.accepted_state.accumulated_plastic_dissipation_j()),
    });
    Ok(NonlinearLoadBlockResult {
        complete: cursor.complete(),
        result,
        continuation: cursor,
    })
}
