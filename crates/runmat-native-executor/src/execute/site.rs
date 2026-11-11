use runmat_mir::MirStmtKind;
use runmat_native_codegen::{
    NativeEdge, NativeEdgeArgument, NativeInstruction, NativeOperation, NativeTerminator,
    NativeTerminatorKind,
};
use runmat_runtime::native::{
    NativeCall, NativeExit, NativeSiteOutcome, NativeSitePhase, NativeSiteRequest, NativeValueRef,
};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::operand::evaluate_rvalue;
use super::state::HostState;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "site/terminator_tests.rs"]
mod terminator_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "site/retirement_tests.rs"]
mod retirement_tests;

pub(super) fn execute(
    state: &mut HostState,
    call: &mut NativeCall,
    request: NativeSiteRequest,
    exit: &mut NativeExit,
) -> NativeExecutorResult<NativeSiteOutcome> {
    if request.function != state.function.portable_id.0 {
        return Err(NativeExecutorError::Host(
            "native site function identity mismatch".into(),
        ));
    }
    // A resumed generated block re-enters from its first site. Skip verified
    // prefix sites before matching the completed site's retirement payload.
    if state.skip_before_resume(request)? {
        return Ok(NativeSiteOutcome::continue_execution());
    }
    // A completed retirement resumes the already-executed site's exact POD
    // control-flow decision. Never replay its mutation or edge binding.
    if let Some(outcome) = super::call_suspension::take_completed_retirement(state, request)? {
        return Ok(outcome);
    }
    if state.region_observation.is_some() && !request_matches_function(state, request) {
        state.region_observation = None;
    } else {
        match super::region::observation::before_site(state, request) {
            Ok(()) => {}
            Err(NativeExecutorError::CallSuspended) => {
                return super::call_suspension::publish(state, call, request, exit)
            }
            Err(error) => return Err(error),
        }
    }
    let initial_candidates = state.runtime.has_cleanup_candidates();
    let mut cleanup = state.runtime.begin_value_retirement()?;
    let _initial_selected = if initial_candidates {
        Some(cleanup.selected_context(&state.runtime)?.enter())
    } else {
        None
    };
    // The observation tail has settled and no current operand borrow exists.
    // Retire only obsolete generations, before the next operation can borrow
    // arena payloads or produce an outgoing exception token.
    state.retire_obsolete_generations(call)?;
    let result = execute_inner(state, call, request, exit);
    let observed = super::region::observation::after_site(state, request, &result);
    runmat_runtime::numeric_region::service::set_native_region_for_current_poll(
        super::region::observation::active_audit(state),
    );
    let mut result = match (result, observed) {
        (Err(first), secondary) => with_secondary(Err(first), secondary),
        (Ok(_), Err(error)) => Err(error),
        (Ok(outcome), Ok(())) => Ok(outcome),
    };
    // Select the exact completion before any post-site slot retirement or
    // Base publication can make an active cleanup target unreachable. The
    // guard covers this synchronous retirement only, never a suspension.
    let selected_guard = if state.runtime.has_cleanup_candidates() {
        let selected = match cleanup.selected_context(&state.runtime) {
            Ok(selected) => selected,
            Err(error) => return with_secondary(result, Err(error.into())),
        };
        Some(selected.enter())
    } else {
        None
    };
    if result
        .as_ref()
        .is_ok_and(|outcome| outcome.kind == runmat_runtime::native::NativeSiteOutcomeKind::EXIT)
    {
        return result;
    }
    if result.is_err() {
        let retired = state.abandon_site_continuations();
        result = with_secondary(result, retired);
    }
    if state.runtime.has_cleanup_candidates()
        && !result.as_ref().is_ok_and(|outcome| {
            outcome.kind == runmat_runtime::native::NativeSiteOutcomeKind::EDGE
        })
    {
        let site = state
            .function
            .blocks
            .iter()
            .find(|block| block.id.0 == request.block)
            .and_then(|block| {
                block
                    .instructions
                    .iter()
                    .find(|instruction| site_matches(instruction, request))
            })
            .map(|instruction| instruction.site.clone());
        if let Some(site) = site {
            result = with_secondary(result, state.retire_site_values(&site));
        }
    }
    result = with_secondary(result, state.retire_obsolete_generations(call));
    if state.runtime.has_cleanup_candidates() {
        // The native frame stages its authentic Base mutation through the
        // current workspace service. Publish that admitted prefix before the
        // cleanup completion checks whether the displaced owner is still live.
        let publication = cleanup
            .selected_context(&state.runtime)
            .and_then(|selected| {
                selected
                    .service_ports()
                    .workspace()
                    .map_or(Ok(()), |workspace| {
                        workspace.publish_admitted_mutations_with_context(selected)
                    })
            })
            .map_err(NativeExecutorError::from);
        result = with_secondary(result, publication);
    }
    let _ = &selected_guard;
    match cleanup.admit_retired_values(&state.runtime) {
        Ok(true) => {
            super::call_suspension::begin_retirement(state, request, cleanup, result)?;
            super::call_suspension::publish(state, call, request, exit)
        }
        Ok(false) => result,
        Err(error) => with_secondary(result, Err(error.into())),
    }
}

fn with_secondary<T>(
    result: NativeExecutorResult<T>,
    secondary: NativeExecutorResult<()>,
) -> NativeExecutorResult<T> {
    match (result, secondary) {
        (result, Ok(())) => result,
        (Ok(_), Err(error)) => Err(error),
        (Err(original), Err(secondary)) => runmat_runtime::runtime_error::complete_with_cleanup(
            Err(original.into_runtime_error()),
            Err(secondary.into_runtime_error()),
        )
        .map_err(Into::into),
    }
}

fn request_matches_function(state: &HostState, request: NativeSiteRequest) -> bool {
    request.function == state.function.portable_id.0
        && state
            .function
            .blocks
            .iter()
            .find(|block| block.id.0 == request.block)
            .is_some_and(|block| {
                if request.phase == NativeSitePhase::TERMINATOR {
                    ensure_site(&block.terminator, request).is_ok()
                } else {
                    block
                        .instructions
                        .iter()
                        .any(|instruction| site_matches(instruction, request))
                }
            })
}

fn execute_inner(
    state: &mut HostState,
    call: &mut NativeCall,
    request: NativeSiteRequest,
    exit: &mut NativeExit,
) -> NativeExecutorResult<NativeSiteOutcome> {
    if let Some(cancelled) = super::control::poll(&state.runtime.cancellation()) {
        *exit = cancelled;
        return Ok(NativeSiteOutcome::exit());
    }
    if state.skip_optimized_site(request) {
        return Ok(NativeSiteOutcome::continue_execution());
    }
    let block = state
        .function
        .blocks
        .iter()
        .find(|block| block.id.0 == request.block)
        .cloned()
        .ok_or_else(|| {
            NativeExecutorError::Host(format!("native block {} is unavailable", request.block))
        })?;
    if request.phase == NativeSitePhase::TERMINATOR {
        ensure_site(&block.terminator, request)?;
        let _catch_guard = enter_site(
            state,
            call,
            &block.terminator.site,
            &block.terminator.source,
            block.terminator.frame_state.side_effect_epoch.0,
            request,
        )?;
        if super::deoptimization::checkpoint(
            state,
            call,
            &block.terminator.site,
            &block.terminator.frame_state,
            &block.region_boundaries,
            block.terminator.safepoint,
            exit,
        )? {
            return Ok(NativeSiteOutcome::exit());
        }
        if matches!(block.terminator.kind, NativeTerminatorKind::For { .. })
            && super::osr::checkpoint(
                state,
                call,
                &block.terminator.site,
                &block.terminator.frame_state,
                exit,
            )?
        {
            return Ok(NativeSiteOutcome::exit());
        }
        let outcome = match execute_terminator(state, call, block.id, &block.terminator, exit) {
            Ok(outcome) => {
                state.finish_contextual_site()?;
                outcome
            }
            Err(NativeExecutorError::CallSuspended) => {
                return super::call_suspension::publish(state, call, request, exit);
            }
            Err(error) => {
                state.finish_contextual_site()?;
                return Err(error);
            }
        };
        refresh_frame_roots(state, call)?;
        Ok(outcome)
    } else {
        let instruction = block
            .instructions
            .iter()
            .find(|instruction| site_matches(instruction, request))
            .cloned()
            .ok_or_else(|| {
                NativeExecutorError::Host("native instruction site is unavailable".into())
            })?;
        let _catch_guard = enter_site(
            state,
            call,
            &instruction.site,
            &instruction.source,
            instruction
                .frame_state
                .as_ref()
                .map_or(0, |frame| frame.side_effect_epoch.0),
            request,
        )?;
        let frame_state = instruction
            .frame_state
            .as_ref()
            .unwrap_or(&block.terminator.frame_state);
        if super::deoptimization::checkpoint(
            state,
            call,
            &instruction.site,
            frame_state,
            &block.region_boundaries,
            instruction.safepoint,
            exit,
        )? {
            return Ok(NativeSiteOutcome::exit());
        }
        if request.phase == NativeSitePhase::TERMINATOR_RVALUE && instruction.site.ordinal == 0 {
            if let NativeTerminatorKind::ParFor { region, .. } = &block.terminator.kind {
                super::deoptimization::install_exit_for_target(
                    state,
                    call,
                    &instruction.site,
                    frame_state,
                    super::deoptimization::NativeTransfer::interpreter_slow_path(
                        parallel_region_token(*region),
                    ),
                    exit,
                )?;
                return Ok(NativeSiteOutcome::exit());
            }
        }
        if request.phase == NativeSitePhase::RVALUE {
            let region = match super::region::checkpoint(state, request, exit) {
                Ok(region) => region,
                Err(NativeExecutorError::CallSuspended) => {
                    return super::call_suspension::publish(state, call, request, exit);
                }
                Err(error) => return Err(error),
            };
            if let Some(outcome) = region {
                // The region completed this site's work without entering the
                // ordinary instruction path. Release its expression scope and
                // publish the updated roots before the next native site.
                state.finish_contextual_site()?;
                refresh_frame_roots(state, call)?;
                return Ok(outcome);
            }
        }
        // A MATLAB `for` iterable is evaluated once. Native IR represents its
        // evaluation as terminator-rvalue sites in the loop header, so a body
        // backedge must retain their first SSA results instead of replaying
        // calls or effects before the next column is selected.
        let retained_for_iterable = request.phase == NativeSitePhase::TERMINATOR_RVALUE
            && state.has_for_loop(block.id)
            && matches!(block.terminator.kind, NativeTerminatorKind::For { .. });
        if !retained_for_iterable {
            match execute_instruction(state, &instruction) {
                Ok(()) => state.finish_contextual_site()?,
                Err(NativeExecutorError::CallSuspended) => {
                    return super::call_suspension::publish(state, call, request, exit);
                }
                Err(error) => {
                    state.finish_contextual_site()?;
                    return Err(error);
                }
            }
        } else {
            state.finish_contextual_site()?;
        }
        refresh_frame_roots(state, call)?;
        Ok(NativeSiteOutcome::continue_execution())
    }
}

fn enter_site(
    state: &mut HostState,
    call: &mut NativeCall,
    site: &runmat_native_codegen::NativeMirSite,
    source: &runmat_native_codegen::NativeSourceLocation,
    side_effect_epoch: u32,
    request: NativeSiteRequest,
) -> NativeExecutorResult<runmat_runtime::context::ActiveCatchGuard> {
    state.hit_coverage(site);
    state.current_source = runtime_source(source);
    state.current_request = Some(request);
    state.begin_contextual_site(request)?;
    state.enter_site_block(runmat_native_codegen::NativeBlockId(request.block));
    // SAFETY: NativeCall was validated before entry and its frame/resume
    // backing allocations live for the complete synchronous invocation.
    let frame = unsafe { &mut *call.frame };
    if frame.resume.is_null() {
        return Err(NativeExecutorError::Host(
            "native frame has no resume state".into(),
        ));
    }
    // SAFETY: checked above and owned by the invoking executor.
    let resume = unsafe { &mut *frame.resume };
    resume.function = request.function;
    resume.block = request.block;
    resume.position = request.position;
    resume.phase = request.phase.0;
    resume.ordinal = request.ordinal;
    resume.local_count = state.locals.len() as u32;
    resume.side_effect_epoch = u64::from(side_effect_epoch);
    resume.source = state.current_source;
    refresh_frame_roots(state, call)?;
    // The machine entry began before this MIR block was selected. Scope every
    // synchronous semantic operation at the site to its actual lexical catch,
    // including dynamic calls that do not enter call_suspension.
    Ok(state
        .runtime
        .enter_active_catch_shared(state.active_catch_exception()))
}

fn refresh_frame_roots(state: &mut HostState, call: &mut NativeCall) -> NativeExecutorResult<()> {
    if call.frame.is_null() {
        return Err(NativeExecutorError::Host("native call has no frame".into()));
    }
    let roots = state.refresh_roots()?;
    // SAFETY: checked above and retained by the invoking executor.
    unsafe { (*call.frame).roots = roots };
    Ok(())
}

fn runtime_source(
    source: &runmat_native_codegen::NativeSourceLocation,
) -> runmat_runtime::native::NativeSourceLocation {
    runmat_runtime::native::NativeSourceLocation {
        source: source.source.0,
        reserved: 0,
        start: source.span.start,
        end: source.span.end,
    }
}

fn execute_instruction(
    state: &mut HostState,
    instruction: &NativeInstruction,
) -> NativeExecutorResult<()> {
    match &instruction.operation {
        NativeOperation::Rvalue {
            value,
            result: runmat_native_codegen::NativeRvalueResult::MultiAssignment(targets),
        } if targets
            .targets
            .iter()
            .any(|target| matches!(target, runmat_mir::MirOutputTarget::Sequence(_))) =>
        {
            let prepared = match state.prepared_multi_assignment.take() {
                Some(prepared) => {
                    prepared.skip_preparation_regions(state)?;
                    prepared
                }
                None => super::mutation::PreparedMultiAssignment::prepare(state, targets)?,
            };
            let results = match evaluate_rvalue(state, value, prepared.cardinality(), None) {
                Ok(results) => results,
                Err(error) => {
                    // A suspending RHS resumes this exact rvalue site. Retain the
                    // already-materialized targets so selector and protocol
                    // preparation is never replayed before completion routing.
                    state.prepared_multi_assignment = Some(prepared);
                    return Err(error);
                }
            };
            let prepared = prepared.bind_values(state, results)?;
            state.prepared_multi_assignment = Some(prepared);
        }
        NativeOperation::Rvalue {
            value,
            result: runmat_native_codegen::NativeRvalueResult::SequenceAssignment(target),
        } => {
            if state.sequence_assignment_register.is_some() {
                return Err(NativeExecutorError::Host(
                    "native sequence register was overwritten before consumption".into(),
                ));
            }
            let prepared = match state.pending_sequence_rvalue.take() {
                Some(prepared) => {
                    super::mutation::skip_prepared_sequence_region(state)?;
                    prepared
                }
                None => {
                    let (destination, count) =
                        super::mutation::prepare_sequence_assignment(state, target)?;
                    super::mutation::PreparedSequenceRvalue { destination, count }
                }
            };
            let results = match evaluate_rvalue(state, value, prepared.count, None) {
                Ok(results) => results,
                Err(NativeExecutorError::CallSuspended) => {
                    state.pending_sequence_rvalue = Some(prepared);
                    return Err(NativeExecutorError::CallSuspended);
                }
                Err(error) => return Err(error),
            };
            state.sequence_assignment_register =
                Some(super::mutation::PreparedSequenceAssignment::with_values(
                    prepared.destination,
                    results,
                )?);
        }
        NativeOperation::Rvalue { value, .. } => {
            let output_local = (instruction.outputs.len() == 1)
                .then(|| instruction.outputs[0].local)
                .flatten();
            let results = evaluate_rvalue(state, value, instruction.outputs.len(), output_local)?;
            if instruction.outputs.len() != results.len() {
                return Err(NativeExecutorError::Host(format!(
                    "native rvalue produced {} values for {} SSA outputs",
                    results.len(),
                    instruction.outputs.len()
                )));
            }
            for (output, result) in instruction.outputs.iter().zip(results) {
                state.values.insert(output.value, result);
            }
        }
        NativeOperation::Statement(statement) => execute_statement(state, instruction, statement)?,
    }
    Ok(())
}

fn execute_statement(
    state: &mut HostState,
    instruction: &NativeInstruction,
    statement: &MirStmtKind,
) -> NativeExecutorResult<()> {
    if super::mutation::execute(state, instruction, statement)? {
        return Ok(());
    }
    if super::workspace::execute(state, instruction, statement)? {
        return Ok(());
    }
    match statement {
        MirStmtKind::StatementOutput(output) => {
            if !super::call_suspension::take_completed_statement_output(state)? {
                super::call_suspension::begin_statement_output(state, output)?;
            }
        }
        MirStmtKind::Expr(_) => {
            state.pending_assign = None;
            state.pending_plain_multi = None;
            state.pending_place_mutation = None;
            if let Some(value) = instruction
                .inputs
                .first()
                .and_then(|input| state.values.get(input))
                .copied()
            {
                state.record_expression(value);
            }
        }
        MirStmtKind::CaptureSequence {
            destination,
            source,
        } => super::call::capture_sequence(state, *destination, source)?,
        MirStmtKind::PrepareFixedAssignment(_)
        | MirStmtKind::PreparedAssign { .. }
        | MirStmtKind::Assign { .. }
        | MirStmtKind::MultiAssign { .. }
        | MirStmtKind::SequenceAssign { .. }
        | MirStmtKind::PlaceMutation(_) => {
            return Err(NativeExecutorError::Host(
                "native mutation statement was not consumed by its semantic adapter".into(),
            ))
        }
        MirStmtKind::WorkspaceEffect { .. } | MirStmtKind::EnvironmentEffect(_) => {
            return Err(NativeExecutorError::Host(
                "native environment statement was not consumed by its semantic adapter".into(),
            ))
        }
    }
    Ok(())
}

fn execute_terminator(
    state: &mut HostState,
    call: &mut NativeCall,
    block: runmat_native_codegen::NativeBlockId,
    terminator: &NativeTerminator,
    exit: &mut NativeExit,
) -> NativeExecutorResult<NativeSiteOutcome> {
    match &terminator.kind {
        NativeTerminatorKind::Goto { edge } => take_edge(state, edge, 0, None, None),
        NativeTerminatorKind::Branch {
            condition,
            then_edge,
            else_edge,
        } => {
            let truth = match super::call_suspension::take_completed_terminator(state)?
                .map(super::call_suspension::TerminatorReady::into_parts)
            {
                Some((super::call_suspension::TerminatorReadyStage::BranchTruth(truth), None)) => {
                    truth
                }
                Some(_) => {
                    return Err(NativeExecutorError::Host(
                        "native branch resumed with another terminator stage".into(),
                    ));
                }
                None => {
                    let condition = value_for(state, *condition)?;
                    let value = state.arena.get(condition)?.clone();
                    return super::call_suspension::begin_branch_truth(state, value);
                }
            };
            if truth {
                take_edge(state, then_edge, 0, None, None)
            } else {
                take_edge(state, else_edge, 1, None, None)
            }
        }
        NativeTerminatorKind::Switch {
            discriminant,
            cases,
            otherwise,
        } => {
            use super::call_suspension::TerminatorReadyStage;
            let (index, discriminant) =
                match super::call_suspension::take_completed_terminator(state)?
                    .map(super::call_suspension::TerminatorReady::into_parts)
                {
                    Some((
                        TerminatorReadyStage::SwitchEquality {
                            index,
                            discriminant,
                        },
                        Some(equality),
                    )) => {
                        if index >= cases.len() {
                            return Err(NativeExecutorError::Host(
                                "resumed switch case index is unavailable".into(),
                            ));
                        }
                        return super::call_suspension::begin_switch_truth(
                            state,
                            index,
                            discriminant,
                            equality,
                        );
                    }
                    Some((
                        TerminatorReadyStage::SwitchTruth {
                            index,
                            discriminant,
                            truth,
                        },
                        None,
                    )) => {
                        if index >= cases.len() {
                            return Err(NativeExecutorError::Host(
                                "resumed switch truth index is unavailable".into(),
                            ));
                        }
                        if truth {
                            return take_edge(state, &cases[index].1, index as u32, None, None);
                        }
                        (index + 1, discriminant)
                    }
                    Some(_) => {
                        return Err(NativeExecutorError::Host(
                            "native switch resumed with another terminator stage".into(),
                        ));
                    }
                    None => {
                        let value = state.arena.get(value_for(state, *discriminant)?)?.clone();
                        let rooted = runmat_runtime::execution::RootedValue::new(value).map_err(
                            |error| NativeExecutorError::from(error.into_runtime_error()),
                        )?;
                        (0, rooted)
                    }
                };
            if let Some((case, _)) = cases.get(index) {
                // Read this case only after the previous equality and truth
                // completions reconciled workspace effects.
                let case = state.arena.get(value_for(state, *case)?)?.clone();
                return super::call_suspension::begin_switch_equality(
                    state,
                    index,
                    discriminant,
                    case,
                );
            }
            take_edge(state, otherwise, cases.len() as u32, None, None)
        }
        NativeTerminatorKind::For {
            iterable,
            binding,
            body,
            exit,
        } => {
            match super::call_suspension::take_completed_terminator(state)?
                .map(super::call_suspension::TerminatorReady::into_parts)
            {
                Some((
                    super::call_suspension::TerminatorReadyStage::ForCapture(iterator),
                    None,
                )) => {
                    state.start_for_loop(block, body.target, iterator);
                }
                Some((super::call_suspension::TerminatorReadyStage::ForNext(next), None)) => {
                    if let Some(reference) = next {
                        state.observe_loop_backedge(terminator.site.point);
                        return take_edge(state, body, 0, Some((*binding, reference)), None);
                    }
                    return take_edge(state, exit, 1, None, None);
                }
                Some(_) => {
                    return Err(NativeExecutorError::Host(
                        "native for-loop resumed with another terminator stage".into(),
                    ));
                }
                None if !state.has_for_loop(block) => {
                    let source = state.arena.get(value_for(state, *iterable)?)?.clone();
                    return super::call_suspension::begin_for_capture(state, source);
                }
                None => {}
            }
            super::call_suspension::begin_for_next(state, block)
        }
        NativeTerminatorKind::TryCatch {
            try_edge,
            catch_edge,
            ..
        } => {
            state.enter_exception_handler(try_edge, catch_edge)?;
            take_edge(state, try_edge, 0, None, None)
        }
        NativeTerminatorKind::Return { values } => {
            let requested_outputs = call.requested_outputs as usize;
            if requested_outputs > call.result_capacity {
                return Err(NativeExecutorError::Host(
                    "native return cannot satisfy the requested output window".into(),
                ));
            }
            let fixed_outputs = if state.function.abi.fixed_outputs.is_empty()
                && state.function.abi.varargout.is_none()
            {
                values
                    .iter()
                    .map(|value| state.arena.get(value_for(state, *value)?).cloned())
                    .collect::<NativeExecutorResult<Vec<_>>>()?
            } else {
                let output_locals = state.function.abi.fixed_outputs.clone();
                output_locals
                    .into_iter()
                    .map(|local| return_local_value(state, terminator, local))
                    .collect::<NativeExecutorResult<Vec<_>>>()?
            };
            let varargout = state
                .function
                .abi
                .varargout
                .map(|local| return_local_value(state, terminator, local))
                .transpose()?;
            let outputs = runmat_runtime::call::function_abi::collect_function_outputs(
                &state.function.name,
                &fixed_outputs,
                varargout.as_ref(),
                requested_outputs,
            )?;
            for (index, value) in outputs.into_iter().enumerate() {
                let reference = state.arena.insert(value, Some(&state.runtime))?;
                // SAFETY: NativeCall validation guarantees a writable result
                // slice of result_capacity elements for this invocation.
                unsafe { *call.results.add(index) = reference };
            }
            *exit = NativeExit::completed(call.requested_outputs);
            Ok(NativeSiteOutcome::exit())
        }
        NativeTerminatorKind::Unreachable => Err(NativeExecutorError::Host(
            "reached a Native IR unreachable terminator".into(),
        )),
        NativeTerminatorKind::Await { future, resume, .. } => {
            let value = state.arena.get(value_for(state, *future)?)?.clone();
            match super::awaiting::begin(state, value, resume.clone())? {
                super::awaiting::AwaitStart::Ready(value) => {
                    let value = value
                        .with_owned(|value| state.arena.insert(value, Some(&state.runtime)))?;
                    take_edge(state, resume, 0, None, Some(value))
                }
                super::awaiting::AwaitStart::Suspended {
                    continuation,
                    generation,
                } => {
                    if call.frame.is_null() {
                        return Err(NativeExecutorError::Host(
                            "native await has no frame".into(),
                        ));
                    }
                    let roots = state.refresh_roots()?;
                    // SAFETY: NativeCall validation guarantees a live frame and resume
                    // record for this synchronous entry. The pointer is validated before
                    // returning from this entry and is not retained by the Rust driver.
                    let resume_state = unsafe { (*call.frame).resume };
                    if resume_state.is_null() {
                        return Err(NativeExecutorError::Host(
                            "native await has no resume state".into(),
                        ));
                    }
                    let target = state.resume_request_for_block(resume.target)?;
                    // SAFETY: the resume record belongs to this invocation and remains
                    // writable for the complete synchronous entry. Publish the exact
                    // post-await site in the ABI exit; the Rust continuation transfers
                    // the awaited value before generated code re-enters that same site.
                    unsafe {
                        (*resume_state).function = target.function;
                        (*resume_state).block = target.block;
                        (*resume_state).position = target.position;
                        (*resume_state).phase = target.phase.0;
                        (*resume_state).ordinal = target.ordinal;
                        (*call.frame).roots = roots;
                    }
                    *exit = NativeExit::suspended(runmat_runtime::native::NativeSuspension {
                        continuation,
                        generation,
                        resume: resume_state,
                        roots,
                    });
                    Ok(NativeSiteOutcome::exit())
                }
            }
        }
        NativeTerminatorKind::ParFor { region, .. } => {
            super::deoptimization::install_exit_for_target(
                state,
                call,
                &terminator.site,
                &terminator.frame_state,
                super::deoptimization::NativeTransfer::interpreter_slow_path(
                    parallel_region_token(*region),
                ),
                exit,
            )?;
            Ok(NativeSiteOutcome::exit())
        }
        NativeTerminatorKind::Spmd { .. } => Err(NativeExecutorError::UnsupportedSite(
            "SPMD terminator requires the distributed native executor".into(),
        )),
    }
}

fn parallel_region_token(region: runmat_types::ParallelRegionId) -> u64 {
    (u64::from(region.0.function.0) << 32) | u64::from(region.0.ordinal)
}

fn return_local_value(
    state: &mut HostState,
    terminator: &NativeTerminator,
    local: runmat_native_codegen::NativeLocalId,
) -> NativeExecutorResult<runmat_value::Value> {
    if let Some(reference) = state.declared_session_reference(local.0 as usize)? {
        return state.arena.get(reference).cloned();
    }
    let value = terminator
        .frame_state
        .locals
        .iter()
        .find(|candidate| candidate.local == local)
        .map(|candidate| candidate.value)
        .ok_or_else(|| {
            NativeExecutorError::Host("native return frame omits an ABI output local".into())
        })?;
    state.arena.get(value_for(state, value)?).cloned()
}

pub(in crate::execute) fn take_edge(
    state: &mut HostState,
    edge: &NativeEdge,
    index: u32,
    loop_iteration: Option<(runmat_native_codegen::NativeLocalId, NativeValueRef)>,
    transferred_value: Option<NativeValueRef>,
) -> NativeExecutorResult<NativeSiteOutcome> {
    let parameters = state
        .function
        .blocks
        .iter()
        .find(|block| block.id == edge.target)
        .ok_or_else(|| NativeExecutorError::Host("native edge target is unavailable".into()))?
        .parameters
        .clone();
    if parameters.len() != edge.arguments.len() {
        return Err(NativeExecutorError::Host(
            "native edge arity mismatch".into(),
        ));
    }
    let mut transferred = Vec::with_capacity(parameters.len());
    for (parameter, argument) in parameters.iter().zip(&edge.arguments) {
        let value = match argument {
            NativeEdgeArgument::Value(value) => value_for(state, *value)?,
            NativeEdgeArgument::LoopIteration { local } => loop_iteration
                .filter(|(binding, _)| binding == local)
                .map(|(_, value)| value)
                .ok_or_else(|| {
                    NativeExecutorError::Host(
                        "native loop iteration edge binding is unavailable".into(),
                    )
                })?,
            NativeEdgeArgument::CaughtException { .. } => transferred_value.ok_or_else(|| {
                NativeExecutorError::Host("native catch edge has no materialized exception".into())
            })?,
            NativeEdgeArgument::AwaitResult { .. } => transferred_value.ok_or_else(|| {
                NativeExecutorError::Host("native await edge has no completed value".into())
            })?,
        };
        transferred.push((
            parameter.clone(),
            value,
            matches!(argument, NativeEdgeArgument::Value(_)),
        ));
    }
    for (parameter, value, inherited_snapshot) in transferred {
        let bound = state.bind_edge_local(parameter.local.0 as usize, value, inherited_snapshot)?;
        state.values.insert(parameter.value, bound);
    }
    state.take_control_edge(edge.target);
    state.retire_edge_values(edge.target)?;
    Ok(NativeSiteOutcome::edge(index))
}

pub(super) fn resume_await(
    state: &mut HostState,
    completion: super::awaiting::AwaitCompletion,
) -> NativeExecutorResult<runmat_runtime::native::NativeSiteRequest> {
    let value = completion
        .value
        .with_owned(|value| state.arena.insert(value, Some(&state.runtime)))?;
    let target = completion.edge.target;
    let _ = take_edge(state, &completion.edge, 0, None, Some(value))?;
    state.resume_request_for_block(target)
}

pub(super) fn redirect_exception(
    state: &mut HostState,
    exception: runmat_runtime::native::NativeException,
) -> NativeExecutorResult<Option<runmat_runtime::native::NativeSiteRequest>> {
    state.abandon_site_continuations()?;
    let Some(handler) = state.take_exception_handler() else {
        return Ok(None);
    };
    let reference = NativeValueRef {
        handle: exception.handle,
        generation: exception.generation,
    };
    let exception = match state.arena.get(reference)? {
        runmat_value::Value::MException(exception) => exception.clone(),
        _ => {
            return Err(NativeExecutorError::Host(
                "native exception exit does not reference an MException".into(),
            ))
        }
    };
    let target = handler.catch_edge.target;
    state.bind_catch_exception(target, exception)?;
    let _ = take_edge(state, &handler.catch_edge, 1, None, Some(reference))?;
    state.resume_request_for_block(target).map(Some)
}

fn value_for(
    state: &HostState,
    value: runmat_native_codegen::NativeValueId,
) -> NativeExecutorResult<NativeValueRef> {
    state.values.get(&value).copied().ok_or_else(|| {
        NativeExecutorError::Host(format!("native value {} is unavailable", value.0))
    })
}

fn site_matches(instruction: &NativeInstruction, request: NativeSiteRequest) -> bool {
    instruction.site.point.block == request.block
        && instruction.site.point.position == request.position
        && phase(instruction.site.phase) == request.phase
        && instruction.site.ordinal == request.ordinal
}

fn ensure_site(
    terminator: &NativeTerminator,
    request: NativeSiteRequest,
) -> NativeExecutorResult<()> {
    if terminator.site.point.block == request.block
        && terminator.site.point.position == request.position
        && phase(terminator.site.phase) == request.phase
        && terminator.site.ordinal == request.ordinal
    {
        Ok(())
    } else {
        Err(NativeExecutorError::Host(
            "native terminator site identity mismatch".into(),
        ))
    }
}

fn phase(phase: runmat_native_codegen::NativeSitePhase) -> NativeSitePhase {
    match phase {
        runmat_native_codegen::NativeSitePhase::Rvalue => NativeSitePhase::RVALUE,
        runmat_native_codegen::NativeSitePhase::Statement => NativeSitePhase::STATEMENT,
        runmat_native_codegen::NativeSitePhase::TerminatorRvalue => {
            NativeSitePhase::TERMINATOR_RVALUE
        }
        runmat_native_codegen::NativeSitePhase::Terminator => NativeSitePhase::TERMINATOR,
    }
}
