use runmat_mir::MirStmtKind;
use runmat_native_codegen::{
    NativeEdge, NativeEdgeArgument, NativeInstruction, NativeOperation, NativeTerminator,
    NativeTerminatorKind,
};
use runmat_runtime::native::{
    NativeCall, NativeExit, NativeSiteOutcome, NativeSitePhase, NativeSiteRequest, NativeValueRef,
};

use crate::{JitError, JitResult};

use super::operand::evaluate_rvalue;
use super::state::HostState;

pub(super) fn execute(
    state: &mut HostState,
    call: &mut NativeCall,
    request: NativeSiteRequest,
    exit: &mut NativeExit,
) -> JitResult<NativeSiteOutcome> {
    if request.function != state.function.id.0 {
        return Err(JitError::Host(
            "native site function identity mismatch".into(),
        ));
    }
    if state.skip_before_resume(request)? {
        return Ok(NativeSiteOutcome::continue_execution());
    }
    if state
        .runtime
        .cancellation()
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        *exit = NativeExit::cancelled(runmat_runtime::native::NativeCancellation {
            reason: runmat_runtime::native::NativeCancellationReason::REQUESTED,
            reserved: 0,
            generation: 1,
        });
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
        .ok_or_else(|| JitError::Host(format!("native block {} is unavailable", request.block)))?;
    if request.phase == NativeSitePhase::TERMINATOR {
        ensure_site(&block.terminator, request)?;
        enter_site(
            state,
            call,
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
        let outcome = execute_terminator(state, call, block.id, &block.terminator, exit)?;
        refresh_frame_roots(state, call)?;
        Ok(outcome)
    } else {
        let instruction = block
            .instructions
            .iter()
            .find(|instruction| site_matches(instruction, request))
            .cloned()
            .ok_or_else(|| JitError::Host("native instruction site is unavailable".into()))?;
        enter_site(
            state,
            call,
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
        if request.phase == NativeSitePhase::RVALUE {
            if let Some(outcome) = super::region::checkpoint(state, request, exit)? {
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
            execute_instruction(state, &instruction)?;
        }
        refresh_frame_roots(state, call)?;
        Ok(NativeSiteOutcome::continue_execution())
    }
}

fn enter_site(
    state: &mut HostState,
    call: &mut NativeCall,
    source: &runmat_native_codegen::NativeSourceLocation,
    side_effect_epoch: u32,
    request: NativeSiteRequest,
) -> JitResult<()> {
    state.current_source = runtime_source(source);
    state.enter_site_block(runmat_native_codegen::NativeBlockId(request.block));
    // SAFETY: NativeCall was validated before entry and its frame/resume
    // backing allocations live for the complete synchronous invocation.
    let frame = unsafe { &mut *call.frame };
    if frame.resume.is_null() {
        return Err(JitError::Host("native frame has no resume state".into()));
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
    refresh_frame_roots(state, call)
}

fn refresh_frame_roots(state: &mut HostState, call: &mut NativeCall) -> JitResult<()> {
    if call.frame.is_null() {
        return Err(JitError::Host("native call has no frame".into()));
    }
    let roots = state.refresh_roots();
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

fn execute_instruction(state: &mut HostState, instruction: &NativeInstruction) -> JitResult<()> {
    match &instruction.operation {
        NativeOperation::Rvalue { value, .. } => {
            let output_local = (instruction.outputs.len() == 1)
                .then(|| instruction.outputs[0].local)
                .flatten();
            let results = evaluate_rvalue(state, value, instruction.outputs.len(), output_local)?;
            if instruction.outputs.len() != results.len() {
                return Err(JitError::Host(format!(
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
) -> JitResult<()> {
    if super::mutation::execute(state, instruction, statement)? {
        return Ok(());
    }
    if super::workspace::execute(state, instruction, statement)? {
        return Ok(());
    }
    match statement {
        MirStmtKind::Expr(_) => {
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
        MirStmtKind::Assign { .. }
        | MirStmtKind::MultiAssign { .. }
        | MirStmtKind::PlaceMutation(_) => {
            return Err(JitError::Host(
                "native mutation statement was not consumed by its semantic adapter".into(),
            ))
        }
        MirStmtKind::WorkspaceEffect { .. } | MirStmtKind::EnvironmentEffect(_) => {
            return Err(JitError::Host(
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
) -> JitResult<NativeSiteOutcome> {
    match &terminator.kind {
        NativeTerminatorKind::Goto { edge } => take_edge(state, edge, 0, None, None),
        NativeTerminatorKind::Branch {
            condition,
            then_edge,
            else_edge,
        } => {
            let condition = value_for(state, *condition)?;
            let value = state.arena.get(condition)?;
            let truth = super::sync::complete(
                &state.runtime,
                runmat_runtime::condition::logical_truth_from_value(value, "branch condition"),
                "branch condition",
            )?;
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
            let discriminant = state.arena.get(value_for(state, *discriminant)?)?.clone();
            for (index, (case, edge)) in cases.iter().enumerate() {
                let case = state.arena.get(value_for(state, *case)?)?.clone();
                let equal = super::sync::complete(
                    &state.runtime,
                    runmat_runtime::call_builtin_async("eq", &[discriminant.clone(), case]),
                    "switch comparison",
                )?;
                if super::sync::complete(
                    &state.runtime,
                    runmat_runtime::condition::logical_truth_from_value(
                        &equal,
                        "switch case comparison",
                    ),
                    "switch truth evaluation",
                )? {
                    return take_edge(state, edge, index as u32, None, None);
                }
            }
            take_edge(state, otherwise, cases.len() as u32, None, None)
        }
        NativeTerminatorKind::For {
            iterable,
            binding,
            body,
            exit,
        } => {
            if !state.has_for_loop(block) {
                let source = state.arena.get(value_for(state, *iterable)?)?.clone();
                let iterator = super::sync::complete(
                    &state.runtime,
                    runmat_runtime::iteration::ForColumnIterator::new(source),
                    "for iterable capture",
                )?;
                state.start_for_loop(block, body.target, iterator);
            }
            let runtime = state.runtime.clone();
            let next = {
                let active = state.for_loop_mut(block)?;
                super::sync::complete(&runtime, active.iterator.next(), "for iteration")?
            };
            if let Some(value) = next {
                state.observe_loop_backedge(terminator.site.point);
                let reference = state.arena.insert(value);
                take_edge(state, body, 0, Some((*binding, reference)), None)
            } else {
                take_edge(state, exit, 1, None, None)
            }
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
                return Err(JitError::Host(
                    "native return cannot satisfy the requested output window".into(),
                ));
            }
            let fixed_outputs = if state.function.abi.fixed_outputs.is_empty()
                && state.function.abi.varargout.is_none()
            {
                values
                    .iter()
                    .map(|value| state.arena.get(value_for(state, *value)?).cloned())
                    .collect::<JitResult<Vec<_>>>()?
            } else {
                state
                    .function
                    .abi
                    .fixed_outputs
                    .iter()
                    .map(|local| return_local_value(state, terminator, *local))
                    .collect::<JitResult<Vec<_>>>()?
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
                let reference = state.arena.insert(value);
                // SAFETY: NativeCall validation guarantees a writable result
                // slice of result_capacity elements for this invocation.
                unsafe { *call.results.add(index) = reference };
            }
            *exit = NativeExit::completed(call.requested_outputs);
            Ok(NativeSiteOutcome::exit())
        }
        NativeTerminatorKind::Unreachable => Err(JitError::Host(
            "reached a Native IR unreachable terminator".into(),
        )),
        NativeTerminatorKind::Await { future, resume, .. } => {
            let value = state.arena.get(value_for(state, *future)?)?.clone();
            match super::awaiting::begin(state, value, resume.clone())? {
                super::awaiting::AwaitStart::Ready(value) => {
                    let value = state.arena.insert(value);
                    take_edge(state, resume, 0, None, Some(value))
                }
                super::awaiting::AwaitStart::Suspended {
                    continuation,
                    generation,
                } => {
                    if call.frame.is_null() {
                        return Err(JitError::Host("native await has no frame".into()));
                    }
                    let roots = state.refresh_roots();
                    // SAFETY: NativeCall validation guarantees a live frame and resume
                    // record for this synchronous entry. The pointer is validated before
                    // returning from this entry and is not retained by the Rust driver.
                    let resume_state = unsafe { (*call.frame).resume };
                    if resume_state.is_null() {
                        return Err(JitError::Host("native await has no resume state".into()));
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
        NativeTerminatorKind::ParFor { .. } | NativeTerminatorKind::Spmd { .. } => {
            Err(JitError::UnsupportedSite(
                "parallel terminator requires the R27 native parallel executor".into(),
            ))
        }
    }
}

fn return_local_value(
    state: &HostState,
    terminator: &NativeTerminator,
    local: runmat_native_codegen::NativeLocalId,
) -> JitResult<runmat_value::Value> {
    let value = terminator
        .frame_state
        .locals
        .iter()
        .find(|candidate| candidate.local == local)
        .map(|candidate| candidate.value)
        .ok_or_else(|| JitError::Host("native return frame omits an ABI output local".into()))?;
    state.arena.get(value_for(state, value)?).cloned()
}

fn take_edge(
    state: &mut HostState,
    edge: &NativeEdge,
    index: u32,
    loop_iteration: Option<(runmat_native_codegen::NativeLocalId, NativeValueRef)>,
    transferred_value: Option<NativeValueRef>,
) -> JitResult<NativeSiteOutcome> {
    let parameters = state
        .function
        .blocks
        .iter()
        .find(|block| block.id == edge.target)
        .ok_or_else(|| JitError::Host("native edge target is unavailable".into()))?
        .parameters
        .clone();
    if parameters.len() != edge.arguments.len() {
        return Err(JitError::Host("native edge arity mismatch".into()));
    }
    let mut transferred = Vec::with_capacity(parameters.len());
    for (parameter, argument) in parameters.iter().zip(&edge.arguments) {
        let value = match argument {
            NativeEdgeArgument::Value(value) => value_for(state, *value)?,
            NativeEdgeArgument::LoopIteration { local } => loop_iteration
                .filter(|(binding, _)| binding == local)
                .map(|(_, value)| value)
                .ok_or_else(|| {
                    JitError::Host("native loop iteration edge binding is unavailable".into())
                })?,
            NativeEdgeArgument::CaughtException { .. } => transferred_value.ok_or_else(|| {
                JitError::Host("native catch edge has no materialized exception".into())
            })?,
            NativeEdgeArgument::AwaitResult { .. } => transferred_value
                .ok_or_else(|| JitError::Host("native await edge has no completed value".into()))?,
        };
        transferred.push((parameter.clone(), value));
    }
    for (parameter, value) in transferred {
        state.values.insert(parameter.value, value);
        state.set_local(parameter.local.0 as usize, value)?;
    }
    state.take_control_edge(edge.target);
    Ok(NativeSiteOutcome::edge(index))
}

pub(super) fn resume_await(
    state: &mut HostState,
    completion: super::awaiting::AwaitCompletion,
) -> JitResult<runmat_runtime::native::NativeSiteRequest> {
    let value = state.arena.insert(completion.value);
    let target = completion.edge.target;
    let _ = take_edge(state, &completion.edge, 0, None, Some(value))?;
    state.resume_request_for_block(target)
}

pub(super) fn redirect_exception(
    state: &mut HostState,
    exception: runmat_runtime::native::NativeException,
) -> JitResult<Option<runmat_runtime::native::NativeSiteRequest>> {
    let Some(handler) = state.take_exception_handler() else {
        return Ok(None);
    };
    let reference = NativeValueRef {
        handle: exception.handle,
        generation: exception.generation,
    };
    if !matches!(
        state.arena.get(reference)?,
        runmat_value::Value::MException(_)
    ) {
        return Err(JitError::Host(
            "native exception exit does not reference an MException".into(),
        ));
    }
    let target = handler.catch_edge.target;
    let _ = take_edge(state, &handler.catch_edge, 1, None, Some(reference))?;
    state.resume_request_for_block(target).map(Some)
}

fn value_for(
    state: &HostState,
    value: runmat_native_codegen::NativeValueId,
) -> JitResult<NativeValueRef> {
    state
        .values
        .get(&value)
        .copied()
        .ok_or_else(|| JitError::Host(format!("native value {} is unavailable", value.0)))
}

fn site_matches(instruction: &NativeInstruction, request: NativeSiteRequest) -> bool {
    instruction.site.point.block == request.block
        && instruction.site.point.position == request.position
        && phase(instruction.site.phase) == request.phase
        && instruction.site.ordinal == request.ordinal
}

fn ensure_site(terminator: &NativeTerminator, request: NativeSiteRequest) -> JitResult<()> {
    if terminator.site.point.block == request.block
        && terminator.site.point.position == request.position
        && phase(terminator.site.phase) == request.phase
        && terminator.site.ordinal == request.ordinal
    {
        Ok(())
    } else {
        Err(JitError::Host(
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
