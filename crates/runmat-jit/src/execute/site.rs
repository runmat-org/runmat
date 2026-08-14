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
        MirStmtKind::Expr(_) => state.pending_place_mutation = None,
        other => {
            return Err(JitError::UnsupportedSite(format!(
                "statement {other:?} is not in the first generic-host cohort"
            )))
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
        NativeTerminatorKind::Goto { edge } => take_edge(state, edge, 0, None),
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
                take_edge(state, then_edge, 0, None)
            } else {
                take_edge(state, else_edge, 1, None)
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
                    return take_edge(state, edge, index as u32, None);
                }
            }
            take_edge(state, otherwise, cases.len() as u32, None)
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
                let reference = state.arena.insert(value);
                take_edge(state, body, 0, Some((*binding, reference)))
            } else {
                take_edge(state, exit, 1, None)
            }
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
        other => Err(JitError::UnsupportedSite(format!(
            "terminator {other:?} is not in the first generic-host cohort"
        ))),
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
            other => {
                return Err(JitError::UnsupportedSite(format!(
                    "edge argument {other:?} requires structured continuation state"
                )))
            }
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
