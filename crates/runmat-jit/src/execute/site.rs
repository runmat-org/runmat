use futures::executor::block_on;
use runmat_mir::{MirPlace, MirStmtKind};
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
        let outcome = execute_terminator(state, call, &block.terminator, exit)?;
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
        execute_instruction(state, &instruction)?;
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
            let result = evaluate_rvalue(state, value)?;
            match instruction.outputs.as_slice() {
                [] => {}
                [output] => {
                    state.values.insert(output.value, result);
                }
                _ => {
                    return Err(JitError::UnsupportedSite(
                        "multiple rvalue outputs require generic call-shape execution".into(),
                    ))
                }
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
    match statement {
        MirStmtKind::Assign {
            place: MirPlace::Local(local),
            ..
        } => {
            let source = instruction
                .inputs
                .first()
                .and_then(|value| state.values.get(value))
                .copied()
                .ok_or_else(|| JitError::Host("assignment result is unavailable".into()))?;
            let slot = state
                .locals
                .get_mut(local.0)
                .ok_or_else(|| JitError::Host("assignment local is out of bounds".into()))?;
            *slot = source;
            for output in &instruction.outputs {
                state.values.insert(output.value, source);
            }
        }
        MirStmtKind::Expr(_) => {}
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
    terminator: &NativeTerminator,
    exit: &mut NativeExit,
) -> JitResult<NativeSiteOutcome> {
    match &terminator.kind {
        NativeTerminatorKind::Goto { edge } => take_edge(state, edge, 0),
        NativeTerminatorKind::Branch {
            condition,
            then_edge,
            else_edge,
        } => {
            let condition = value_for(state, *condition)?;
            let value = state.arena.get(condition)?;
            let truth = block_on(state.runtime.scope(
                runmat_runtime::condition::logical_truth_from_value(value, "branch condition"),
            ))
            .map_err(JitError::from)?;
            if truth {
                take_edge(state, then_edge, 0)
            } else {
                take_edge(state, else_edge, 1)
            }
        }
        NativeTerminatorKind::Return { values } => {
            if values.len() > call.requested_outputs as usize || values.len() > call.result_capacity
            {
                return Err(JitError::Host(
                    "native return exceeds the transactional result window".into(),
                ));
            }
            for (index, value) in values.iter().enumerate() {
                let reference = value_for(state, *value)?;
                // SAFETY: NativeCall validation guarantees a writable result
                // slice of result_capacity elements for this invocation.
                unsafe { *call.results.add(index) = reference };
            }
            *exit = NativeExit::completed(values.len() as u32);
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

fn take_edge(state: &mut HostState, edge: &NativeEdge, index: u32) -> JitResult<NativeSiteOutcome> {
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
        state.locals[parameter.local.0 as usize] = value;
    }
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
