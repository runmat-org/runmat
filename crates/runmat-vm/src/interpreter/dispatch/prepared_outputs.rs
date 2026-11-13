use super::SequenceState;
use crate::interpreter::vars::InterpreterVars;
use futures::FutureExt;
use runmat_runtime::RuntimeError;

type PreparedResult = Result<(), RuntimeError>;
type PreparedDispatchFuture<'a> =
    Result<runmat_runtime::execution::RuntimeContinuationFuture<'a, ()>, RuntimeError>;

// The selected concrete handler is admitted before construction or polling.
// Runtime owns its original single Box and keeps the receipt outside it so
// physical deallocation precedes refund on completion, error, or abandonment.
#[inline(never)]
fn admit<'a, F>(
    runtime: &runmat_runtime::context::RuntimeContext,
    make_future: impl FnOnce() -> F,
) -> PreparedDispatchFuture<'a>
where
    F: std::future::Future<Output = PreparedResult> + 'a,
{
    runmat_runtime::execution::admit_domain_continuation_with(runtime, make_future)
}

mod commit;
mod control;
mod endpoint;
mod legacy;
mod path;
mod publication;
mod support;

pub(super) struct PreparedOutputContext<'a> {
    pub bytecode: &'a crate::bytecode::Bytecode,
    pub authority: super::slot_authority::AssignmentSlotAuthority<'a>,
}

pub(super) struct Request<'a> {
    pub instruction: &'a crate::bytecode::Instr,
    pub stack: &'a mut crate::interpreter::stack::InterpreterStack,
    pub vars: &'a mut InterpreterVars,
    pub state: &'a mut SequenceState,
    pub context: PreparedOutputContext<'a>,
    pub before_overwrite: &'a mut dyn FnMut(usize),
    pub after_store: &'a mut dyn FnMut(usize),
}

// Select the concrete handler before admitting its suspended-instruction Box.
// No operands, hooks, roots, or destination state are inspected until polling.
pub(super) fn dispatch<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    use crate::bytecode::Instr;
    match request.instruction {
        Instr::BeginOutputAssignment { .. } => control::begin_output_assignment(request, map),
        Instr::BeginFixedAssignment { .. } => control::begin_fixed_assignment(request, map),
        Instr::PrepareFixedOutputTarget => control::prepare_fixed_output_target(request, map),
        Instr::PrepareDiscardOutputTarget => control::prepare_discard_output_target(request, map),
        Instr::BeginSequenceOutputTarget { .. } => {
            control::begin_sequence_output_target(request, map)
        }
        Instr::PrepareMemberPathStep { .. } => path::prepare_member_path_step(request, map),
        Instr::PrepareDynamicMemberPathStep { .. } => {
            path::prepare_dynamic_member_path_step(request, map)
        }
        Instr::BeginPreparedIndexSelectors { .. } => {
            control::begin_prepared_index_selectors(request, map)
        }
        Instr::LoadPreparedIndexEnd { .. } => control::load_prepared_index_end(request, map),
        Instr::BeginContextualIndexSelectors { .. } => {
            control::begin_contextual_index_selectors(request, map)
        }
        Instr::LoadContextualIndexEnd { .. } => control::load_contextual_index_end(request, map),
        Instr::FinishContextualIndexSelectors { .. } => {
            control::finish_contextual_index_selectors(request, map)
        }
        Instr::PrepareParenthesesPathStep { .. } => {
            path::prepare_parentheses_path_step(request, map)
        }
        Instr::PrepareBracesPathStep { .. } => path::prepare_braces_path_step(request, map),
        Instr::FinishMemberSequenceOutputTarget(..) => {
            endpoint::finish_member_sequence_output_target(request, map)
        }
        Instr::FinishFixedAssignmentTarget { .. } => {
            endpoint::finish_fixed_assignment_target(request, map)
        }
        Instr::FinishDynamicMemberSequenceOutputTarget => {
            endpoint::finish_dynamic_member_sequence_output_target(request, map)
        }
        Instr::FinishCellContentsSequenceOutputTarget { .. } => {
            endpoint::finish_cell_contents_sequence_output_target(request, map)
        }
        Instr::LoadPreparedOutputCardinality => {
            control::load_prepared_output_cardinality(request, map)
        }
        Instr::CommitPreparedOutputTargets { .. } => {
            publication::commit_prepared_output_targets(request, map)
        }
        Instr::CommitFixedAssignment { .. } => publication::commit_fixed_assignment(request, map),
        Instr::PrepareMemberSequenceOutputTarget { .. }
        | Instr::PrepareMemberDynamicSequenceOutputTarget { .. }
        | Instr::PrepareIndexedMemberSequenceOutputTarget { .. }
        | Instr::PrepareIndexedMemberDynamicSequenceOutputTarget { .. }
        | Instr::PrepareCellContentsSequenceOutputTarget { .. } => suspend_legacy(request, map),
        _ => {
            let context = request.context.authority.context;
            admit(&context.runtime, move || async { Ok(false) }.map(map))
        }
    }
}

#[inline(never)]
fn suspend_legacy<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        legacy::dispatch(
            request.instruction,
            request.stack,
            request.vars,
            request.state,
            &context.runtime,
            request.context.bytecode,
            Some(request.context.authority.current_function_name),
        )
        .map(map)
    })
}
