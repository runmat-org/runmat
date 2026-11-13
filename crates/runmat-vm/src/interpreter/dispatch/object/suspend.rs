//! Exact opcode selection at the existing object-instruction suspension owner.
use super::{capture, read, subscript, write, ObjectDispatchContext};
use runmat_runtime::RuntimeError;

pub(in crate::interpreter::dispatch) struct Request<'a> {
    pub instruction: &'a crate::bytecode::Instr,
    pub stack: &'a mut crate::interpreter::stack::InterpreterStack,
    pub sequence_register: &'a mut super::super::SequenceState,
    pub context: ObjectDispatchContext<'a>,
}

// Select the concrete opcode before admitting its original single suspension.
// Selection reads only the opcode, never operands, roots, or user callbacks.
pub(in crate::interpreter::dispatch) fn dispatch_object<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> Result<(), RuntimeError> + 'a,
) -> Result<runmat_runtime::execution::RuntimeContinuationFuture<'a, ()>, RuntimeError> {
    use futures::FutureExt;
    let runtime = request.context.runtime;
    match request.instruction {
        crate::bytecode::Instr::LoadMember(..) => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                read::load_member(request).map(map)
            })
        }
        crate::bytecode::Instr::LoadMemberOrInit(..) => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                read::load_member_or_init(request).map(map)
            })
        }
        crate::bytecode::Instr::LoadMemberDynamic => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                read::load_member_dynamic(request).map(map)
            })
        }
        crate::bytecode::Instr::LoadMemberDynamicOrInit => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                read::load_member_dynamic_or_init(request).map(map)
            })
        }
        crate::bytecode::Instr::LoadMemberSequence { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                read::load_member_sequence(request).map(map)
            })
        }
        crate::bytecode::Instr::LoadMemberDynamicSequence { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                read::load_member_dynamic_sequence(request).map(map)
            })
        }
        crate::bytecode::Instr::MemberSequenceCardinality => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                read::member_sequence_cardinality(request).map(map)
            })
        }
        crate::bytecode::Instr::LoadMemberSequenceUsingOutputSlot { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                read::load_member_sequence_using_output_slot(request).map(map)
            })
        }
        crate::bytecode::Instr::LoadMemberDynamicSequenceUsingOutputSlot { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                read::load_member_dynamic_sequence_using_output_slot(request).map(map)
            })
        }
        crate::bytecode::Instr::StoreMember(..) => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                write::store_member(request).map(map)
            })
        }
        crate::bytecode::Instr::StoreMemberOrInit(..) => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                write::store_member_or_init(request).map(map)
            })
        }
        crate::bytecode::Instr::StoreMemberDynamic => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                write::store_member_dynamic(request).map(map)
            })
        }
        crate::bytecode::Instr::StoreMemberDynamicOrInit => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                write::store_member_dynamic_or_init(request).map(map)
            })
        }
        crate::bytecode::Instr::StoreMemberSequence(..) => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                write::store_member_sequence(request).map(map)
            })
        }
        crate::bytecode::Instr::StoreMemberDynamicSequence => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                write::store_member_dynamic_sequence(request).map(map)
            })
        }
        crate::bytecode::Instr::MoveCapturedSequence { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                capture::move_captured_sequence(request).map(map)
            })
        }
        crate::bytecode::Instr::CaptureCallOutputSequence => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                capture::capture_call_output_sequence(request).map(map)
            })
        }
        crate::bytecode::Instr::CaptureScalarSequence => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                capture::capture_scalar_sequence(request).map(map)
            })
        }
        crate::bytecode::Instr::CaptureMemberSequence { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                capture::capture_member_sequence(request).map(map)
            })
        }
        crate::bytecode::Instr::CaptureMemberDynamicSequence { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                capture::capture_member_dynamic_sequence(request).map(map)
            })
        }
        crate::bytecode::Instr::CaptureCellContentsSequence { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                capture::capture_cell_contents_sequence(request).map(map)
            })
        }
        crate::bytecode::Instr::CaptureReturnedOutputsSequence { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                capture::capture_returned_outputs_sequence(request).map(map)
            })
        }
        crate::bytecode::Instr::BeginSubscriptEndReceiver { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                subscript::begin_subscript_end_receiver(request).map(map)
            })
        }
        crate::bytecode::Instr::LoadSubscriptEnd { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                subscript::load_subscript_end(request).map(map)
            })
        }
        crate::bytecode::Instr::FinishSubscriptEndReceiver { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                subscript::finish_subscript_end_receiver(request).map(map)
            })
        }
        crate::bytecode::Instr::FinishDottedInvokeReceiver { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                subscript::finish_dotted_invoke_receiver(request).map(map)
            })
        }
        crate::bytecode::Instr::ReadSubscriptPath { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                subscript::read_subscript_path(request).map(map)
            })
        }
        crate::bytecode::Instr::CaptureSubscriptPath { .. } => {
            runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
                subscript::capture_subscript_path(request).map(map)
            })
        }
        _ => runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
            futures::future::ready(Ok(false)).map(map)
        }),
    }
}
