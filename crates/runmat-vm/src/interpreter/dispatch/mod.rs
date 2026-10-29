mod arithmetic;
mod arrays;
mod call_output_sequence;
mod calls;
mod control_flow;
mod exceptions;
mod future;
mod indexing;
mod lexical;
mod object;
mod parallel;
mod prepared_outputs;
mod route;
mod sequence_captures;
mod slot_authority;
pub(crate) use parallel::encode_spmd_interpreter_output;
mod stack;

use crate::bytecode::Instr;
use crate::interpreter::debug;
use crate::interpreter::persistent::{self, PersistentAlias};
use crate::interpreter::vars::InterpreterVars;
use crate::runtime::workspace::{workspace_slot_assigned, workspace_slot_name};
use runmat_runtime::RuntimeError;
use runmat_value::{Tensor, Value};
use std::collections::{HashMap, HashSet};
use std::future::Future;

pub(crate) use arrays::MatrixConstructionAtSource;
pub use arrays::{create_matrix, create_range, pack_to_col, pack_to_row, unpack};
pub(crate) use calls::checked_class_registration;
pub use calls::{
    build_builtin_expand_multi_args, build_feval_expand_multi_args,
    build_user_function_expand_multi_args, handle_builtin_call_multi, handle_create_closure,
    handle_create_semantic_closure, handle_load_method, handle_load_static_property,
    handle_method_or_member_index_expand_multi_call, handle_method_or_member_index_multi_call,
    handle_prepared_user_function_call, handle_register_class, handle_user_function_call,
    handle_workspace_first_prepared_call, BuiltinHandling, UserCallHandling,
};
pub use control_flow::{apply_control_flow_action, DispatchDecision};
pub use exceptions::{redirect_exception_to_catch, ExceptionHandling};
pub use runmat_runtime::condition::logical_truth_from_value;
pub use stack::{
    emit_stack_top, emit_var, load_bool, load_char_row, load_complex, load_const, load_int,
    load_local, load_string, load_var, store_local, store_var,
};
pub(crate) use stack::{publish_var_value, BindingPublicationFrame};

use runmat_runtime::builtin::builtin_constant_value;

pub enum DispatchHandled {
    Generic(DispatchDecision),
    ReturnValue(DispatchDecision),
    Return(DispatchDecision),
}

pub struct DispatchMeta<'a> {
    pub instr: &'a Instr,
    pub bytecode: &'a crate::bytecode::Bytecode,
    #[cfg(feature = "native-accel")]
    pub private_temp_load: Option<crate::bytecode::fusion_private_temp::FusionPrivateTempBridge>,
    pub var_names: &'a HashMap<usize, String>,
    pub function_registry: &'a crate::bytecode::FunctionRegistry,
    pub source_id: Option<runmat_hir::SourceId>,
    pub call_arg_spans: Option<runmat_runtime::callsite::CallsiteSpans>,
    pub invocation_arity: runmat_runtime::context::RuntimeInvocationArity,
    pub current_function_name: &'a str,
}

pub struct DispatchState<'a> {
    pub stack: &'a mut crate::interpreter::stack::InterpreterStack,
    pub vars: &'a mut InterpreterVars,
    pub sequence_register: &'a mut SequenceState,
    pub context: &'a mut crate::bytecode::program::ExecutionContext,
    pub try_stack: &'a mut Vec<crate::interpreter::state::ActiveTryHandler>,
    pub last_exception: &'a mut Option<runmat_value::MException>,
    pub imports: &'a mut Vec<(Vec<String>, bool)>,
    pub global_aliases: &'a runmat_runtime::workspace::bindings::FrameGlobalBindings,
    pub persistent_aliases: &'a mut crate::interpreter::persistent::PersistentAliases,
    pub missing_input_slots: &'a mut HashSet<usize>,
    pub pc: &'a mut usize,
}

/// A verified transient window in the operand stack.
///
/// Sequence-producing bytecode must be followed immediately by its consuming
/// store. Keeping only the stack window here makes its values ordinary VM GC
/// roots while preventing a comma-separated sequence from becoming a language
/// `Value` that can escape into a variable or workspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceRegister {
    pub start: usize,
    pub len: usize,
}

pub struct SequenceState {
    pub assignment: Option<SequenceRegister>,
    call_outputs: Option<SequenceRegister>,
    captures: crate::runtime::workspace::FrameMap<usize, SequenceRegister>,
    capture_gate: runmat_runtime::context::host_allocation::FrameCollectionGate,
    capture_context: Option<runmat_runtime::context::RuntimeContext>,
    output_targets: Vec<PreparedOutputTarget>,
    fixed_target: Option<PreparedOutputTarget>,
    expected_output_targets: Option<usize>,
    fixed_assignment_owner: Option<(runmat_types::StmtId, usize)>,
    destination_builder: Option<PendingSequenceDestination>,
    prepared_index_component_count: Option<usize>,
    prepared_index_receiver: Option<runmat_runtime::object::protocol::PreparedSubscriptReceiver>,
    contextual_indices:
        crate::runtime::workspace::FrameVec<(usize, crate::runtime::workspace::FrameVec<usize>)>,
    cleanup_windows: crate::runtime::workspace::FrameVec<SequenceRegister>,
    subscript_receivers: crate::runtime::workspace::FrameVec<(
        runmat_runtime::object::protocol::PreparedSubscriptReceiver,
        usize,
        usize,
    )>,
}

struct PendingSequenceDestination {
    root_slot: usize,
    builder: runmat_runtime::sequence::SequenceDestinationBuilder,
    absent_root: Option<Value>,
    root_creation: runmat_mir::SequenceRootCreation,
    prepared_root: Option<Value>,
}

#[expect(
    clippy::large_enum_variant,
    reason = "Keep prepared destination and rooted receivers inline without a per-assignment allocation"
)]
pub(super) enum PreparedOutputTarget {
    Fixed,
    Discard,
    Sequence {
        root_slot: usize,
        endpoint: runmat_runtime::sequence::PreparedSequenceDestination,
        absent_root: Option<Value>,
        root_creation: runmat_mir::SequenceRootCreation,
        prepared_root: Option<Value>,
    },
}

impl SequenceState {
    /// Non-stack roots retained while an indexing destination spans calls.
    pub(crate) fn visit_retained_values(&self, visit: &mut impl FnMut(&Value)) {
        for target in self.output_targets.iter().chain(self.fixed_target.iter()) {
            match target {
                PreparedOutputTarget::Sequence {
                    endpoint,
                    absent_root,
                    prepared_root,
                    ..
                } => {
                    endpoint.visit_retained_values(visit);
                    if let Some(root) = absent_root {
                        visit(root);
                    }
                    if let Some(root) = prepared_root {
                        visit(root);
                    }
                }
                PreparedOutputTarget::Fixed | PreparedOutputTarget::Discard => {}
            }
        }
        if let Some(pending) = &self.destination_builder {
            pending.builder.visit_retained_values(visit);
            if let Some(root) = &pending.prepared_root {
                visit(root);
            }
            if let Some(root) = &pending.absent_root {
                visit(root);
            }
        }
        for (receiver, _, _) in &self.subscript_receivers {
            visit(receiver.receiver());
        }
        if let Some(receiver) = &self.prepared_index_receiver {
            visit(receiver.receiver());
        }
    }

    pub(super) fn begin_output_assignment(
        &mut self,
        target_count: usize,
    ) -> Result<(), RuntimeError> {
        if self.expected_output_targets.is_some()
            || !self.output_targets.is_empty()
            || self.fixed_target.is_some()
        {
            return Err(crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "output-assignment destination state was overwritten before consumption",
            ));
        }
        self.output_targets.clear();
        self.destination_builder = None;
        self.prepared_index_component_count = None;
        self.expected_output_targets = Some(target_count);
        Ok(())
    }

    pub(super) fn push_output_target(
        &mut self,
        target: PreparedOutputTarget,
    ) -> Result<(), RuntimeError> {
        if self.fixed_assignment_owner.is_some() {
            if self.fixed_target.is_some() {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:PreparedAssignmentOwner",
                    "fixed assignment has multiple destinations",
                ));
            }
            self.fixed_target = Some(target);
        } else {
            self.output_targets.push(target);
        }
        Ok(())
    }

    pub(super) fn begin_destination(
        &mut self,
        context: &runmat_runtime::context::RuntimeContext,
        root_slot: usize,
        root: &Value,
        absent_root: Option<Value>,
        root_creation: runmat_mir::SequenceRootCreation,
        bytecode: &crate::bytecode::Bytecode,
    ) -> Result<(), RuntimeError> {
        if self.destination_builder.is_some() {
            return Err(crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "a prepared destination path is already live",
            ));
        }
        let builder =
            runmat_runtime::sequence::SequenceDestinationBuilder::new_in_context(context, root)?;
        let builder = if bytecode.active_class_method_owner.is_some() {
            builder.with_method_frame_factory_in_context(context, || {
                object::frame_access(bytecode, context)
            })?
        } else {
            builder
        };
        let builder = match root_creation {
            runmat_mir::SequenceRootCreation::Ordinary(creation) => {
                builder.with_assignment_creation(creation)
            }
            runmat_mir::SequenceRootCreation::OrdinaryContainer(_) => builder
                .with_assignment_creation(
                    runmat_types::AssignmentCreationPolicy::CreateStructFieldPath,
                ),
            runmat_mir::SequenceRootCreation::OrdinaryCell => builder.with_assignment_creation(
                runmat_types::AssignmentCreationPolicy::CreateArrayByIndex,
            ),
            _ => builder,
        };
        self.destination_builder = Some(PendingSequenceDestination {
            root_slot,
            builder,
            absent_root,
            root_creation,
            prepared_root: root_creation
                .retains_original_receiver()
                .then(|| root.clone()),
        });
        Ok(())
    }

    pub(super) fn destination_builder_mut(
        &mut self,
    ) -> Result<&mut runmat_runtime::sequence::SequenceDestinationBuilder, RuntimeError> {
        self.destination_builder
            .as_mut()
            .map(|pending| &mut pending.builder)
            .ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "RunMat:CommaSeparatedListState",
                    "no prepared destination path is live",
                )
            })
    }

    pub(super) fn begin_index_selectors(&mut self, count: usize) -> Result<(), RuntimeError> {
        if self.prepared_index_component_count.replace(count).is_some() {
            return Err(crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "prepared index-selector context was overwritten",
            ));
        }
        Ok(())
    }

    pub(super) fn finish_index_selectors(&mut self, count: usize) -> Result<(), RuntimeError> {
        if self.prepared_index_component_count.take() != Some(count) {
            return Err(crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "prepared index-selector context is incomplete or inconsistent",
            ));
        }
        self.prepared_index_receiver = None;
        Ok(())
    }

    pub(super) fn index_selector_count(&self) -> Result<usize, RuntimeError> {
        self.prepared_index_component_count.ok_or_else(|| {
            crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "no prepared index-selector context is live",
            )
        })
    }

    pub(super) fn begin_contextual_index(
        &mut self,
        component_count: usize,
        shape: &[usize],
    ) -> Result<(), RuntimeError> {
        let mut retained_shape = self.frame_vector(shape.len())?;
        retained_shape.extend_from_slice(shape);
        if self.contextual_indices.len() == self.contextual_indices.capacity() {
            let capacity = self
                .contextual_indices
                .len()
                .checked_add(1)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "RunMat:ContextualIndexState",
                        "context depth overflows",
                    )
                })?;
            let mut replacement = self.frame_vector(capacity)?;
            replacement.extend(self.contextual_indices.drain(..));
            self.contextual_indices = replacement;
        }
        self.contextual_indices
            .push((component_count, retained_shape));
        Ok(())
    }

    pub(super) fn contextual_index_extent(&self, component: usize) -> Result<usize, RuntimeError> {
        let (count, shape) = self.contextual_indices.last().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "RunMat:ContextualIndexState",
                "contextual end was evaluated without an index context",
            )
        })?;
        runmat_runtime::indexing::shape::selector_extent(shape, *count, component)
    }

    pub(super) fn finish_contextual_index(
        &mut self,
        component_count: usize,
    ) -> Result<(), RuntimeError> {
        let Some((count, _)) = self.contextual_indices.pop() else {
            return Err(crate::interpreter::errors::mex(
                "RunMat:ContextualIndexState",
                "contextual index state is not live",
            ));
        };
        if count != component_count {
            return Err(crate::interpreter::errors::mex(
                "RunMat:ContextualIndexState",
                "contextual index component count changed during evaluation",
            ));
        }
        Ok(())
    }

    pub(super) fn push_subscript_receiver(
        &mut self,
        receiver: runmat_runtime::object::protocol::PreparedSubscriptReceiver,
        prefix_start: usize,
        stack_index: usize,
    ) -> Result<(), RuntimeError> {
        self.prepare_subscript_receiver()?;
        self.subscript_receivers
            .push((receiver, prefix_start, stack_index));
        Ok(())
    }

    pub(super) fn subscript_receiver(
        &self,
    ) -> Result<&runmat_runtime::object::protocol::PreparedSubscriptReceiver, RuntimeError> {
        self.subscript_receivers
            .last()
            .map(|entry| &entry.0)
            .ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "RunMat:SubscriptPathState",
                    "contextual subscript end has no prepared receiver",
                )
            })
    }

    pub(super) fn pop_subscript_receiver(
        &mut self,
    ) -> Result<
        (
            runmat_runtime::object::protocol::PreparedSubscriptReceiver,
            usize,
            usize,
        ),
        RuntimeError,
    > {
        self.subscript_receivers.pop().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "RunMat:SubscriptPathState",
                "contextual subscript receiver was already consumed",
            )
        })
    }

    pub(super) fn finish_dotted_invoke_receiver(
        &mut self,
        stack: &mut crate::interpreter::stack::InterpreterStack,
        prefix_operand_count: usize,
        argument_operand_count: usize,
    ) -> Result<(), RuntimeError> {
        let (prepared, prefix_start, receiver_index) =
            self.subscript_receivers.last().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "RunMat:SubscriptPathState",
                    "dotted invocation has no prepared receiver",
                )
            })?;
        let (prepared, prefix_start, receiver_index) =
            (prepared.receiver().clone(), *prefix_start, *receiver_index);
        let expected = prefix_start
            .checked_add(prefix_operand_count)
            .and_then(|index| index.checked_add(1));
        let captured_operand_count = self
            .captures
            .values()
            .filter(|window| window.start > receiver_index)
            .try_fold(0usize, |total, window| total.checked_add(window.len));
        if expected != Some(receiver_index)
            || stack.len().checked_sub(receiver_index.saturating_add(1))
                != captured_operand_count
                    .and_then(|count| count.checked_add(argument_operand_count))
        {
            return Err(crate::interpreter::errors::mex(
                "RunMat:SubscriptPathState",
                "dotted invocation receiver or arguments moved during preparation",
            ));
        }
        let removed = receiver_index - prefix_start;
        for window in self
            .captures
            .values()
            .chain(self.assignment.iter())
            .chain(self.call_outputs.iter())
        {
            if window.start >= prefix_start && window.start <= receiver_index {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:CommaSeparatedListState",
                    "sequence window overlaps its prepared receiver",
                ));
            }
        }
        self.subscript_receivers.pop();
        for window in self.captures.values_mut() {
            if window.start > receiver_index {
                window.start -= removed;
            }
        }
        for window in [&mut self.assignment, &mut self.call_outputs] {
            if let Some(window) = window.as_mut() {
                if window.start > receiver_index {
                    window.start -= removed;
                }
            }
        }
        stack.drain(prefix_start..=receiver_index);
        stack.try_insert(prefix_start, prepared)?;
        Ok(())
    }

    fn take_destination_builder(&mut self) -> Result<PendingSequenceDestination, RuntimeError> {
        if self.prepared_index_component_count.is_some() {
            return Err(crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "prepared index-selector context remains live",
            ));
        }
        self.destination_builder.take().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "no prepared destination path is live",
            )
        })
    }

    pub(super) fn output_layout(
        &self,
    ) -> Result<runmat_runtime::sequence::DestinationLayout, RuntimeError> {
        if self.expected_output_targets != Some(self.output_targets.len()) {
            return Err(crate::interpreter::errors::mex(
                "RunMat:CommaSeparatedListState",
                "output-assignment destination plan is incomplete",
            ));
        }
        runmat_runtime::sequence::DestinationLayout::new(self.output_targets.iter().map(|target| {
            match target {
                PreparedOutputTarget::Fixed => {
                    runmat_runtime::sequence::DestinationCardinality::Fixed
                }
                PreparedOutputTarget::Discard => {
                    runmat_runtime::sequence::DestinationCardinality::Discard
                }
                PreparedOutputTarget::Sequence { endpoint, .. } => {
                    runmat_runtime::sequence::DestinationCardinality::Sequence(
                        endpoint.cardinality(),
                    )
                }
            }
        }))
    }

    pub(super) fn output_cardinality(&self) -> Result<usize, RuntimeError> {
        if self.fixed_assignment_owner.is_some() {
            if self.expected_output_targets == Some(1) && self.fixed_target.is_some() {
                return Ok(1);
            }
            return Err(crate::interpreter::errors::mex(
                "RunMat:PreparedAssignmentOwner",
                "fixed destination is incomplete",
            ));
        }
        Ok(self.output_layout()?.total())
    }

    pub(super) fn take_output_targets(&mut self) -> Vec<PreparedOutputTarget> {
        self.expected_output_targets = None;
        self.fixed_assignment_owner = None;
        std::mem::take(&mut self.output_targets)
    }

    pub fn take_captures(
        &mut self,
        stack: &mut crate::interpreter::stack::InterpreterStack,
        slots: &[usize],
    ) -> Result<
        crate::runtime::workspace::FrameMap<usize, runmat_value::OutputListStorage>,
        RuntimeError,
    > {
        let mut windows = self.frame_vector(slots.len())?;
        for slot in slots {
            if windows.iter().any(|(seen, _)| seen == slot) {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:CommaSeparatedListState",
                    "call references the same captured sequence more than once",
                ));
            }
            let window = self.captures.get(slot).copied().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "RunMat:CommaSeparatedListState",
                    "call references an unavailable captured sequence",
                )
            })?;
            let end = window.start.checked_add(window.len).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "RunMat:CommaSeparatedListState",
                    "captured sequence bounds overflowed",
                )
            })?;
            if end > stack.len() {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:CommaSeparatedListState",
                    "captured sequence is outside the operand stack",
                ));
            }
            windows.push((*slot, window));
        }
        windows.sort_unstable_by_key(|(_, window)| std::cmp::Reverse(window.start));
        for pair in windows.windows(2) {
            let later = pair[0].1;
            let earlier = pair[1].1;
            if earlier.start + earlier.len > later.start {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:CommaSeparatedListState",
                    "captured sequence stack windows overlap",
                ));
            }
        }
        let mut values = self.frame_map(slots.len())?;
        for (slot, window) in windows {
            let end = window.start + window.len;
            let output = stack.try_extract_output_range(window.start, end)?;
            self.captures.remove(&slot);
            values.insert(slot, output);
            for remaining in self.captures.values_mut() {
                if remaining.start >= end {
                    remaining.start -= window.len;
                }
            }
            if let Some(assignment) = self.assignment.as_mut() {
                if assignment.start >= end {
                    assignment.start -= window.len;
                }
            }
            if let Some(call) = self.call_outputs.as_mut() {
                if call.start >= end {
                    call.start -= window.len;
                }
            }
        }
        Ok(values)
    }

    pub fn clear(&mut self, stack: &mut crate::interpreter::stack::InterpreterStack) {
        // Every capture/receiver reserved its cleanup slot before publication.
        // Cancellation and catch retirement must not request another grant.
        self.cleanup_windows.clear();
        self.cleanup_windows.extend(self.captures.values().copied());
        self.cleanup_windows
            .extend(
                self.subscript_receivers
                    .iter()
                    .map(|(_, _, start)| SequenceRegister {
                        start: *start,
                        len: 1,
                    }),
            );
        self.cleanup_windows
            .sort_unstable_by_key(|window| window.start);
        let mut inline = [self.assignment.take(), self.call_outputs.take()];
        loop {
            let mut choice = None;
            for (index, window) in inline.iter().enumerate() {
                if let Some(window) = window {
                    if choice.is_none_or(|(_, start)| window.start >= start) {
                        choice = Some((index, window.start));
                    }
                }
            }
            let window = match (self.cleanup_windows.last(), choice) {
                (Some(window), Some((index, start))) if start >= window.start => {
                    inline[index].take()
                }
                (Some(_), _) => self.cleanup_windows.pop(),
                (None, Some((index, _))) => inline[index].take(),
                (None, None) => break,
            };
            let Some(window) = window else { break };
            if let Some(end) = window.start.checked_add(window.len) {
                if end <= stack.len() {
                    stack.drain(window.start..end);
                }
            }
        }
        self.captures.clear();
        self.fixed_assignment_owner = None;
        self.output_targets.clear();
        self.fixed_target = None;
        self.expected_output_targets = None;
        self.destination_builder = None;
        self.prepared_index_component_count = None;
        self.contextual_indices.clear();
        self.prepared_index_receiver = None;
        self.subscript_receivers.clear();
    }
}

pub struct DispatchHooks<'a> {
    pub store_var_before_overwrite: &'a mut dyn FnMut(usize),
    pub store_var_after_store: &'a mut dyn FnMut(usize),
    pub store_local_before_local_overwrite: &'a mut dyn FnMut(&Value, &Value),
    pub store_local_before_var_overwrite: &'a mut dyn FnMut(usize),
    pub store_local_after_store: &'a mut dyn FnMut(usize, &Value),
    pub store_local_after_fallback_store: &'a mut dyn FnMut(&str, usize),
}

fn materialize_aggregate_rows(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    sequence_state: &mut SequenceState,
    row_lengths: &[usize],
    elements: &[crate::bytecode::AggregateElementSpec],
) -> Result<Vec<Vec<Value>>, RuntimeError> {
    let expected_elements = row_lengths
        .iter()
        .try_fold(0usize, |total, length| total.checked_add(*length));
    if expected_elements != Some(elements.len()) {
        return Err(crate::interpreter::errors::mex(
            "AggregateShapeMismatch",
            "aggregate row boundaries do not match the element plan",
        ));
    }
    let capture_slots = elements
        .iter()
        .filter_map(|element| match element {
            crate::bytecode::AggregateElementSpec::CapturedSequence { slot } => Some(*slot),
            crate::bytecode::AggregateElementSpec::Single => None,
        })
        .collect::<Vec<_>>();
    let mut captures = sequence_state.take_captures(stack, &capture_slots)?;
    let single_count = elements
        .iter()
        .filter(|element| matches!(element, crate::bytecode::AggregateElementSpec::Single))
        .count();
    let split = stack.len().checked_sub(single_count).ok_or_else(|| {
        crate::interpreter::errors::mex("StackUnderflow", "aggregate operand stack underflow")
    })?;
    let mut singles = stack
        .drain(split..)
        .collect::<std::collections::VecDeque<_>>();
    let mut rows = Vec::with_capacity(row_lengths.len());
    let mut element_offset = 0usize;
    for &row_length in row_lengths {
        let row_end = element_offset.checked_add(row_length).ok_or_else(|| {
            crate::interpreter::errors::mex(
                "AggregateShapeMismatch",
                "aggregate row boundary overflow",
            )
        })?;
        let mut row = Vec::new();
        for element in &elements[element_offset..row_end] {
            match element {
                crate::bytecode::AggregateElementSpec::Single => {
                    row.push(singles.pop_front().ok_or_else(|| {
                        crate::interpreter::errors::mex(
                            "StackUnderflow",
                            "aggregate operand stack underflow",
                        )
                    })?);
                }
                crate::bytecode::AggregateElementSpec::CapturedSequence { slot } => {
                    row.extend(
                        captures
                            .remove(slot)
                            .ok_or_else(|| {
                                crate::interpreter::errors::mex(
                                    "CommaSeparatedListState",
                                    "aggregate sequence was consumed more than once",
                                )
                            })?
                            .try_into_owned()
                            .map_err(|_| {
                                crate::interpreter::errors::mex(
                                    "RunMat:CommaSeparatedListState",
                                    "captured aggregate output unexpectedly shares storage",
                                )
                            })?,
                    );
                }
            }
        }
        rows.push(row);
        element_offset = row_end;
    }
    Ok(rows)
}

pub(super) fn requested_outputs_from_slot(
    vars: &InterpreterVars,
    slot: usize,
) -> Result<usize, RuntimeError> {
    let value = vars.get(slot).ok_or_else(|| {
        crate::interpreter::errors::mex(
            "OutputCountSlotOutOfBounds",
            "requested output slot is out of bounds",
        )
    })?;
    match &*value {
        Value::Num(n) => {
            if !n.is_finite() || *n < 0.0 || (*n - n.round()).abs() > f64::EPSILON {
                return Err(crate::interpreter::errors::mex(
                    "InvalidOutputCountValue",
                    "requested output count slot must contain a nonnegative integer scalar",
                ));
            }
            Ok(*n as usize)
        }
        Value::Int(i) => i.try_to_usize().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "InvalidOutputCountValue",
                "requested output count slot must contain a nonnegative integer scalar",
            )
        }),
        _ => Err(crate::interpreter::errors::mex(
            "InvalidOutputCountValue",
            "requested output count slot must contain a nonnegative integer scalar",
        )),
    }
}

pub fn dispatch_instruction<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
    hooks: DispatchHooks<'a>,
) -> impl Future<Output = Result<Option<DispatchHandled>, RuntimeError>> + 'a {
    future::dispatch_instruction(meta, state, hooks)
}
fn dispatch_immediate(
    meta: DispatchMeta<'_>,
    state: DispatchState<'_>,
    hooks: DispatchHooks<'_>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    let DispatchMeta {
        instr,
        var_names,
        function_registry,
        current_function_name,
        #[cfg(feature = "native-accel")]
        private_temp_load,
        ..
    } = meta;
    let DispatchState {
        stack,
        vars,
        sequence_register,
        context,
        try_stack,
        imports,
        global_aliases,
        persistent_aliases,
        missing_input_slots,
        pc,
        ..
    } = state;
    let DispatchHooks {
        store_var_before_overwrite,
        store_var_after_store,
        store_local_before_local_overwrite,
        store_local_before_var_overwrite,
        store_local_after_store,
        store_local_after_fallback_store,
    } = hooks;
    match instr {
        Instr::LoadConst(value) => {
            load_const(stack, *value)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadInt(value) => {
            load_int(stack, value.clone())?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadComplex(re, im) => {
            load_complex(stack, *re, *im)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadBool(value) => {
            load_bool(stack, *value)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadString(value) => {
            load_string(stack, value.clone())?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadCharRow(value) => {
            load_char_row(stack, value.clone())?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadLocal(offset) => {
            if context.call_stack.last().is_none() {
                if !lexical::refresh(context, vars, *offset)? {
                    let mut error = crate::interpreter::errors::mex(
                        "UndefinedVariable",
                        &format!(
                            "Undefined variable: {}",
                            var_names
                                .get(offset)
                                .map(String::as_str)
                                .unwrap_or("<lexical binding>")
                        ),
                    );
                    error.failure_origin =
                        runmat_runtime::runtime_error::RuntimeFailureOrigin::LanguageEvaluation;
                    return Err(error);
                }
                persistent::refresh(
                    persistent_aliases,
                    vars,
                    &context.runtime.session_variable_authority(),
                    current_function_name,
                    *offset,
                )?;
            }
            load_local(stack, context, vars, *offset)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadVar(index) => {
            if let Some(global_value) =
                crate::runtime::globals::current_bound_value(global_aliases, *index)
            {
                if *index >= vars.len() {
                    vars.resize_owned(*index + 1, Value::Num(0.0));
                }
                let _retired = vars.replace_owned(*index, global_value);
                crate::runtime::workspace::refresh_workspace_slot_from_vars(*index, vars);
            }
            persistent::refresh(
                persistent_aliases,
                vars,
                &context.runtime.session_variable_authority(),
                current_function_name,
                *index,
            )?;
            if !lexical::refresh(context, vars, *index)? {
                let mut error = crate::interpreter::errors::mex(
                    "UndefinedVariable",
                    &format!(
                        "Undefined variable: {}",
                        var_names
                            .get(index)
                            .map(String::as_str)
                            .unwrap_or("<lexical binding>")
                    ),
                );
                error.failure_origin =
                    runmat_runtime::runtime_error::RuntimeFailureOrigin::LanguageEvaluation;
                return Err(error);
            }
            if missing_input_slots.contains(index) {
                return Err(crate::interpreter::errors::mex(
                    "NotEnoughInputs",
                    "Not enough input arguments.",
                ));
            }
            if let (Some(false), Some(slot_name), Some(var_name)) = (
                workspace_slot_assigned(*index),
                workspace_slot_name(*index),
                var_names.get(index),
            ) {
                if slot_name == *var_name {
                    if let Some(value) = builtin_constant_value(var_name) {
                        debug::trace_load_var(*pc, *index, &value);
                        stack.try_push(value)?;
                        return Ok(Some(DispatchHandled::Generic(
                            DispatchDecision::FallThrough,
                        )));
                    }
                    let mut error = crate::interpreter::errors::mex(
                        "UndefinedVariable",
                        &format!("Undefined variable: {slot_name}"),
                    );
                    error.failure_origin =
                        runmat_runtime::runtime_error::RuntimeFailureOrigin::LanguageEvaluation;
                    return Err(error);
                }
            }
            #[cfg(feature = "native-accel")]
            let move_private = private_temp_load.is_some_and(|bridge| {
                bridge.slot == *index && bridge.load_pc == *pc && bridge.consumer_pc > *pc
            });
            #[cfg(not(feature = "native-accel"))]
            let move_private = false;
            if move_private {
                debug::trace_load_var(
                    *pc,
                    *index,
                    &vars
                        .get(*index)
                        .expect("loaded private variable slot lies within frame extent"),
                );
                stack.try_push_with(|| {
                    vars.take_private_owned(*index).ok_or_else(|| {
                        runmat_runtime::build_runtime_error(
                            "compiler-private fusion temporary is not an owned frame value",
                        )
                        .with_identifier("RunMat:Placement:InvalidDecision")
                        .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                        .build()
                    })
                })?;
            } else {
                let value = vars
                    .get(*index)
                    .expect("loaded variable slot lies within frame extent");
                debug::trace_load_var(*pc, *index, &value);
                load_var(stack, vars, *index)?;
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadVarForIndexAssignment(index)
        | Instr::LoadVarForCellAssignment(index)
        | Instr::LoadVarForStructAssignment(index) => {
            let root_presence = slot_authority::AssignmentSlotAuthority {
                context,
                global_aliases,
                persistent_aliases,
                missing_input_slots,
                var_names,
                current_function_name,
            }
            .refresh(vars, *index)?;
            if let slot_authority::AssignmentRootPresence::Absent(slot_name) = root_presence {
                if matches!(instr, Instr::LoadVarForStructAssignment(_)) {
                    let empty = Value::Struct(runmat_value::StructValue::new());
                    debug::trace_load_var(*pc, *index, &empty);
                    stack.try_push(empty)?;
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::FallThrough,
                    )));
                }
                if matches!(instr, Instr::LoadVarForCellAssignment(_)) {
                    let empty = slot_authority::empty_cell_root();
                    debug::trace_load_var(*pc, *index, &empty);
                    stack.try_push(empty)?;
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::FallThrough,
                    )));
                }
                let empty = Tensor::new(Vec::new(), vec![0, 0]).map_err(|err| {
                        crate::interpreter::errors::mex(
                            "IndexAssignmentBaseInit",
                            &format!(
                                "failed to initialize undefined indexed-assignment base '{slot_name}': {err}"
                            ),
                        )
                    })?;
                debug::trace_load_var(*pc, *index, &Value::Tensor(empty.clone()));
                stack.try_push(Value::Tensor(empty))?;
                return Ok(Some(DispatchHandled::Generic(
                    DispatchDecision::FallThrough,
                )));
            }
            let value = vars
                .get(*index)
                .expect("assignment base slot lies within frame extent")
                .clone();
            debug::trace_load_var(*pc, *index, &value);
            load_var(stack, vars, *index)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::StoreVar(index) => {
            let preview = stack
                .last()
                .cloned()
                .ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?;
            debug::trace_store_var(*pc, *index, &preview);
            let value = crate::interpreter::stack::pop_value(stack)?;
            publish_var_value(
                value,
                *index,
                BindingPublicationFrame {
                    vars,
                    context,
                    var_names,
                    persistent_aliases,
                    missing_input_slots,
                    current_function_name,
                },
                store_var_before_overwrite,
                store_var_after_store,
            )?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::StoreLocal(offset) => {
            if context.call_stack.last().is_none() {
                store_var(
                    stack,
                    vars,
                    *offset,
                    var_names,
                    store_local_before_var_overwrite,
                    |stored_index| {
                        store_local_after_fallback_store("<main>", stored_index);
                    },
                )?;
                missing_input_slots.remove(offset);
                persistent::publish_store(
                    persistent_aliases,
                    vars,
                    &context.runtime.session_variable_authority(),
                    current_function_name,
                    *offset,
                )?;
                lexical::publish(context, vars, *offset)?;
            } else {
                store_local(
                    stack,
                    context,
                    vars,
                    *offset,
                    store_local_before_local_overwrite,
                    store_local_before_var_overwrite,
                    store_local_after_store,
                    store_local_after_fallback_store,
                )?;
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Swap => {
            crate::ops::stack::swap(stack)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Pop => {
            drop(stack.pop());
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Jump(target) => Ok(Some(DispatchHandled::Generic(apply_control_flow_action(
            crate::ops::control_flow::jump(*target),
            pc,
        )))),
        Instr::EnterTry {
            scope,
            catch_pc,
            catch_var,
        } => {
            crate::ops::control_flow::enter_try(
                try_stack,
                *scope,
                *catch_pc,
                *catch_var,
                stack.len(),
            );
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LeaveTry(scope) => {
            crate::ops::control_flow::leave_try(try_stack, *scope);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::ReturnValue => Ok(Some(DispatchHandled::ReturnValue(
            apply_control_flow_action(crate::ops::control_flow::return_value(stack)?, pc),
        ))),
        Instr::Return => Ok(Some(DispatchHandled::Return(apply_control_flow_action(
            crate::ops::control_flow::return_void(),
            pc,
        )))),
        Instr::EnterScope(local_count) => {
            crate::ops::control_flow::enter_scope(&mut context.locals, *local_count);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::ExitScope(local_count) => {
            for _ in 0..*local_count {
                drop(context.locals.pop());
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::RegisterImport { path, wildcard } => {
            imports.push((path.clone(), *wildcard));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::DeclareGlobal(indices) => {
            crate::runtime::globals::declare_global(indices, vars, global_aliases)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::DeclareGlobalNamed(indices, names) => {
            crate::runtime::globals::declare_global_named(indices, names, vars, global_aliases)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::DeclarePersistent(indices) => {
            let authority = context.runtime.session_variable_authority();
            for &slot in indices {
                let prepared = crate::runtime::workspace::prepare_workspace_persistent_slot(
                    context.runtime.local_identity(),
                    slot,
                    &context.runtime,
                )?;
                if let Some(prepared) = &prepared {
                    prepared.checkpoint_before_publication()?;
                }
                persistent::declare(
                    persistent_aliases,
                    vars,
                    &authority,
                    current_function_name,
                    slot,
                    PersistentAlias::LegacySlot,
                )?;
                if let Some(prepared) = prepared {
                    let value = vars
                        .get(slot)
                        .expect("persistent slot lies within frame extent")
                        .clone();
                    prepared.commit_owned_value(value);
                }
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::DeclarePersistentNamed(indices, names) => {
            let authority = context.runtime.session_variable_authority();
            for (position, &slot) in indices.iter().enumerate() {
                let name = names
                    .get(position)
                    .cloned()
                    .unwrap_or_else(|| format!("var_{slot}"));
                let prepared = crate::runtime::workspace::prepare_workspace_persistent_slot(
                    context.runtime.local_identity(),
                    slot,
                    &context.runtime,
                )?;
                if let Some(prepared) = &prepared {
                    prepared.checkpoint_before_publication()?;
                }
                persistent::declare(
                    persistent_aliases,
                    vars,
                    &authority,
                    current_function_name,
                    slot,
                    PersistentAlias::Named(name),
                )?;
                if let Some(prepared) = prepared {
                    let value = vars
                        .get(slot)
                        .expect("persistent slot lies within frame extent")
                        .clone();
                    prepared.commit_owned_value(value);
                }
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::PackToRow(count) => {
            pack_to_row(stack, *count)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::PackToCol(count) => {
            pack_to_col(stack, *count)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Unpack(out_count) => {
            if *out_count > 0 {
                unpack(stack, *out_count)?;
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateMatrix(rows, cols) => {
            create_matrix(stack, *rows, *cols)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateCell2D(rows, cols) => {
            let mut elems = Vec::with_capacity(*rows * *cols);
            for _ in 0..(*rows * *cols) {
                elems.push(stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?);
            }
            elems.reverse();
            stack.try_push(runmat_runtime::object::cell::create_cell_2d(
                elems, *rows, *cols,
            )?)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateCellFromSequences {
            row_lengths,
            elements,
            ..
        } => {
            let rows = materialize_aggregate_rows(stack, sequence_register, row_lengths, elements)?;
            let columns = rows.first().map_or(0, Vec::len);
            if rows.iter().any(|row| row.len() != columns) {
                return Err(crate::interpreter::errors::mex(
                    "CellLiteralShapeMismatch",
                    "cell literal rows realize different widths after comma-separated-list expansion",
                ));
            }
            let row_count = rows.len();
            let values = rows.into_iter().flatten().collect();
            stack.try_push(runmat_runtime::object::cell::create_cell_2d(
                values, row_count, columns,
            )?)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateStructLiteral(fields) => {
            let start = stack.len().checked_sub(fields.len()).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow while building aggregate literal",
                )
            })?;
            let values = stack.try_extract_output(start)?;
            let values = runmat_runtime::execution::RootedValues::from_output_storage_in_context(
                values,
                &context.runtime,
            )?;
            let structure = runmat_runtime::execution::construct_struct_in_context(
                &context.runtime,
                fields.iter().map(String::as_str),
                values,
            )?;
            structure.with_owned(|value| stack.try_push(value))?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateObjectLiteral { class_name, fields } => {
            let start = stack.len().checked_sub(fields.len()).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow while building aggregate literal",
                )
            })?;
            let values = stack.try_extract_output(start)?;
            let values = runmat_runtime::execution::RootedValues::from_output_storage_in_context(
                values,
                &context.runtime,
            )?;
            let object = runmat_runtime::class_registry::construct_class_literal_in_context(
                &context.runtime,
                class_name,
                fields,
                values,
            )?;
            object.with_owned(|value| stack.try_push(value))?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadMethod(name) => {
            handle_load_method(stack, name.0.clone(), current_function_name)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateClosure(func_name, capture_count) => {
            handle_create_closure(stack, func_name.clone(), *capture_count)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateSemanticClosure(function, display_name, capture_count, source_role) => {
            let inherits_shared_bindings = *source_role
                == runmat_types::FunctionSourceRole::Anonymous
                && function_registry.get(*function).is_some_and(|target| {
                    target.lexical_binding_slots.iter().any(|binding| {
                        binding.kind == crate::bytecode::program::LexicalBindingKind::Transit
                    })
                });
            handle_create_semantic_closure(
                stack,
                *function,
                display_name.clone(),
                *capture_count,
                *source_role,
                (source_role == &runmat_types::FunctionSourceRole::Nested
                    || inherits_shared_bindings)
                    .then(|| context.lexical_environment.clone())
                    .flatten(),
            )?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateMethodFunctionHandle(name) => {
            stack.try_push(Value::method_function_handle(name.clone()))?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateBoundFunctionHandle(function, name, source_role)
        | Instr::CreateExternalBoundFunctionHandle(function, name, source_role) => {
            let site = runmat_execution::value::GraphStaticReferenceSite {
                instruction: u32::try_from(*pc).map_err(|_| {
                    crate::interpreter::errors::mex(
                        "StaticSiteInvalid",
                        "instruction site exceeds selected graph domain",
                    )
                })?,
                occurrence: 0,
            };
            let selected = context
                .runtime
                .selected_static_binding(site)
                .map_err(|message| {
                    crate::interpreter::errors::mex("StaticImportUnavailable", message)
                })?;
            if selected.is_none()
                && context.runtime.has_selected_static_site_resolver()
                && matches!(instr, Instr::CreateExternalBoundFunctionHandle(..))
            {
                return Err(crate::interpreter::errors::mex(
                    "StaticImportUnavailable",
                    "external handle has no exact selected graph target",
                ));
            }
            let (selected_function, authority, source_file) = if let Some(binding) = selected {
                (binding.function, binding.authority, None)
            } else {
                (
                    function.0,
                    runmat_runtime::user_functions::bound_function_authority_for(function.0),
                    runmat_runtime::user_functions::source_file_for(function.0),
                )
            };
            stack.try_push(Value::bound_function_handle(
                name.clone(),
                selected_function,
                authority,
                *source_role,
                source_file,
            ))?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadStaticProperty(class_name, prop) => {
            handle_load_static_property(stack, class_name, prop.display_name())?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadWorkspaceFirstStaticProperty {
            name,
            class_name,
            property,
        } => {
            if let Some(value) = crate::runtime::workspace::workspace_lookup(name) {
                stack.try_push(value)?;
            } else {
                handle_load_static_property(stack, class_name, property.display_name())?;
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::RegisterClass {
            name,
            super_class,
            is_sealed,
            is_abstract,
            properties,
            methods,
            events,
            enumerations,
        } => {
            handle_register_class(crate::object::class_def::ClassRegistration {
                name: name.clone(),
                super_class: super_class.clone(),
                is_sealed: *is_sealed,
                is_abstract: *is_abstract,
                properties: properties.clone(),
                methods: methods.clone(),
                events: events.clone(),
                enumerations: enumerations.clone(),
            })?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::logical_truth_from_value;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, Tensor, Value};

    #[test]
    fn logical_truth_reads_typed_integer_tensor_storage_exactly() {
        let zero = Tensor::new_integer(IntegerStorage::U64(vec![0]), vec![1, 1])
            .expect("zero integer tensor");
        assert!(!block_on(logical_truth_from_value(
            &Value::Tensor(zero),
            "if condition"
        ))
        .unwrap());

        let nonzero = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1])
            .expect("nonzero integer tensor");
        assert!(block_on(logical_truth_from_value(
            &Value::Tensor(nonzero),
            "if condition"
        ))
        .unwrap());
    }

    #[test]
    fn logical_truth_accepts_typed_integer_scalar_with_cleared_mirror() {
        let value = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("integer scalar");

        assert!(block_on(logical_truth_from_value(
            &Value::Tensor(value),
            "if condition"
        ))
        .unwrap());
    }
}
