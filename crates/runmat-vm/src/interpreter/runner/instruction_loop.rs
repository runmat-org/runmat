//! Heap owner for the interpreter instruction loop.
//!
//! The constructor receives only a bounded set of frame references. Building
//! the boxed owner in this module prevents the parent poll state from embedding
//! the concrete instruction-loop future in every nested semantic Process frame.

use super::*;
use std::future::Future;
use std::pin::Pin;

#[path = "instruction_loop/fusion_publish.rs"]
mod fusion_publish;
use fusion_publish::{after_global_store, after_local_fallback_store, after_local_store};
#[cfg(feature = "native-accel")]
use fusion_publish::{publish_fusion_stores, FusionPublicationFrame};

pub(super) type InstructionLoopResult = Result<(), RuntimeError>;

pub(super) struct InstructionLoopContext<'a> {
    pub(super) bytecode: &'a Bytecode,
    pub(super) pc: &'a mut usize,
    pub(super) stack: &'a mut Vec<Value>,
    pub(super) vars: &'a mut Vec<Value>,
    pub(super) completion_boundary:
        Option<crate::interpreter::state::InterpreterCompletionBoundary>,
    pub(super) completion_assigned_slots: &'a mut Option<HashSet<usize>>,
    pub(super) context: &'a mut crate::bytecode::program::ExecutionContext,
    pub(super) try_stack: &'a mut Vec<crate::interpreter::state::ActiveTryHandler>,
    pub(super) last_exception: &'a mut Option<runmat_value::MException>,
    pub(super) imports: &'a mut Vec<(Vec<String>, bool)>,
    pub(super) global_aliases: &'a runmat_runtime::workspace::bindings::FrameGlobalBindings,
    pub(super) persistent_aliases: &'a mut crate::interpreter::PersistentAliases,
    pub(super) missing_input_slots: &'a mut HashSet<usize>,
    pub(super) current_function_name: &'a str,
    pub(super) invocation_arity: RuntimeInvocationArity,
    pub(super) function_registry: &'a FunctionRegistry,
    pub(super) catch_runtime: &'a runmat_runtime::context::RuntimeContext,
    pub(super) catch_exceptions:
        &'a mut HashMap<runmat_mir::CatchScopeId, std::rc::Rc<runmat_value::MException>>,
    pub(super) catch_entries: &'a HashMap<usize, runmat_mir::CatchScopeId>,
    pub(super) sequence_register: &'a mut interp_dispatch::SequenceState,
    pub(super) debug_stack: bool,
    pub(super) interpreter_timing: &'a mut InterpreterTiming,
    #[cfg(feature = "native-accel")]
    pub(super) fusion_accel_graph: &'a Option<runmat_accelerate::graph::AccelGraph>,
    #[cfg(feature = "native-accel")]
    pub(super) fusion_observations:
        &'a mut crate::interpreter::fusion_observation::FusionRegionObservations,
}

pub(super) type BoxedInstructionLoop<'a> =
    Pin<Box<dyn Future<Output = InstructionLoopResult> + 'a>>;

#[inline(never)]
pub(super) fn boxed(context: InstructionLoopContext<'_>) -> BoxedInstructionLoop<'_> {
    Box::pin(run(context))
}

// Fusion planning owns a different execution future from ordinary instruction
// dispatch. Allocate it only for a selected fusion candidate, so recursive VM
// polling does not reserve space for its concrete provider future.
#[cfg(feature = "native-accel")]
type FusionExecutionFuture<'a> = Pin<
    Box<
        dyn Future<
                Output = Result<
                    runmat_accelerate::fusion_exec::FusionExecutionOutcome<
                        accel_fusion::FusionVmResult,
                    >,
                    accel_fusion::FusionAttemptError,
                >,
            > + 'a,
    >,
>;

#[cfg(feature = "native-accel")]
struct FusionExecutionFrame<'a> {
    stack: &'a mut Vec<Value>,
    vars: &'a mut Vec<Value>,
    context: &'a mut crate::bytecode::program::ExecutionContext,
    persistent_aliases: &'a crate::interpreter::PersistentAliases,
    current_function_name: &'a str,
}

#[cfg(feature = "native-accel")]
#[inline(never)]
fn boxed_fusion_execution<'a>(
    plan: &'a runmat_accelerate::FusionGroupPlan,
    graph: &'a runmat_accelerate::AccelGraph,
    frame: FusionExecutionFrame<'a>,
    placement: Option<runmat_accelerate::placement::PlacementCorrelationId>,
) -> FusionExecutionFuture<'a> {
    #[cfg(test)]
    fusion_store_tests::record_fusion_execution_entry();
    Box::pin(async move {
        let FusionExecutionFrame {
            stack,
            vars,
            context,
            persistent_aliases,
            current_function_name,
        } = frame;
        // Refresh only frame-owned variable inputs before graph capture. Stack
        // operands remain the values evaluated at their original instructions.
        let authority = context.runtime.session_variable_authority();
        let mut refreshed = HashSet::new();
        for value_id in &plan.inputs {
            if let Some(info) = graph.value(*value_id) {
                if let runmat_accelerate::graph::ValueOrigin::Variable { kind, index } =
                    &info.origin
                {
                    if (*kind == runmat_accelerate::VarKind::Global
                        || context.call_stack.last().is_none())
                        && refreshed.insert(*index)
                    {
                        crate::interpreter::persistent::refresh(
                            persistent_aliases,
                            vars,
                            &authority,
                            current_function_name,
                            *index,
                        )
                        .map_err(accel_fusion::FusionAttemptError::Terminal)?;
                    }
                }
            }
        }
        accel_fusion::try_execute_fusion_group(plan, graph, stack, vars, context, placement).await
    })
}

async fn run(loop_context: InstructionLoopContext<'_>) -> InstructionLoopResult {
    let InstructionLoopContext {
        bytecode,
        pc,
        stack,
        vars,
        completion_boundary,
        completion_assigned_slots,
        context,
        try_stack,
        last_exception,
        imports,
        global_aliases,
        persistent_aliases,
        missing_input_slots,
        current_function_name,
        invocation_arity,
        function_registry,
        catch_runtime,
        catch_exceptions,
        catch_entries,
        sequence_register,
        debug_stack,
        interpreter_timing,
        #[cfg(feature = "native-accel")]
        fusion_accel_graph,
        #[cfg(feature = "native-accel")]
        fusion_observations,
    } = loop_context;
    while *pc < bytecode.instructions.len() {
        crate::runtime::workspace::reconcile_workspace_vars(&mut *vars);
        if let (Some(scope), Some(exception)) = (catch_entries.get(&*pc), last_exception.as_ref()) {
            catch_exceptions.insert(*scope, std::rc::Rc::new(exception.clone()));
        }
        if completion_boundary.is_some_and(|boundary| boundary.pc() == *pc) {
            break;
        }
        if let Some(sites) = bytecode.coverage_sites.get(*pc) {
            runmat_runtime::coverage::hit_sites(sites);
        }
        set_vm_pc(*pc);
        #[cfg(feature = "native-accel")]
        set_current_pc(*pc);
        interp_engine::check_cancelled()?;
        let active_catch = bytecode
            .catch_scopes
            .get(*pc)
            .copied()
            .flatten()
            .and_then(|scope| catch_exceptions.get(&scope).cloned());
        #[cfg(feature = "native-accel")]
        if let (Some(plan), Some(graph)) = (active_group_plan_clone(), fusion_accel_graph.as_ref())
        {
            if plan.group.span.start == *pc && !fusion_observations.contains_start(*pc) {
                let span = plan.group.span.clone();
                let crosses_catch_scope = !bytecode.catch_scopes.is_empty()
                    && bytecode
                        .catch_scopes
                        .get(span.start..=span.end)
                        .is_none_or(|scopes| {
                            scopes
                                .first()
                                .is_some_and(|first| scopes.iter().any(|scope| scope != first))
                        });
                let has_barrier = !plan.kernel.supported
                    || crosses_catch_scope
                    || accel_fusion::fusion_span_has_vm_barrier(&bytecode.instructions, &span)
                    || accel_fusion::final_only_span_requires_ordinary_stores(
                        &plan.group.kind,
                        &bytecode.instructions,
                        &span,
                    );
                #[cfg(feature = "native-accel")]
                {
                    interp_engine::note_fusion_gate(
                        &mut *interpreter_timing,
                        &plan,
                        bytecode,
                        *pc,
                        has_barrier,
                        accel_fusion::fusion_span_live_result_count(
                            &bytecode.instructions,
                            &plan.group.span,
                        ),
                    );
                }
                let _fusion_span = info_span!(
                    "fusion.execute",
                    span_start = plan.group.span.start,
                    span_end = plan.group.span.end,
                    kind = ?plan.group.kind
                )
                .entered();
                let fusion_started = Instant::now();
                let trace = begin_trace(fusion_operation_token(&plan.group.kind), None);
                trace.event(
                    PlacementEventKind::Candidate,
                    Some(PlacementVariant::ProviderFusion),
                    Some("legacy_fusion_candidate"),
                    None,
                    None,
                    &[],
                );
                trace.event(
                    PlacementEventKind::Candidate,
                    Some(PlacementVariant::SharedRuntime),
                    Some("bytecode_fallback_candidate"),
                    None,
                    None,
                    &[],
                );
                if !has_barrier {
                    let _consumed_fusion = crate::runtime::gc::ManagedRoots::new(&*stack)
                        .map_err(RuntimeError::new)?;
                    let fusion_future = boxed_fusion_execution(
                        &plan,
                        graph,
                        FusionExecutionFrame {
                            stack: &mut *stack,
                            vars: &mut *vars,
                            context: &mut *context,
                            persistent_aliases: &*persistent_aliases,
                            current_function_name,
                        },
                        Some(trace.correlation()),
                    );
                    let fusion_result = if let Some(exception) = active_catch.clone() {
                        catch_runtime
                            .scope_active_catch_unboxed_shared(Some(exception), fusion_future)
                            .await
                    } else {
                        fusion_future.await
                    };
                    match fusion_result {
                        Ok(runmat_accelerate::fusion_exec::FusionExecutionOutcome::Executed(
                            result,
                        )) => {
                            trace.submitted(
                                PlacementVariant::ProviderFusion,
                                Some(duration_ns_saturating(fusion_started.elapsed())),
                            );
                            if let Err(err) = publish_fusion_stores(
                                result.ordered_stores,
                                FusionPublicationFrame {
                                    vars: &mut *vars,
                                    context: &mut *context,
                                    var_names: &bytecode.var_names,
                                    completion_assigned_slots,
                                    global_aliases,
                                    persistent_aliases,
                                    current_function_name,
                                    missing_input_slots,
                                },
                            ) {
                                fusion_observations.abort();
                                sequence_register.clear(&mut *stack);
                                match interp_dispatch::redirect_exception_to_catch(
                                    err,
                                    &mut *try_stack,
                                    &mut *vars,
                                    &mut *last_exception,
                                    &mut *pc,
                                    refresh_workspace_state,
                                ) {
                                    interp_dispatch::ExceptionHandling::Caught => continue,
                                    interp_dispatch::ExceptionHandling::Uncaught(err) => {
                                        return Err(*err)
                                    }
                                }
                            }
                            stack.push(result.final_value);
                            let executed_pc = *pc;
                            *pc = plan.group.span.end + 1;
                            fusion_observations.instruction_exit(executed_pc, *pc);
                            continue;
                        }
                        Ok(
                            runmat_accelerate::fusion_exec::FusionExecutionOutcome::SharedRuntime(
                                ticket,
                            ),
                        ) => {
                            fusion_observations.begin(span.clone(), Some(ticket), trace);
                            // Planning may yield or request cancellation. Install
                            // the observation first so cancellation retires the
                            // selected ticket as failure, even for a one-instruction
                            // CPU region that could otherwise complete this turn.
                            interp_engine::check_cancelled()?;
                        }
                        Err(accel_fusion::FusionAttemptError::Terminal(err)) => {
                            fusion_observations.abort();
                            sequence_register.clear(&mut *stack);
                            match interp_dispatch::redirect_exception_to_catch(
                                err,
                                &mut *try_stack,
                                &mut *vars,
                                &mut *last_exception,
                                &mut *pc,
                                refresh_workspace_state,
                            ) {
                                interp_dispatch::ExceptionHandling::Caught => continue,
                                interp_dispatch::ExceptionHandling::Uncaught(err) => {
                                    return Err(*err)
                                }
                            }
                        }
                        Err(accel_fusion::FusionAttemptError::Deopt(err)) => {
                            trace.event(
                                PlacementEventKind::Fallback,
                                Some(PlacementVariant::SharedRuntime),
                                Some("fusion_execution_failed"),
                                Some(duration_ns_saturating(fusion_started.elapsed())),
                                None,
                                &[],
                            );
                            fusion_observations.begin(span.clone(), None, trace);
                            log::debug!("fusion fallback at pc {}: {}", *pc, err);
                        }
                    }
                } else {
                    trace.event(
                        PlacementEventKind::Selected,
                        Some(PlacementVariant::SharedRuntime),
                        Some("fusion_vm_barrier"),
                        None,
                        None,
                        &[],
                    );
                    fusion_observations.begin(span.clone(), None, trace);
                    interp_engine::note_fusion_skip(*pc, &span);
                }
            }
        }
        interp_engine::note_pre_dispatch(
            &mut *interpreter_timing,
            debug_stack,
            *pc,
            &bytecode.instructions[*pc],
            stack.len(),
        );
        let store_var_global_aliases = match &bytecode.instructions[*pc] {
            Instr::StoreVar(_) => Some(global_aliases.clone()),
            _ => None,
        };
        let store_local_global_aliases = match &bytecode.instructions[*pc] {
            Instr::StoreLocal(_) => Some(global_aliases.clone()),
            _ => None,
        };
        let mut store_var_before_overwrite = |_current: &Value, _incoming: &Value| {};
        let mut store_var_after_store = |stored_index: usize, stored_value: &Value| {
            if let Some(ref aliases) = store_var_global_aliases {
                after_global_store(
                    completion_assigned_slots,
                    aliases,
                    stored_index,
                    stored_value,
                );
            } else if let Some(assigned_slots) = completion_assigned_slots.as_mut() {
                assigned_slots.insert(stored_index);
            }
        };
        let mut store_local_before_local_overwrite = |_current: &Value, _incoming: &Value| {};
        let mut store_local_before_var_overwrite = |_current: &Value, _incoming: &Value| {};
        let mut store_local_after_store = |stored_offset: usize, stored_value: &Value| {
            if let Some(ref aliases) = store_local_global_aliases {
                after_local_store(aliases, stored_offset, stored_value);
            }
        };
        let mut store_local_after_fallback_store =
            |_func_name: &str, stored_offset: usize, stored_value: &Value| {
                if let Some(ref aliases) = store_local_global_aliases {
                    after_local_fallback_store(aliases, stored_offset, stored_value);
                }
            };
        #[cfg(feature = "native-accel")]
        let executed_pc = *pc;
        let dispatch_future = interp_dispatch::dispatch_instruction(
            interp_dispatch::DispatchMeta {
                instr: &bytecode.instructions[*pc],
                bytecode,
                var_names: &bytecode.var_names,
                function_registry,
                source_id: bytecode
                    .source_id
                    .and_then(|source| source.resolve_in(&context.runtime)),
                call_arg_spans: bytecode.call_arg_spans.get(*pc).cloned().flatten(),
                invocation_arity,
                current_function_name,
            },
            interp_dispatch::DispatchState {
                stack: &mut *stack,
                vars: &mut *vars,
                sequence_register: &mut *sequence_register,
                context: &mut *context,
                try_stack: &mut *try_stack,
                last_exception: &mut *last_exception,
                imports: &mut *imports,
                global_aliases,
                persistent_aliases: &mut *persistent_aliases,
                missing_input_slots: &mut *missing_input_slots,
                pc: &mut *pc,
            },
            interp_dispatch::DispatchHooks {
                store_var_before_overwrite: &mut store_var_before_overwrite,
                store_var_after_store: &mut store_var_after_store,
                store_local_before_local_overwrite: &mut store_local_before_local_overwrite,
                store_local_before_var_overwrite: &mut store_local_before_var_overwrite,
                store_local_after_store: &mut store_local_after_store,
                store_local_after_fallback_store: &mut store_local_after_fallback_store,
            },
        );
        let dispatch_result = if let Some(exception) = active_catch {
            catch_runtime
                .scope_active_catch_unboxed_shared(Some(exception), dispatch_future)
                .await
        } else {
            dispatch_future.await
        };
        let dispatch_result = match dispatch_result {
            Ok(result) => result,
            Err(err) => {
                #[cfg(feature = "native-accel")]
                fusion_observations.abort();
                sequence_register.clear(&mut *stack);
                match interp_dispatch::redirect_exception_to_catch(
                    err,
                    &mut *try_stack,
                    &mut *vars,
                    &mut *last_exception,
                    &mut *pc,
                    refresh_workspace_state,
                ) {
                    interp_dispatch::ExceptionHandling::Caught => {
                        continue;
                    }
                    interp_dispatch::ExceptionHandling::Uncaught(err) => return Err(*err),
                }
            }
        };
        if let Some(decision) = dispatch_result {
            match decision {
                interp_dispatch::DispatchHandled::Generic(DispatchDecision::ContinueLoop) => {
                    #[cfg(feature = "native-accel")]
                    fusion_observations.instruction_exit(executed_pc, *pc);
                    continue;
                }
                interp_dispatch::DispatchHandled::Generic(DispatchDecision::FallThrough) => {
                    *pc += 1;
                    #[cfg(feature = "native-accel")]
                    fusion_observations.instruction_exit(executed_pc, *pc);
                    continue;
                }
                interp_dispatch::DispatchHandled::Generic(DispatchDecision::Return) => {
                    interpreter_timing.flush_host_span("return", None);
                    break;
                }
                interp_dispatch::DispatchHandled::ReturnValue(DispatchDecision::ContinueLoop)
                | interp_dispatch::DispatchHandled::Return(DispatchDecision::ContinueLoop) => {
                    #[cfg(feature = "native-accel")]
                    fusion_observations.instruction_exit(executed_pc, *pc);
                    continue;
                }
                interp_dispatch::DispatchHandled::ReturnValue(DispatchDecision::Return) => {
                    interpreter_timing.flush_host_span("return_value", None);
                    break;
                }
                interp_dispatch::DispatchHandled::Return(DispatchDecision::Return) => {
                    interpreter_timing.flush_host_span("return", None);
                    break;
                }
                interp_dispatch::DispatchHandled::ReturnValue(DispatchDecision::FallThrough)
                | interp_dispatch::DispatchHandled::Return(DispatchDecision::FallThrough) => {
                    *pc += 1;
                    #[cfg(feature = "native-accel")]
                    fusion_observations.instruction_exit(executed_pc, *pc);
                    continue;
                }
            }
        }
        match bytecode.instructions[*pc].clone() {
            Instr::EmitStackTop { .. }
            | Instr::EmitVar { .. }
            | Instr::AndAnd(_)
            | Instr::OrOr(_)
            | Instr::JumpIfFalse(_)
            | Instr::Jump(_)
            | Instr::LoadConst(_)
            | Instr::LoadInt(_)
            | Instr::LoadComplex(_, _)
            | Instr::LoadBool(_)
            | Instr::LoadString(_)
            | Instr::LoadCharRow(_)
            | Instr::LoadLocal(_)
            | Instr::LoadVar(_)
            | Instr::LoadVarForIndexAssignment(_)
            | Instr::LoadVarForStructAssignment(_)
            | Instr::StoreVar(_)
            | Instr::StoreLocal(_)
            | Instr::Swap
            | Instr::Pop
            | Instr::EnterTry { .. }
            | Instr::LeaveTry(_)
            | Instr::ReturnValue
            | Instr::Return
            | Instr::EnterScope(_)
            | Instr::LoadMember(_)
            | Instr::LoadMemberOrInit(_)
            | Instr::LoadMemberDynamic
            | Instr::LoadMemberDynamicOrInit
            | Instr::LoadMemberSequence { .. }
            | Instr::LoadMemberDynamicSequence { .. }
            | Instr::MemberSequenceCardinality
            | Instr::LoadMemberSequenceUsingOutputSlot { .. }
            | Instr::LoadMemberDynamicSequenceUsingOutputSlot { .. }
            | Instr::CaptureCallOutputSequence
            | Instr::CaptureScalarSequence
            | Instr::CaptureMemberSequence { .. }
            | Instr::CaptureMemberDynamicSequence { .. }
            | Instr::CaptureCellContentsSequence { .. }
            | Instr::CaptureReturnedOutputsSequence { .. }
            | Instr::ReadSubscriptPath { .. }
            | Instr::CaptureSubscriptPath { .. }
            | Instr::BeginSubscriptEndReceiver { .. }
            | Instr::LoadSubscriptEnd { .. }
            | Instr::FinishSubscriptEndReceiver { .. }
            | Instr::FinishDottedInvokeReceiver { .. }
            | Instr::BeginOutputAssignment { .. }
            | Instr::PrepareFixedOutputTarget
            | Instr::PrepareDiscardOutputTarget
            | Instr::PrepareMemberSequenceOutputTarget { .. }
            | Instr::PrepareMemberDynamicSequenceOutputTarget { .. }
            | Instr::PrepareIndexedMemberSequenceOutputTarget { .. }
            | Instr::PrepareIndexedMemberDynamicSequenceOutputTarget { .. }
            | Instr::PrepareCellContentsSequenceOutputTarget { .. }
            | Instr::BeginSequenceOutputTarget { .. }
            | Instr::PrepareMemberPathStep(_)
            | Instr::PrepareDynamicMemberPathStep
            | Instr::BeginPreparedIndexSelectors { .. }
            | Instr::LoadPreparedIndexEnd { .. }
            | Instr::BeginContextualIndexSelectors { .. }
            | Instr::LoadContextualIndexEnd { .. }
            | Instr::FinishContextualIndexSelectors { .. }
            | Instr::PrepareParenthesesPathStep { .. }
            | Instr::PrepareBracesPathStep { .. }
            | Instr::FinishMemberSequenceOutputTarget(_)
            | Instr::FinishDynamicMemberSequenceOutputTarget
            | Instr::FinishCellContentsSequenceOutputTarget { .. }
            | Instr::LoadPreparedOutputCardinality
            | Instr::CommitPreparedOutputTargets { .. }
            | Instr::StoreMember(_)
            | Instr::StoreMemberOrInit(_)
            | Instr::StoreMemberDynamic
            | Instr::StoreMemberDynamicOrInit
            | Instr::StoreMemberSequence(_)
            | Instr::StoreMemberDynamicSequence
            | Instr::Index(_)
            | Instr::IndexSlice(_, _, _, _)
            | Instr::IndexCell { .. }
            | Instr::IndexCellExpand { .. }
            | Instr::IndexCellList { .. }
            | Instr::StoreIndex(_)
            | Instr::StoreIndexCell { .. }
            | Instr::StoreIndexDelete(_)
            | Instr::StoreIndexCellDelete { .. }
            | Instr::StoreSlice(_, _, _, _)
            | Instr::StoreSliceDelete(_, _, _, _)
            | Instr::CallMethodOrMemberIndexMulti { .. }
            | Instr::CallMethodOrMemberIndexExpandMultiOutput { .. }
            | Instr::LoadMethod(_)
            | Instr::CreateFunctionHandle(_)
            | Instr::CreateExternalFunctionHandle(_)
            | Instr::CreateMethodFunctionHandle(_)
            | Instr::CreateBoundFunctionHandle(_, _, _)
            | Instr::CreateExternalBoundFunctionHandle(_, _, _)
            | Instr::CreateClosure(_, _)
            | Instr::CreateSemanticClosure(_, _, _, _)
            | Instr::LoadStaticProperty(_, _)
            | Instr::LoadWorkspaceFirstStaticProperty { .. }
            | Instr::RegisterClass { .. }
            | Instr::CallFevalMulti(_, _)
            | Instr::CallFevalMultiUsingOutputSlot(_, _)
            | Instr::CallFevalExpandMultiOutput(_, _)
            | Instr::CallFevalExpandMultiOutputUsingOutputSlot(_, _)
            | Instr::CreateSemanticFuture(_, _, _)
            | Instr::CreateSemanticFutureExpandMultiOutput(_, _, _)
            | Instr::ScheduleFeval { .. }
            | Instr::Spawn
            | Instr::SpawnOn
            | Instr::Await
            | Instr::FetchOutputs { .. }
            | Instr::FetchNext { .. }
            | Instr::EnsurePool(_)
            | Instr::CurrentPool(_)
            | Instr::CallBuiltinMulti(_, _, _)
            | Instr::CallBuiltinMultiUsingOutputSlot(_, _, _)
            | Instr::CallSuperConstructorMulti { .. }
            | Instr::CallSuperMethodMulti { .. }
            | Instr::CallSemanticFunctionMulti(_, _, _)
            | Instr::CallSemanticFunctionMultiUsingOutputSlot(_, _, _)
            | Instr::CallSemanticNestedFunctionMulti { .. }
            | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { .. }
            | Instr::CallFunctionMulti { .. }
            | Instr::CallFunctionMultiUsingOutputSlot { .. }
            | Instr::CallFunctionExpandMultiOutput { .. }
            | Instr::CallWorkspaceFirstMulti { .. }
            | Instr::CallWorkspaceFirstMultiUsingOutputSlot { .. }
            | Instr::CallWorkspaceFirstExpandMultiOutput { .. }
            | Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { .. }
            | Instr::CallSemanticFunctionExpandMultiOutput(_, _, _)
            | Instr::CallSemanticNestedFunctionExpandMultiOutput { .. }
            | Instr::CallBuiltinExpandMultiOutput(_, _, _)
            | Instr::CallSuperConstructorExpandMultiOutput { .. }
            | Instr::CallSuperMethodExpandMultiOutput { .. }
            | Instr::ExitScope(_)
            | Instr::RegisterImport { .. }
            | Instr::DeclareGlobal(_)
            | Instr::DeclareGlobalNamed(_, _)
            | Instr::DeclarePersistent(_)
            | Instr::DeclarePersistentNamed(_, _)
            | Instr::CreateCell2D(_, _)
            | Instr::CreateCellFromSequences { .. }
            | Instr::CreateStructLiteral(_)
            | Instr::CreateObjectLiteral { .. }
            | Instr::Add
            | Instr::Sub
            | Instr::Mul
            | Instr::ElemMul
            | Instr::ElemDiv
            | Instr::ElemPow
            | Instr::ElemLeftDiv
            | Instr::Neg
            | Instr::UPlus
            | Instr::Transpose
            | Instr::ConjugateTranspose
            | Instr::Pow
            | Instr::RightDiv
            | Instr::LeftDiv
            | Instr::LessEqual
            | Instr::Less
            | Instr::Greater
            | Instr::GreaterEqual
            | Instr::Equal
            | Instr::NotEqual
            | Instr::LogicalNot
            | Instr::LogicalAnd
            | Instr::LogicalOr
            | Instr::ExecuteParfor { .. }
            | Instr::ExecuteSpmd { .. }
            | Instr::Distributed(_)
            | Instr::Collective { .. }
            | Instr::Unpack(_)
            | Instr::CreateMatrix(_, _)
            | Instr::CreateMatrixFromSequences { .. }
            | Instr::CreateMatrixDynamic(_)
            | Instr::CreateRange(_)
            | Instr::PackToRow(_)
            | Instr::PackToCol(_) => unreachable!("handled by dispatch_instruction"),
            Instr::StochasticEvolution => {
                let steps_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let scale_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let drift_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let state_value = stack
                    .pop()
                    .ok_or(mex("StackUnderflow", "stack underflow"))?;
                let evolved =
                    crate::accel::idioms::stochastic_evolution::execute_stochastic_evolution(
                        state_value,
                        drift_value,
                        scale_value,
                        steps_value,
                    )
                    .await?;
                stack.push(evolved);
            }
        }
        if debug_stack {
            debug!(pc = *pc, stack_len = stack.len(), "[vm] after exec");
        }
        *pc += 1;
        #[cfg(feature = "native-accel")]
        fusion_observations.instruction_exit(executed_pc, *pc);
    }
    Ok::<(), RuntimeError>(())
}

#[cfg(all(test, feature = "native-accel"))]
#[path = "instruction_loop/fusion_store_tests.rs"]
mod fusion_store_tests;

#[cfg(all(test, feature = "native-accel"))]
#[path = "instruction_loop/persistent_fusion_tests.rs"]
mod persistent_fusion_tests;
