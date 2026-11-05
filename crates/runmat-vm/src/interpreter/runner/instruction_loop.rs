//! Heap owner for the interpreter instruction loop.
//!
//! The constructor receives only a bounded set of frame references. Building
//! the boxed owner in this module prevents the parent poll state from embedding
//! the concrete instruction-loop future in every nested semantic Process frame.

use super::*;
use crate::interpreter::vars::InterpreterVars;
#[cfg(feature = "native-accel")]
use runmat_runtime::context::VmFusionHostInstruction;
#[cfg(feature = "wgpu")]
use std::future::Future;
#[cfg(feature = "wgpu")]
use std::pin::Pin;

#[path = "instruction_loop/fusion_publish.rs"]
mod fusion_publish;
use fusion_publish::after_global_store;
#[path = "instruction_loop/stochastic_source.rs"]
mod stochastic_source;
#[cfg(feature = "native-accel")]
use fusion_publish::{publish_fusion_stores, FusionPublicationFrame};
#[cfg(feature = "native-accel")]
#[path = "instruction_loop/image_stages.rs"]
mod image_stages;

pub(super) type InstructionLoopResult = Result<(), RuntimeError>;

pub(super) struct InstructionLoopContext<'a> {
    pub(super) bytecode: &'a Bytecode,
    pub(super) pc: &'a mut usize,
    pub(super) stack: &'a mut crate::interpreter::stack::InterpreterStack,
    pub(super) vars: &'a mut InterpreterVars,
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
    pub(super) fusion_plan: &'a Option<runmat_accelerate::FusionPlanRef>,
    #[cfg(feature = "native-accel")]
    pub(super) fusion_installation: &'a Option<crate::bytecode::program::VmFusionPlanInstallation>,
    #[cfg(feature = "native-accel")]
    pub(super) image_installation: &'a Option<crate::bytecode::program::VmImagePlanInstallation>,
    #[cfg(feature = "native-accel")]
    pub(super) fusion_observations:
        &'a mut crate::interpreter::fusion_observation::FusionRegionObservations,
    #[cfg(all(
        test,
        not(target_arch = "wasm32"),
        feature = "native-accel",
        feature = "wgpu"
    ))]
    pub(super) fusion_publication_fault:
        &'a mut Option<crate::interpreter::state::VmFusionPublicationFault>,
}

pub(super) type BoxedInstructionLoop<'a> =
    runmat_runtime::execution::RuntimeContinuationFuture<'a, ()>;

#[cfg(feature = "wgpu")]
pin_project_lite::pin_project! {
    struct AuditedInstructionLoop<F> {
        #[pin]
        inner: F,
        slot: Option<runmat_accelerate_api::observation::SourceRegionRouteAuditSlot>,
    }
}

#[cfg(feature = "wgpu")]
impl<F: Future> Future for AuditedInstructionLoop<F> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.project();
        runmat_accelerate_api::observation::with_source_region_poll(this.slot.as_ref(), || {
            this.inner.poll(cx)
        })
    }
}

#[inline(never)]
pub(super) fn boxed(
    context: InstructionLoopContext<'_>,
) -> Result<BoxedInstructionLoop<'_>, RuntimeError> {
    let runtime = context.catch_runtime;
    runmat_runtime::execution::admit_domain_continuation_with(runtime, move || {
        #[cfg(feature = "wgpu")]
        {
            let slot = context
                .fusion_observations
                .route_audit_slot(context.image_installation.is_some());
            AuditedInstructionLoop {
                inner: run(context),
                slot,
            }
        }
        #[cfg(not(feature = "wgpu"))]
        {
            run(context)
        }
    })
}

// Fusion planning owns a different execution future from ordinary instruction
// dispatch. Allocate it only for a selected fusion candidate, so recursive VM
// polling does not reserve space for its concrete provider future.
#[cfg(feature = "native-accel")]
type FusionExecutionFuture<'a> = runmat_runtime::execution::RuntimeContinuationFuture<
    'a,
    runmat_accelerate::fusion_exec::FusionExecutionOutcome<accel_fusion::FusionVmResult>,
    accel_fusion::FusionAttemptError,
>;

#[cfg(feature = "native-accel")]
struct FusionExecutionFrame<'a> {
    stack: &'a mut crate::interpreter::stack::InterpreterStack,
    vars: &'a mut InterpreterVars,
    context: &'a mut crate::bytecode::program::ExecutionContext,
    persistent_aliases: &'a crate::interpreter::PersistentAliases,
    global_aliases: &'a runmat_runtime::workspace::bindings::FrameGlobalBindings,
    missing_input_slots: &'a HashSet<usize>,
}

#[cfg(feature = "native-accel")]
#[inline(never)]
fn boxed_fusion_execution<'a>(
    runtime: &runmat_runtime::context::RuntimeContext,
    plan: &'a runmat_accelerate::FusionGroupPlan,
    graph: &'a runmat_accelerate::AccelGraph,
    frame: FusionExecutionFrame<'a>,
    placement: Option<runmat_accelerate::placement::PlacementCorrelationId>,
    invocation: accel_fusion::VmFusionInvocation<'a>,
) -> Result<FusionExecutionFuture<'a>, accel_fusion::FusionAttemptError> {
    runmat_runtime::execution::admit_native_hostcall_with_contextual_error(runtime, move || {
        #[cfg(test)]
        fusion_store_tests::record_fusion_execution_entry();
        async move {
            let FusionExecutionFrame {
                stack,
                vars,
                context,
                persistent_aliases,
                global_aliases,
                missing_input_slots,
            } = frame;
            if invocation.transport.is_none_or(|transport| {
                !transport.permits_frame_reads(
                    context,
                    missing_input_slots,
                    global_aliases,
                    persistent_aliases,
                )
            }) {
                return Err(accel_fusion::FusionAttemptError::Deopt(
                    crate::interpreter::errors::mex(
                        "FusionInputTransport",
                        "fusion region retains an ordinary semantic load",
                    ),
                ));
            }
            accel_fusion::try_execute_fusion_group_with_inputs(
                plan, graph, stack, vars, context, placement, invocation,
            )
            .await
        }
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
        fusion_plan,
        #[cfg(feature = "native-accel")]
        fusion_installation,
        #[cfg(feature = "native-accel")]
        image_installation,
        #[cfg(feature = "native-accel")]
        fusion_observations,
        #[cfg(all(
            test,
            not(target_arch = "wasm32"),
            feature = "native-accel",
            feature = "wgpu"
        ))]
        fusion_publication_fault,
    } = loop_context;
    #[cfg(feature = "native-accel")]
    let private_temp_bridges = if fusion_plan.is_some() {
        crate::bytecode::fusion_private_temp::bridges(bytecode)
    } else {
        Vec::new()
    };
    #[cfg(feature = "native-accel")]
    let private_empty_controls = if fusion_plan.is_some() {
        crate::bytecode::fusion_private_temp::empty_controls(bytecode)
    } else {
        Vec::new()
    };
    #[cfg(all(feature = "native-accel", not(feature = "wgpu")))]
    let mut original_image_span: Option<runmat_accelerate::InstrSpan> = None;
    // A browser adapter must regain its event loop even when every opcode
    // completes synchronously. Native NeverYield keeps no task-turn owner.
    let mut cpu_turn = match context.runtime.execution().cpu_cooperation_policy() {
        runmat_runtime::execution::CpuCooperationPolicy::NeverYield => None,
        runmat_runtime::execution::CpuCooperationPolicy::WorkQuantum(_) => {
            Some(context.runtime.cpu_cooperation())
        }
    };
    while *pc < bytecode.instructions.len() {
        #[cfg(feature = "wgpu")]
        if let Some(route) = fusion_observations.take_completed_original() {
            route.complete().await;
        }
        if let Some(turn) = &mut cpu_turn {
            turn.checkpoint(1)
                .await
                .map_err(|error| error.into_runtime_error_in_context(Some(&context.runtime)))?;
        }
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
        interp_engine::check_cancelled(&context.runtime)?;
        let active_catch = bytecode
            .catch_scopes
            .get(*pc)
            .copied()
            .flatten()
            .and_then(|scope| catch_exceptions.get(&scope).cloned());
        #[cfg(all(feature = "native-accel", not(feature = "wgpu")))]
        if original_image_span
            .as_ref()
            .is_some_and(|span| *pc <= span.start || *pc > span.end)
        {
            original_image_span = None;
        }
        #[cfg(all(feature = "native-accel", not(feature = "wgpu")))]
        let mut image_original_source = original_image_span.is_some();
        #[cfg(all(feature = "native-accel", feature = "wgpu"))]
        let mut image_original_source = false;
        #[cfg(feature = "native-accel")]
        if let Some(installation) = image_installation.as_ref() {
            let prepared_image = image_stages::prepare(
                installation,
                *pc,
                image_stages::ImageStageFrame {
                    vars: &mut *vars,
                    context: &mut *context,
                    var_names: &bytecode.var_names,
                    completion_assigned_slots,
                    persistent_aliases: &*persistent_aliases,
                    current_function_name,
                    missing_input_slots,
                },
            );
            let image_result = match prepared_image {
                Err(error) => Err(error),
                Ok(None) => Ok(image_stages::ImageStageOutcome::Declined),
                Ok(Some(selected)) => {
                    let runtime = selected.runtime().clone();
                    let mut outcome = image_stages::ImageStageOutcome::Declined;
                    let outcome_slot = &mut outcome;
                    let completed = match runmat_runtime::execution::admit_domain_continuation_with(
                        &runtime,
                        move || async move {
                            *outcome_slot = image_stages::execute_selected(selected).await?;
                            Ok(())
                        },
                    ) {
                        Err(error) => Err(error),
                        Ok(continuation) => continuation.await,
                    };
                    completed.map(|()| outcome)
                }
            };
            match image_result {
                Ok(image_stages::ImageStageOutcome::Declined) => {}
                #[cfg(feature = "wgpu")]
                Ok(image_stages::ImageStageOutcome::Original {
                    span,
                    publication,
                    route,
                }) => {
                    fusion_observations.begin_original(span, publication, route);
                    image_original_source = true;
                    interp_engine::check_cancelled(&context.runtime)?;
                }
                #[cfg(not(feature = "wgpu"))]
                Ok(image_stages::ImageStageOutcome::OriginalSource { span }) => {
                    original_image_span = Some(span);
                    image_original_source = true;
                    interp_engine::check_cancelled(&context.runtime)?;
                }
                Ok(image_stages::ImageStageOutcome::Executed { next_pc }) => {
                    *pc = next_pc;
                    continue;
                }
                Err(error) => {
                    fusion_observations.abort();
                    sequence_register.clear(stack);
                    match interp_dispatch::redirect_exception_to_catch(
                        error,
                        &mut *try_stack,
                        &mut *vars,
                        &mut *last_exception,
                        &mut *pc,
                        stack,
                    ) {
                        interp_dispatch::ExceptionHandling::Caught => continue,
                        interp_dispatch::ExceptionHandling::Uncaught(error) => return Err(*error),
                    }
                }
            }
        }
        #[cfg(feature = "native-accel")]
        if let (Some(plan), Some(graph)) = (
            fusion_plan
                .as_ref()
                .and_then(|plan| plan.group_for_pc_clone(*pc)),
            fusion_accel_graph.as_ref(),
        ) {
            if plan.group.span.start == *pc
                && !fusion_observations.contains_start(*pc)
                && !image_original_source
            {
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
                let private_store_pcs =
                    crate::bytecode::fusion_private_temp::private_reduction_store_pcs(
                        &plan.group,
                        &private_temp_bridges,
                        &private_empty_controls,
                    );
                let has_barrier = !plan.kernel.supported
                    || crosses_catch_scope
                    || accel_fusion::fusion_span_has_vm_barrier(&bytecode.instructions, &span)
                    || accel_fusion::final_only_span_requires_ordinary_stores_with_private(
                        &plan.group.kind,
                        &bytecode.instructions,
                        &span,
                        &private_store_pcs,
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
                    let right =
                        if plan.group.kind.is_reduction() || plan.group.kind.is_elementwise() {
                            fusion_installation.as_ref().and_then(|installation| {
                                fusion_plan.as_ref().and_then(|active| {
                                    installation.right_for(active, &plan, &context.runtime)
                                })
                            })
                        } else {
                            None
                        };
                    let decline = fusion_installation.as_ref().and_then(|installation| {
                        fusion_plan.as_ref().and_then(|active| {
                            installation.decline_for(active, &plan, &context.runtime)
                        })
                    });
                    let publication_pc =
                        if plan.group.kind.is_reduction() || plan.group.kind.is_elementwise() {
                            fusion_installation.as_ref().and_then(|installation| {
                                fusion_plan.as_ref().and_then(|active| {
                                    installation.publication_pc_for(active, &plan)
                                })
                            })
                        } else {
                            None
                        };
                    let input_uses = fusion_installation.as_ref().and_then(|installation| {
                        fusion_plan
                            .as_ref()
                            .and_then(|active| installation.host_input_uses(active, &plan))
                    });
                    let transport = fusion_installation.as_ref().and_then(|installation| {
                        fusion_plan
                            .as_ref()
                            .and_then(|active| installation.input_transport(active, &plan))
                    });
                    let fusion_result = {
                        let fusion_future = boxed_fusion_execution(
                            catch_runtime,
                            &plan,
                            graph,
                            FusionExecutionFrame {
                                stack,
                                vars: &mut *vars,
                                context: &mut *context,
                                persistent_aliases: &*persistent_aliases,
                                global_aliases,
                                missing_input_slots,
                            },
                            Some(trace.correlation()),
                            accel_fusion::VmFusionInvocation {
                                right,
                                decline,
                                input_uses,
                                transport,
                            },
                        );
                        match fusion_future {
                            Ok(fusion_future) => {
                                if let Some(exception) = active_catch.clone() {
                                    catch_runtime
                                        .scope_active_catch_unboxed_shared(
                                            Some(exception),
                                            fusion_future,
                                        )
                                        .await
                                } else {
                                    fusion_future.await
                                }
                            }
                            Err(error) => Err(error),
                        }
                    };
                    match fusion_result {
                        Ok(runmat_accelerate::fusion_exec::FusionExecutionOutcome::Executed(
                            mut result,
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
                                    persistent_aliases,
                                    current_function_name,
                                    missing_input_slots,
                                },
                            ) {
                                fusion_observations.abort();
                                sequence_register.clear(stack);
                                match interp_dispatch::redirect_exception_to_catch(
                                    err,
                                    &mut *try_stack,
                                    &mut *vars,
                                    &mut *last_exception,
                                    &mut *pc,
                                    stack,
                                ) {
                                    interp_dispatch::ExceptionHandling::Caught => continue,
                                    interp_dispatch::ExceptionHandling::Uncaught(err) => {
                                        return Err(*err)
                                    }
                                }
                            }
                            stack.try_push(result.final_value)?;
                            let executed_pc = plan.group.span.end;
                            *pc = plan.group.span.end + 1;
                            if let Some(completion) = result.completion.take() {
                                fusion_observations.begin_provider_publication(
                                    span.clone(),
                                    completion,
                                    fusion_started,
                                    publication_pc,
                                );
                            }
                            fusion_observations.instruction_exit(executed_pc, *pc);
                            continue;
                        }
                        Ok(
                            runmat_accelerate::fusion_exec::FusionExecutionOutcome::SharedRuntime(
                                ticket,
                            ),
                        ) => {
                            fusion_observations.begin_at(
                                span.clone(),
                                Some(ticket),
                                trace,
                                fusion_started,
                                publication_pc,
                            );
                            // Planning may yield or request cancellation. Install
                            // the observation first so cancellation retires the
                            // selected ticket as failure, even for a one-instruction
                            // CPU region that could otherwise complete this turn.
                            interp_engine::check_cancelled(&context.runtime)?;
                        }
                        Err(accel_fusion::FusionAttemptError::Terminal(err)) => {
                            fusion_observations.abort();
                            sequence_register.clear(stack);
                            match interp_dispatch::redirect_exception_to_catch(
                                err,
                                &mut *try_stack,
                                &mut *vars,
                                &mut *last_exception,
                                &mut *pc,
                                stack,
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
                            fusion_observations.begin_at(
                                span.clone(),
                                None,
                                trace,
                                fusion_started,
                                None,
                            );
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
                    fusion_observations.begin_at(span.clone(), None, trace, fusion_started, None);
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
        let mut store_var_before_overwrite = |_stored_index: usize| {};
        let mut store_var_after_store = |stored_index: usize| {
            after_global_store(completion_assigned_slots, stored_index);
        };
        let mut store_local_before_local_overwrite = |_current: &Value, _incoming: &Value| {};
        let mut store_local_before_var_overwrite = |_stored_index: usize| {};
        let mut store_local_after_store = |_stored_offset: usize, _stored_value: &Value| {};
        let mut store_local_after_fallback_store = |_func_name: &str, _stored_offset: usize| {};
        #[cfg(feature = "native-accel")]
        let executed_pc = *pc;
        // The installed region supplies only request-local, instruction-scoped
        // host evidence. The ordinary dispatcher still owns execution and
        // restores the outer runtime before any result or catch is handled.
        #[cfg(feature = "native-accel")]
        let outer_runtime = context.runtime.clone();
        #[cfg(feature = "native-accel")]
        let host_kind = host_instruction_kind(
            &bytecode.instructions[*pc],
            *pc,
            &private_temp_bridges,
            &private_empty_controls,
            fusion_observations.is_terminal(*pc),
            match &bytecode.instructions[*pc] {
                Instr::CallBuiltinMultiUsingOutputSlot(_, _, slot) => {
                    interp_dispatch::requested_outputs_from_slot(vars, *slot).ok()
                }
                _ => None,
            },
            stack.as_slice(),
        );
        #[cfg(feature = "native-accel")]
        if let Some(route) = fusion_observations.instruction_route(*pc, host_kind) {
            if matches!(
                host_kind,
                VmFusionHostInstruction::ElementwiseBinary
                    | VmFusionHostInstruction::ElementwiseCall
                    | VmFusionHostInstruction::Intrinsic(_)
                    | VmFusionHostInstruction::Reduction
            ) {
                context.runtime = outer_runtime.clone().with_vm_fusion_host_route(Some(route));
            }
        }
        #[cfg(all(
            test,
            not(target_arch = "wasm32"),
            feature = "native-accel",
            feature = "wgpu"
        ))]
        if fusion_publication_fault.as_ref().is_some_and(|fault| {
            fusion_observations.checked_publication_store_pending(*pc, fault.route())
        }) && matches!(&bytecode.instructions[*pc], Instr::StoreVar(_))
        {
            if let Some(fault) = fusion_publication_fault.take() {
                use crate::interpreter::state::VmFusionPublicationFault;
                match fault {
                    VmFusionPublicationFault::Reject(_, reached) => {
                        reached.set(true);
                        *stack.last_mut().expect("checked publication has a result") =
                            Value::output_list(vec![Value::Num(0.0)]);
                    }
                    VmFusionPublicationFault::Cancel(_, source, reached) => {
                        reached.set(true);
                        source.cancel(runmat_execution::CancellationReason::User);
                        std::future::pending::<()>().await;
                    }
                    VmFusionPublicationFault::Suspend(_, reached) => {
                        reached.set(true);
                        std::future::pending::<()>().await;
                    }
                }
            }
        }
        let call_arg_spans = bytecode
            .call_arg_spans
            .get(*pc)
            .and_then(Option::as_deref)
            .map(|spans| {
                runmat_runtime::callsite::CallsiteSpans::from_borrowed_in_context(
                    spans,
                    &context.runtime,
                )
                .map_err(|source| {
                    runmat_runtime::context::host_allocation::host_allocation_runtime_error(
                        Some(&context.runtime),
                        "RunMat:VM:CallsiteProjection",
                        "VM callsite metadata admission failed",
                        source,
                    )
                })
            })
            .transpose()?;
        // Bracket actual slot/operand retirement, including temporary drops
        // inside a wholly-ready instruction. The registered frame follows the
        // original slot/stack owners, so surviving aliases remain visible.
        let cleanup = context.runtime.function_cleanup_scope()?;
        let dispatch_future = interp_dispatch::dispatch_instruction(
            interp_dispatch::DispatchMeta {
                instr: &bytecode.instructions[*pc],
                bytecode,
                #[cfg(feature = "native-accel")]
                private_temp_load: private_temp_bridges.iter().copied().find(|bridge| {
                    bridge.load_pc == *pc
                        && matches!(
                            &bytecode.instructions[*pc],
                            Instr::LoadVar(slot) if *slot == bridge.slot
                        )
                }),
                var_names: &bytecode.var_names,
                function_registry,
                source_id: bytecode
                    .source_id
                    .and_then(|source| source.resolve_in(&context.runtime)),
                call_arg_spans,
                invocation_arity,
                current_function_name,
            },
            interp_dispatch::DispatchState {
                stack,
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
        let cleanup_runtime = &catch_runtime;
        let dispatch_future = async move {
            // The inner dispatch future releases its completed captures before
            // the post-poll publication inspects surviving workspace aliases.
            let dispatch_future = async move { dispatch_future.await };
            let (result, cleanup) = cleanup
                .run_with_post_poll(cleanup_runtime, dispatch_future, |selected| {
                    // Completed slot changes must release their canonical owners
                    // before cleanup admission inspects surviving aliases. This
                    // publication keeps the live frame and never snapshots it.
                    // The selected completion must exist before its GC observers
                    // can queue a callback.
                    if selected.has_cleanup_candidates() {
                        selected
                            .service_ports()
                            .workspace()
                            .map_or(Ok(()), |workspace| {
                                workspace.publish_admitted_mutations_with_context(selected)
                            })
                    } else {
                        Ok(())
                    }
                })
                .await?;
            runmat_runtime::runtime_error::complete_with_cleanup(result, cleanup)
        };
        let dispatch_result = if let Some(exception) = active_catch {
            catch_runtime
                .scope_active_catch_unboxed_shared(Some(exception), dispatch_future)
                .await
        } else {
            dispatch_future.await
        };
        #[cfg(feature = "native-accel")]
        {
            context.runtime = outer_runtime;
        }
        #[cfg(feature = "native-accel")]
        let dispatch_result = match dispatch_result {
            Ok(result) => match fusion_observations.substitute_loaded_input(
                executed_pc,
                &bytecode.instructions[executed_pc],
                stack,
            ) {
                Ok(()) => Ok(result),
                // Load validation boxes only the failure path. Ordinary
                // dispatch retains its existing owned runtime-error boundary.
                Err(error) => Err(*error),
            },
            Err(error) => Err(error),
        };
        let dispatch_result = match dispatch_result {
            Ok(result) => result,
            Err(err) => {
                #[cfg(feature = "native-accel")]
                fusion_observations.abort();
                sequence_register.clear(stack);
                match interp_dispatch::redirect_exception_to_catch(
                    err,
                    &mut *try_stack,
                    &mut *vars,
                    &mut *last_exception,
                    &mut *pc,
                    stack,
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
        // The source descriptor may own randn argument vectors and text. Its
        // speculative preflight must borrow the instruction before the
        // ordinary by-value dispatch clones any opcode payload.
        if let instruction @ Instr::StochasticEvolution { fallback_pc, .. } =
            &bytecode.instructions[*pc]
        {
            let prepared = stochastic_source::prepare(
                instruction,
                stochastic_source::StochasticSourceFrame {
                    vars: &mut *vars,
                    context: &mut *context,
                    var_names: &bytecode.var_names,
                    completion_assigned_slots: &mut *completion_assigned_slots,
                    global_aliases,
                    persistent_aliases: &mut *persistent_aliases,
                    missing_input_slots: &mut *missing_input_slots,
                    current_function_name,
                },
            );
            let result = match prepared {
                Err(error) => Err(error),
                Ok(None) => Ok(stochastic_source::StochasticDispatch::Declined),
                Ok(Some(selected)) => {
                    let runtime = selected.runtime().clone();
                    let mut outcome = stochastic_source::StochasticDispatch::Declined;
                    let outcome_slot = &mut outcome;
                    let completed = match runmat_runtime::execution::admit_domain_continuation_with(
                        &runtime,
                        move || async move {
                            *outcome_slot = stochastic_source::execute_selected(selected).await?;
                            Ok(())
                        },
                    ) {
                        Err(error) => Err(error),
                        Ok(continuation) => continuation.await,
                    };
                    completed.map(|()| outcome)
                }
            };
            match result {
                Ok(stochastic_source::StochasticDispatch::Declined) => {
                    *pc = *fallback_pc;
                    #[cfg(feature = "native-accel")]
                    fusion_observations.instruction_exit(executed_pc, *pc);
                    continue;
                }
                Ok(stochastic_source::StochasticDispatch::Completed) => {
                    *pc += 1;
                    #[cfg(feature = "native-accel")]
                    fusion_observations.instruction_exit(executed_pc, *pc);
                    continue;
                }
                Err(err) => {
                    #[cfg(feature = "native-accel")]
                    fusion_observations.abort();
                    sequence_register.clear(stack);
                    match interp_dispatch::redirect_exception_to_catch(
                        err,
                        &mut *try_stack,
                        &mut *vars,
                        &mut *last_exception,
                        &mut *pc,
                        stack,
                    ) {
                        interp_dispatch::ExceptionHandling::Caught => continue,
                        interp_dispatch::ExceptionHandling::Uncaught(err) => return Err(*err),
                    }
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
            | Instr::LoadVarForCellAssignment(_)
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
            | Instr::MoveCapturedSequence { .. }
            | Instr::ReadSubscriptPath { .. }
            | Instr::CaptureSubscriptPath { .. }
            | Instr::BeginSubscriptEndReceiver { .. }
            | Instr::LoadSubscriptEnd { .. }
            | Instr::FinishSubscriptEndReceiver { .. }
            | Instr::FinishDottedInvokeReceiver { .. }
            | Instr::BeginOutputAssignment { .. }
            | Instr::BeginFixedAssignment { .. }
            | Instr::CommitFixedAssignment { .. }
            | Instr::PrepareFixedOutputTarget
            | Instr::PrepareDiscardOutputTarget
            | Instr::PrepareMemberSequenceOutputTarget { .. }
            | Instr::PrepareMemberDynamicSequenceOutputTarget { .. }
            | Instr::PrepareIndexedMemberSequenceOutputTarget { .. }
            | Instr::PrepareIndexedMemberDynamicSequenceOutputTarget { .. }
            | Instr::PrepareCellContentsSequenceOutputTarget { .. }
            | Instr::BeginSequenceOutputTarget { .. }
            | Instr::PrepareMemberPathStep { .. }
            | Instr::PrepareDynamicMemberPathStep { .. }
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
            | Instr::FinishFixedAssignmentTarget { .. }
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
            | Instr::CreateExternalFuture(_, _, _)
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
            | Instr::CallBuiltinExpandMultiOutputUsingOutputSlot(_, _, _)
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
            | Instr::CreateRange(_)
            | Instr::PackToRow(_)
            | Instr::PackToCol(_) => unreachable!("handled by dispatch_instruction"),
            Instr::StochasticEvolution { .. } => {
                unreachable!("stochastic source is dispatched before opcode cloning")
            }
        }
    }
    #[cfg(feature = "wgpu")]
    if let Some(route) = fusion_observations.take_completed_original() {
        route.complete().await;
    }
    Ok::<(), RuntimeError>(())
}

#[cfg(feature = "native-accel")]
fn host_instruction_kind(
    instr: &Instr,
    pc: usize,
    private_bridges: &[crate::bytecode::fusion_private_temp::FusionPrivateTempBridge],
    private_controls: &[crate::bytecode::fusion_private_temp::FusionPrivateEmptyControl],
    selected_terminal: bool,
    dynamic_output_count: Option<usize>,
    operands: &[Value],
) -> VmFusionHostInstruction {
    match instr {
        Instr::Add
        | Instr::Sub
        | Instr::ElemMul
        | Instr::ElemDiv
        | Instr::ElemLeftDiv
        | Instr::ElemPow => VmFusionHostInstruction::ElementwiseBinary,
        // The graph refines matrix division to elementwise division only for
        // a scalar divisor. Recheck the actual operands with the dispatcher's
        // predicate; the opcode alone must not admit a matrix solve. The
        // selected canonical binding still authenticates the executed body.
        Instr::RightDiv
            if matches!(operands, [.., _, right]
                if crate::ops::arithmetic::is_scalarish_for_division(right)) =>
        {
            VmFusionHostInstruction::ElementwiseBinary
        }
        Instr::LeftDiv
            if matches!(operands, [.., left, _]
                if crate::ops::arithmetic::is_scalarish_for_division(left)) =>
        {
            VmFusionHostInstruction::ElementwiseBinary
        }
        Instr::Neg => VmFusionHostInstruction::Intrinsic(runmat_types::OperatorKind::UnaryMinus),
        Instr::UPlus => VmFusionHostInstruction::Intrinsic(runmat_types::OperatorKind::UnaryPlus),
        Instr::Mul => {
            VmFusionHostInstruction::Intrinsic(runmat_types::OperatorKind::MatrixMultiply)
        }
        // The actual selected builtin identity is checked by the runtime
        // dispatcher and the host reduction branch. This opcode alone never
        // certifies a reduction or a CPU sample.
        Instr::CallBuiltinMulti(name, arguments, _)
        | Instr::CallBuiltinMultiUsingOutputSlot(name, arguments, _) => {
            let outputs = match instr {
                Instr::CallBuiltinMulti(_, _, outputs) => Some(*outputs),
                _ => dynamic_output_count,
            };
            match outputs.and_then(|outputs| {
                crate::accel::fusion_declaration::call_kind(name, *arguments, outputs)
            }) {
                Some(crate::bytecode::FusionInstructionKind::Elementwise) => {
                    VmFusionHostInstruction::ElementwiseCall
                }
                Some(crate::bytecode::FusionInstructionKind::Reduction) if selected_terminal => {
                    VmFusionHostInstruction::Reduction
                }
                _ => VmFusionHostInstruction::Unproved,
            }
        }
        Instr::StoreVar(slot)
            if private_bridges
                .iter()
                .any(|bridge| bridge.store_pc == pc && bridge.slot == *slot)
                || private_controls
                    .iter()
                    .any(|control| control.store_pc == pc && control.slot == *slot) =>
        {
            VmFusionHostInstruction::Structural
        }
        Instr::CreateMatrix(0, 0)
            if private_controls
                .iter()
                .any(|control| control.producer_pc == pc) =>
        {
            VmFusionHostInstruction::Structural
        }
        Instr::LoadVar(_)
        | Instr::LoadLocal(_)
        | Instr::LoadConst(_)
        | Instr::LoadInt(_)
        | Instr::LoadBool(_)
        | Instr::Swap
        | Instr::Pop => VmFusionHostInstruction::Structural,
        _ => VmFusionHostInstruction::Unproved,
    }
}

#[cfg(all(test, feature = "native-accel"))]
#[path = "instruction_loop/fusion_store_tests.rs"]
mod fusion_store_tests;

#[cfg(all(test, feature = "native-accel"))]
#[path = "instruction_loop/persistent_fusion_tests.rs"]
mod persistent_fusion_tests;

#[cfg(all(test, feature = "native-accel"))]
#[path = "instruction_loop/host_route_tests.rs"]
mod host_route_tests;

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "native-accel",
    feature = "wgpu"
))]
#[path = "instruction_loop/publication_lifecycle_tests.rs"]
mod publication_lifecycle_tests;
