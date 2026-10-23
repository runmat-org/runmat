//! Region observations live across bytecode dispatch, not just fusion selection.
use runmat_accelerate::fusion_exec::{FusionCpuObservation, FusionProviderCompletion};
use runmat_accelerate::placement::{PlacementTraceContext, PlacementVariant};
use runmat_accelerate::InstrSpan;
use runmat_runtime::context::{VmFusionHostInstruction, VmFusionHostRoute};
use runmat_time::{duration_ns_saturating, Instant};

#[derive(Default)]
pub(super) struct FusionRegionObservations(
    Vec<RegionObservation>,
    #[cfg(feature = "wgpu")] Option<runmat_accelerate_api::observation::SourceRegionRouteAuditSlot>,
    #[cfg(feature = "wgpu")] Option<Box<runmat_accelerate::fusion_exec::ImageVmOriginalRoute>>,
);

#[cfg(feature = "wgpu")]
struct OriginalSourceProof {
    route: Box<runmat_accelerate::fusion_exec::ImageVmOriginalRoute>,
    publication: std::rc::Rc<crate::bytecode::ImageNormalizePublicationJoin>,
    next_store: usize,
}

struct RegionObservation {
    span: InstrSpan,
    ticket: Option<FusionCpuObservation>,
    provider_completion: Option<FusionProviderCompletion>,
    trace: Option<PlacementTraceContext>,
    started: Instant,
    host_route: Option<VmFusionHostRoute>,
    publication_pc: Option<usize>,
    awaiting_publication: bool,
    #[cfg(feature = "wgpu")]
    original: Option<OriginalSourceProof>,
}

#[derive(Debug, PartialEq, Eq)]
enum RegionExit {
    Running,
    Completed,
    Aborted,
}

fn region_exit(span: &InstrSpan, executed_pc: usize, next_pc: usize) -> RegionExit {
    if executed_pc == span.end && span.end.checked_add(1) == Some(next_pc) {
        RegionExit::Completed
    } else if next_pc < span.start || next_pc > span.end {
        RegionExit::Aborted
    } else {
        RegionExit::Running
    }
}

impl FusionRegionObservations {
    #[cfg(feature = "wgpu")]
    pub(super) fn route_audit_slot(
        &mut self,
        image_capable: bool,
    ) -> Option<runmat_accelerate_api::observation::SourceRegionRouteAuditSlot> {
        if image_capable && self.1.is_none() {
            self.1 = Some(std::sync::Arc::new(std::sync::Mutex::new(None)));
        }
        self.1.as_ref().map(std::sync::Arc::clone)
    }

    #[cfg(feature = "wgpu")]
    fn clear_route_audit(&self) {
        if let Some(slot) = &self.1 {
            if let Ok(mut active) = slot.lock() {
                *active = None;
            }
        }
        runmat_accelerate_api::observation::set_source_region_for_current_poll(None);
    }

    #[cfg(feature = "wgpu")]
    pub(super) fn begin_original(
        &mut self,
        span: InstrSpan,
        publication: std::rc::Rc<crate::bytecode::ImageNormalizePublicationJoin>,
        route: Box<runmat_accelerate::fusion_exec::ImageVmOriginalRoute>,
    ) {
        if let Some(slot) = &self.1 {
            if let Ok(mut active) = slot.lock() {
                let audit = route.audit();
                *active = Some(audit.clone());
                runmat_accelerate_api::observation::set_source_region_for_current_poll(Some(audit));
            }
        }
        self.0.push(RegionObservation {
            span,
            ticket: None,
            provider_completion: None,
            trace: None,
            started: Instant::now(),
            host_route: None,
            publication_pc: None,
            awaiting_publication: false,
            original: Some(OriginalSourceProof {
                route,
                publication,
                next_store: 0,
            }),
        });
    }

    #[cfg(feature = "wgpu")]
    pub(super) fn take_completed_original(
        &mut self,
    ) -> Option<Box<runmat_accelerate::fusion_exec::ImageVmOriginalRoute>> {
        self.2.take()
    }

    /// Dispatch has already refreshed and checked the ordinary binding. Only
    /// now may an exact certified load replace its pushed stack value with the
    /// selected region's private host mirror.
    pub(super) fn substitute_loaded_input(
        &self,
        pc: usize,
        instruction: &crate::bytecode::Instr,
        stack: &mut crate::interpreter::stack::InterpreterStack,
    ) -> Result<(), Box<runmat_runtime::RuntimeError>> {
        use runmat_accelerate::graph::{VarBinding, VarKind};
        let binding = match instruction {
            crate::bytecode::Instr::LoadVar(index) => VarBinding {
                kind: VarKind::Global,
                index: *index,
            },
            crate::bytecode::Instr::LoadLocal(index) => VarBinding {
                kind: VarKind::Local,
                index: *index,
            },
            _ => return Ok(()),
        };
        let Some(observation) = self
            .0
            .iter()
            .rev()
            .find(|observation| observation.span.start <= pc && pc <= observation.span.end)
        else {
            return Ok(());
        };
        let Some(ticket) = observation.ticket.as_ref() else {
            return Ok(());
        };
        let Some(actual) = stack.last_mut() else {
            return Err(Box::new(
                runmat_runtime::build_runtime_error("certified ordinary load produced no operand")
                    .build(),
            ));
        };
        if let Some(value) = ticket.substitute_vm_host_load(pc, &binding, actual)? {
            *actual = value;
        }
        Ok(())
    }
    #[cfg(all(test, not(target_arch = "wasm32"), feature = "wgpu"))]
    pub(super) fn checked_publication_store_pending(
        &self,
        pc: usize,
        route: &crate::interpreter::state::VmFusionPublicationFaultRoute,
    ) -> bool {
        self.0.iter().any(|observation| {
            observation.awaiting_publication
                && observation.publication_pc == Some(pc)
                && match route {
                    crate::interpreter::state::VmFusionPublicationFaultRoute::AnySelected => true,
                    crate::interpreter::state::VmFusionPublicationFaultRoute::SharedRuntime => {
                        observation.host_route.is_some()
                    }
                }
        })
    }

    pub(super) fn contains_start(&self, pc: usize) -> bool {
        self.0
            .iter()
            .any(|observation| observation.span.start == pc)
    }

    pub(super) fn is_terminal(&self, pc: usize) -> bool {
        self.0
            .iter()
            .rev()
            .find(|observation| observation.span.start <= pc && pc <= observation.span.end)
            .is_some_and(|observation| {
                observation.host_route.is_some() && observation.span.end == pc
            })
    }

    #[cfg(test)]
    pub(super) fn begin(
        &mut self,
        span: InstrSpan,
        ticket: Option<FusionCpuObservation>,
        trace: PlacementTraceContext,
    ) {
        self.begin_at(span, ticket, trace, Instant::now(), None);
    }

    pub(super) fn begin_at(
        &mut self,
        span: InstrSpan,
        ticket: Option<FusionCpuObservation>,
        trace: PlacementTraceContext,
        started: Instant,
        publication_pc: Option<usize>,
    ) {
        let host_route = ticket
            .as_ref()
            .and_then(FusionCpuObservation::vm_host_route);
        self.0.push(RegionObservation {
            span,
            ticket,
            provider_completion: None,
            trace: Some(trace),
            started,
            host_route,
            publication_pc,
            awaiting_publication: false,
            #[cfg(feature = "wgpu")]
            original: None,
        });
    }

    pub(super) fn begin_provider_publication(
        &mut self,
        span: InstrSpan,
        completion: FusionProviderCompletion,
        started: Instant,
        publication_pc: Option<usize>,
    ) {
        self.0.push(RegionObservation {
            span,
            ticket: None,
            provider_completion: Some(completion),
            trace: None,
            started,
            host_route: None,
            publication_pc,
            awaiting_publication: false,
            #[cfg(feature = "wgpu")]
            original: None,
        });
    }

    pub(super) fn instruction_route(
        &self,
        pc: usize,
        instruction: VmFusionHostInstruction,
    ) -> Option<VmFusionHostRoute> {
        let route = self
            .0
            .iter()
            .rev()
            .find(|observation| observation.span.start <= pc && pc <= observation.span.end)
            .and_then(|observation| observation.host_route.clone())?;
        route.begin_instruction(pc, instruction);
        Some(route)
    }

    pub(super) fn instruction_exit(&mut self, executed_pc: usize, next_pc: usize) {
        let mut index = 0;
        while index < self.0.len() {
            #[cfg(feature = "wgpu")]
            if let Some(proof) = self.0[index].original.as_mut() {
                if let Some(expected) = proof.publication.ordered_stores().get(proof.next_store) {
                    if expected.store.pc == executed_pc
                        && executed_pc.checked_add(1) == Some(next_pc)
                    {
                        proof.next_store += 1;
                    } else if expected.store.pc < next_pc || expected.store.pc <= executed_pc {
                        self.0.remove(index);
                        self.clear_route_audit();
                        continue;
                    }
                }
            }
            if self.0[index].awaiting_publication {
                let published = self.0[index].publication_pc == Some(executed_pc)
                    && executed_pc.checked_add(1) == Some(next_pc);
                let mut observation = self.0.remove(index);
                if published {
                    if let Some(ticket) = observation.ticket.take() {
                        ticket.complete_vm(observation.started);
                    }
                    if let Some(completion) = observation.provider_completion.take() {
                        completion.complete(observation.started);
                    }
                    if let Some(trace) = observation.trace.take() {
                        trace.complete(
                            Some(PlacementVariant::SharedRuntime),
                            Some(duration_ns_saturating(observation.started.elapsed())),
                        );
                    }
                }
                continue;
            }
            if let Some(route) = &self.0[index].host_route {
                if route.contains_pc(executed_pc) {
                    route.instruction_exit(executed_pc);
                }
            }
            match region_exit(&self.0[index].span, executed_pc, next_pc) {
                RegionExit::Running => index += 1,
                RegionExit::Completed => {
                    let mut observation = self.0.remove(index);
                    #[cfg(feature = "wgpu")]
                    if let Some(proof) = observation.original.take() {
                        self.clear_route_audit();
                        if proof.next_store == proof.publication.ordered_stores().len() {
                            self.2 = Some(proof.route);
                        }
                        continue;
                    }
                    if observation.publication_pc == executed_pc.checked_add(1)
                        && (observation.host_route.is_some()
                            || observation.provider_completion.is_some())
                    {
                        observation.awaiting_publication = true;
                        self.0.insert(index, observation);
                        index += 1;
                    } else if observation.host_route.is_none()
                        && observation.provider_completion.is_none()
                    {
                        if let Some(ticket) = observation.ticket.take() {
                            ticket.complete();
                        }
                        if let Some(trace) = observation.trace.take() {
                            trace.complete(
                                Some(PlacementVariant::SharedRuntime),
                                Some(duration_ns_saturating(observation.started.elapsed())),
                            );
                        }
                    }
                }
                RegionExit::Aborted => {
                    #[cfg(feature = "wgpu")]
                    if self.0[index].original.is_some() {
                        self.clear_route_audit();
                    }
                    self.0.remove(index);
                }
            }
        }
    }

    pub(super) fn abort(&mut self) {
        self.0.clear();
        #[cfg(feature = "wgpu")]
        {
            self.clear_route_audit();
            self.2 = None;
        }
    }
}

// Dropping on error, cancellation, early return, or an unpolled future retires
// VM-bound tickets without positive feedback. Legacy observations retain their
// existing failure reporting; unfinished traces record fallback only.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_execution_of_final_instruction_completes_a_region() {
        let span = InstrSpan { start: 4, end: 8 };
        assert_eq!(region_exit(&span, 8, 9), RegionExit::Completed);
        assert_eq!(region_exit(&span, 7, 9), RegionExit::Aborted);
        assert_eq!(region_exit(&span, 8, 10), RegionExit::Aborted);
        assert_eq!(region_exit(&span, 6, 3), RegionExit::Aborted);
        assert_eq!(region_exit(&span, 8, 4), RegionExit::Running);
        assert_eq!(region_exit(&span, 4, 5), RegionExit::Running);
    }

    #[test]
    fn nested_and_repeated_regions_have_independent_lifetimes() {
        use runmat_accelerate::placement::begin_trace;
        let mut observations = FusionRegionObservations::default();
        observations.begin(
            InstrSpan { start: 4, end: 8 },
            None,
            begin_trace("outer", None),
        );
        observations.begin(
            InstrSpan { start: 5, end: 6 },
            None,
            begin_trace("inner", None),
        );
        observations.instruction_exit(6, 7);
        assert_eq!(observations.0.len(), 1);
        assert!(observations.contains_start(4));
        observations.instruction_exit(8, 9);
        assert!(observations.0.is_empty());
        observations.begin(
            InstrSpan { start: 4, end: 8 },
            None,
            begin_trace("repeated", None),
        );
        observations.instruction_exit(8, 9);
        assert!(observations.0.is_empty());
    }

    #[test]
    fn caught_error_or_cancellation_aborts_all_active_regions() {
        use runmat_accelerate::placement::begin_trace;
        let mut observations = FusionRegionObservations::default();
        observations.begin(
            InstrSpan { start: 4, end: 8 },
            None,
            begin_trace("cancelled", None),
        );
        observations.abort();
        observations.instruction_exit(8, 9);
        assert!(observations.0.is_empty());
    }
}
