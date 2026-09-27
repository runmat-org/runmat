//! Region observations live across bytecode dispatch, not just fusion selection.
use runmat_accelerate::fusion_exec::FusionCpuObservation;
use runmat_accelerate::placement::{PlacementTraceContext, PlacementVariant};
use runmat_accelerate::InstrSpan;
use runmat_time::{duration_ns_saturating, Instant};

#[derive(Default)]
pub(super) struct FusionRegionObservations(Vec<RegionObservation>);

struct RegionObservation {
    span: InstrSpan,
    ticket: Option<FusionCpuObservation>,
    trace: Option<PlacementTraceContext>,
    started: Instant,
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
    pub(super) fn contains_start(&self, pc: usize) -> bool {
        self.0
            .iter()
            .any(|observation| observation.span.start == pc)
    }

    pub(super) fn begin(
        &mut self,
        span: InstrSpan,
        ticket: Option<FusionCpuObservation>,
        trace: PlacementTraceContext,
    ) {
        self.0.push(RegionObservation {
            span,
            ticket,
            trace: Some(trace),
            started: Instant::now(),
        });
    }

    pub(super) fn instruction_exit(&mut self, executed_pc: usize, next_pc: usize) {
        let mut index = 0;
        while index < self.0.len() {
            match region_exit(&self.0[index].span, executed_pc, next_pc) {
                RegionExit::Running => index += 1,
                RegionExit::Completed => {
                    let mut observation = self.0.remove(index);
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
                RegionExit::Aborted => {
                    self.0.remove(index);
                }
            }
        }
    }

    pub(super) fn abort(&mut self) {
        self.0.clear();
    }
}

// Dropping the scope (error, cancellation, early return, or dropped future)
// drops unfinished tickets as failures and unfinished traces as fallbacks.

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
