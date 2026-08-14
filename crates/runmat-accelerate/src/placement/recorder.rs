use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use once_cell::sync::Lazy;
use runmat_time::{duration_ns_saturating, Instant};
use runmat_types::RegionId;

use super::{
    PlacementAttribute, PlacementCorrelationId, PlacementEvent, PlacementEventKind,
    PlacementReport, PlacementTrace, PlacementVariant, PLACEMENT_REPORT_SCHEMA_VERSION,
};

const TRACE_CAPACITY: usize = 256;
const EVENT_CAPACITY: usize = 64;
const OPERATION_MAX_BYTES: usize = 128;
const REASON_MAX_BYTES: usize = 128;
const ATTRIBUTE_CAPACITY: usize = 16;

struct ActiveTrace {
    started: Instant,
    trace: PlacementTrace,
}

struct RecorderState {
    next_correlation: u64,
    active: HashMap<PlacementCorrelationId, ActiveTrace>,
    completed: VecDeque<PlacementTrace>,
    dropped_traces: u64,
}

impl Default for RecorderState {
    fn default() -> Self {
        Self {
            next_correlation: 1,
            active: HashMap::new(),
            completed: VecDeque::new(),
            dropped_traces: 0,
        }
    }
}

static RECORDER: Lazy<Mutex<RecorderState>> = Lazy::new(|| Mutex::new(RecorderState::default()));

#[derive(Debug)]
pub struct PlacementTraceContext {
    correlation: PlacementCorrelationId,
    enabled: bool,
    finished: bool,
}

impl PlacementTraceContext {
    pub fn correlation(&self) -> PlacementCorrelationId {
        self.correlation
    }

    pub fn event(
        &self,
        kind: PlacementEventKind,
        variant: Option<PlacementVariant>,
        reason: Option<&str>,
        duration_ns: Option<u64>,
        bytes: Option<u64>,
        attributes: &[PlacementAttribute],
    ) {
        if !self.enabled {
            return;
        }
        record_event(
            self.correlation,
            kind,
            variant,
            reason,
            duration_ns,
            bytes,
            attributes,
        );
    }

    pub fn complete(mut self, variant: Option<PlacementVariant>, duration_ns: Option<u64>) {
        if !self.enabled {
            self.finished = true;
            return;
        }
        record_event(
            self.correlation,
            PlacementEventKind::Complete,
            variant,
            None,
            duration_ns,
            None,
            &[],
        );
        complete_trace(self.correlation);
        self.finished = true;
    }

    pub fn fallback(
        mut self,
        variant: Option<PlacementVariant>,
        reason: &str,
        duration_ns: Option<u64>,
    ) {
        if !self.enabled {
            self.finished = true;
            return;
        }
        record_event(
            self.correlation,
            PlacementEventKind::Fallback,
            variant,
            Some(reason),
            duration_ns,
            None,
            &[],
        );
        complete_trace(self.correlation);
        self.finished = true;
    }
}

impl Drop for PlacementTraceContext {
    fn drop(&mut self) {
        if self.enabled && !self.finished {
            record_event(
                self.correlation,
                PlacementEventKind::Fallback,
                None,
                Some("trace_scope_dropped"),
                None,
                None,
                &[],
            );
            complete_trace(self.correlation);
        }
    }
}

pub fn begin_trace(operation: &str, region: Option<RegionId>) -> PlacementTraceContext {
    let operation = bounded_token(operation, OPERATION_MAX_BYTES, "unknown");
    let mut state = RECORDER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    while state.active.len().saturating_add(state.completed.len()) >= TRACE_CAPACITY {
        if state.completed.pop_front().is_some() {
            state.dropped_traces = state.dropped_traces.saturating_add(1);
        } else {
            break;
        }
    }
    if state.active.len() >= TRACE_CAPACITY {
        state.dropped_traces = state.dropped_traces.saturating_add(1);
        return PlacementTraceContext {
            correlation: PlacementCorrelationId(0),
            enabled: false,
            finished: false,
        };
    }
    let correlation = PlacementCorrelationId(state.next_correlation.max(1));
    state.next_correlation = state.next_correlation.saturating_add(1).max(1);
    state.active.insert(
        correlation,
        ActiveTrace {
            started: Instant::now(),
            trace: PlacementTrace {
                correlation,
                region,
                operation,
                events: Vec::new(),
                dropped_events: 0,
                complete: false,
            },
        },
    );
    PlacementTraceContext {
        correlation,
        enabled: true,
        finished: false,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn record_event(
    correlation: PlacementCorrelationId,
    kind: PlacementEventKind,
    variant: Option<PlacementVariant>,
    reason: Option<&str>,
    duration_ns: Option<u64>,
    bytes: Option<u64>,
    attributes: &[PlacementAttribute],
) {
    let mut state = RECORDER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(active) = state.active.get_mut(&correlation) else {
        return;
    };
    if active.trace.events.len() >= EVENT_CAPACITY {
        active.trace.dropped_events = active.trace.dropped_events.saturating_add(1);
        return;
    }
    let sequence = u32::try_from(active.trace.events.len()).unwrap_or(u32::MAX);
    active.trace.events.push(PlacementEvent {
        sequence,
        elapsed_ns: duration_ns_saturating(active.started.elapsed()),
        kind,
        variant,
        reason: reason.map(|reason| bounded_token(reason, REASON_MAX_BYTES, "unknown")),
        duration_ns,
        bytes,
        attributes: attributes
            .iter()
            .take(ATTRIBUTE_CAPACITY)
            .cloned()
            .collect(),
    });
}

pub fn complete_trace(correlation: PlacementCorrelationId) {
    let mut state = RECORDER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(mut active) = state.active.remove(&correlation) else {
        return;
    };
    active.trace.complete = true;
    while state.completed.len() >= TRACE_CAPACITY {
        state.completed.pop_front();
        state.dropped_traces = state.dropped_traces.saturating_add(1);
    }
    state.completed.push_back(active.trace);
}

pub fn report() -> PlacementReport {
    let state = RECORDER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut traces = state.completed.iter().cloned().collect::<Vec<_>>();
    traces.extend(state.active.values().map(|active| active.trace.clone()));
    traces.sort_by_key(|trace| trace.correlation);
    PlacementReport {
        schema_version: PLACEMENT_REPORT_SCHEMA_VERSION,
        trace_capacity: TRACE_CAPACITY,
        event_capacity: EVENT_CAPACITY,
        traces,
        dropped_traces: state.dropped_traces,
    }
}

pub fn reset() {
    *RECORDER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = RecorderState::default();
}

fn bounded_token(value: &str, max_bytes: usize, fallback: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        return fallback.to_string();
    }
    let mut end = value.len().min(max_bytes);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn correlated_reports_are_bounded_ordered_and_redacted_by_contract() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset();
        let trace = begin_trace("elementwise", None);
        trace.event(
            PlacementEventKind::Candidate,
            Some(PlacementVariant::SharedRuntime),
            Some("legacy_threshold"),
            Some(7),
            None,
            &[PlacementAttribute {
                key: "elements".to_string(),
                value: 42,
            }],
        );
        trace.complete(Some(PlacementVariant::SharedRuntime), Some(9));

        let report = report();
        assert_eq!(report.schema_version, PLACEMENT_REPORT_SCHEMA_VERSION);
        assert_eq!(report.traces.len(), 1);
        assert_eq!(report.traces[0].events.len(), 2);
        assert_eq!(report.traces[0].events[0].sequence, 0);
        assert_eq!(report.traces[0].events[1].sequence, 1);
        assert!(report.traces[0].complete);
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains("tensor_contents"));
        assert!(report.render_text().contains("elementwise"));
    }

    #[test]
    fn event_overflow_is_counted_without_growing_the_trace() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset();
        let trace = begin_trace("bounded", None);
        for _ in 0..EVENT_CAPACITY + 5 {
            trace.event(PlacementEventKind::Queue, None, None, None, None, &[]);
        }
        let correlation = trace.correlation();
        drop(trace);
        let report = report();
        let trace = report
            .traces
            .iter()
            .find(|trace| trace.correlation == correlation)
            .unwrap();
        assert_eq!(trace.events.len(), EVENT_CAPACITY);
        assert_eq!(trace.dropped_events, 6);
    }

    #[test]
    fn active_trace_overflow_is_counted_without_growing_the_report() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset();
        let traces = (0..TRACE_CAPACITY + 3)
            .map(|_| begin_trace("bounded-active", None))
            .collect::<Vec<_>>();

        let report = report();
        assert_eq!(report.traces.len(), TRACE_CAPACITY);
        assert_eq!(report.dropped_traces, 3);
        drop(traces);
    }

    #[test]
    fn completed_and_active_traces_share_one_capacity() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset();
        for _ in 0..TRACE_CAPACITY {
            begin_trace("complete", None).complete(None, None);
        }

        let active = begin_trace("active", None);
        let report = report();
        assert_eq!(report.traces.len(), TRACE_CAPACITY);
        assert_eq!(report.dropped_traces, 1);
        assert!(report
            .traces
            .iter()
            .any(|trace| trace.correlation == active.correlation()));
        drop(active);
    }
}
