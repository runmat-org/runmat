use runmat_types::RegionId;
use serde::{Deserialize, Serialize};

pub const PLACEMENT_REPORT_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PlacementCorrelationId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementVariant {
    SharedRuntime,
    GenericNativeCpu,
    SpecializedNativeCpu,
    ProviderOperation,
    ProviderLibrary,
    ProviderGraph,
    ProviderFusion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementEventKind {
    Candidate,
    Selected,
    Compile,
    Prepare,
    Upload,
    Queue,
    Kernel,
    Synchronize,
    Download,
    Complete,
    Fallback,
}

/// One bounded, low-cardinality numeric attribute.
///
/// Placement observation intentionally excludes source text, paths, tensor
/// contents, user identifiers, and arbitrary error messages. Keys and labels
/// come from compiler/provider-owned stable vocabularies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementAttribute {
    pub key: String,
    pub value: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementEvent {
    pub sequence: u32,
    pub elapsed_ns: u64,
    pub kind: PlacementEventKind,
    pub variant: Option<PlacementVariant>,
    pub reason: Option<String>,
    pub duration_ns: Option<u64>,
    pub bytes: Option<u64>,
    pub attributes: Vec<PlacementAttribute>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementTrace {
    pub correlation: PlacementCorrelationId,
    pub region: Option<RegionId>,
    pub operation: String,
    pub events: Vec<PlacementEvent>,
    pub dropped_events: u64,
    pub complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementReport {
    pub schema_version: u16,
    pub trace_capacity: usize,
    pub event_capacity: usize,
    pub traces: Vec<PlacementTrace>,
    pub dropped_traces: u64,
}

impl PlacementReport {
    pub fn render_text(&self) -> String {
        let mut lines = vec![format!(
            "placement report v{}: {} trace(s), {} dropped",
            self.schema_version,
            self.traces.len(),
            self.dropped_traces
        )];
        for trace in &self.traces {
            lines.push(format!(
                "#{} {} region={} events={} dropped={} complete={}",
                trace.correlation.0,
                trace.operation,
                trace
                    .region
                    .map(|region| format!("{}:{}", region.function.0, region.ordinal))
                    .unwrap_or_else(|| "legacy".to_string()),
                trace.events.len(),
                trace.dropped_events,
                trace.complete
            ));
            for event in &trace.events {
                lines.push(format!(
                    "  {:02} +{}ns {:?} variant={:?} reason={} duration_ns={} bytes={}",
                    event.sequence,
                    event.elapsed_ns,
                    event.kind,
                    event.variant,
                    event.reason.as_deref().unwrap_or("-"),
                    event
                        .duration_ns
                        .map_or_else(|| "-".to_string(), |value| value.to_string()),
                    event
                        .bytes
                        .map_or_else(|| "-".to_string(), |value| value.to_string())
                ));
            }
        }
        lines.join("\n")
    }
}
