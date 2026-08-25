use runmat_types::ForeignCapability;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForeignTelemetryOutcome {
    Admitted,
    Rejected,
    Succeeded,
    Failed,
    Cancelled,
    Released,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignTelemetryEvent {
    pub adapter: String,
    pub operation: String,
    pub capability: Option<ForeignCapability>,
    pub isolated: bool,
    pub outcome: ForeignTelemetryOutcome,
    pub error_identifier: Option<String>,
}

pub trait ForeignTelemetrySink {
    fn record(&self, event: ForeignTelemetryEvent);
}

#[derive(Debug, Default)]
pub struct NoopForeignTelemetry;

impl ForeignTelemetrySink for NoopForeignTelemetry {
    fn record(&self, _event: ForeignTelemetryEvent) {}
}
