//! Observe shared recovery progress without adding a native-host dependency.
use tracing::{span, Event, Metadata, Subscriber};

pub(in crate::cdt) struct RecoveryDiagnostics;
impl RecoveryDiagnostics {
    pub(in crate::cdt) fn segment_recovery() -> Self {
        Self
    }
}

impl Subscriber for RecoveryDiagnostics {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.target() == "runmat::meshing::segment_recovery"
    }

    fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
        span::Id::from_u64(1)
    }
    fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
    fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
    fn event(&self, event: &Event<'_>) {
        let mut fields = Fields(String::new());
        event.record(&mut fields);
        eprintln!(
            "case={} {}",
            std::thread::current().name().unwrap_or("unnamed"),
            fields.0
        );
    }
    fn enter(&self, _: &span::Id) {}
    fn exit(&self, _: &span::Id) {}
}

struct Fields(String);
impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write;
        let _ = write!(&mut self.0, "{}={value:?} ", field.name());
    }
}
