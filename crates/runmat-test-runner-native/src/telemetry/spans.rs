use runmat_test_runner::telemetry::{TelemetryFields, TelemetryPort};

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeTelemetry;

impl TelemetryPort for NativeTelemetry {
    fn event(&self, name: &'static str, fields: &TelemetryFields) {
        log::debug!(target: "runmat::test", "{name} {:?}", fields.values);
    }
}
