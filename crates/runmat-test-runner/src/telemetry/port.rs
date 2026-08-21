use super::TelemetryFields;

pub trait TelemetryPort {
    fn event(&self, name: &'static str, fields: &TelemetryFields);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoopTelemetry;

impl TelemetryPort for NoopTelemetry {
    fn event(&self, _name: &'static str, _fields: &TelemetryFields) {}
}
