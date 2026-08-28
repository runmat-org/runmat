use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TelemetryFields {
    pub values: BTreeMap<String, String>,
}

impl TelemetryFields {
    pub fn public(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.values.insert(key.into(), value.into());
        self
    }
}
