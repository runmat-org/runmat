use serde::{Deserialize, Serialize};

pub(crate) mod builders;
pub(crate) mod material_assignment;
pub(crate) mod thermal;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaDiagnosticSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct FeaDiagnostic {
    pub code: String,
    pub severity: FeaDiagnosticSeverity,
    pub message: String,
}

#[derive(Serialize)]
#[serde(rename = "FeaDiagnostic")]
pub(crate) struct DiagnosticSerialization<'a, T: Serialize> {
    pub(crate) code: &'a str,
    pub(crate) severity: FeaDiagnosticSeverity,
    pub(crate) message: T,
}

impl Serialize for FeaDiagnostic {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        DiagnosticSerialization {
            code: &self.code,
            severity: self.severity,
            message: &self.message,
        }
        .serialize(serializer)
    }
}
