use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearAlgebraBackendKind {
    CpuReference,
    RuntimeTensor,
}

impl LinearAlgebraBackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            LinearAlgebraBackendKind::CpuReference => "cpu_reference",
            LinearAlgebraBackendKind::RuntimeTensor => "runtime_tensor",
        }
    }
}

impl std::fmt::Display for LinearAlgebraBackendKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}
