#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
