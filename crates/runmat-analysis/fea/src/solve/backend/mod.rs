pub mod cpu_reference;
pub mod kind;
pub mod linear_algebra;

use cpu_reference::CpuReferenceBackend;
use kind::LinearAlgebraBackendKind;
use linear_algebra::LinearAlgebraBackend;

pub fn build_backend(kind: LinearAlgebraBackendKind) -> Box<dyn LinearAlgebraBackend> {
    match kind {
        LinearAlgebraBackendKind::CpuReference => Box::new(CpuReferenceBackend),
        // Solver-owned accelerator paths intercept this selection before numerical work. Families
        // without an admitted native provider path execute their explicitly reported CPU fallback
        // through this reference algebra implementation.
        LinearAlgebraBackendKind::Accelerator => Box::new(CpuReferenceBackend),
    }
}
