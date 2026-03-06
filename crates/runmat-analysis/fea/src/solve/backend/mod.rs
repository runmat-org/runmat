pub mod cpu_reference;
pub mod kind;
pub mod linear_algebra;
pub mod runtime_tensor;

use cpu_reference::CpuReferenceBackend;
use kind::LinearAlgebraBackendKind;
use linear_algebra::LinearAlgebraBackend;
use runtime_tensor::RuntimeTensorBackend;

pub fn build_backend(kind: LinearAlgebraBackendKind) -> Box<dyn LinearAlgebraBackend> {
    match kind {
        LinearAlgebraBackendKind::CpuReference => Box::new(CpuReferenceBackend),
        LinearAlgebraBackendKind::RuntimeTensor => Box::new(RuntimeTensorBackend),
    }
}
