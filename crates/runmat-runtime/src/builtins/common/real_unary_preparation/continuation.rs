//! Physical publication policy supplied by the original numerical family.
//! This does not classify values, authorize a kernel, or certify completion.
use runmat_accelerate_api::{GpuHandleProvenance, GpuTensorHandle};
use runmat_types::NumericDomain;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum RealUnaryPublication {
    Restore,
    /// Log2's legacy/automatic host fallback is not explicitly resident.
    NonExplicitHost,
    /// Inverse families keep automatic complex results on host, but retain
    /// the historical resident policy for unannotated/explicit handles.
    AutomaticComplexHost,
}

impl RealUnaryPublication {
    pub(crate) fn publishes_host(self, source: &GpuTensorHandle, domain: NumericDomain) -> bool {
        match self {
            Self::Restore => false,
            Self::NonExplicitHost => !runmat_accelerate_api::handle_is_explicit(source),
            Self::AutomaticComplexHost => {
                domain == NumericDomain::Complex
                    && runmat_accelerate_api::handle_provenance(source)
                        == Some(GpuHandleProvenance::Automatic)
            }
        }
    }
}
