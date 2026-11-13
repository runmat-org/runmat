//! The grid test double answers no exact query: hidden-line projection, proximity and STEP
//! export need an exact kernel (OCCT), and the double says so instead of approximating.

use super::FakeKernel;
use runmat_geometry_kernel::op::KernelError;
use runmat_geometry_kernel::queries::{
    ExactQueries, ExchangeDocument, ExchangeFormat, Placed, ProjectedCurve, ProjectionRequest,
    Proximity,
};

fn unsupported(what: &str) -> KernelError {
    KernelError::NotSupported {
        what: format!("{what} on the grid test kernel"),
    }
}

impl ExactQueries for FakeKernel {
    fn project(
        &self,
        _: &[Placed<'_>],
        _: &ProjectionRequest,
    ) -> Result<Vec<ProjectedCurve>, KernelError> {
        Err(unsupported("exact hidden-line projection"))
    }
    fn proximity(&self, _: &Placed<'_>, _: &Placed<'_>) -> Result<Proximity, KernelError> {
        Err(unsupported("exact proximity"))
    }
    fn export(&self, _: &ExchangeDocument<'_>, _: ExchangeFormat) -> Result<Vec<u8>, KernelError> {
        Err(unsupported("STEP export"))
    }
}
