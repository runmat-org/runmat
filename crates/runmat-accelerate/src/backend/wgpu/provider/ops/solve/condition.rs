//! Factor-retaining square `linsolve` condition readback. The solution remains
//! device-resident; only requested condition evidence crosses to the host.

use super::*;
use runmat_runtime::builtins::math::linalg::solve::linsolve::{
    square_factor_rcond_for_provider, ProviderSquareFactor,
};

pub(super) enum SquareDeviceFactor<'a> {
    Qr {
        q: &'a GpuTensorHandle,
        r: &'a GpuTensorHandle,
    },
    Cholesky {
        r: &'a GpuTensorHandle,
    },
}

impl WgpuProvider {
    pub(super) async fn square_rcond_from_factors(
        &self,
        effective_matrix: &GpuTensorHandle,
        factor: SquareDeviceFactor<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<f64> {
        // The QR branch passes its effective (possibly transposed) A; POSDEF
        // passes the original symmetric A. These readbacks happen only for a
        // requested second output or an explicit RCOND threshold.
        let matrix = self
            .host_tensor_from_handle("linsolve_square_condition_matrix", effective_matrix)
            .await?;
        let estimate = match factor {
            SquareDeviceFactor::Qr { q, r } => {
                let q = self
                    .host_tensor_from_handle("linsolve_square_condition_q", q)
                    .await?;
                let r = self
                    .host_tensor_from_handle("linsolve_square_condition_r", r)
                    .await?;
                square_factor_rcond_for_provider(&matrix, ProviderSquareFactor::Qr { q: &q, r: &r })
            }
            SquareDeviceFactor::Cholesky { r } => {
                let r = self
                    .host_tensor_from_handle("linsolve_square_condition_r", r)
                    .await?;
                square_factor_rcond_for_provider(&matrix, ProviderSquareFactor::Cholesky { r: &r })
            }
        };
        estimate.map_err(runtime_flow_to_provider_error)
    }
}
