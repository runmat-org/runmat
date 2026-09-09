use futures::executor::block_on;
use runmat_accelerate_api::GpuTensorHandle;

use crate::algebra::RealSparseMatrix;

pub(super) struct CsrDeviceOperatorContext<'a> {
    pub(super) matrix: &'a RealSparseMatrix,
    pub(super) values: &'a GpuTensorHandle,
}

pub(super) struct DeviceOperatorContext<'a> {
    pub(super) provider: &'a dyn runmat_accelerate_api::AccelProvider,
    pub(super) csr: CsrDeviceOperatorContext<'a>,
}

pub(super) fn apply_k_device(
    ctx: &DeviceOperatorContext<'_>,
    x: &GpuTensorHandle,
) -> Option<GpuTensorHandle> {
    apply_csr_device(ctx.provider, &ctx.csr, x)
}

fn apply_csr_device(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    csr: &CsrDeviceOperatorContext<'_>,
    x: &GpuTensorHandle,
) -> Option<GpuTensorHandle> {
    let output = provider.scalar_mul(x, 0.0).ok()?;
    for row in 0..csr.matrix.rows() {
        let range = csr.matrix.pattern().row_range(row);
        let start = range.start;
        let end = range.end;
        if start == end {
            continue;
        }
        let columns = (start..end)
            .map(|entry| u32::try_from(csr.matrix.pattern().column(entry)).ok())
            .collect::<Option<Vec<_>>>()?;
        let entries = (start..end)
            .map(|entry| u32::try_from(entry).ok())
            .collect::<Option<Vec<_>>>()?;
        let row_shape = [end - start];
        let x_row = provider.gather_linear(x, &columns, &row_shape).ok()?;
        let values_row = provider
            .gather_linear(csr.values, &entries, &row_shape)
            .ok()?;
        let products = block_on(provider.elem_mul(&x_row, &values_row)).ok()?;
        let sum = block_on(provider.reduce_sum(&products)).ok()?;
        provider
            .scatter_linear(&output, &[u32::try_from(row).ok()?], &sum)
            .ok()?;
        let _ = provider.free(&sum);
        let _ = provider.free(&products);
        let _ = provider.free(&values_row);
        let _ = provider.free(&x_row);
    }
    Some(output)
}

pub(super) fn dot_handle(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    a: &GpuTensorHandle,
    b: &GpuTensorHandle,
    host_sync_count: &mut u32,
) -> Option<f64> {
    let mul = block_on(provider.elem_mul(a, b)).ok()?;
    let sum = block_on(provider.reduce_sum(&mul)).ok()?;
    let out = match provider.read_scalar(&sum, 0) {
        Ok(value) => Some(value),
        Err(_) => {
            *host_sync_count = host_sync_count.saturating_add(1);
            block_on(provider.download(&sum))
                .ok()
                .and_then(|host| host.data.first().copied())
        }
    };
    let _ = provider.free(&sum);
    let _ = provider.free(&mul);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_csr_constraints_form_identity_rows_and_zero_columns() {
        let csr = RealSparseMatrix::from_dense(
            &[4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 4.0],
            3,
            crate::algebra::OperatorRole::Stiffness,
            crate::algebra::MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let normalized = csr
            .with_identity_constraints(&[true, false, false])
            .unwrap();
        assert_eq!(
            normalized.to_dense(),
            vec![1.0, 0.0, 0.0, 0.0, 4.0, -1.0, 0.0, -1.0, 4.0]
        );
    }
}
