use futures::executor::block_on;
use runmat_accelerate_api::GpuTensorHandle;

pub(super) struct DeviceOperatorContext<'a> {
    pub(super) provider: &'a dyn runmat_accelerate_api::AccelProvider,
    pub(super) diag: &'a GpuTensorHandle,
    pub(super) upper_left: &'a GpuTensorHandle,
    pub(super) upper_right: &'a GpuTensorHandle,
    pub(super) constrained_mask: &'a GpuTensorHandle,
    pub(super) unconstrained_mask: &'a GpuTensorHandle,
    pub(super) prev_indices: &'a [u32],
    pub(super) next_indices: &'a [u32],
    pub(super) shape: &'a [usize],
}

pub(super) fn apply_k_device(
    ctx: &DeviceOperatorContext<'_>,
    x: &GpuTensorHandle,
) -> Option<GpuTensorHandle> {
    let x_prev = ctx
        .provider
        .gather_linear(x, ctx.prev_indices, ctx.shape)
        .ok()?;
    let x_next = ctx
        .provider
        .gather_linear(x, ctx.next_indices, ctx.shape)
        .ok()?;

    let diag_term = block_on(ctx.provider.elem_mul(ctx.diag, x)).ok()?;
    let left_term = block_on(ctx.provider.elem_mul(ctx.upper_left, &x_prev)).ok()?;
    let right_term = block_on(ctx.provider.elem_mul(ctx.upper_right, &x_next)).ok()?;
    let tmp = block_on(ctx.provider.elem_sub(&diag_term, &left_term)).ok()?;
    let unconstrained_value = block_on(ctx.provider.elem_sub(&tmp, &right_term)).ok()?;

    let unconstrained_part = block_on(
        ctx.provider
            .elem_mul(ctx.unconstrained_mask, &unconstrained_value),
    )
    .ok()?;
    let constrained_part = block_on(ctx.provider.elem_mul(ctx.constrained_mask, x)).ok()?;
    let y = block_on(
        ctx.provider
            .elem_add(&unconstrained_part, &constrained_part),
    )
    .ok()?;

    let _ = ctx.provider.free(&constrained_part);
    let _ = ctx.provider.free(&unconstrained_part);
    let _ = ctx.provider.free(&unconstrained_value);
    let _ = ctx.provider.free(&tmp);
    let _ = ctx.provider.free(&right_term);
    let _ = ctx.provider.free(&left_term);
    let _ = ctx.provider.free(&diag_term);
    let _ = ctx.provider.free(&x_next);
    let _ = ctx.provider.free(&x_prev);
    Some(y)
}

pub(super) fn apply_k_host_from_prepared(
    diag: &[f64],
    upper_left: &[f64],
    upper_right: &[f64],
    constrained_mask: &[f64],
    unconstrained_mask: &[f64],
    x: &[f64],
) -> Vec<f64> {
    let n = x.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let prev = if i == 0 { x[0] } else { x[i - 1] };
        let next = if i + 1 >= n { x[n - 1] } else { x[i + 1] };
        let unconstrained_value = diag[i] * x[i] - upper_left[i] * prev - upper_right[i] * next;
        y[i] = unconstrained_mask[i] * unconstrained_value + constrained_mask[i] * x[i];
    }
    y
}

pub(super) fn linear_shift_indices(n: usize, shift: isize) -> Option<Vec<u32>> {
    if n > u32::MAX as usize {
        return None;
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let shifted = (i as isize) + shift;
        let index = if shifted < 0 {
            0
        } else if shifted >= n as isize {
            n.saturating_sub(1)
        } else {
            shifted as usize
        };
        out.push(index as u32);
    }
    Some(out)
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
