use futures::executor::block_on;

use runmat_accelerate_api::{provider, HostTensorView};

use super::{cpu_reference::CpuReferenceBackend, linear_algebra::LinearAlgebraBackend};

#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeTensorBackend;

impl LinearAlgebraBackend for RuntimeTensorBackend {
    fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        if let Some(value) = try_dot_gpu(a, b) {
            return value;
        }
        CpuReferenceBackend.dot(a, b)
    }

    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        if let Some(out) = try_axpy_gpu(alpha, x, y) {
            if out.len() == y.len() {
                y.copy_from_slice(&out);
                return;
            }
        }
        CpuReferenceBackend.axpy(alpha, x, y)
    }

    fn vec_sub(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        if let Some(out) = try_vec_sub_gpu(a, b) {
            return out;
        }
        CpuReferenceBackend.vec_sub(a, b)
    }
}

fn try_dot_gpu(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let provider = provider()?;
    let shape = [a.len()];
    let ah = provider
        .upload(&HostTensorView {
            data: a,
            shape: &shape,
        })
        .ok()?;
    let bh = provider
        .upload(&HostTensorView {
            data: b,
            shape: &shape,
        })
        .ok()?;
    let mul = block_on(provider.elem_mul(&ah, &bh)).ok()?;
    let sum = block_on(provider.reduce_sum(&mul)).ok()?;

    let scalar = match provider.read_scalar(&sum, 0) {
        Ok(value) => Some(value),
        Err(_) => block_on(provider.download(&sum))
            .ok()
            .and_then(|host| host.data.first().copied()),
    };

    let _ = provider.free(&sum);
    let _ = provider.free(&mul);
    let _ = provider.free(&bh);
    let _ = provider.free(&ah);
    scalar
}

fn try_axpy_gpu(alpha: f64, x: &[f64], y: &[f64]) -> Option<Vec<f64>> {
    if x.len() != y.len() {
        return None;
    }
    let provider = provider()?;
    let shape = [x.len()];
    let xh = provider
        .upload(&HostTensorView {
            data: x,
            shape: &shape,
        })
        .ok()?;
    let yh = provider
        .upload(&HostTensorView {
            data: y,
            shape: &shape,
        })
        .ok()?;
    let scaled = provider.scalar_mul(&xh, alpha).ok()?;
    let out_h = block_on(provider.elem_add(&scaled, &yh)).ok()?;
    let host = block_on(provider.download(&out_h)).ok()?;

    let _ = provider.free(&out_h);
    let _ = provider.free(&scaled);
    let _ = provider.free(&yh);
    let _ = provider.free(&xh);
    Some(host.data)
}

fn try_vec_sub_gpu(a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    if a.len() != b.len() {
        return None;
    }
    let provider = provider()?;
    let shape = [a.len()];
    let ah = provider
        .upload(&HostTensorView {
            data: a,
            shape: &shape,
        })
        .ok()?;
    let bh = provider
        .upload(&HostTensorView {
            data: b,
            shape: &shape,
        })
        .ok()?;
    let out_h = block_on(provider.elem_sub(&ah, &bh)).ok()?;
    let host = block_on(provider.download(&out_h)).ok()?;

    let _ = provider.free(&out_h);
    let _ = provider.free(&bh);
    let _ = provider.free(&ah);
    Some(host.data)
}
