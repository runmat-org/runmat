use super::*;

pub(in crate::backend::wgpu::provider::backend) fn elu_shader(
    precision: NumericPrecision,
) -> &'static str {
    match precision {
        NumericPrecision::F64 => ELU_SHADER_F64,
        NumericPrecision::F32 => ELU_SHADER_F32,
    }
}

impl WgpuProvider {
    pub(crate) fn activation_adjoint_exec(
        &self,
        input: &GpuTensorHandle,
        upstream: &GpuTensorHandle,
        operation: runmat_accelerate_api::ProviderActivationAdjoint,
    ) -> Result<GpuTensorHandle> {
        let x = self.get_entry(input)?;
        let dy = self.get_entry(upstream)?;
        ensure!(
            x.storage == GpuTensorStorage::Real
                && dy.storage == GpuTensorStorage::Real
                && x.shape == dy.shape
                && x.len == dy.len
                && runmat_accelerate_api::handle_precision(input)
                    == runmat_accelerate_api::handle_precision(upstream),
            "activation_adjoint: input and upstream must have the same real shape and precision"
        );
        let shader = match (self.precision, operation) {
            (NumericPrecision::F32, runmat_accelerate_api::ProviderActivationAdjoint::Relu) => {
                std::borrow::Cow::Borrowed(RELU_ADJOINT_F32)
            }
            (NumericPrecision::F64, runmat_accelerate_api::ProviderActivationAdjoint::Relu) => {
                self.require_ieee_words(self.precision)?;
                std::borrow::Cow::Owned(crate::backend::wgpu::shaders::ieee_words::inject(
                    RELU_ADJOINT_F64,
                ))
            }
            (
                NumericPrecision::F32,
                runmat_accelerate_api::ProviderActivationAdjoint::Elu { .. },
            ) => std::borrow::Cow::Borrowed(ELU_ADJOINT_F32),
            (
                NumericPrecision::F64,
                runmat_accelerate_api::ProviderActivationAdjoint::Elu { .. },
            ) => std::borrow::Cow::Borrowed(ELU_ADJOINT_F64),
        };
        match operation {
            runmat_accelerate_api::ProviderActivationAdjoint::Relu => self
                .fused_elementwise_with_telemetry_exec(
                    &shader,
                    &[input.clone(), upstream.clone()],
                    &x.shape,
                    x.len,
                ),
            runmat_accelerate_api::ProviderActivationAdjoint::Elu { alpha } => {
                ensure!(
                    alpha.is_finite() && alpha >= 0.0,
                    "activation_adjoint: invalid Alpha"
                );
                if self.precision == NumericPrecision::F32 {
                    ensure!(
                        (alpha as f32).is_finite(),
                        "activation_adjoint: Alpha exceeds single precision"
                    );
                }
                let alpha_handle = self.fill_exec(&x.shape, alpha)?;
                let result = self.fused_elementwise_with_telemetry_exec(
                    &shader,
                    &[input.clone(), upstream.clone(), alpha_handle.clone()],
                    &x.shape,
                    x.len,
                );
                drop(alpha_handle);
                result
            }
        }
    }
    pub(crate) fn relu_exec(&self, input: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(input)?;
        ensure!(
            entry.storage == GpuTensorStorage::Real
                && runmat_accelerate_api::handle_storage(input) == GpuTensorStorage::Real,
            "activation_relu: real storage is required"
        );
        let shader = match self.precision {
            NumericPrecision::F64 => RELU_SHADER_F64,
            NumericPrecision::F32 => RELU_SHADER_F32,
        };
        self.fused_elementwise_with_telemetry_exec(
            shader,
            std::slice::from_ref(input),
            &entry.shape,
            entry.len,
        )
    }

    pub(crate) fn elu_exec(&self, input: &GpuTensorHandle, alpha: f64) -> Result<GpuTensorHandle> {
        ensure!(
            alpha.is_finite()
                && alpha >= 0.0
                && (self.precision != NumericPrecision::F32 || (alpha as f32).is_finite()),
            "activation_elu: invalid alpha for device precision"
        );
        let entry = self.get_entry(input)?;
        ensure!(
            entry.storage != GpuTensorStorage::ComplexInterleaved
                && runmat_accelerate_api::handle_storage(input)
                    != GpuTensorStorage::ComplexInterleaved,
            "activation_elu: complex inputs are not supported"
        );
        let alpha_tensor = self.fill_exec(&entry.shape, alpha)?;
        let shader = match self.precision {
            NumericPrecision::F64 => ELU_SHADER_F64,
            NumericPrecision::F32 => ELU_SHADER_F32,
        };
        let result = self.fused_elementwise_with_telemetry_exec(
            shader,
            &[input.clone(), alpha_tensor.clone()],
            &entry.shape,
            entry.len,
        );
        drop(alpha_tensor);
        result
    }
}

const RELU_SHADER_F64: &str = r#"
struct Tensor { data: array<f64> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let value = input0.data[idx];
  output.data[idx] = select(value, f64(0.0), value <= f64(0.0));
}
"#;

const RELU_SHADER_F32: &str = r#"
struct Tensor { data: array<f32> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let value = input0.data[idx];
  output.data[idx] = select(value, 0.0, value <= 0.0);
}
"#;

const RELU_ADJOINT_F64: &str = r#"
struct Tensor { data: array<f64> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> upstream: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let x = input0.data[idx];
  let high = runmat_f64_high(x);
  let low = runmat_f64_low(x);
  let nan = (high & 0x7ff00000u) == 0x7ff00000u &&
            ((high & 0x000fffffu) != 0u || low != 0u);
  let derivative = select(f64(0.0), f64(1.0), x > f64(0.0));
  output.data[idx] = upstream.data[idx] * select(derivative, x, nan);
}
"#;

const RELU_ADJOINT_F32: &str = r#"
struct Tensor { data: array<f32> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> upstream: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let x = input0.data[idx];
  let bits = bitcast<u32>(x);
  let nan = (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u;
  let derivative = select(0.0, 1.0, x > 0.0);
  output.data[idx] = upstream.data[idx] * select(derivative, x, nan);
}
"#;

const ELU_ADJOINT_F64: &str = r#"
struct Tensor { data: array<f64> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> upstream: Tensor;
@group(0) @binding(2) var<storage, read> alpha: Tensor;
@group(0) @binding(3) var<storage, read_write> output: Tensor;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let x = input0.data[idx];
  let derivative = select(alpha.data[idx] * exp(x), f64(1.0), x >= f64(0.0));
  output.data[idx] = upstream.data[idx] * derivative;
}
"#;

const ELU_ADJOINT_F32: &str = r#"
struct Tensor { data: array<f32> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> upstream: Tensor;
@group(0) @binding(2) var<storage, read> alpha: Tensor;
@group(0) @binding(3) var<storage, read_write> output: Tensor;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let x = input0.data[idx];
  let derivative = select(alpha.data[idx] * exp(x), 1.0, x >= 0.0);
  output.data[idx] = upstream.data[idx] * derivative;
}
"#;

const ELU_SHADER_F64: &str = r#"
struct Tensor { data: array<f64> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> alpha: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let value = input0.data[idx];
  output.data[idx] = select(alpha.data[idx] * (exp(value) - f64(1.0)), value, value >= f64(0.0));
}
"#;

const ELU_SHADER_F32: &str = r#"
struct Tensor { data: array<f32> };
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> alpha: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x; if (idx >= params.len) { return; }
  let value = input0.data[idx];
  output.data[idx] = select(alpha.data[idx] * (exp(value) - 1.0), value, value >= 0.0);
}
"#;

#[cfg(test)]
#[path = "activation/validation_tests.rs"]
mod validation_tests;
