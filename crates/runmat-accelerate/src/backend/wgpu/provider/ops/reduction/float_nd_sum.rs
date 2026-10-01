//! Dimensional F64 reduction for rank greater than two. The input has already
//! crossed any integer-to-floating boundary and is contiguous column-major.
use super::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    len: u32,
    offset: u32,
    stride_before: u32,
    reduce_len: u32,
}

const SHADER: &str = r#"
struct Doubles { data: array<f64> };
struct Params { len: u32, offset: u32, stride_before: u32, reduce_len: u32 };
@group(0) @binding(0) var<storage, read> input: Doubles;
@group(0) @binding(1) var<storage, read_write> output: Doubles;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.len) { return; }
    let index = params.offset + id.x;
    let before = params.stride_before;
    let base = (index % before) + (index / before) * before * params.reduce_len;
    var sum = f64(0.0);
    for (var k = 0u; k < params.reduce_len; k = k + 1u) {
        sum = sum + input.data[base + k * before];
    }
    output.data[index] = sum;
}
"#;

impl WgpuProvider {
    pub(crate) fn reduce_nd_sum_f64_exec(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(input)?;
        ensure!(
            self.precision == NumericPrecision::F64 && entry.storage == GpuTensorStorage::Real,
            "N-D floating sum requires real F64 storage"
        );
        ensure!(
            input.shape() == entry.shape
                && runmat_accelerate_api::handle_transpose_info(input).is_none(),
            "N-D floating sum requires contiguous logical storage"
        );
        ensure!(
            dim < entry.shape.len(),
            "N-D floating sum dimension out of bounds"
        );
        let before = entry.shape[..dim]
            .iter()
            .try_fold(1usize, |acc, &size| acc.checked_mul(size))
            .ok_or_else(|| anyhow!("N-D floating sum stride overflow"))?;
        let reduce_len = entry.shape[dim];
        let after = entry.shape[dim + 1..]
            .iter()
            .try_fold(1usize, |acc, &size| acc.checked_mul(size))
            .ok_or_else(|| anyhow!("N-D floating sum shape overflow"))?;
        let out_len = before
            .checked_mul(after)
            .ok_or_else(|| anyhow!("N-D floating sum output overflow"))?;
        ensure!(
            product_checked(&entry.shape) == Some(entry.len),
            "N-D floating sum allocation shape mismatch"
        );
        ensure!(
            entry.precision == NumericPrecision::F64 && entry.integer_type().is_none(),
            "N-D floating sum requires physical double elements"
        );
        u32::try_from(entry.len)
            .map_err(|_| gpu_dispatch_length_limit_error("N-D floating sum input", entry.len))?;
        let input_bytes = u64::try_from(entry.len)?
            .checked_mul(8)
            .ok_or_else(|| anyhow!("N-D floating sum input byte extent overflow"))?;
        let output_bytes = u64::try_from(out_len)?
            .checked_mul(8)
            .ok_or_else(|| anyhow!("N-D floating sum output byte extent overflow"))?;
        let binding_limit = u64::from(self.adapter_limits.max_storage_buffer_binding_size);
        ensure!(
            input_bytes <= binding_limit && output_bytes <= binding_limit,
            "N-D floating sum storage binding exceeds this device's {binding_limit}-byte limit"
        );
        ensure!(
            input_bytes <= self.adapter_limits.max_buffer_size
                && output_bytes <= self.adapter_limits.max_buffer_size,
            "N-D floating sum buffer exceeds this device's per-buffer limit"
        );
        let before = u32::try_from(before)
            .map_err(|_| gpu_dispatch_length_limit_error("N-D floating sum", before))?;
        let reduce_len = u32::try_from(reduce_len)
            .map_err(|_| gpu_dispatch_length_limit_error("N-D floating sum", reduce_len))?;
        u32::try_from(out_len)
            .map_err(|_| gpu_dispatch_length_limit_error("N-D floating sum", out_len))?;
        let mut out_shape = entry.shape.clone();
        out_shape[dim] = 1;
        let output = self.with_emission(|emission, pending| -> Result<Arc<wgpu::Buffer>> {
            let output =
                self.create_storage_buffer_checked(emission, out_len, "runmat-nd-sum-f64")?;
            ensure!(
                !Arc::ptr_eq(&output, &entry.buffer),
                "N-D floating sum output aliases its input"
            );
            if out_len == 0 {
                return Ok(output);
            }
            let workgroup = crate::backend::wgpu::config::effective_workgroup_size();
            let shader = SHADER.replace("@WG@", &workgroup.to_string());
            let layout =
                emission
                    .device()
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("runmat-nd-sum-f64-bgl"),
                        entries: &[
                            crate::backend::wgpu::bindings::storage_read_entry(0),
                            crate::backend::wgpu::bindings::storage_read_write_entry(1),
                            crate::backend::wgpu::bindings::uniform_entry(2),
                        ],
                    });
            let pipeline_layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
                emission,
                "runmat-nd-sum-f64-pl",
                &layout,
            );
            let module = crate::backend::wgpu::pipelines::create_shader_module(
                emission,
                "runmat-nd-sum-f64-module",
                &shader,
            );
            let key = self.compute_pipeline_hash_bytes(
                shader.as_bytes(),
                "runmat-nd-sum-f64-bgl",
                Some(workgroup),
            );
            let pipeline = self.get_or_create_pipeline(
                emission,
                key,
                &pipeline_layout,
                &module,
                "runmat-nd-sum-f64",
                Some(shader.as_bytes()),
                Some("runmat-nd-sum-f64-bgl"),
                Some(workgroup),
            )?;
            let capacity = crate::backend::wgpu::config::elementwise_chunk_capacity(workgroup)
                .ok_or_else(|| anyhow!("N-D floating sum has invalid workgroup geometry"))?
                as usize;
            let mut offset = 0;
            while offset < out_len {
                let count = (out_len - offset).min(capacity);
                let params = self.uniform_buffer(
                    emission,
                    &Params {
                        len: count as u32,
                        offset: offset as u32,
                        stride_before: before,
                        reduce_len,
                    },
                    "runmat-nd-sum-f64-params",
                )?;
                let group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-nd-sum-f64-bg"),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: output.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params.as_entire_binding(),
                            },
                        ],
                    });
                let groups =
                    crate::backend::wgpu::dispatch::common::dispatch_size(count as u32, workgroup);
                crate::backend::wgpu::dispatch::elementwise::run(
                    emission, pending, &pipeline, &group, groups,
                )?;
                offset += count;
            }
            Ok(output)
        })??;
        self.register_existing_buffer(output, out_shape, out_len)
    }
}
