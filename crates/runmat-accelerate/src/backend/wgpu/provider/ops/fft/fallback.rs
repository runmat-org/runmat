use super::*;

impl WgpuProvider {
    pub(super) async fn fft_dim_exec_host_fallback(
        &self,
        handle: &GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
        inverse: bool,
    ) -> Result<GpuTensorHandle> {
        let HostTensorOwned {
            data,
            mut shape,
            storage,
        } = self.download_exec(handle).await?;
        let mut complex_axis = false;
        if storage == GpuTensorStorage::ComplexInterleaved {
            complex_axis = true;
        }
        if shape.is_empty() {
            if complex_axis {
                let inferred = data.len() / 2;
                shape = vec![inferred];
            } else if data.is_empty() {
                shape = vec![0];
            } else {
                shape = vec![data.len()];
            }
        }
        let origin_rank = shape.len();
        while shape.len() <= dim {
            shape.push(1);
        }
        let current_len = shape.get(dim).copied().unwrap_or(0);
        let target_len = len.unwrap_or(current_len);

        let inner_stride = shape[..dim]
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let outer_stride = shape[dim + 1..]
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let num_slices = inner_stride.saturating_mul(outer_stride);

        let copy_len = current_len.min(target_len);

        let mut out_shape = shape.clone();
        if dim < out_shape.len() {
            out_shape[dim] = target_len;
        }

        if target_len == 0 || num_slices == 0 {
            fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
            let buffer = self.create_storage_buffer(0, "runmat-fft-empty");
            return Ok(self.register_existing_buffer_with_storage(
                buffer,
                out_shape,
                0,
                GpuTensorStorage::ComplexInterleaved,
            ));
        }

        let total_elems = shape
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let mut input = Vec::with_capacity(total_elems);
        if complex_axis {
            for idx in 0..total_elems {
                let base = idx * 2;
                let re = data.get(base).copied().unwrap_or(0.0);
                let im = data.get(base + 1).copied().unwrap_or(0.0);
                input.push(Complex::new(re, im));
            }
        } else {
            for idx in 0..total_elems {
                let re = data.get(idx).copied().unwrap_or(0.0);
                input.push(Complex::new(re, 0.0));
            }
        }

        let mut planner = FftPlanner::<f64>::new();
        let fft_plan = if target_len > 1 {
            Some(if inverse {
                planner.plan_fft_inverse(target_len)
            } else {
                planner.plan_fft_forward(target_len)
            })
        } else {
            None
        };

        let mut buffer_line = vec![Complex::new(0.0, 0.0); target_len];
        let mut output = vec![Complex::new(0.0, 0.0); target_len.saturating_mul(num_slices)];

        for outer in 0..outer_stride {
            let base_in = outer.saturating_mul(current_len.saturating_mul(inner_stride));
            let base_out = outer.saturating_mul(target_len.saturating_mul(inner_stride));
            for inner in 0..inner_stride {
                buffer_line.fill(Complex::new(0.0, 0.0));
                for (k, slot) in buffer_line.iter_mut().enumerate().take(copy_len) {
                    let src_idx = base_in + inner + k * inner_stride;
                    if src_idx < input.len() {
                        *slot = input[src_idx];
                    }
                }
                if let Some(plan) = &fft_plan {
                    plan.process(&mut buffer_line);
                }
                for (k, value) in buffer_line.iter().enumerate().take(target_len) {
                    let dst_idx = base_out + inner + k * inner_stride;
                    if dst_idx < output.len() {
                        output[dst_idx] = if inverse {
                            *value * (1.0 / (target_len as f64))
                        } else {
                            *value
                        };
                    }
                }
            }
        }

        fft_trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));
        let mut packed = Vec::with_capacity(output.len() * 2);
        for complex in output {
            packed.push(complex.re);
            packed.push(complex.im);
        }

        let buffer = self.create_storage_buffer(packed.len(), "runmat-fft-host-fallback-out");
        match self.precision {
            NumericPrecision::F64 => {
                self.queue_ref()
                    .write_buffer(buffer.as_ref(), 0, cast_slice(&packed))
            }
            NumericPrecision::F32 => {
                let packed32: Vec<f32> = packed.iter().map(|&v| v as f32).collect();
                self.queue_ref()
                    .write_buffer(buffer.as_ref(), 0, cast_slice(&packed32));
            }
        }
        Ok(self.register_existing_buffer_with_storage(
            buffer,
            out_shape,
            packed.len(),
            GpuTensorStorage::ComplexInterleaved,
        ))
    }
}
