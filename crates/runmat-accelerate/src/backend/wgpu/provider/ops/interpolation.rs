use super::*;
use runmat_accelerate_api::{
    ProviderInterp1Extrapolation, ProviderInterp1Method, ProviderInterp1Request,
};

impl WgpuProvider {
    pub(crate) async fn interp1_exec(
        &self,
        request: &ProviderInterp1Request<'_>,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            request.sample_len >= 2,
            "interp1: sample_len must be at least 2"
        );
        ensure!(
            request.series_count >= 1,
            "interp1: series_count must be positive"
        );
        let output_len = request
            .query_len
            .checked_mul(request.series_count)
            .ok_or_else(|| anyhow!("interp1: output length overflow"))?;
        let expected = product_checked(request.output_shape)
            .ok_or_else(|| anyhow!("interp1: output shape product overflow"))?;
        ensure!(
            expected == output_len,
            "interp1: output shape does not match query/series count"
        );
        ensure!(
            request.sample_len <= u32::MAX as usize
                && request.query_len <= u32::MAX as usize
                && request.series_count <= u32::MAX as usize
                && output_len <= u32::MAX as usize,
            "interp1: tensor exceeds GPU limits"
        );

        let x_entry = self.get_entry(request.x)?;
        let y_entry = self.get_entry(request.y)?;
        let xq_entry = self.get_entry(request.xq)?;
        ensure!(
            x_entry.precision == self.precision
                && y_entry.precision == self.precision
                && xq_entry.precision == self.precision,
            "interp1: precision mismatch between tensors and provider"
        );
        ensure!(
            x_entry.len == request.sample_len,
            "interp1: X length does not match sample_len"
        );
        let y_len = request
            .sample_len
            .checked_mul(request.series_count)
            .ok_or_else(|| anyhow!("interp1: Y length overflow"))?;
        ensure!(
            y_entry.len == y_len,
            "interp1: Y length does not match sample/series count"
        );
        ensure!(
            xq_entry.len == request.query_len,
            "interp1: Xq length does not match query_len"
        );
        ensure!(
            request.x.device_id == request.y.device_id
                && request.y.device_id == request.xq.device_id,
            "interp1: input tensors must be on the same device"
        );
        ensure!(
            request.query_len > 0 || output_len == 0,
            "interp1: query/output length mismatch"
        );

        let out_buffer = self.create_storage_buffer_checked(output_len, "runmat-interp1-out")?;
        if output_len == 0 {
            return Ok(self.register_existing_buffer(
                out_buffer,
                request.output_shape.to_vec(),
                output_len,
            ));
        }

        let method = match request.method {
            ProviderInterp1Method::Linear => 0u32,
            ProviderInterp1Method::Nearest => 1u32,
        };
        let extrapolation = match request.extrapolation {
            ProviderInterp1Extrapolation::Nan => 0u32,
            ProviderInterp1Extrapolation::Extrapolate => 1u32,
            ProviderInterp1Extrapolation::Value => 2u32,
        };

        let params_buffer = match self.precision {
            NumericPrecision::F64 => {
                let params = Interp1ParamsF64 {
                    sample_len: request.sample_len as u32,
                    query_len: request.query_len as u32,
                    series_count: request.series_count as u32,
                    output_len: output_len as u32,
                    method,
                    extrapolation,
                    _pad0: 0,
                    _pad1: 0,
                    extrapolation_value: request.extrapolation_value,
                    _pad2: 0.0,
                };
                self.uniform_buffer(&params, "runmat-interp1-params")
            }
            NumericPrecision::F32 => {
                let params = Interp1ParamsF32 {
                    sample_len: request.sample_len as u32,
                    query_len: request.query_len as u32,
                    series_count: request.series_count as u32,
                    output_len: output_len as u32,
                    method,
                    extrapolation,
                    _pad0: 0,
                    _pad1: 0,
                    extrapolation_value: request.extrapolation_value as f32,
                    _pad2: 0.0,
                    _pad3: 0.0,
                    _pad4: 0.0,
                };
                self.uniform_buffer(&params, "runmat-interp1-params")
            }
        };

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-interp1-bind"),
                layout: &self.pipelines.interp1.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: y_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: xq_entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            output_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::creation::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.interp1.pipeline,
            &bind_group,
            workgroups,
            "runmat-interp1-encoder",
            "runmat-interp1-pass",
        );

        Ok(self.register_existing_buffer(out_buffer, request.output_shape.to_vec(), output_len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::provider::register_test_wgpu_provider;
    use runmat_accelerate_api::{AccelProvider, HostTensorView};

    fn with_wgpu_provider<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&'static dyn AccelProvider) -> R,
    {
        let Ok(provider) = register_test_wgpu_provider(WgpuProviderOptions::default()) else {
            return None;
        };
        Some(f(provider.provider()))
    }

    fn assert_values_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let difference = (actual - expected).abs();
            assert!(
                difference <= tolerance,
                "value {index} differs: actual={actual}, expected={expected}, difference={difference}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn interp1_provider_linear_and_nearest_match_column_major_series() {
        let Some(()) = with_wgpu_provider(|provider| {
            let x = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0],
                    shape: &[1, 3],
                })
                .expect("upload x");
            let y = provider
                .upload(&HostTensorView {
                    data: &[10.0, 20.0, 40.0, 100.0, 200.0, 400.0],
                    shape: &[3, 2],
                })
                .expect("upload y");
            let xq = provider
                .upload(&HostTensorView {
                    data: &[1.2, 2.5],
                    shape: &[1, 2],
                })
                .expect("upload xq");
            let output_shape = [1, 2, 2];

            let linear = pollster::block_on(provider.interp1(&ProviderInterp1Request {
                x: &x,
                y: &y,
                xq: &xq,
                sample_len: 3,
                series_count: 2,
                query_len: 2,
                output_shape: &output_shape,
                method: ProviderInterp1Method::Linear,
                extrapolation: ProviderInterp1Extrapolation::Nan,
                extrapolation_value: f64::NAN,
            }))
            .expect("linear interp1");
            let nearest = pollster::block_on(provider.interp1(&ProviderInterp1Request {
                x: &x,
                y: &y,
                xq: &xq,
                sample_len: 3,
                series_count: 2,
                query_len: 2,
                output_shape: &output_shape,
                method: ProviderInterp1Method::Nearest,
                extrapolation: ProviderInterp1Extrapolation::Nan,
                extrapolation_value: f64::NAN,
            }))
            .expect("nearest interp1");

            let linear_host = pollster::block_on(provider.download(&linear)).expect("download");
            let nearest_host = pollster::block_on(provider.download(&nearest)).expect("download");
            assert_eq!(linear_host.shape, output_shape);
            assert_eq!(nearest_host.shape, output_shape);
            assert_values_close(&linear_host.data, &[12.0, 30.0, 120.0, 300.0], 1.0e-5);
            assert_eq!(nearest_host.data, vec![10.0, 20.0, 100.0, 200.0]);
            provider.free(&x).ok();
            provider.free(&y).ok();
            provider.free(&xq).ok();
            provider.free(&linear).ok();
            provider.free(&nearest).ok();
        }) else {
            return;
        };
    }
}
