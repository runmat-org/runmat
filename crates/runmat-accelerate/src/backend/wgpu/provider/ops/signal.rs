use super::*;
use crate::backend::wgpu::shaders::signal::{
    analytic_signal_mask_shader, envelope_analytic_bounds_shader,
    envelope_analytic_fir_bounds_shader, envelope_analytic_mask_shader,
    envelope_center_real_to_complex_shader, envelope_rms_bounds_shader, spectral_frame_shader,
    spectral_power_shader, spectral_select_shader, SpectralFrameShaderConfig,
    SpectralFrameShaderMode, SpectralRangeShaderMode,
};
use runmat_accelerate_api::{
    ProviderEnvelopeMethod, ProviderEnvelopeRequest, ProviderEnvelopeResult, ProviderHilbertRequest,
};

impl WgpuProvider {
    pub(crate) async fn uniform_spectral_estimate_exec(
        &self,
        request: &ProviderSpectralRequest<'_>,
    ) -> Result<ProviderSpectralResult> {
        ensure!(
            !request.window.is_empty()
                && request.nfft > 0
                && request.frame_count > 0
                && request.denominator.is_finite()
                && request.denominator > 0.0,
            "uniform_spectral_estimate: invalid request"
        );

        let window_shape = [request.window.len(), 1usize];
        let window = self.upload_exec(&HostTensorView {
            data: request.window,
            shape: &window_shape,
        })?;

        let framed_len = request
            .nfft
            .checked_mul(request.frame_count)
            .and_then(|len| len.checked_mul(2))
            .ok_or_else(|| anyhow!("uniform_spectral_estimate: frame too large"))?;
        let frame_shader = spectral_frame_shader(
            &SpectralFrameShaderConfig {
                input_complex: request.input_complex,
                mode: spectral_frame_shader_mode(request.frame_mode),
                window_len: request.window.len(),
                nfft: request.nfft,
                frame_count: request.frame_count,
                input_len: request.input_len,
            },
            self.precision,
        );
        let frame_inputs = [request.input.clone(), window.clone()];
        let framed = match self.fused_elementwise_with_telemetry_exec(
            &frame_shader,
            &frame_inputs,
            &[request.nfft, request.frame_count],
            framed_len,
        ) {
            Ok(framed) => framed,
            Err(err) => {
                self.free_exec(&window).ok();
                return Err(err);
            }
        };
        self.free_exec(&window).ok();
        runmat_accelerate_api::set_handle_storage(&framed, GpuTensorStorage::ComplexInterleaved);

        let spectrum = match self.fft_dim_exec(&framed, None, 0).await {
            Ok(spectrum) => spectrum,
            Err(err) => {
                self.free_exec(&framed).ok();
                return Err(err);
            }
        };
        self.free_exec(&framed).ok();

        let rows = spectral_selected_frequency_len(request.nfft, request.range);
        let range = spectral_range_shader_mode(request.range);
        let selected = if matches!(request.range, ProviderSpectralRange::Twosided) {
            spectrum
        } else {
            let selected_len = rows
                .checked_mul(request.frame_count)
                .and_then(|len| len.checked_mul(2))
                .ok_or_else(|| anyhow!("uniform_spectral_estimate: output too large"))?;
            let shader = spectral_select_shader(request.nfft, rows, range, self.precision);
            let handle = match self.fused_elementwise_with_telemetry_exec(
                &shader,
                std::slice::from_ref(&spectrum),
                &[rows, request.frame_count],
                selected_len,
            ) {
                Ok(handle) => handle,
                Err(err) => {
                    self.free_exec(&spectrum).ok();
                    return Err(err);
                }
            };
            self.free_exec(&spectrum).ok();
            runmat_accelerate_api::set_handle_storage(
                &handle,
                GpuTensorStorage::ComplexInterleaved,
            );
            handle
        };

        let ps_len = rows
            .checked_mul(request.frame_count)
            .ok_or_else(|| anyhow!("uniform_spectral_estimate: power output too large"))?;
        let ps_shader = spectral_power_shader(
            rows,
            range,
            request.nfft.is_multiple_of(2),
            request.denominator,
            self.precision,
        );
        let ps = match self.fused_elementwise_with_telemetry_exec(
            &ps_shader,
            std::slice::from_ref(&selected),
            &[rows, request.frame_count],
            ps_len,
        ) {
            Ok(ps) => ps,
            Err(err) => {
                self.free_exec(&selected).ok();
                return Err(err);
            }
        };

        Ok(ProviderSpectralResult {
            s: selected,
            ps,
            rows,
            cols: request.frame_count,
        })
    }

    pub(crate) async fn signal_envelope_exec(
        &self,
        request: &ProviderEnvelopeRequest<'_>,
    ) -> Result<ProviderEnvelopeResult> {
        let output_len = request
            .output_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("signal_envelope: invalid request"))?;
        let input_len = request
            .channel_len
            .checked_mul(request.channel_count)
            .ok_or_else(|| anyhow!("signal_envelope: invalid request"))?;
        ensure!(
            request.channel_len > 0 && request.channel_count > 0 && output_len == input_len,
            "signal_envelope: invalid request"
        );
        let entry = self.get_entry(request.input)?;
        ensure!(
            entry.precision == self.precision,
            "signal_envelope: mixed precision tensors are not supported"
        );
        ensure!(
            entry.storage == GpuTensorStorage::Real
                && runmat_accelerate_api::handle_storage(request.input) == GpuTensorStorage::Real,
            "signal_envelope: complex input tensors are not supported"
        );
        ensure!(
            entry.len == input_len,
            "signal_envelope: input length mismatch"
        );
        ensure!(
            envelope_input_shape_matches(&entry.shape, request.channel_len, request.channel_count)
                && envelope_input_shape_matches(
                    &request.input.shape,
                    request.channel_len,
                    request.channel_count
                ),
            "signal_envelope: input shape mismatch"
        );

        match request.method {
            ProviderEnvelopeMethod::Analytic => self.signal_envelope_analytic_exec(request).await,
            ProviderEnvelopeMethod::AnalyticFir { filter_len } => {
                ensure!(filter_len > 0, "signal_envelope: invalid request");
                self.signal_envelope_analytic_fir_exec(request, filter_len)
            }
            ProviderEnvelopeMethod::Rms { window_len } => {
                ensure!(window_len > 0, "signal_envelope: invalid request");
                self.signal_envelope_rms_exec(request, window_len)
            }
        }
    }

    pub(crate) async fn signal_hilbert_exec(
        &self,
        request: &ProviderHilbertRequest<'_>,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(request.input)?;
        ensure!(
            entry.precision == self.precision,
            "signal_hilbert: mixed precision tensors are not supported"
        );
        ensure!(
            entry.storage == GpuTensorStorage::Real
                && runmat_accelerate_api::handle_storage(request.input) == GpuTensorStorage::Real,
            "signal_hilbert: complex input tensors are not supported"
        );
        ensure!(request.length != Some(0), "signal_hilbert: invalid request");

        let spectrum = self
            .fft_dim_exec(request.input, request.length, request.dim)
            .await?;
        let filtered_result = (|| -> Result<GpuTensorHandle> {
            let spectrum_entry = self.get_entry(&spectrum)?;
            ensure!(
                spectrum_entry.storage == GpuTensorStorage::ComplexInterleaved,
                "signal_hilbert: fft output was not complex"
            );
            ensure!(
                spectrum_entry.len <= u32::MAX as usize,
                "signal_hilbert: output too large"
            );

            let mut shape = spectrum_entry.shape.clone();
            while shape.len() <= request.dim {
                shape.push(1);
            }
            let transform_len = shape[request.dim];
            ensure!(
                transform_len > 0,
                "signal_hilbert: invalid transform length"
            );
            let inner_stride = product_checked(&shape[..request.dim])
                .ok_or_else(|| anyhow!("signal_hilbert: shape overflow"))?;
            ensure!(
                inner_stride > 0
                    && inner_stride <= u32::MAX as usize
                    && transform_len <= u32::MAX as usize,
                "signal_hilbert: dimensions exceed GPU kernel limits"
            );

            ensure!(
                spectrum_entry.len % 2 == 0,
                "signal_hilbert: invalid complex output length"
            );
            let shader = analytic_signal_mask_shader(
                transform_len,
                inner_stride,
                spectrum_entry.len,
                self.precision,
            );
            self.fused_elementwise_with_telemetry_exec(
                &shader,
                std::slice::from_ref(&spectrum),
                &spectrum_entry.shape,
                spectrum_entry.len,
            )
        })();
        let filtered = match filtered_result {
            Ok(filtered) => filtered,
            Err(err) => {
                self.free_exec(&spectrum).ok();
                return Err(err);
            }
        };
        self.free_exec(&spectrum).ok();
        runmat_accelerate_api::set_handle_storage(&filtered, GpuTensorStorage::ComplexInterleaved);

        let analytic = match self.ifft_dim_exec(&filtered, None, request.dim).await {
            Ok(analytic) => analytic,
            Err(err) => {
                self.free_exec(&filtered).ok();
                return Err(err);
            }
        };
        self.free_exec(&filtered).ok();
        runmat_accelerate_api::set_handle_storage(&analytic, GpuTensorStorage::ComplexInterleaved);
        Ok(analytic)
    }

    async fn signal_envelope_analytic_exec(
        &self,
        request: &ProviderEnvelopeRequest<'_>,
    ) -> Result<ProviderEnvelopeResult> {
        let complex_len = request
            .channel_len
            .checked_mul(request.channel_count)
            .and_then(|len| len.checked_mul(2))
            .ok_or_else(|| anyhow!("signal_envelope: analytic output too large"))?;
        let channel_shape = [request.channel_len, request.channel_count];
        let center_shader = envelope_center_real_to_complex_shader(
            request.channel_len,
            request.channel_count,
            self.precision,
        );
        let centered = self.fused_elementwise_with_telemetry_exec(
            &center_shader,
            std::slice::from_ref(request.input),
            &channel_shape,
            complex_len,
        )?;
        runmat_accelerate_api::set_handle_storage(&centered, GpuTensorStorage::ComplexInterleaved);

        let spectrum = match self.fft_dim_exec(&centered, None, 0).await {
            Ok(spectrum) => spectrum,
            Err(err) => {
                self.free_exec(&centered).ok();
                return Err(err);
            }
        };
        self.free_exec(&centered).ok();

        let mask_shader = envelope_analytic_mask_shader(
            request.channel_len,
            request.channel_count,
            self.precision,
        );
        let filtered = match self.fused_elementwise_with_telemetry_exec(
            &mask_shader,
            std::slice::from_ref(&spectrum),
            &channel_shape,
            complex_len,
        ) {
            Ok(filtered) => filtered,
            Err(err) => {
                self.free_exec(&spectrum).ok();
                return Err(err);
            }
        };
        self.free_exec(&spectrum).ok();
        runmat_accelerate_api::set_handle_storage(&filtered, GpuTensorStorage::ComplexInterleaved);

        let analytic = match self.ifft_dim_exec(&filtered, None, 0).await {
            Ok(analytic) => analytic,
            Err(err) => {
                self.free_exec(&filtered).ok();
                return Err(err);
            }
        };
        self.free_exec(&filtered).ok();

        let real_len = request
            .channel_len
            .checked_mul(request.channel_count)
            .ok_or_else(|| anyhow!("signal_envelope: output too large"))?;
        let bounds_shader = envelope_analytic_bounds_shader(
            request.channel_len,
            request.channel_count,
            self.precision,
        );
        let inputs = [request.input.clone(), analytic.clone()];
        let mut outputs = match self.fused_elementwise_multi_with_telemetry_exec(
            &bounds_shader,
            &inputs,
            request.output_shape,
            real_len,
            2,
        ) {
            Ok(outputs) => outputs,
            Err(err) => {
                self.free_exec(&analytic).ok();
                return Err(err);
            }
        };
        self.free_exec(&analytic).ok();
        ensure!(outputs.len() == 2, "signal_envelope: missing outputs");
        let lower = outputs.pop().expect("lower output");
        let upper = outputs.pop().expect("upper output");
        Ok(ProviderEnvelopeResult { upper, lower })
    }

    fn signal_envelope_analytic_fir_exec(
        &self,
        request: &ProviderEnvelopeRequest<'_>,
        filter_len: usize,
    ) -> Result<ProviderEnvelopeResult> {
        ensure!(filter_len > 0, "signal_envelope: invalid filter length");
        let real_len = request
            .channel_len
            .checked_mul(request.channel_count)
            .ok_or_else(|| anyhow!("signal_envelope: output too large"))?;
        let kernel_data = envelope_hilbert_fir_kernel(filter_len);
        let kernel_shape = [filter_len, 1usize];
        let kernel = self.upload_exec(&HostTensorView {
            data: &kernel_data,
            shape: &kernel_shape,
        })?;
        let shader = envelope_analytic_fir_bounds_shader(
            request.channel_len,
            request.channel_count,
            filter_len,
            self.precision,
        );
        let inputs = [request.input.clone(), kernel.clone()];
        let mut outputs = match self.fused_elementwise_multi_with_telemetry_exec(
            &shader,
            &inputs,
            request.output_shape,
            real_len,
            2,
        ) {
            Ok(outputs) => outputs,
            Err(err) => {
                self.free_exec(&kernel).ok();
                return Err(err);
            }
        };
        self.free_exec(&kernel).ok();
        ensure!(outputs.len() == 2, "signal_envelope: missing outputs");
        let lower = outputs.pop().expect("lower output");
        let upper = outputs.pop().expect("upper output");
        Ok(ProviderEnvelopeResult { upper, lower })
    }

    fn signal_envelope_rms_exec(
        &self,
        request: &ProviderEnvelopeRequest<'_>,
        window_len: usize,
    ) -> Result<ProviderEnvelopeResult> {
        ensure!(window_len > 0, "signal_envelope: invalid window length");
        let real_len = request
            .channel_len
            .checked_mul(request.channel_count)
            .ok_or_else(|| anyhow!("signal_envelope: output too large"))?;
        let shader = envelope_rms_bounds_shader(
            request.channel_len,
            request.channel_count,
            window_len,
            self.precision,
        );
        let mut outputs = self.fused_elementwise_multi_with_telemetry_exec(
            &shader,
            std::slice::from_ref(request.input),
            request.output_shape,
            real_len,
            2,
        )?;
        ensure!(outputs.len() == 2, "signal_envelope: missing outputs");
        let lower = outputs.pop().expect("lower output");
        let upper = outputs.pop().expect("upper output");
        Ok(ProviderEnvelopeResult { upper, lower })
    }

    pub(crate) fn conv2d_exec(
        &self,
        _signal: &GpuTensorHandle,
        _kernel: &GpuTensorHandle,
        _mode: ProviderConvMode,
    ) -> Result<GpuTensorHandle> {
        Err(anyhow!("conv2d not implemented for the WGPU provider yet"))
    }

    pub(crate) fn conv1d_exec(
        &self,
        signal: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: ProviderConv1dOptions,
    ) -> Result<GpuTensorHandle> {
        let entry_signal = self.get_entry(signal)?;
        let entry_kernel = self.get_entry(kernel)?;

        ensure!(
            entry_signal.precision == self.precision && entry_kernel.precision == self.precision,
            "conv1d: mixed precision tensors are not supported"
        );

        let signal_len = entry_signal.len;
        let kernel_len = entry_kernel.len;

        let (output_len, start_offset, _) = conv1d_window(signal_len, kernel_len, options.mode)?;

        if output_len == 0 {
            let out_shape = conv1d_output_shape(0, options.orientation);
            let out_buffer = self.create_storage_buffer(0, "runmat-conv1d-empty");
            return Ok(self.register_existing_buffer(out_buffer, out_shape, 0));
        }

        ensure!(
            signal_len <= u32::MAX as usize
                && kernel_len <= u32::MAX as usize
                && output_len <= u32::MAX as usize
                && start_offset <= u32::MAX as usize,
            "conv1d: tensor exceeds GPU kernel limits"
        );

        let out_shape = conv1d_output_shape(output_len, options.orientation);
        let out_buffer = self.create_storage_buffer_checked(output_len, "runmat-conv1d-out")?;

        let params = Conv1dParams {
            signal_len: signal_len as u32,
            kernel_len: kernel_len as u32,
            output_len: output_len as u32,
            start_offset: start_offset as u32,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-conv1d-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-conv1d-bind"),
                layout: &self.pipelines.conv1d.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry_signal.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: entry_kernel.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            output_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::conv::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.conv1d.pipeline,
            &bind_group,
            workgroups,
        );

        let handle = self.register_existing_buffer(out_buffer, out_shape, output_len);

        Ok(handle)
    }
    pub(crate) async fn iir_filter_exec(
        &self,
        b: &GpuTensorHandle,
        a: &GpuTensorHandle,
        x: &GpuTensorHandle,
        options: ProviderIirFilterOptions,
    ) -> Result<ProviderIirFilterResult> {
        let ProviderIirFilterOptions { dim, zi } = options;

        let entry_b = self.get_entry(b)?;
        let entry_a = self.get_entry(a)?;
        let entry_x = self.get_entry(x)?;

        ensure!(
            entry_b.precision == self.precision
                && entry_a.precision == self.precision
                && entry_x.precision == self.precision,
            "iir_filter: mixed precision tensors are not supported"
        );

        let nb = entry_b.len;
        let na = entry_a.len;
        ensure!(
            nb > 0,
            "iir_filter: numerator coefficients must not be empty"
        );
        ensure!(
            na > 0,
            "iir_filter: denominator coefficients must not be empty"
        );

        let b_host = self.download_exec(b).await?;
        let a_host = self.download_exec(a).await?;
        let a0 = *a_host
            .data
            .first()
            .ok_or_else(|| anyhow!("iir_filter: denominator coefficients cannot be empty"))?;
        ensure!(
            a0 != 0.0,
            "iir_filter: denominator coefficient a(1) must be non-zero"
        );

        let order = nb.max(na);
        ensure!(
            order <= u32::MAX as usize,
            "iir_filter: filter order exceeds GPU limits"
        );

        let mut b_norm = vec![0.0f64; order];
        let mut a_norm = vec![0.0f64; order];
        for i in 0..order {
            let b_coeff = if i < nb { b_host.data[i] } else { 0.0 };
            b_norm[i] = b_coeff / a0;
            if i == 0 {
                a_norm[0] = 1.0;
            } else {
                let a_coeff = if i < na { a_host.data[i] } else { 0.0 };
                a_norm[i] = a_coeff / a0;
            }
        }

        let state_len = order.saturating_sub(1);

        let mut shape_ext = entry_x.shape.clone();
        if dim >= shape_ext.len() {
            shape_ext.extend(std::iter::repeat_n(1, dim + 1 - shape_ext.len()));
        }
        ensure!(
            dim < shape_ext.len(),
            "iir_filter: dimension argument exceeds tensor rank"
        );
        let dim_idx = dim;
        let dim_len = shape_ext[dim_idx];

        let leading = if dim_idx == 0 {
            1usize
        } else {
            product_checked(&shape_ext[..dim_idx])
                .ok_or_else(|| anyhow!("iir_filter: tensor exceeds GPU limits"))?
        };
        let trailing = if dim_idx + 1 >= shape_ext.len() {
            1usize
        } else {
            product_checked(&shape_ext[dim_idx + 1..])
                .ok_or_else(|| anyhow!("iir_filter: tensor exceeds GPU limits"))?
        };
        let channel_count = leading
            .checked_mul(trailing)
            .ok_or_else(|| anyhow!("iir_filter: tensor exceeds GPU limits"))?;

        ensure!(
            shape_ext.len() <= crate::backend::wgpu::params::FILTER_MAX_RANK,
            "iir_filter: tensors exceed supported rank for GPU kernel"
        );

        let state_shape = filter_state_shape(shape_ext.clone(), dim_idx, state_len);
        ensure!(
            state_shape.len() <= crate::backend::wgpu::params::FILTER_MAX_RANK,
            "iir_filter: filter state rank exceeds GPU limits"
        );

        let state_total = if state_len == 0 {
            0usize
        } else {
            product_checked(&state_shape)
                .ok_or_else(|| anyhow!("iir_filter: filter state exceeds GPU limits"))?
        };

        if let Some(ref zi_handle) = zi {
            let zi_entry = self.get_entry(zi_handle)?;
            ensure!(
                zi_entry.precision == self.precision,
                "iir_filter: initial conditions use incompatible precision"
            );
            ensure!(
                shapes_compatible(&state_shape, &zi_entry.shape),
                "iir_filter: initial conditions are not compatible with the requested dimension"
            );
            let zi_dim = if dim_idx < zi_entry.shape.len() {
                zi_entry.shape[dim_idx]
            } else {
                1
            };
            ensure!(
                zi_dim == state_len,
                "iir_filter: initial conditions must have {} states along dimension {}",
                state_len,
                dim + 1
            );
            if state_total == 0 {
                ensure!(
                    zi_entry.len == 0,
                    "iir_filter: initial conditions have {} elements but zero were expected",
                    zi_entry.len
                );
            } else {
                ensure!(
                    zi_entry.len == state_total,
                    "iir_filter: initial state vector length mismatch (expected {}, found {})",
                    state_total,
                    zi_entry.len
                );
            }
        }

        ensure!(
            entry_x.len <= u32::MAX as usize,
            "iir_filter: signal length exceeds GPU limits"
        );
        ensure!(
            leading <= u32::MAX as usize
                && trailing <= u32::MAX as usize
                && channel_count <= u32::MAX as usize,
            "iir_filter: tensor exceeds GPU kernel limits"
        );
        ensure!(
            dim_len <= u32::MAX as usize,
            "iir_filter: dimension length exceeds GPU limits"
        );
        ensure!(
            state_len <= u32::MAX as usize,
            "iir_filter: filter order exceeds GPU limits"
        );
        ensure!(
            state_total <= u32::MAX as usize,
            "iir_filter: filter state size exceeds GPU limits"
        );

        let state_buffer_len = if state_len == 0 {
            0usize
        } else {
            state_len
                .checked_mul(channel_count)
                .ok_or_else(|| anyhow!("iir_filter: state buffer length overflow"))?
        };
        ensure!(
            state_buffer_len <= u32::MAX as usize,
            "iir_filter: state buffer length exceeds GPU limits"
        );

        let mut cleanup_handles: Vec<GpuTensorHandle> = Vec::new();
        let result = (|| -> Result<ProviderIirFilterResult> {
            let b_shape = [order, 1usize];
            let b_view = HostTensorView {
                data: &b_norm,
                shape: &b_shape,
            };
            let b_norm_handle = self.upload_exec(&b_view)?;
            cleanup_handles.push(b_norm_handle.clone());

            let a_shape = [order, 1usize];
            let a_view = HostTensorView {
                data: &a_norm,
                shape: &a_shape,
            };
            let a_norm_handle = self.upload_exec(&a_view)?;
            cleanup_handles.push(a_norm_handle.clone());

            let b_norm_entry = self.get_entry(&b_norm_handle)?;
            let a_norm_entry = self.get_entry(&a_norm_handle)?;

            let out_buffer = self.create_storage_buffer(entry_x.len, "runmat-iir-filter-out");
            let states_buffer =
                self.create_storage_buffer(state_buffer_len, "runmat-iir-filter-state");
            let final_state_buffer =
                self.create_storage_buffer(state_total, "runmat-iir-filter-final");

            let (zi_buffer, zi_present_flag) = if let Some(ref zi_handle) = zi {
                let zi_entry = self.get_entry(zi_handle)?;
                (zi_entry.buffer, 1u32)
            } else {
                (
                    self.create_storage_buffer(state_total, "runmat-iir-filter-zi"),
                    0u32,
                )
            };

            let mut signal_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
                crate::backend::wgpu::params::FILTER_MAX_RANK];
            for (idx, dim_len) in shape_ext.iter().enumerate() {
                signal_shape_arr[idx] =
                    crate::backend::wgpu::params::AlignedU32::new(*dim_len as u32);
            }
            let mut state_shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
                crate::backend::wgpu::params::FILTER_MAX_RANK];
            for (idx, dim_len) in state_shape.iter().enumerate() {
                state_shape_arr[idx] =
                    crate::backend::wgpu::params::AlignedU32::new(*dim_len as u32);
            }

            let params = FilterParams {
                dim_len: dim_len as u32,
                leading: leading as u32,
                trailing: trailing as u32,
                order: order as u32,
                state_len: state_len as u32,
                signal_len: entry_x.len as u32,
                channel_count: channel_count as u32,
                zi_present: zi_present_flag,
                dim_idx: dim_idx as u32,
                rank: shape_ext.len() as u32,
                state_rank: state_shape.len() as u32,
                _pad: 0,
                signal_shape: signal_shape_arr,
                state_shape: state_shape_arr,
            };

            let params_buffer = self.uniform_buffer(&params, "runmat-iir-filter-params");

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-iir-filter-bind"),
                    layout: &self.pipelines.filter.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry_x.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_norm_entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: a_norm_entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: zi_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: states_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: final_state_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                channel_count as u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::filter::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.filter.pipeline,
                &bind_group,
                workgroups,
            );

            let output_handle =
                self.register_existing_buffer(out_buffer, entry_x.shape.clone(), entry_x.len);
            let final_state_handle =
                self.register_existing_buffer(final_state_buffer, state_shape.clone(), state_total);

            Ok(ProviderIirFilterResult {
                output: output_handle,
                final_state: Some(final_state_handle),
            })
        })();

        for handle in cleanup_handles {
            let _ = self.free_exec(&handle);
        }

        result
    }
    pub(crate) fn diff_once_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;

        let mut ext_shape = if entry.shape.is_empty() {
            vec![if entry.len == 0 { 1 } else { entry.len }]
        } else {
            entry.shape.clone()
        };
        while ext_shape.len() <= dim {
            ext_shape.push(1);
        }

        let len_dim = ext_shape[dim];

        let mut out_shape = entry.shape.clone();
        while out_shape.len() <= dim {
            out_shape.push(1);
        }

        if len_dim <= 1 || entry.len == 0 {
            out_shape[dim] = out_shape[dim].saturating_sub(1);
            let out_len = product_checked(&out_shape).unwrap_or(0);
            let out_buffer = self.create_storage_buffer(out_len, "runmat-diff-empty");
            return Ok(self.register_existing_buffer(out_buffer, out_shape, out_len));
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            product_checked(&ext_shape[..dim])
                .ok_or_else(|| anyhow!("diff: stride computation overflow"))?
                .max(1)
        };
        let stride_after = if dim + 1 >= ext_shape.len() {
            1usize
        } else {
            product_checked(&ext_shape[dim + 1..])
                .ok_or_else(|| anyhow!("diff: stride computation overflow"))?
                .max(1)
        };

        let expected_len = stride_before
            .checked_mul(len_dim)
            .and_then(|v| v.checked_mul(stride_after))
            .ok_or_else(|| anyhow!("diff: tensor size exceeds GPU limits"))?;
        ensure!(
            expected_len == entry.len,
            "diff: tensor shape mismatch (expected {} elements, got {})",
            expected_len,
            entry.len
        );

        let segment_out = len_dim - 1;
        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("diff: segment count exceeds GPU limits"))?;
        let out_len = segments
            .checked_mul(segment_out)
            .ok_or_else(|| anyhow!("diff: output size exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(len_dim)
            .ok_or_else(|| anyhow!("diff: block size exceeds GPU limits"))?;

        ensure!(
            len_dim <= u32::MAX as usize
                && stride_before <= u32::MAX as usize
                && stride_after <= u32::MAX as usize
                && segments <= u32::MAX as usize
                && block <= u32::MAX as usize
                && out_len <= u32::MAX as usize
                && entry.len <= u32::MAX as usize,
            "diff: tensor exceeds GPU kernel limits"
        );

        let out_buffer = self.create_storage_buffer(out_len, "runmat-diff-out");
        out_shape[dim] = len_dim - 1;

        let params = DiffParams {
            stride_before: stride_before as u32,
            segments: segments as u32,
            segment_len: len_dim as u32,
            segment_out: segment_out as u32,
            block: block as u32,
            total_out: out_len as u32,
            total_in: entry.len as u32,
            _pad: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-diff-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-diff-bind"),
                layout: &self.pipelines.diff.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            out_len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::diff::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.diff.pipeline,
            &bind_group,
            workgroups,
        );

        Ok(self.register_existing_buffer(out_buffer, out_shape, out_len))
    }

    pub(crate) fn diff_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        order: usize,
    ) -> Result<GpuTensorHandle> {
        if order == 0 {
            return Ok(handle.clone());
        }

        let mut current = handle.clone();
        let mut owns_current = false;
        for _ in 0..order {
            let next = self.diff_once_exec(&current, dim)?;
            if owns_current {
                let _ = self.free_exec(&current);
            }
            current = next;
            owns_current = true;

            let entry = self.get_entry(&current)?;
            if entry.len == 0 {
                break;
            }
        }
        Ok(current)
    }

    pub(crate) fn gradient_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        spacing: f64,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry(handle)?;
        let complex_storage = entry.storage == GpuTensorStorage::ComplexInterleaved
            || runmat_accelerate_api::handle_storage(handle)
                == GpuTensorStorage::ComplexInterleaved;
        ensure!(
            !complex_storage || entry.len % 2 == 0,
            "gradient: complex-interleaved buffers must have even length"
        );
        let logical_len = if complex_storage {
            entry.len / 2
        } else {
            entry.len
        };

        let mut ext_shape = normalize_gradient_shape(&entry.shape, logical_len);
        if ext_shape.is_empty() {
            ext_shape = vec![0, 0];
        }
        while ext_shape.len() <= dim {
            ext_shape.push(1);
        }

        let len_dim = ext_shape[dim];
        let mut out_shape = normalize_gradient_shape(&entry.shape, logical_len);
        if out_shape.is_empty() {
            out_shape = vec![0, 0];
        }
        while out_shape.len() <= dim {
            out_shape.push(1);
        }

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-gradient-out");
        if entry.len == 0 {
            return Ok(self.register_existing_buffer_with_storage(
                out_buffer,
                out_shape,
                0,
                if complex_storage {
                    GpuTensorStorage::ComplexInterleaved
                } else {
                    GpuTensorStorage::Real
                },
            ));
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            product_checked(&ext_shape[..dim])
                .ok_or_else(|| anyhow!("gradient: stride computation overflow"))?
                .max(1)
        };
        let stride_after = if dim + 1 >= ext_shape.len() {
            1usize
        } else {
            product_checked(&ext_shape[dim + 1..])
                .ok_or_else(|| anyhow!("gradient: stride computation overflow"))?
                .max(1)
        };

        let expected_len = stride_before
            .checked_mul(len_dim.max(1))
            .and_then(|v| v.checked_mul(stride_after))
            .ok_or_else(|| anyhow!("gradient: tensor size exceeds GPU limits"))?;
        ensure!(
            expected_len == logical_len,
            "gradient: tensor shape mismatch (expected {} elements, got {})",
            expected_len,
            logical_len
        );

        let block = stride_before
            .checked_mul(len_dim.max(1))
            .ok_or_else(|| anyhow!("gradient: block size exceeds GPU limits"))?;
        let lane_multiplier = if complex_storage { 2usize } else { 1usize };
        let kernel_stride_before = stride_before
            .checked_mul(lane_multiplier)
            .ok_or_else(|| anyhow!("gradient: stride computation overflow"))?;
        let kernel_block = block
            .checked_mul(lane_multiplier)
            .ok_or_else(|| anyhow!("gradient: block size exceeds GPU limits"))?;
        ensure!(
            len_dim <= u32::MAX as usize
                && kernel_stride_before <= u32::MAX as usize
                && kernel_block <= u32::MAX as usize
                && entry.len <= u32::MAX as usize,
            "gradient: tensor exceeds GPU kernel limits"
        );

        let params_buffer = match self.precision {
            NumericPrecision::F64 => self.uniform_buffer(
                &GradientParamsF64 {
                    stride_before: kernel_stride_before as u32,
                    segment_len: len_dim as u32,
                    block: kernel_block as u32,
                    total: entry.len as u32,
                    spacing,
                    _pad0: 0.0,
                    _pad1: 0.0,
                    _pad2: 0.0,
                },
                "runmat-gradient-params",
            ),
            NumericPrecision::F32 => self.uniform_buffer(
                &GradientParamsF32 {
                    meta0: crate::backend::wgpu::params::PackedU32([
                        kernel_stride_before as u32,
                        len_dim as u32,
                        kernel_block as u32,
                        entry.len as u32,
                    ]),
                    meta1: crate::backend::wgpu::params::PackedF32([spacing as f32, 0.0, 0.0, 0.0]),
                },
                "runmat-gradient-params",
            ),
        };

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-gradient-bind"),
                layout: &self.pipelines.gradient.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
            entry.len as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::gradient::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.gradient.pipeline,
            &bind_group,
            workgroups,
        );

        Ok(self.register_existing_buffer_with_storage(
            out_buffer,
            out_shape,
            entry.len,
            if complex_storage {
                GpuTensorStorage::ComplexInterleaved
            } else {
                GpuTensorStorage::Real
            },
        ))
    }

    pub(crate) fn cumsum_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        // For reverse scans, compute as flip → forward-scan → flip to preserve exact semantics
        if matches!(direction, ProviderScanDirection::Reverse) {
            let flipped_in = self.flip_exec(handle, &[dim])?;
            let forward =
                self.cumsum_exec(&flipped_in, dim, ProviderScanDirection::Forward, nan_mode)?;
            let _ = self.free_exec(&flipped_in);
            let flipped_out = self.flip_exec(&forward, &[dim])?;
            let _ = self.free_exec(&forward);
            return Ok(flipped_out);
        }
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cumsum: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-cumsum-empty");
            return Ok(self.register_existing_buffer(out_buffer, entry.shape, 0));
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cumsum: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cumsum: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cumsum: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cumsum: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cumsum: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cumsum: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cumsum: tensor too large for GPU kernel"
        );

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-cumsum-out");
        if entry.len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry.shape, 0));
        }

        let mut flags = 0u32;
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CumsumParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-cumsum-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-cumsum-bind"),
                layout: &self.pipelines.cumsum.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            segments as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scan::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.cumsum.pipeline,
            &bind_group,
            groups,
            "runmat-cumsum-encoder",
            "runmat-cumsum-pass",
        );
        Ok(self.register_existing_buffer(out_buffer, entry.shape, entry.len))
    }
    pub(crate) fn cumprod_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        // For reverse scans, compute as flip → forward-scan → flip to preserve exact semantics
        if matches!(direction, ProviderScanDirection::Reverse) {
            let flipped_in = self.flip_exec(handle, &[dim])?;
            let forward =
                self.cumprod_exec(&flipped_in, dim, ProviderScanDirection::Forward, nan_mode)?;
            let _ = self.free_exec(&flipped_in);
            let flipped_out = self.flip_exec(&forward, &[dim])?;
            let _ = self.free_exec(&forward);
            return Ok(flipped_out);
        }
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            return Ok(handle.clone());
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cumprod: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let out_buffer = self.create_storage_buffer(0, "runmat-cumprod-empty");
            return Ok(self.register_existing_buffer(out_buffer, entry.shape, 0));
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cumprod: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cumprod: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cumprod: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cumprod: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cumprod: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cumprod: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cumprod: tensor too large for GPU kernel"
        );

        let out_buffer = self.create_storage_buffer(entry.len, "runmat-cumprod-out");
        if entry.len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, entry.shape, 0));
        }

        let mut flags = 0u32;
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CumprodParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-cumprod-params");
        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-cumprod-bind"),
                layout: &self.pipelines.cumprod.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            segments as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scan::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.cumprod.pipeline,
            &bind_group,
            groups,
            "runmat-cumprod-encoder",
            "runmat-cumprod-pass",
        );
        Ok(self.register_existing_buffer(out_buffer, entry.shape, entry.len))
    }
    pub(crate) fn cummin_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<ProviderCumminResult> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            let values_buffer = self.create_storage_buffer(0, "runmat-cummin-values-empty");
            let indices_buffer = self.create_storage_buffer(0, "runmat-cummin-indices-empty");
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
            return Ok(ProviderCumminResult { values, indices });
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cummin: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let values_buffer = self.create_storage_buffer(0, "runmat-cummin-values-empty");
            let indices_buffer = self.create_storage_buffer(0, "runmat-cummin-indices-empty");
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
            return Ok(ProviderCumminResult { values, indices });
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cummin: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cummin: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cummin: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cummin: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cummin: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cummin: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cummin: tensor too large for GPU kernel"
        );

        let values_buffer = self.create_storage_buffer(entry.len, "runmat-cummin-values");
        let indices_buffer = self.create_storage_buffer(entry.len, "runmat-cummin-indices");

        let mut flags = 0u32;
        if matches!(direction, ProviderScanDirection::Reverse) {
            flags |= 1;
        }
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CumminParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-cummin-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-cummin-bind"),
                layout: &self.pipelines.cummin.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: values_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: indices_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            segments as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scan::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.cummin.pipeline,
            &bind_group,
            groups,
            "runmat-cummin-encoder",
            "runmat-cummin-pass",
        );

        let values = self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
        let indices = self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
        Ok(ProviderCumminResult { values, indices })
    }
    pub(crate) fn cummax_exec(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> Result<ProviderCummaxResult> {
        let entry = self.get_entry(handle)?;
        if entry.len == 0 {
            let values_buffer = self.create_storage_buffer(0, "runmat-cummax-values-empty");
            let indices_buffer = self.create_storage_buffer(0, "runmat-cummax-indices-empty");
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
            return Ok(ProviderCummaxResult { values, indices });
        }

        let work_shape = if entry.shape.is_empty() {
            vec![entry.len]
        } else {
            entry.shape.clone()
        };

        if dim >= work_shape.len() {
            return Err(anyhow!(
                "cummax: dimension {} exceeds tensor rank {}",
                dim + 1,
                work_shape.len()
            ));
        }

        let segment_len = work_shape[dim];
        if segment_len == 0 {
            let values_buffer = self.create_storage_buffer(0, "runmat-cummax-values-empty");
            let indices_buffer = self.create_storage_buffer(0, "runmat-cummax-indices-empty");
            let values =
                self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
            let indices =
                self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
            return Ok(ProviderCummaxResult { values, indices });
        }

        let stride_before = if dim == 0 {
            1usize
        } else {
            work_shape[..dim].iter().copied().product::<usize>().max(1)
        };
        let stride_after = if dim + 1 >= work_shape.len() {
            1usize
        } else {
            work_shape[dim + 1..]
                .iter()
                .copied()
                .product::<usize>()
                .max(1)
        };

        let segments = stride_before
            .checked_mul(stride_after)
            .ok_or_else(|| anyhow!("cummax: segment count exceeds GPU limits"))?;
        let block = stride_before
            .checked_mul(segment_len)
            .ok_or_else(|| anyhow!("cummax: segment stride exceeds GPU limits"))?;

        ensure!(
            segment_len <= u32::MAX as usize,
            "cummax: dimension length exceeds GPU kernel limits"
        );
        ensure!(
            stride_before <= u32::MAX as usize,
            "cummax: stride_before exceeds GPU kernel limits"
        );
        ensure!(
            segments <= u32::MAX as usize,
            "cummax: segment count exceeds GPU limits"
        );
        ensure!(
            block <= u32::MAX as usize,
            "cummax: block size exceeds GPU limits"
        );
        ensure!(
            entry.len <= u32::MAX as usize,
            "cummax: tensor too large for GPU kernel"
        );

        let values_buffer = self.create_storage_buffer(entry.len, "runmat-cummax-values");
        let indices_buffer = self.create_storage_buffer(entry.len, "runmat-cummax-indices");

        let mut flags = 0u32;
        if matches!(direction, ProviderScanDirection::Reverse) {
            flags |= 1;
        }
        if matches!(nan_mode, ProviderNanMode::Omit) {
            flags |= 2;
        }

        let params = CummaxParams {
            segment_len: segment_len as u32,
            segments: segments as u32,
            stride_before: stride_before as u32,
            block: block as u32,
            flags,
            total_len: entry.len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-cummax-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-cummax-bind"),
                layout: &self.pipelines.cummax.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: entry.buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: values_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: indices_buffer.as_ref().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let groups = crate::backend::wgpu::dispatch::common::dispatch_size(
            segments as u32,
            crate::backend::wgpu::config::WORKGROUP_SIZE,
        );
        crate::backend::wgpu::dispatch::scan::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.cummax.pipeline,
            &bind_group,
            groups,
            "runmat-cummax-encoder",
            "runmat-cummax-pass",
        );

        let values = self.register_existing_buffer(values_buffer, entry.shape.clone(), entry.len);
        let indices = self.register_existing_buffer(indices_buffer, entry.shape.clone(), entry.len);
        Ok(ProviderCummaxResult { values, indices })
    }
}

fn spectral_selected_frequency_len(nfft: usize, range: ProviderSpectralRange) -> usize {
    match range {
        ProviderSpectralRange::Onesided => nfft / 2 + 1,
        ProviderSpectralRange::Twosided | ProviderSpectralRange::Centered => nfft,
    }
}

fn envelope_hilbert_fir_kernel(filter_len: usize) -> Vec<f64> {
    if filter_len == 0 {
        return Vec::new();
    }
    let center = (filter_len as f64 - 1.0) / 2.0;
    let denominator = envelope_modified_bessel_i0(8.0);
    (0..filter_len)
        .map(|idx| {
            let k = idx as f64 - center;
            let ideal = if k.abs() <= f64::EPSILON {
                0.0
            } else {
                let rounded = k.round();
                if (rounded - k).abs() <= f64::EPSILON && (rounded as i64).rem_euclid(2) == 0 {
                    0.0
                } else {
                    2.0 / (std::f64::consts::PI * k)
                }
            };
            ideal * envelope_kaiser_window(idx, filter_len, denominator)
        })
        .collect()
}

fn envelope_input_shape_matches(shape: &[usize], channel_len: usize, channel_count: usize) -> bool {
    if channel_count == 1 {
        return match shape {
            [len] => *len == channel_len,
            [rows, cols] => {
                (*rows == channel_len && *cols == 1) || (*rows == 1 && *cols == channel_len)
            }
            _ => false,
        };
    }

    matches!(shape, [rows, cols] if *rows == channel_len && *cols == channel_count)
}

fn envelope_kaiser_window(idx: usize, len: usize, denominator: f64) -> f64 {
    if len <= 1 {
        return 1.0;
    }
    let ratio = 2.0 * idx as f64 / (len - 1) as f64 - 1.0;
    let argument = 8.0 * (1.0 - ratio * ratio).max(0.0).sqrt();
    envelope_modified_bessel_i0(argument) / denominator
}

fn envelope_modified_bessel_i0(x: f64) -> f64 {
    let y = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..=32 {
        term *= y / ((k * k) as f64);
        sum += term;
        if term.abs() <= sum.abs() * 1.0e-15 {
            break;
        }
    }
    sum
}

fn spectral_frame_shader_mode(mode: ProviderSpectralFrameMode) -> SpectralFrameShaderMode {
    match mode {
        ProviderSpectralFrameMode::Sliding { hop } => SpectralFrameShaderMode::Sliding { hop },
        ProviderSpectralFrameMode::ColumnSliding {
            hop,
            input_rows,
            frames_per_column,
        } => SpectralFrameShaderMode::ColumnSliding {
            hop,
            input_rows,
            frames_per_column,
        },
        ProviderSpectralFrameMode::FoldedColumns { input_rows } => {
            SpectralFrameShaderMode::FoldedColumns { input_rows }
        }
    }
}

fn spectral_range_shader_mode(range: ProviderSpectralRange) -> SpectralRangeShaderMode {
    match range {
        ProviderSpectralRange::Onesided => SpectralRangeShaderMode::Onesided,
        ProviderSpectralRange::Twosided => SpectralRangeShaderMode::Twosided,
        ProviderSpectralRange::Centered => SpectralRangeShaderMode::Centered,
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use num_complex::Complex;
    use runmat_accelerate_api::{
        AccelProvider, GpuTensorStorage, HostTensorView, ProviderEnvelopeMethod,
        ProviderEnvelopeRequest, ProviderHilbertRequest,
    };
    use runmat_builtins::ComplexTensor;
    use runmat_runtime::builtins::math::reduction::gradient_complex_tensor_host;
    use rustfft::FftPlanner;
    use std::sync::{Mutex, OnceLock};

    fn with_wgpu_provider<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&'static dyn AccelProvider) -> R,
    {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = LOCK.get_or_init(|| Mutex::new(())).lock().expect("lock");
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return None;
        };
        Some(f(provider))
    }

    fn run_envelope(method: ProviderEnvelopeMethod, values: &[f64], shape: &[usize]) -> Vec<f64> {
        with_wgpu_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: values,
                    shape,
                })
                .expect("upload");
            let result = pollster::block_on(provider.signal_envelope(&ProviderEnvelopeRequest {
                input: &input,
                channel_len: values.len(),
                channel_count: 1,
                output_shape: shape,
                method,
            }))
            .expect("envelope");
            let upper = pollster::block_on(provider.download(&result.upper)).expect("download");
            provider.free(&input).ok();
            provider.free(&result.upper).ok();
            provider.free(&result.lower).ok();
            upper.data
        })
        .unwrap_or_default()
    }

    fn run_hilbert(
        values: &[f64],
        shape: &[usize],
        length: Option<usize>,
        dim: usize,
    ) -> Vec<(f64, f64)> {
        with_wgpu_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: values,
                    shape,
                })
                .expect("upload");
            let output = pollster::block_on(provider.signal_hilbert(&ProviderHilbertRequest {
                input: &input,
                length,
                dim,
            }))
            .expect("hilbert");
            let host = pollster::block_on(provider.download(&output)).expect("download");
            assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
            assert_eq!(host.data.len() % 2, 0);
            provider.free(&input).ok();
            provider.free(&output).ok();
            host.data
                .chunks_exact(2)
                .map(|pair| (pair[0], pair[1]))
                .collect()
        })
        .unwrap_or_default()
    }

    fn run_complex_gradient(
        values: &[(f64, f64)],
        shape: &[usize],
        dim: usize,
        spacing: f64,
    ) -> Vec<(f64, f64)> {
        with_wgpu_provider(|provider| {
            let mut interleaved = Vec::with_capacity(values.len() * 2);
            for &(re, im) in values {
                interleaved.push(re);
                interleaved.push(im);
            }
            let input = provider
                .upload(&HostTensorView {
                    data: &interleaved,
                    shape,
                })
                .expect("upload");
            runmat_accelerate_api::set_handle_storage(&input, GpuTensorStorage::ComplexInterleaved);
            let output = provider
                .gradient_dim(&input, dim, spacing)
                .expect("gradient");
            let host = pollster::block_on(provider.download(&output)).expect("download");
            assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
            assert_eq!(host.data.len() % 2, 0);
            provider.free(&input).ok();
            provider.free(&output).ok();
            host.data
                .chunks_exact(2)
                .map(|pair| (pair[0], pair[1]))
                .collect()
        })
        .unwrap_or_default()
    }

    fn host_hilbert(
        values: &[f64],
        shape: &[usize],
        length: Option<usize>,
        dim: usize,
    ) -> Vec<(f64, f64)> {
        let mut ext_shape = shape.to_vec();
        while ext_shape.len() <= dim {
            ext_shape.push(1);
        }
        let current_len = ext_shape[dim];
        let target_len = length.unwrap_or(current_len);
        let inner_stride = ext_shape[..dim].iter().copied().product::<usize>();
        let outer_stride = ext_shape[dim + 1..].iter().copied().product::<usize>();
        let mut out = vec![Complex::new(0.0, 0.0); target_len * inner_stride * outer_stride];
        let input = values
            .iter()
            .copied()
            .map(|re| Complex::new(re, 0.0))
            .collect::<Vec<_>>();
        let mut planner = FftPlanner::<f64>::new();
        let forward = (target_len > 1).then(|| planner.plan_fft_forward(target_len));
        let inverse = (target_len > 1).then(|| planner.plan_fft_inverse(target_len));
        let mut buffer = vec![Complex::new(0.0, 0.0); target_len];
        let copy_len = current_len.min(target_len);

        for outer in 0..outer_stride {
            let base_in = outer * current_len * inner_stride;
            let base_out = outer * target_len * inner_stride;
            for inner in 0..inner_stride {
                buffer.fill(Complex::new(0.0, 0.0));
                for (freq, slot) in buffer.iter_mut().enumerate().take(copy_len) {
                    *slot = input[base_in + inner + freq * inner_stride];
                }
                if let Some(plan) = &forward {
                    plan.process(&mut buffer);
                }
                for (freq, slot) in buffer.iter_mut().enumerate() {
                    *slot *= analytic_test_multiplier(freq, target_len);
                }
                if let Some(plan) = &inverse {
                    plan.process(&mut buffer);
                }
                let scale = if target_len > 0 {
                    1.0 / target_len as f64
                } else {
                    1.0
                };
                for (freq, value) in buffer.iter().enumerate() {
                    out[base_out + inner + freq * inner_stride] = *value * scale;
                }
            }
        }

        out.into_iter().map(|value| (value.re, value.im)).collect()
    }

    fn analytic_test_multiplier(freq: usize, len: usize) -> f64 {
        if freq == 0 {
            1.0
        } else if len.is_multiple_of(2) {
            if freq < len / 2 {
                2.0
            } else if freq == len / 2 {
                1.0
            } else {
                0.0
            }
        } else if freq <= len / 2 {
            2.0
        } else {
            0.0
        }
    }

    fn assert_complex_slices_close(actual: &[(f64, f64)], expected: &[(f64, f64)], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual.0 - expected.0).abs() <= tol,
                "real mismatch at {idx}: actual={} expected={}",
                actual.0,
                expected.0
            );
            assert!(
                (actual.1 - expected.1).abs() <= tol,
                "imag mismatch at {idx}: actual={} expected={}",
                actual.1,
                expected.1
            );
        }
    }

    #[test]
    fn signal_envelope_rms_provider_matches_centered_window() {
        let values = [0.0, 3.0, 4.0, 0.0];
        let upper = run_envelope(
            ProviderEnvelopeMethod::Rms { window_len: 3 },
            &values,
            &[4, 1],
        );
        if upper.is_empty() {
            return;
        }
        assert!((upper[1] - (25.0f64 / 3.0).sqrt()).abs() < 1.0e-5);
    }

    #[test]
    fn signal_envelope_provider_rejects_layout_mismatch_with_same_len() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let Some(()) = with_wgpu_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &[3, 2],
                })
                .expect("upload");
            let output_shape = [2, 3];
            let err = pollster::block_on(provider.signal_envelope(&ProviderEnvelopeRequest {
                input: &input,
                channel_len: 2,
                channel_count: 3,
                output_shape: &output_shape,
                method: ProviderEnvelopeMethod::Rms { window_len: 3 },
            }))
            .expect_err("layout mismatch should be rejected");

            provider.free(&input).ok();
            assert!(
                err.to_string().contains("input shape mismatch"),
                "unexpected error: {err:#}"
            );
        }) else {
            return;
        };
    }

    #[test]
    fn signal_envelope_analytic_provider_keeps_constant_bounds() {
        let values = [2.5; 8];
        let upper = run_envelope(ProviderEnvelopeMethod::Analytic, &values, &[8, 1]);
        if upper.is_empty() {
            return;
        }
        for value in upper {
            assert!((value - 2.5).abs() < 1.0e-10);
        }
    }

    #[test]
    fn signal_envelope_analytic_fir_provider_keeps_constant_bounds() {
        let values = [1.25; 8];
        let upper = run_envelope(
            ProviderEnvelopeMethod::AnalyticFir { filter_len: 5 },
            &values,
            &[8, 1],
        );
        if upper.is_empty() {
            return;
        }
        for value in upper {
            assert!((value - 1.25).abs() < 1.0e-10);
        }
    }

    #[test]
    fn signal_hilbert_provider_matches_row_cosine() {
        let values = [1.0, 0.0, -1.0, 0.0];
        let out = run_hilbert(&values, &[1, 4], None, 1);
        if out.is_empty() {
            return;
        }
        let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        for (idx, (actual, expected)) in out.iter().copied().zip(expected).enumerate() {
            assert!(
                (actual.0 - expected.0).abs() < 1.0e-5,
                "real mismatch at {idx}: actual={} expected={}",
                actual.0,
                expected.0
            );
            assert!(
                (actual.1 - expected.1).abs() < 1.0e-5,
                "imag mismatch at {idx}: actual={} expected={}",
                actual.1,
                expected.1
            );
        }
    }

    #[test]
    fn signal_hilbert_provider_operates_down_columns() {
        let values = [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0];
        let out = run_hilbert(&values, &[4, 2], None, 0);
        if out.is_empty() {
            return;
        }
        let expected = [
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
            (0.0, -1.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
        ];
        for (idx, (actual, expected)) in out.iter().copied().zip(expected).enumerate() {
            assert!(
                (actual.0 - expected.0).abs() < 1.0e-5,
                "real mismatch at {idx}: actual={} expected={}",
                actual.0,
                expected.0
            );
            assert!(
                (actual.1 - expected.1).abs() < 1.0e-5,
                "imag mismatch at {idx}: actual={} expected={}",
                actual.1,
                expected.1
            );
        }
    }

    #[test]
    fn signal_hilbert_provider_matches_host_for_padded_inner_stride() {
        let values = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let out = run_hilbert(&values, &[2, 3], Some(5), 1);
        if out.is_empty() {
            return;
        }
        let expected = host_hilbert(&values, &[2, 3], Some(5), 1);
        assert_complex_slices_close(&out, &expected, 1.0e-5);
    }

    #[test]
    fn signal_hilbert_provider_matches_host_for_truncated_odd_signal() {
        let values = [1.0, 2.0, 0.5, -1.0, 0.25];
        let out = run_hilbert(&values, &[1, 5], Some(3), 1);
        if out.is_empty() {
            return;
        }
        let expected = host_hilbert(&values, &[1, 5], Some(3), 1);
        assert_complex_slices_close(&out, &expected, 1.0e-5);
    }

    #[test]
    fn gradient_provider_complex_matches_host_and_stays_interleaved() {
        let values = [
            (1.0, 1.0),
            (2.0, -1.0),
            (4.0, 3.0),
            (6.0, 2.0),
            (9.0, 6.0),
            (12.0, 4.0),
        ];
        let out = run_complex_gradient(&values, &[2, 3], 1, 2.0);
        if out.is_empty() {
            return;
        }
        let expected = gradient_complex_tensor_host(
            ComplexTensor::new(values.to_vec(), vec![2, 3]).expect("complex tensor"),
            2,
            2.0,
        )
        .expect("host gradient");
        assert_complex_slices_close(&out, &expected.data, 1.0e-5);
    }
}
