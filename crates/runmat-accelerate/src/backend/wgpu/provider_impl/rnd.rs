use super::*;

impl WgpuProvider {
    pub(crate) fn random_exponential_exec(
        &self,
        mu: f64,
        shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("exprnd: tensor size exceeds GPU limits"))?;
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-rng-exprnd-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "exprnd: tensor length too large"
        );

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut rng_guard = rng_state()
            .lock()
            .map_err(|_| anyhow!("exprnd: provider RNG mutex poisoned"))?;
        let mut chunk_state = *rng_guard;

        let mut offset = 0usize;
        while offset < total_len {
            let remaining = total_len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let chunk_u32 = u32::try_from(chunk_len)
                .map_err(|_| anyhow!("exprnd: chunk length exceeds GPU dispatch limits"))?;
            let offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("exprnd: tensor offset exceeds GPU limits"))?;
            let (key0, key1) = philox_keys_from_state(chunk_state);

            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::RandomDistParamsF64 {
                        offset: offset_u32,
                        chunk: chunk_u32,
                        key0,
                        key1,
                        param1: mu,
                        param2: 0.0,
                    };
                    self.uniform_buffer(&params, "runmat-rng-exprnd-f64-params")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::RandomDistParamsF32 {
                        offset: offset_u32,
                        chunk: chunk_u32,
                        key0,
                        key1,
                        param1: mu as f32,
                        param2: 0.0,
                        _pad0: 0,
                        _pad1: 0,
                    };
                    self.uniform_buffer(&params, "runmat-rng-exprnd-f32-params")
                }
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-rng-exprnd-bind"),
                    layout: &self.pipelines.random_exprnd.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.random_exprnd.pipeline,
                &bind_group,
                workgroups,
                "runmat-rng-exprnd-encoder",
                "runmat-rng-exprnd-pass",
            );

            chunk_state = advance_rng_state(chunk_state, u64::from(chunk_u32));
            offset += chunk_len;
        }

        *rng_guard = chunk_state;
        drop(rng_guard);

        Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), total_len))
    }

    pub(crate) fn random_normrnd_exec(
        &self,
        mu: f64,
        sigma: f64,
        shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("normrnd: tensor size exceeds GPU limits"))?;
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-rng-normrnd-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "normrnd: tensor length too large"
        );

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut rng_guard = rng_state()
            .lock()
            .map_err(|_| anyhow!("normrnd: provider RNG mutex poisoned"))?;
        let mut chunk_state = *rng_guard;

        let mut offset = 0usize;
        while offset < total_len {
            let remaining = total_len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let chunk_u32 = u32::try_from(chunk_len)
                .map_err(|_| anyhow!("normrnd: chunk length exceeds GPU dispatch limits"))?;
            let offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("normrnd: tensor offset exceeds GPU limits"))?;
            let (key0, key1) = philox_keys_from_state(chunk_state);

            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::RandomDistParamsF64 {
                        offset: offset_u32,
                        chunk: chunk_u32,
                        key0,
                        key1,
                        param1: mu,
                        param2: sigma,
                    };
                    self.uniform_buffer(&params, "runmat-rng-normrnd-f64-params")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::RandomDistParamsF32 {
                        offset: offset_u32,
                        chunk: chunk_u32,
                        key0,
                        key1,
                        param1: mu as f32,
                        param2: sigma as f32,
                        _pad0: 0,
                        _pad1: 0,
                    };
                    self.uniform_buffer(&params, "runmat-rng-normrnd-f32-params")
                }
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-rng-normrnd-bind"),
                    layout: &self.pipelines.random_normrnd.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.random_normrnd.pipeline,
                &bind_group,
                workgroups,
                "runmat-rng-normrnd-encoder",
                "runmat-rng-normrnd-pass",
            );

            // Box-Muller emits two normals per two uniform samples.
            let delta = u64::from(chunk_u32).div_ceil(2) * 2;
            chunk_state = advance_rng_state(chunk_state, delta);
            offset += chunk_len;
        }

        *rng_guard = chunk_state;
        drop(rng_guard);

        Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), total_len))
    }

    pub(crate) fn random_unifrnd_exec(
        &self,
        a: f64,
        b: f64,
        shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("unifrnd: tensor size exceeds GPU limits"))?;
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-rng-unifrnd-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "unifrnd: tensor length too large"
        );

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut rng_guard = rng_state()
            .lock()
            .map_err(|_| anyhow!("unifrnd: provider RNG mutex poisoned"))?;
        let mut chunk_state = *rng_guard;

        let mut offset = 0usize;
        while offset < total_len {
            let remaining = total_len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let chunk_u32 = u32::try_from(chunk_len)
                .map_err(|_| anyhow!("unifrnd: chunk length exceeds GPU dispatch limits"))?;
            let offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("unifrnd: tensor offset exceeds GPU limits"))?;
            let (key0, key1) = philox_keys_from_state(chunk_state);

            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::RandomDistParamsF64 {
                        offset: offset_u32,
                        chunk: chunk_u32,
                        key0,
                        key1,
                        param1: a,
                        param2: b,
                    };
                    self.uniform_buffer(&params, "runmat-rng-unifrnd-f64-params")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::RandomDistParamsF32 {
                        offset: offset_u32,
                        chunk: chunk_u32,
                        key0,
                        key1,
                        param1: a as f32,
                        param2: b as f32,
                        _pad0: 0,
                        _pad1: 0,
                    };
                    self.uniform_buffer(&params, "runmat-rng-unifrnd-f32-params")
                }
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-rng-unifrnd-bind"),
                    layout: &self.pipelines.random_unifrnd.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: out_buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let workgroups = crate::backend::wgpu::dispatch::common::dispatch_size(
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &self.pipelines.random_unifrnd.pipeline,
                &bind_group,
                workgroups,
                "runmat-rng-unifrnd-encoder",
                "runmat-rng-unifrnd-pass",
            );

            chunk_state = advance_rng_state(chunk_state, u64::from(chunk_u32));
            offset += chunk_len;
        }

        *rng_guard = chunk_state;
        drop(rng_guard);

        Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), total_len))
    }
}
