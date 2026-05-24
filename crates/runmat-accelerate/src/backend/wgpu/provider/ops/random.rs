use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_time::Instant;

use super::{
    advance_rng_state, philox_keys_from_state, product_checked, rng_state, seed_from_state,
    NumericPrecision, WgpuProvider, MAX_SAFE_INTEGER,
};

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
    pub(crate) fn random_uniform_exec(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("rand: tensor size exceeds GPU limits"))?;
        if std::env::var("RUNMAT_DISABLE_RNG").is_ok()
            || std::env::var("RUNMAT_DISABLE_RAND").is_ok()
            || std::env::var("RUNMAT_DISABLE_RANDUNIFORM").is_ok()
        {
            // Debug-only CPU fallback: fill with deterministic values in [0,1)
            let mut out = vec![0.0f64; total_len];
            for (i, value) in out.iter_mut().enumerate().take(total_len) {
                *value = ((i as u64).wrapping_mul(1664525).wrapping_add(1013904223) % (1u64 << 32))
                    as f64
                    / 4294967296.0f64;
            }
            return self.upload(&HostTensorView { data: &out, shape });
        }
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-rng-uniform-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "rand: tensor length too large"
        );

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut rng_guard = rng_state()
            .lock()
            .map_err(|_| anyhow!("rand: provider RNG mutex poisoned"))?;
        let mut chunk_state = *rng_guard;

        let mut offset = 0usize;
        while offset < total_len {
            let remaining = total_len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let chunk_u32 = u32::try_from(chunk_len)
                .map_err(|_| anyhow!("rand: chunk length exceeds GPU dispatch limits"))?;
            let offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("rand: tensor offset exceeds GPU limits"))?;
            let (key0, key1) = philox_keys_from_state(chunk_state);

            let params = crate::backend::wgpu::params::RandomScalarParams {
                offset: offset_u32,
                chunk: chunk_u32,
                key0,
                key1,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-rng-uniform-params");

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-rng-uniform-bind"),
                    layout: &self.pipelines.random_uniform.layout,
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
                &self.pipelines.random_uniform.pipeline,
                &bind_group,
                workgroups,
                "runmat-rng-uniform-encoder",
                "runmat-rng-uniform-pass",
            );

            chunk_state = advance_rng_state(chunk_state, u64::from(chunk_u32));
            offset += chunk_len;
        }

        *rng_guard = chunk_state;
        drop(rng_guard);

        Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), total_len))
    }

    pub(crate) fn random_normal_exec(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("randn: tensor size exceeds GPU limits"))?;
        if std::env::var("RUNMAT_DISABLE_RNG").is_ok()
            || std::env::var("RUNMAT_DISABLE_RAND").is_ok()
            || std::env::var("RUNMAT_DISABLE_RANDN").is_ok()
        {
            // Debug-only CPU fallback: simple deterministic normal-ish via Box-Muller on uint LCG
            let mut out = vec![0.0f64; total_len];
            let mut state: u64 = 0x9e3779b97f4a7c15;
            let next_u32 = |s: &mut u64| -> u32 {
                *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *s as u32
            };
            let mut i = 0usize;
            while i < total_len {
                let u1 = (next_u32(&mut state) as f64 + 1.0) / 4294967297.0;
                let u2 = (next_u32(&mut state) as f64 + 1.0) / 4294967297.0;
                let r = (-2.0f64 * u1.ln()).sqrt();
                let theta = 2.0f64 * std::f64::consts::PI * u2;
                let z0 = r * theta.cos();
                let z1 = r * theta.sin();
                out[i] = z0;
                if i + 1 < total_len {
                    out[i + 1] = z1;
                }
                i += 2;
            }
            return self.upload(&HostTensorView { data: &out, shape });
        }
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-rng-normal-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "randn: tensor length too large"
        );

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut rng_guard = rng_state()
            .lock()
            .map_err(|_| anyhow!("randn: provider RNG mutex poisoned"))?;
        let mut chunk_state = *rng_guard;

        let mut offset = 0usize;
        while offset < total_len {
            let remaining = total_len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let chunk_u32 = u32::try_from(chunk_len)
                .map_err(|_| anyhow!("randn: chunk length exceeds GPU dispatch limits"))?;
            let offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("randn: tensor offset exceeds GPU limits"))?;
            let (key0, key1) = philox_keys_from_state(chunk_state);

            let params = crate::backend::wgpu::params::RandomScalarParams {
                offset: offset_u32,
                chunk: chunk_u32,
                key0,
                key1,
            };
            let params_buffer = self.uniform_buffer(&params, "runmat-rng-normal-params");

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-rng-normal-bind"),
                    layout: &self.pipelines.random_normal.layout,
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
                &self.pipelines.random_normal.pipeline,
                &bind_group,
                workgroups,
                "runmat-rng-normal-encoder",
                "runmat-rng-normal-pass",
            );

            let delta = u64::from(chunk_u32) * 2;
            chunk_state = advance_rng_state(chunk_state, delta);
            offset += chunk_len;
        }

        *rng_guard = chunk_state;
        drop(rng_guard);

        Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), total_len))
    }

    pub(crate) fn stochastic_evolution_exec(
        &self,
        state: &GpuTensorHandle,
        drift: f64,
        scale: f64,
        steps: u32,
    ) -> Result<GpuTensorHandle> {
        let total_len = product_checked(&state.shape)
            .ok_or_else(|| anyhow!("stochastic_evolution: tensor size exceeds GPU limits"))?;
        let start = Instant::now();
        let state_entry = self.get_entry(state)?;
        let out_buffer =
            self.create_storage_buffer_checked(total_len, "runmat-stochastic-evolution-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, state_entry.shape.clone(), 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "stochastic_evolution: tensor length too large"
        );

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut rng_guard = rng_state()
            .lock()
            .map_err(|_| anyhow!("stochastic_evolution: provider RNG mutex poisoned"))?;
        let mut chunk_state = *rng_guard;

        let pipeline = &self.pipelines.stochastic_evolution;
        let mut offset = 0usize;
        while offset < total_len {
            let remaining = total_len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let chunk_u32 = u32::try_from(chunk_len).map_err(|_| {
                anyhow!("stochastic_evolution: chunk length exceeds GPU dispatch limits")
            })?;
            let offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("stochastic_evolution: offset exceeds GPU limits"))?;
            let len_u32 = u32::try_from(total_len)
                .map_err(|_| anyhow!("stochastic_evolution: len overflow"))?;
            let (key0, key1) = philox_keys_from_state(chunk_state);

            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::StochasticEvolutionParamsF64 {
                        offset: offset_u32,
                        chunk: chunk_u32,
                        len: len_u32,
                        steps,
                        key0,
                        key1,
                        _pad0: 0,
                        _pad1: 0,
                        drift,
                        scale,
                    };
                    self.uniform_buffer(&params, "runmat-stochastic-evolution-f64-params")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::StochasticEvolutionParamsF32 {
                        offset: offset_u32,
                        chunk: chunk_u32,
                        len: len_u32,
                        steps,
                        key0,
                        key1,
                        _pad0: 0,
                        _pad1: 0,
                        drift: drift as f32,
                        scale: scale as f32,
                    };
                    self.uniform_buffer(&params, "runmat-stochastic-evolution-f32-params")
                }
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-stochastic-evolution-bind"),
                    layout: &pipeline.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: state_entry.buffer.as_ref().as_entire_binding(),
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
                chunk_u32,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
            );
            crate::backend::wgpu::dispatch::creation::run(
                self.device_ref(),
                self.queue_ref(),
                &pipeline.pipeline,
                &bind_group,
                workgroups,
                "runmat-stochastic-evolution-encoder",
                "runmat-stochastic-evolution-pass",
            );

            if steps > 0 {
                let advance = u64::from(chunk_u32) * u64::from(steps);
                chunk_state = advance_rng_state(chunk_state, advance);
            }
            offset += chunk_len;
        }

        *rng_guard = chunk_state;
        drop(rng_guard);

        self.telemetry
            .record_fused_elementwise_duration(start.elapsed());
        Ok(self.register_existing_buffer(out_buffer, state_entry.shape.clone(), total_len))
    }

    pub(crate) fn random_integer_range_exec(
        &self,
        lower: i64,
        upper: i64,
        shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        ensure!(lower <= upper, "randi: lower bound must be <= upper bound");
        let total_len = product_checked(shape)
            .ok_or_else(|| anyhow!("randi: tensor size exceeds GPU limits"))?;
        let out_buffer = self.create_storage_buffer_checked(total_len, "runmat-rng-int-out")?;
        if total_len == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), 0));
        }
        ensure!(
            total_len <= u32::MAX as usize,
            "randi: tensor length too large"
        );
        let span_i128 = (upper as i128)
            .checked_sub(lower as i128)
            .and_then(|d| d.checked_add(1))
            .ok_or_else(|| anyhow!("randi: integer range overflow"))?;
        ensure!(span_i128 > 0, "randi: integer range must be non-empty");
        ensure!(
            span_i128 <= (1i128 << 53),
            "randi: range cannot exceed 2^53"
        );
        let span_minus_one = span_i128 - 1;
        let span_u64 = span_i128 as u64;

        let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

        let mut rng_guard = rng_state()
            .lock()
            .map_err(|_| anyhow!("randi: provider RNG mutex poisoned"))?;
        let mut chunk_state = *rng_guard;

        let mut offset = 0usize;
        while offset < total_len {
            let remaining = total_len - offset;
            let chunk_len = remaining.min(chunk_capacity);
            let chunk_u32 = u32::try_from(chunk_len)
                .map_err(|_| anyhow!("randi: chunk length exceeds GPU dispatch limits"))?;
            let offset_u32 = u32::try_from(offset)
                .map_err(|_| anyhow!("randi: tensor offset exceeds GPU limits"))?;
            let seed = seed_from_state(chunk_state);

            let params_buffer = match self.precision {
                NumericPrecision::F64 => {
                    let params = crate::backend::wgpu::params::RandomIntParamsF64 {
                        lower: lower as f64,
                        upper: upper as f64,
                        span: span_u64 as f64,
                        span_minus_one: span_minus_one as f64,
                        offset: offset_u32,
                        chunk: chunk_u32,
                        seed,
                        _pad: 0,
                    };
                    self.uniform_buffer(&params, "runmat-rng-int-params-f64")
                }
                NumericPrecision::F32 => {
                    let params = crate::backend::wgpu::params::RandomIntParamsF32 {
                        lower: lower as f32,
                        upper: upper as f32,
                        span: span_u64 as f32,
                        span_minus_one: span_minus_one as f32,
                        offset: offset_u32,
                        chunk: chunk_u32,
                        seed,
                        _pad: 0,
                    };
                    self.uniform_buffer(&params, "runmat-rng-int-params-f32")
                }
            };

            let bind_group = self
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-rng-int-bind"),
                    layout: &self.pipelines.random_int.layout,
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
                &self.pipelines.random_int.pipeline,
                &bind_group,
                workgroups,
                "runmat-rng-int-encoder",
                "runmat-rng-int-pass",
            );

            chunk_state = advance_rng_state(chunk_state, u64::from(chunk_u32));
            offset += chunk_len;
        }

        *rng_guard = chunk_state;
        drop(rng_guard);

        Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), total_len))
    }
    pub(crate) fn randperm_exec(&self, n: usize, k: usize) -> Result<GpuTensorHandle> {
        ensure!(k <= n, "randperm: K must satisfy 0 <= K <= N");
        ensure!((n as u64) <= MAX_SAFE_INTEGER, "randperm: N exceeds 2^53");
        ensure!(
            n <= u32::MAX as usize,
            "randperm: N exceeds GPU dispatch limits"
        );
        ensure!(
            k <= u32::MAX as usize,
            "randperm: K exceeds GPU dispatch limits"
        );

        let effective_k = k.min(n);
        let shape_vec = vec![1, effective_k];
        let out_buffer = self.create_storage_buffer_checked(effective_k, "runmat-randperm-out")?;
        if effective_k == 0 {
            return Ok(self.register_existing_buffer(out_buffer, shape_vec, 0));
        }
        let mut rng_guard = rng_state()
            .lock()
            .map_err(|_| anyhow!("randperm: provider RNG mutex poisoned"))?;
        let seed = seed_from_state(*rng_guard);

        let params = crate::backend::wgpu::params::RandPermParams {
            n: n as u32,
            k: effective_k as u32,
            seed,
            _pad: 0,
        };
        let params_buffer = self.uniform_buffer(&params, "runmat-randperm-params");

        let bind_group = self
            .device_ref()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runmat-randperm-bind"),
                layout: &self.pipelines.randperm.layout,
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

        crate::backend::wgpu::dispatch::creation::run(
            self.device_ref(),
            self.queue_ref(),
            &self.pipelines.randperm.pipeline,
            &bind_group,
            1,
            "runmat-randperm-encoder",
            "runmat-randperm-pass",
        );

        *rng_guard = advance_rng_state(*rng_guard, effective_k as u64);
        drop(rng_guard);

        Ok(self.register_existing_buffer(out_buffer, shape_vec, effective_k))
    }
}
