use super::*;

impl WgpuProvider {
    pub(crate) async fn reduce_nd_mean_exec(
        &self,
        a: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<GpuTensorHandle> {
        // If input is a known square of a base tensor, reuse or compute ex2 via moments
        if let Ok(map) = self.pow2_of.lock() {
            if let Some(&base_id) = map.get(&a.buffer_id) {
                drop(map);
                // Try cache
                if let Ok(cache) = self.moments_cache.lock() {
                    if let Some((_mean_h, ex2_h)) = cache.get(&(base_id, dims_zero_based.to_vec()))
                    {
                        return Ok(ex2_h.clone());
                    }
                }
                // Build a handle for the base buffer id
                let base_entry = {
                    let guard = self.buffers.lock().expect("buffer mutex poisoned");
                    guard.get(&base_id).cloned()
                };
                if let Some(entry) = base_entry {
                    let base_handle = GpuTensorHandle {
                        shape: entry.shape.clone(),
                        device_id: self.runtime_device_id,
                        buffer_id: base_id,
                    };
                    let moments = self.reduce_moments_nd_exec(&base_handle, dims_zero_based)?;
                    if let Ok(mut cache2) = self.moments_cache.lock() {
                        cache2.insert(
                            (base_id, dims_zero_based.to_vec()),
                            (moments.mean.clone(), moments.ex2.clone()),
                        );
                    }
                    return Ok(moments.ex2);
                }
            }
        }
        // Prefer computing moments once and caching ex2 for future reuse
        if let Ok(cache) = self.moments_cache.lock() {
            let key = (a.buffer_id, dims_zero_based.to_vec());
            if let Some((mean_h, _ex2_h)) = cache.get(&key) {
                return Ok(mean_h.clone());
            }
            // Compute moments and store
            drop(cache);
            let moments = self.reduce_moments_nd_exec(a, dims_zero_based)?;
            if let Ok(mut cache2) = self.moments_cache.lock() {
                cache2.insert(
                    (a.buffer_id, dims_zero_based.to_vec()),
                    (moments.mean.clone(), moments.ex2.clone()),
                );
            }
            return Ok(moments.mean);
        }
        let entry = self.get_entry(a)?;
        let rank = entry.shape.len();
        ensure!(rank > 0, "reduce_mean_nd: rank must be > 0");
        let mut reduce: Vec<usize> = dims_zero_based
            .iter()
            .copied()
            .filter(|&d| d < rank)
            .collect();
        reduce.sort_unstable();
        reduce.dedup();
        ensure!(
            !reduce.is_empty(),
            "reduce_mean_nd: no valid dims to reduce"
        );
        let kept: Vec<usize> = (0..rank).filter(|i| !reduce.contains(i)).collect();

        // Compute strides (MATLAB/column-major)
        let mut strides: Vec<usize> = vec![0; rank];
        let mut s = 1usize;
        for (i, stride_slot) in strides.iter_mut().enumerate().take(rank) {
            *stride_slot = s;
            s = s
                .checked_mul(entry.shape[i])
                .ok_or_else(|| anyhow!("reduce_mean_nd: shape too large"))?;
        }

        let kept_sizes: Vec<u32> = kept.iter().map(|&i| entry.shape[i] as u32).collect();
        let reduce_sizes: Vec<u32> = reduce.iter().map(|&i| entry.shape[i] as u32).collect();
        let kept_strides: Vec<u32> = kept.iter().map(|&i| strides[i] as u32).collect();
        let reduce_strides: Vec<u32> = reduce.iter().map(|&i| strides[i] as u32).collect();

        let rows: usize = reduce
            .iter()
            .fold(1usize, |acc, &i| acc.saturating_mul(entry.shape[i]));
        let cols: usize = kept
            .iter()
            .fold(1usize, |acc, &i| acc.saturating_mul(entry.shape[i]));
        ensure!(rows > 0 && cols > 0, "reduce_mean_nd: empty tensor");
        if rows as u64 > u32::MAX as u64 || cols as u64 > u32::MAX as u64 {
            return Err(anyhow!("reduce_mean_nd: tensor exceeds GPU limits"));
        }

        // Heuristic fallback: for very large row extents, prefer sequenced dim-reductions.
        if rows >= self.two_pass_threshold() {
            let mut current = a.clone();
            let mut owned = false;
            for &d in &reduce {
                let next = self.reduce_mean_dim(&current, d).await?;
                if owned {
                    let _ = self.free_exec(&current);
                }
                current = next;
                owned = true;
            }
            return Ok(current);
        }

        let mut out_buffer = self.create_storage_buffer(cols, "runmat-reduce-nd-mean-out");
        // Prevent aliasing: output must not be the same as input buffer
        if std::ptr::eq(out_buffer.as_ref(), entry.buffer.as_ref()) {
            let size_bytes = (cols * self.element_size) as u64;
            out_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runmat-reduce-nd-mean-out-unique"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let mut out_shape = entry.shape.clone();
        for &d in &reduce {
            out_shape[d] = 1;
        }

        match entry.precision {
            NumericPrecision::F64 => {
                let mut params = crate::backend::wgpu::params::ReduceNdParams {
                    rank: rank as u32,
                    kept_count: kept.len() as u32,
                    reduce_count: reduce.len() as u32,
                    _pad: 0,
                    rows: rows as u32,
                    cols: cols as u32,
                    _pad2: [0; 2],
                    kept_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    kept_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                };
                for (i, v) in kept_sizes.iter().enumerate() {
                    params.kept_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_sizes.iter().enumerate() {
                    params.reduce_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in kept_strides.iter().enumerate() {
                    params.kept_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_strides.iter().enumerate() {
                    params.reduce_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                let pbuf = self.uniform_buffer(&params, "runmat-reduce-nd-mean-params-f64");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-reduce-nd-mean-bind-f64"),
                    layout: &self.pipelines.reduce_nd_mean.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    eprintln!(
                        "[reduce-nd-mean f64] in ptr={:p} out ptr={:p} rows={} cols={}",
                        entry.buffer.as_ref(),
                        out_buffer.as_ref(),
                        rows,
                        cols
                    );
                }
                // One workgroup per output column (kept slice)
                let groups_x = cols as u32;
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_nd_mean.pipeline,
                    &bind_group,
                    groups_x,
                );
            }
            NumericPrecision::F32 => {
                let mut params = crate::backend::wgpu::params::ReduceNdParams {
                    rank: rank as u32,
                    kept_count: kept.len() as u32,
                    reduce_count: reduce.len() as u32,
                    _pad: 0,
                    rows: rows as u32,
                    cols: cols as u32,
                    _pad2: [0; 2],
                    kept_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    kept_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                };
                for (i, v) in kept_sizes.iter().enumerate() {
                    params.kept_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_sizes.iter().enumerate() {
                    params.reduce_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in kept_strides.iter().enumerate() {
                    params.kept_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_strides.iter().enumerate() {
                    params.reduce_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                let pbuf = self.uniform_buffer(&params, "runmat-reduce-nd-mean-params-f32");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-reduce-nd-mean-bind-f32"),
                    layout: &self.pipelines.reduce_nd_mean.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });
                if std::env::var("RUNMAT_DEBUG_REDUCTION").is_ok() {
                    eprintln!(
                        "[reduce-nd-mean f32] in ptr={:p} out ptr={:p} rows={} cols={}",
                        entry.buffer.as_ref(),
                        out_buffer.as_ref(),
                        rows,
                        cols
                    );
                }
                // One workgroup per output column (kept slice)
                let groups_x = cols as u32;
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_nd_mean.pipeline,
                    &bind_group,
                    groups_x,
                );
            }
        }

        Ok(self.register_existing_buffer(out_buffer, out_shape, cols))
    }

    pub(crate) fn reduce_moments_nd_exec(
        &self,
        a: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<runmat_accelerate_api::ProviderMoments2> {
        let entry = self.get_entry(a)?;
        let rank = entry.shape.len();
        ensure!(rank > 0, "reduce_moments_nd: rank must be > 0");
        let mut reduce: Vec<usize> = dims_zero_based
            .iter()
            .copied()
            .filter(|&d| d < rank)
            .collect();
        reduce.sort_unstable();
        reduce.dedup();
        ensure!(
            !reduce.is_empty(),
            "reduce_moments_nd: no valid dims to reduce"
        );
        let kept: Vec<usize> = (0..rank).filter(|i| !reduce.contains(i)).collect();

        // Strides in column-major
        let mut strides: Vec<usize> = vec![0; rank];
        let mut s = 1usize;
        for (i, stride_slot) in strides.iter_mut().enumerate().take(rank) {
            *stride_slot = s;
            s = s
                .checked_mul(entry.shape[i])
                .ok_or_else(|| anyhow!("reduce_moments_nd: shape too large"))?;
        }

        let kept_sizes: Vec<u32> = kept.iter().map(|&i| entry.shape[i] as u32).collect();
        let reduce_sizes: Vec<u32> = reduce.iter().map(|&i| entry.shape[i] as u32).collect();
        let kept_strides: Vec<u32> = kept.iter().map(|&i| strides[i] as u32).collect();
        let reduce_strides: Vec<u32> = reduce.iter().map(|&i| strides[i] as u32).collect();

        let rows: usize = reduce
            .iter()
            .fold(1usize, |acc, &i| acc.saturating_mul(entry.shape[i]));
        let cols: usize = kept
            .iter()
            .fold(1usize, |acc, &i| acc.saturating_mul(entry.shape[i]));
        ensure!(rows > 0 && cols > 0, "reduce_moments_nd: empty tensor");
        if rows as u64 > u32::MAX as u64 || cols as u64 > u32::MAX as u64 {
            return Err(anyhow!("reduce_moments_nd: tensor exceeds GPU limits"));
        }

        // Allocate outputs
        let mean_out = self.create_storage_buffer(cols, "runmat-reduce-nd-moments-mean");
        let ex2_out = self.create_storage_buffer(cols, "runmat-reduce-nd-moments-ex2");
        let mut out_shape = entry.shape.clone();
        for &d in &reduce {
            out_shape[d] = 1;
        }

        match entry.precision {
            NumericPrecision::F64 => {
                let mut params = crate::backend::wgpu::params::ReduceNdParams {
                    rank: rank as u32,
                    kept_count: kept.len() as u32,
                    reduce_count: reduce.len() as u32,
                    _pad: 0,
                    rows: rows as u32,
                    cols: cols as u32,
                    _pad2: [0; 2],
                    kept_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    kept_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                };
                for (i, v) in kept_sizes.iter().enumerate() {
                    params.kept_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_sizes.iter().enumerate() {
                    params.reduce_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in kept_strides.iter().enumerate() {
                    params.kept_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_strides.iter().enumerate() {
                    params.reduce_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                let pbuf = self.uniform_buffer(&params, "runmat-reduce-nd-moments-params-f64");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-reduce-nd-moments-bind-f64"),
                    layout: &self.pipelines.reduce_nd_moments.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: mean_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: ex2_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });
                // One workgroup per output column (kept slice)
                let groups_x = cols as u32;
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_nd_moments.pipeline,
                    &bind_group,
                    groups_x,
                );
            }
            NumericPrecision::F32 => {
                let mut params = crate::backend::wgpu::params::ReduceNdParams {
                    rank: rank as u32,
                    kept_count: kept.len() as u32,
                    reduce_count: reduce.len() as u32,
                    _pad: 0,
                    rows: rows as u32,
                    cols: cols as u32,
                    _pad2: [0; 2],
                    kept_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_sizes: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    kept_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                    reduce_strides: [crate::backend::wgpu::params::AlignedU32::default();
                        crate::backend::wgpu::params::BCAST_MAX_RANK],
                };
                for (i, v) in kept_sizes.iter().enumerate() {
                    params.kept_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_sizes.iter().enumerate() {
                    params.reduce_sizes[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in kept_strides.iter().enumerate() {
                    params.kept_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                for (i, v) in reduce_strides.iter().enumerate() {
                    params.reduce_strides[i] = crate::backend::wgpu::params::AlignedU32::new(*v);
                }
                let pbuf = self.uniform_buffer(&params, "runmat-reduce-nd-moments-params-f32");
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-reduce-nd-moments-bind-f32"),
                    layout: &self.pipelines.reduce_nd_moments.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: entry.buffer.as_ref().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: mean_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: ex2_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });
                let groups_x = cols as u32;
                crate::backend::wgpu::dispatch::elementwise::run(
                    self.device_ref(),
                    self.queue_ref(),
                    &self.pipelines.reduce_nd_moments.pipeline,
                    &bind_group,
                    groups_x,
                );
            }
        }

        let mean_handle = self.register_existing_buffer(mean_out, out_shape.clone(), cols);
        let ex2_handle = self.register_existing_buffer(ex2_out, out_shape, cols);
        Ok(runmat_accelerate_api::ProviderMoments2 {
            mean: mean_handle,
            ex2: ex2_handle,
        })
    }
}
