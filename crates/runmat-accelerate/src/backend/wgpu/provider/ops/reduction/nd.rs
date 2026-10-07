use super::*;

#[path = "nd/geometry.rs"]
mod geometry;
use geometry::NdReductionGeometry;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "nd/lazy_tests.rs"]
mod lazy_tests;

impl WgpuProvider {
    pub(crate) async fn reduce_nd_mean_exec(
        &self,
        a: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<GpuTensorHandle> {
        // Cache hits must not bypass the original input's owner/live/precision admission.
        self.get_entry(a)?;
        if self.reduction_reuse.is_poisoned() {
            // The original uncached ND kernel remains authoritative when only
            // optional reuse bookkeeping is unavailable; make the loss visible.
            log::error!("WGPU reduction reuse cache poisoned; executing uncached ND mean");
        } else {
            // If input is a known square of a base tensor, reuse or compute ex2 via moments.
            let base = self.cached_square_source(a)?;
            if let Some(base) = base {
                if let Ok(base_handle) = base.acquire() {
                    self.get_entry(&base_handle)?;
                    if let Some((_, ex2)) =
                        self.cached_moment_pair(&base_handle, dims_zero_based)?
                    {
                        return Ok(ex2);
                    }
                    let moments = self.reduce_moments_nd_exec(&base_handle, dims_zero_based)?;
                    self.remember_moments(
                        &base_handle,
                        dims_zero_based,
                        &moments.mean,
                        &moments.ex2,
                    )?;
                    return Ok(moments.ex2);
                }
            }
            // Prefer computing moments once and caching ex2 for future reuse.
            if let Some((mean, _)) = self.cached_moment_pair(a, dims_zero_based)? {
                return Ok(mean);
            }
            let moments = self.reduce_moments_nd_exec(a, dims_zero_based)?;
            self.remember_moments(a, dims_zero_based, &moments.mean, &moments.ex2)?;
            return Ok(moments.mean);
        }
        let entry = self.get_entry(a)?;
        let NdReductionGeometry {
            rank,
            reduce,
            kept,
            kept_sizes,
            reduce_sizes,
            kept_strides,
            reduce_strides,
            rows,
            cols,
            out_shape,
        } = NdReductionGeometry::new(&entry.shape, dims_zero_based, "reduce_mean_nd")?;

        // Geometry above is logical. The stride-based kernel (and sequenced fallback)
        // needs canonical storage, not the backing order of a lazy transpose.
        // Keep this owner local; cache identity remains the caller's original view.
        let materialized = self.materialize_nd_reduction_input(a)?;
        let a = materialized.as_ref().unwrap_or(a);
        let entry = self.get_entry(a)?;

        // Heuristic fallback: for very large row extents, prefer sequenced dim-reductions.
        if rows >= self.two_pass_threshold() {
            let mut current = a.clone();
            let mut owned = false;
            for &d in &reduce {
                let next = self.reduce_mean_dim(&current, d).await?;
                if owned {
                    drop(current);
                }
                current = next;
                owned = true;
            }
            return Ok(current);
        }

        let (out_buffer, out_shape) = self.with_emission(|emission, pending| -> Result<_> {
            let mut out_buffer =
                self.create_storage_buffer(emission, cols, "runmat-reduce-nd-mean-out")?;
            // Prevent aliasing: output must not be the same as input buffer
            if std::ptr::eq(out_buffer.as_ref(), entry.buffer.as_ref()) {
                let size_bytes = (cols * self.element_size) as u64;
                out_buffer = Arc::new(emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("runmat-reduce-nd-mean-out-unique"),
                    size: size_bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            let out_shape = out_shape.clone();

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
                        params.reduce_strides[i] =
                            crate::backend::wgpu::params::AlignedU32::new(*v);
                    }
                    let pbuf =
                        self.uniform_buffer(emission, &params, "runmat-reduce-nd-mean-params-f64")?;
                    let bind_group =
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-reduce-nd-mean-bind-f64"),
                                layout: &self.pipelines.reduce_nd_mean.get(emission)?.layout,
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
                        emission,
                        pending,
                        &self.pipelines.reduce_nd_mean.get(emission)?.pipeline,
                        &bind_group,
                        groups_x,
                    )?;
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
                        params.reduce_strides[i] =
                            crate::backend::wgpu::params::AlignedU32::new(*v);
                    }
                    let pbuf =
                        self.uniform_buffer(emission, &params, "runmat-reduce-nd-mean-params-f32")?;
                    let bind_group =
                        emission
                            .device()
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("runmat-reduce-nd-mean-bind-f32"),
                                layout: &self.pipelines.reduce_nd_mean.get(emission)?.layout,
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
                        emission,
                        pending,
                        &self.pipelines.reduce_nd_mean.get(emission)?.pipeline,
                        &bind_group,
                        groups_x,
                    )?;
                }
            }

            Ok((out_buffer, out_shape))
        })??;
        self.register_existing_buffer(out_buffer, out_shape, cols)
    }

    pub(crate) fn reduce_moments_nd_exec(
        &self,
        a: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<runmat_accelerate_api::ProviderMoments2> {
        let entry = self.get_entry(a)?;
        let NdReductionGeometry {
            rank,
            reduce,
            kept,
            kept_sizes,
            reduce_sizes,
            kept_strides,
            reduce_strides,
            rows,
            cols,
            out_shape,
        } = NdReductionGeometry::new(&entry.shape, dims_zero_based, "reduce_moments_nd")?;

        // Mean's normal cache-miss path and direct moments/variance consumers
        // share this boundary. Validate geometry before issuing the storage move.
        let materialized = self.materialize_nd_reduction_input(a)?;
        let a = materialized.as_ref().unwrap_or(a);
        let entry = self.get_entry(a)?;

        let (mean_out, ex2_out, out_shape) =
            self.with_emission(|emission, pending| -> Result<_> {
                // Allocate outputs
                let mean_out =
                    self.create_storage_buffer(emission, cols, "runmat-reduce-nd-moments-mean")?;
                let ex2_out =
                    self.create_storage_buffer(emission, cols, "runmat-reduce-nd-moments-ex2")?;
                let out_shape = out_shape.clone();

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
                            params.kept_sizes[i] =
                                crate::backend::wgpu::params::AlignedU32::new(*v);
                        }
                        for (i, v) in reduce_sizes.iter().enumerate() {
                            params.reduce_sizes[i] =
                                crate::backend::wgpu::params::AlignedU32::new(*v);
                        }
                        for (i, v) in kept_strides.iter().enumerate() {
                            params.kept_strides[i] =
                                crate::backend::wgpu::params::AlignedU32::new(*v);
                        }
                        for (i, v) in reduce_strides.iter().enumerate() {
                            params.reduce_strides[i] =
                                crate::backend::wgpu::params::AlignedU32::new(*v);
                        }
                        let pbuf = self.uniform_buffer(
                            emission,
                            &params,
                            "runmat-reduce-nd-moments-params-f64",
                        )?;
                        let bind_group =
                            emission
                                .device()
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("runmat-reduce-nd-moments-bind-f64"),
                                    layout: &self.pipelines.reduce_nd_moments.get(emission)?.layout,
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
                            emission,
                            pending,
                            &self.pipelines.reduce_nd_moments.get(emission)?.pipeline,
                            &bind_group,
                            groups_x,
                        )?;
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
                            params.kept_sizes[i] =
                                crate::backend::wgpu::params::AlignedU32::new(*v);
                        }
                        for (i, v) in reduce_sizes.iter().enumerate() {
                            params.reduce_sizes[i] =
                                crate::backend::wgpu::params::AlignedU32::new(*v);
                        }
                        for (i, v) in kept_strides.iter().enumerate() {
                            params.kept_strides[i] =
                                crate::backend::wgpu::params::AlignedU32::new(*v);
                        }
                        for (i, v) in reduce_strides.iter().enumerate() {
                            params.reduce_strides[i] =
                                crate::backend::wgpu::params::AlignedU32::new(*v);
                        }
                        let pbuf = self.uniform_buffer(
                            emission,
                            &params,
                            "runmat-reduce-nd-moments-params-f32",
                        )?;
                        let bind_group =
                            emission
                                .device()
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("runmat-reduce-nd-moments-bind-f32"),
                                    layout: &self.pipelines.reduce_nd_moments.get(emission)?.layout,
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
                            emission,
                            pending,
                            &self.pipelines.reduce_nd_moments.get(emission)?.pipeline,
                            &bind_group,
                            groups_x,
                        )?;
                    }
                }

                Ok((mean_out, ex2_out, out_shape))
            })??;
        let mean_handle = self.register_existing_buffer(mean_out, out_shape.clone(), cols)?;
        let ex2_handle = self.register_existing_buffer(ex2_out, out_shape, cols)?;
        Ok(runmat_accelerate_api::ProviderMoments2 {
            mean: mean_handle,
            ex2: ex2_handle,
        })
    }

    pub(in crate::backend::wgpu::provider) fn materialize_nd_reduction_input(
        &self,
        input: &GpuTensorHandle,
    ) -> Result<Option<GpuTensorHandle>> {
        if runmat_accelerate_api::handle_is_transposed(input) {
            // reshape owns the bit-preserving GPU permutation and its intermediate
            // lease; no download, dtype conversion, or caller view mutation.
            self.reshape_exec(input, input.shape()).map(Some)
        } else {
            Ok(None)
        }
    }
}
