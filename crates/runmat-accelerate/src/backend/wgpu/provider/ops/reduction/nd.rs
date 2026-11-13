use super::*;

#[path = "nd/geometry.rs"]
mod geometry;
use geometry::NdReductionGeometry;

#[path = "nd/selected_mean.rs"]
mod selected_mean;

#[path = "nd/rank_n_dim.rs"]
mod rank_n_dim;

#[path = "nd/ordered_real_mean.rs"]
mod ordered_real_mean;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "nd/lazy_tests.rs"]
mod lazy_tests;

impl WgpuProvider {
    pub(in crate::backend::wgpu::provider::backend) fn validate_nd_moment_pair(
        &self,
        source: &GpuTensorHandle,
        dims: &[usize],
        mean: &GpuTensorHandle,
        ex2: &GpuTensorHandle,
    ) -> Result<()> {
        let source_entry = self.real_floating_entry(source)?;
        let mean_entry = self.real_floating_entry(mean)?;
        let ex2_entry = self.real_floating_entry(ex2)?;
        let geometry = NdReductionGeometry::new(&source_entry.shape, dims, "N-D moments pair")?;
        ensure!(
            source_entry.storage == GpuTensorStorage::Real
                && mean_entry.storage == GpuTensorStorage::Real
                && ex2_entry.storage == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(mean)
                && !runmat_accelerate_api::handle_is_logical(ex2)
                && mean_entry.element_type == source_entry.element_type
                && ex2_entry.element_type == source_entry.element_type
                && mean_entry.shape == geometry.out_shape
                && ex2_entry.shape == geometry.out_shape
                && !mean.shares_allocation_with(source)
                && !ex2.shares_allocation_with(source)
                && !mean.shares_allocation_with(ex2),
            "N-D moments pair violates selected source, class, shape, or allocation ownership"
        );
        Ok(())
    }

    pub(crate) async fn reduce_nd_mean_exec(
        &self,
        a: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<Option<GpuTensorHandle>> {
        // Cache hits must not bypass the original input's owner/live/precision admission.
        let source_entry = self.real_floating_entry(a)?;
        if source_entry.storage != GpuTensorStorage::Real {
            return Ok(None);
        }
        if source_entry.precision != self.precision
            && runmat_accelerate_api::handle_is_transposed(a)
        {
            return Ok(None);
        }
        if self.reduction_reuse.is_poisoned() {
            // The original uncached ND kernel remains authoritative when only
            // optional reuse bookkeeping is unavailable; make the loss visible.
            log::error!("WGPU reduction reuse cache poisoned; executing uncached ND mean");
        } else {
            // If input is a known square of a base tensor, reuse or compute ex2 via moments.
            let base = if source_entry.precision == self.precision {
                self.cached_square_source(a)?
            } else {
                // No alternate-precision square provenance is certified yet;
                // reduce the actual selected input rather than its source.
                None
            };
            if let Some(base) = base {
                if let Ok(base_handle) = base.acquire() {
                    self.real_floating_entry(&base_handle)?;
                    if let Some((_, ex2)) =
                        self.cached_moment_pair(&base_handle, dims_zero_based)?
                    {
                        return Ok(Some(ex2));
                    }
                    let moments = self.reduce_moments_nd_exec(&base_handle, dims_zero_based)?;
                    self.remember_moments(
                        &base_handle,
                        dims_zero_based,
                        &moments.mean,
                        &moments.ex2,
                        None,
                    )?;
                    return Ok(Some(moments.ex2));
                }
            }
            // Prefer computing moments once and caching ex2 for future reuse.
            if let Some((mean, _)) = self.cached_moment_pair(a, dims_zero_based)? {
                return Ok(Some(mean));
            }
            let moments = self.reduce_moments_nd_exec(a, dims_zero_based)?;
            self.remember_moments(a, dims_zero_based, &moments.mean, &moments.ex2, None)?;
            return Ok(Some(moments.mean));
        }
        let entry = self.real_floating_entry(a)?;
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

        if entry.precision != self.precision
            && self.prepare_cold_mean_nd_exec(a, &reduce)?.is_none()
        {
            return Ok(None);
        }

        // Geometry above is logical. The stride-based kernel (and sequenced fallback)
        // needs canonical storage, not the backing order of a lazy transpose.
        // Keep this owner local; cache identity remains the caller's original view.
        let materialized = self.materialize_nd_reduction_input(a)?;
        let a = materialized.as_ref().unwrap_or(a);
        let entry = self.real_floating_entry(a)?;
        let pipeline = self.real_nd_pipeline(entry.precision, false)?;

        // Heuristic fallback: for very large row extents, prefer sequenced dim-reductions.
        if rows >= self.two_pass_threshold() && entry.precision == self.precision {
            let mut current = a.clone();
            let mut owned = false;
            for &d in &reduce {
                let next = self
                    .mean_dim_optional_exec(&current, d)
                    .await?
                    .ok_or_else(|| {
                        anyhow!(
                            "mean: required dimensional stage unavailable after N-D preparation"
                        )
                    })?;
                if owned {
                    drop(current);
                }
                current = next;
                owned = true;
            }
            return Ok(Some(current));
        }

        let (out_buffer, out_shape) = self.with_emission(|emission, pending| -> Result<_> {
            let mut out_buffer = self.create_storage_buffer_checked_with_element_size(
                emission,
                cols,
                "runmat-reduce-nd-mean-out",
                crate::backend::wgpu::residency::BufferUsageClass::Generic,
                entry.element_type.element_size(),
            )?;
            // Prevent aliasing: output must not be the same as input buffer
            if std::ptr::eq(out_buffer.as_ref(), entry.buffer.as_ref()) {
                let size_bytes = u64::try_from(cols)?
                    .checked_mul(entry.element_type.element_size() as u64)
                    .ok_or_else(|| anyhow!("N-D mean output byte extent overflow"))?;
                out_buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                    emission.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("runmat-reduce-nd-mean-out-unique"),
                        size: size_bytes,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                );
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
                                layout: &pipeline.get(emission)?.layout,
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
                        &pipeline.get(emission)?.pipeline,
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
                                layout: &pipeline.get(emission)?.layout,
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
                        &pipeline.get(emission)?.pipeline,
                        &bind_group,
                        groups_x,
                    )?;
                }
            }

            Ok((out_buffer, out_shape))
        })??;
        self.register_real_floating_output(out_buffer, out_shape, cols, entry.precision)
            .map(Some)
    }

    pub(crate) fn reduce_moments_nd_exec(
        &self,
        a: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<runmat_accelerate_api::ProviderMoments2> {
        self.reduce_moments_nd_exec_admitted(a, dims_zero_based, None)
    }

    pub(crate) fn reduce_moments_nd_exec_admitted(
        &self,
        a: &GpuTensorHandle,
        dims_zero_based: &[usize],
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<runmat_accelerate_api::ProviderMoments2> {
        let entry = self.real_floating_entry(a)?;
        if entry.storage != GpuTensorStorage::Real {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "N-D moments requires real physical storage",
            ));
        }
        if entry.precision != self.precision && runmat_accelerate_api::handle_is_transposed(a) {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "alternate-precision N-D moments requires a canonical source view",
            ));
        }
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

        if entry.precision != self.precision
            && self.prepare_cold_mean_nd_exec(a, &reduce)?.is_none()
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "alternate-precision N-D moments has no qualified physical route",
            ));
        }

        // Mean's normal cache-miss path and direct moments/variance consumers
        // share this boundary. Validate geometry before issuing the storage move.
        let materialized = self.materialize_nd_reduction_input(a)?;
        let a = materialized.as_ref().unwrap_or(a);
        let entry = self.real_floating_entry(a)?;
        let pipeline = self.real_nd_pipeline(entry.precision, true)?;

        let (mean_out, ex2_out, out_shape) =
            self.with_emission(|emission, pending| -> Result<_> {
                // Allocate outputs
                let mean_out = self.create_storage_buffer_checked_with_element_size(
                    emission,
                    cols,
                    "runmat-reduce-nd-moments-mean",
                    crate::backend::wgpu::residency::BufferUsageClass::Generic,
                    entry.element_type.element_size(),
                )?;
                let ex2_out = self.create_storage_buffer_checked_with_element_size(
                    emission,
                    cols,
                    "runmat-reduce-nd-moments-ex2",
                    crate::backend::wgpu::residency::BufferUsageClass::Generic,
                    entry.element_type.element_size(),
                )?;
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
                                    layout: &pipeline.get(emission)?.layout,
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
                            &pipeline.get(emission)?.pipeline,
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
                                    layout: &pipeline.get(emission)?.layout,
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
                            &pipeline.get(emission)?.pipeline,
                            &bind_group,
                            groups_x,
                        )?;
                    }
                }

                Ok((mean_out, ex2_out, out_shape))
            })??;
        let mean_handle = self.register_real_floating_output_admitted(
            mean_out,
            out_shape.clone(),
            cols,
            entry.precision,
            admission,
        )?;
        let ex2_handle = self.register_real_floating_output_admitted(
            ex2_out,
            out_shape,
            cols,
            entry.precision,
            admission,
        )?;
        self.validate_nd_moment_pair(a, dims_zero_based, &mean_handle, &ex2_handle)?;
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
