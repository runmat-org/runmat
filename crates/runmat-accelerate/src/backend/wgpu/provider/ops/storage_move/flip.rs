//! Existing flip kernel with pure optional admission and observed output ownership.
use super::*;

impl WgpuProvider {
    /// Required internal stage, using the same pure admission and physical routine.
    pub(crate) fn flip_exec(
        &self,
        handle: &GpuTensorHandle,
        axes: &[usize],
    ) -> Result<GpuTensorHandle> {
        self.flip_optional_exec(handle, axes, &std::cell::Cell::new(false), &|_| Ok(()))?
            .ok_or_else(|| {
                runmat_accelerate_api::unsupported_provider_operation(
                    "flip: unsupported physical request",
                )
            })
    }

    pub(crate) fn flip_optional_exec(
        &self,
        handle: &GpuTensorHandle,
        axes: &[usize],
        allocated: &std::cell::Cell<bool>,
        on_buffer_created: &dyn Fn(&runmat_accelerate_api::WgpuBufferOwner) -> Result<()>,
    ) -> Result<Option<GpuTensorHandle>> {
        if handle.shape().contains(&0) && handle.physical().element_capacity() != 0 {
            self.allocations.validate_empty_view(handle)?;
            return Ok(Some(handle.clone()));
        }
        let entry = self.get_entry_for_bitwise_move(handle, "flip")?;
        let geometry = runmat_types::geometry::FlipGeometry::new(&entry.shape, axes);
        if !geometry.has_effect() {
            return Ok(Some(handle.clone()));
        }
        if handle.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous
            || entry.storage == GpuTensorStorage::ComplexInterleaved
            || (entry.integer_type().is_none() && entry.precision != self.precision)
        {
            return Ok(None);
        }
        let integer_type = entry.integer_type();
        let word_factor = integer_type.map_or(1usize, |element_type| match element_type {
            runmat_accelerate_api::IntegerElementType::I64
            | runmat_accelerate_api::IntegerElementType::U64 => 2,
            _ => 1,
        });
        let ext_shape = if entry.shape.is_empty() {
            vec![1usize]
        } else {
            entry.shape.clone()
        };

        let logical_rank = ext_shape.len();
        if logical_rank == 0 {
            return Ok(Some(handle.clone()));
        }
        let rank = logical_rank + usize::from(word_factor > 1);
        if rank > crate::backend::wgpu::params::FLIP_MAX_RANK {
            return Ok(None);
        }

        let logical_total = product_checked(&ext_shape)
            .ok_or_else(|| anyhow!("flip: dimension product exceeds GPU limits"))?;
        ensure!(
            logical_total == entry.len || (logical_total == 0 && entry.len == 0),
            "flip: shape/product mismatch ({} vs {})",
            logical_total,
            entry.len
        );
        let storage_total = logical_total
            .checked_mul(word_factor)
            .ok_or_else(|| anyhow!("flip: tensor storage length exceeds GPU limits"))?;
        ensure!(
            storage_total <= u32::MAX as usize,
            "flip: tensor too large for GPU kernel"
        );
        ensure!(
            ext_shape.iter().all(|&d| d <= u32::MAX as usize),
            "flip: dimensions exceed GPU kernel limits"
        );

        let flags = geometry.reversed_axes();

        let mut kernel_shape = Vec::with_capacity(rank);
        let mut kernel_flags = Vec::with_capacity(rank);
        if word_factor > 1 {
            kernel_shape.push(word_factor);
            kernel_flags.push(false);
        }
        kernel_shape.extend(ext_shape.iter().copied());
        kernel_flags.extend(flags.iter().copied());

        let mut strides = vec![0usize; rank];
        let mut stride = 1usize;
        for (idx, &dim) in kernel_shape.iter().enumerate() {
            strides[idx] = stride;
            stride = stride
                .checked_mul(dim.max(1))
                .ok_or_else(|| anyhow!("flip: stride computation exceeds GPU limits"))?;
        }
        ensure!(
            strides.iter().all(|&s| s <= u32::MAX as usize),
            "flip: strides exceed GPU kernel limits"
        );

        let mut shape_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::FLIP_MAX_RANK];
        let mut strides_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::FLIP_MAX_RANK];
        let mut flags_arr = [crate::backend::wgpu::params::AlignedU32::new(0);
            crate::backend::wgpu::params::FLIP_MAX_RANK];
        for i in 0..rank {
            shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(kernel_shape[i] as u32);
            strides_arr[i] = crate::backend::wgpu::params::AlignedU32::new(strides[i] as u32);
            flags_arr[i] =
                crate::backend::wgpu::params::AlignedU32::new(if kernel_flags[i] { 1 } else { 0 });
        }
        let pipeline_bundle = if integer_type.is_some() {
            &self.pipelines.integer_flip
        } else {
            &self.pipelines.flip
        };
        let launch = pipeline_bundle
            .linear_launch()
            .ok_or_else(|| anyhow!("flip pipeline has no linear launch"))?;
        let chunk_capacity = launch
            .chunk_capacity()
            .ok_or_else(|| anyhow!("flip chunk capacity overflows"))?;

        let out_buffer = if integer_type.is_some() {
            self.with_emission(|emission, _| {
                let buffer = self.create_integer_word_buffer(
                    emission,
                    storage_total,
                    "runmat-integer-flip-out",
                )?;
                allocated.set(true);
                on_buffer_created(&buffer)?;
                Ok::<_, anyhow::Error>(buffer)
            })??
        } else {
            self.with_emission(|emission, _| {
                let buffer = self.create_storage_buffer(emission, entry.len, "runmat-flip-out")?;
                allocated.set(true);
                on_buffer_created(&buffer)?;
                Ok::<_, anyhow::Error>(buffer)
            })??
        };
        let out_shape = entry.shape.clone();

        self.with_emission(|emission, pending| -> Result<()> {
            let mut enc = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-flip-noop"),
                },
            );
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-flip-noop-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_bundle.get(emission)?.pipeline);
            drop(pass);
            self.submit(emission, pending, enc)?;
            Ok(())
        })??;
        self.access.poll(wgpu::Maintain::Poll);
        self.with_emission(|emission, pending| -> Result<()> {
            {
                let enc = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-flip-flush-gap"),
                    },
                );
                self.submit(emission, pending, enc)?;
            }

            let mut offset = 0usize;
            while offset < storage_total {
                let remaining = storage_total - offset;
                let chunk_len = remaining.min(chunk_capacity);
                let params = crate::backend::wgpu::params::FlipParams {
                    len: chunk_len as u32,
                    offset: offset as u32,
                    rank: rank as u32,
                    _pad: 0,
                    shape: shape_arr,
                    strides: strides_arr,
                    flags: flags_arr,
                };
                let params_buffer = self.uniform_buffer(emission, &params, "runmat-flip-params")?;
                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-flip-bind"),
                        layout: &pipeline_bundle.get(emission)?.layout,
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
                let workgroups = launch
                    .dispatch_groups(chunk_len)
                    .ok_or_else(|| anyhow!("flip launch exceeds enabled limits"))?;
                crate::backend::wgpu::dispatch::flip::run(
                    emission,
                    pending,
                    &pipeline_bundle.get(emission)?.pipeline,
                    &bind_group,
                    workgroups,
                )?;
                offset += chunk_len;
            }

            Ok(())
        })??;

        let allocated_bytes = out_buffer.size();
        self.register_storage_move_buffer(
            out_buffer,
            &entry,
            out_shape,
            logical_total,
            allocated_bytes,
        )
        .map(Some)
    }
}
