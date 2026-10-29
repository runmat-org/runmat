use super::super::backend_types::{NumericBufferRegistration, WgpuProvider};
use crate::backend::wgpu::residency::BufferUsageClass;
use crate::backend::wgpu::types::NumericPrecision;
use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::{
    unsupported_provider_operation, AccelProvider, GpuTensorHandle, GpuTensorStorage,
    HostNumericDataOwned, HostNumericTensorView, ImfilterOptions, ImfilterPadding,
    NumericElementType, ProviderOperationIdentity, ProviderPrecision,
};
use runmat_runtime::builtins::image::filters::kernel::{
    apply_imfilter_native, build_imfilter_plan, floating_accumulation, FilterAccumulation,
};
use runmat_value::Tensor;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "imfilter/tests.rs"]
mod tests;

impl WgpuProvider {
    pub(crate) async fn imfilter_exec(
        &self,
        image: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: &ImfilterOptions,
    ) -> Result<GpuTensorHandle> {
        // Authenticate both allocations before capability selection or readback.
        let image_entry = self.get_entry_raw(image)?;
        let kernel_entry = self.get_entry_raw(kernel)?;
        for (handle, entry) in [(image, &image_entry), (kernel, &kernel_entry)] {
            if entry.storage != GpuTensorStorage::Real
                || handle.view().is_logical()
                || !matches!(
                    entry.element_type,
                    NumericElementType::F32 | NumericElementType::F64
                )
            {
                return Err(unsupported_provider_operation(
                    "imfilter requires real floating operands",
                ));
            }
        }
        let operation = ProviderOperationIdentity::new("imfilter");
        let observed = runmat_accelerate_api::observation::current_provider_observation()
            .filter(|recorder| recorder.expects_operation(&operation));
        // A typed runtime candidate must be re-derived from the current live
        // views. The owner witness is only confirmed after the actual direct
        // filter dispatch, never by the preceding H readback/reshape alone.
        let admitted_route = runmat_runtime::builtins::image::filters::placement::exact_hook_query(
            image, kernel, options,
        )
        .and_then(|query| {
            self.query_feasibility(&query)
                .is_supported()
                .then(|| self.imfilter_physical_evidence(&query))
                .flatten()
        })
        .map(|physical| physical.route);
        let admitted_direct =
            admitted_route == Some(runmat_accelerate_api::ProviderImfilterPhysicalRoute::Direct);
        let admitted_host = matches!(
            admitted_route,
            Some(runmat_accelerate_api::ProviderImfilterPhysicalRoute::HostMediatedEarly
                | runmat_accelerate_api::ProviderImfilterPhysicalRoute::HostMediatedAfterKernelReadback)
        );
        let storage_precision = image_entry.precision;
        let accumulation = floating_accumulation(
            match storage_precision {
                NumericPrecision::F32 => ProviderPrecision::F32,
                NumericPrecision::F64 => ProviderPrecision::F64,
            },
            runmat_accelerate_api::handle_is_explicit(image),
        );
        let accumulation_precision = match accumulation {
            FilterAccumulation::HostDouble => NumericPrecision::F64,
            FilterAccumulation::GpuSingle => NumericPrecision::F32,
        };
        if accumulation_precision == NumericPrecision::F64
            && !self.with_emission(|emission, _| {
                emission
                    .device()
                    .features()
                    .contains(wgpu::Features::SHADER_F64)
            })?
        {
            return Err(unsupported_provider_operation(
                "imfilter requires shader-f64 for the admitted accumulation policy",
            ));
        }
        if admitted_route
            == Some(runmat_accelerate_api::ProviderImfilterPhysicalRoute::HostMediatedEarly)
            || crate::placement::imfilter_shader_disabled()
        {
            return self
                .imfilter_exec_fallback(image, kernel, options, admitted_host)
                .await;
        }
        let kernel_tensor = native_tensor(self.download_numeric_exec(kernel).await?)?;
        if admitted_route == Some(
            runmat_accelerate_api::ProviderImfilterPhysicalRoute::HostMediatedAfterKernelReadback,
        ) {
            return self.imfilter_exec_fallback(image, kernel, options, admitted_host).await;
        }
        let after_kernel_readback = observed.as_ref().map(
            runmat_accelerate_api::observation::ProviderOperationRecorder::submission_checkpoint,
        );

        let image_shape = if image_entry.shape.is_empty() {
            vec![1usize]
        } else {
            image_entry.shape.clone()
        };

        let plan = match build_imfilter_plan(&image_shape, &kernel_tensor, options, "imfilter") {
            Ok(plan) => plan,
            Err(err) => return Err(anyhow!(err)),
        };

        if plan.rank > crate::backend::wgpu::params::IMFILTER_MAX_RANK {
            return self
                .imfilter_exec_fallback(image, kernel, options, admitted_host)
                .await;
        }

        let image_ext_product = plan
            .image_shape_ext
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("imfilter: image dimensions exceed GPU limits"))?;
        if image_ext_product != image_entry.len {
            return self
                .imfilter_exec_fallback(image, kernel, options, admitted_host)
                .await;
        }

        let output_len = plan
            .output_shape_ext
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow!("imfilter: output dimensions exceed GPU limits"))?;
        if output_len > u32::MAX as usize || image_entry.len > u32::MAX as usize {
            return self
                .imfilter_exec_fallback(image, kernel, options, admitted_host)
                .await;
        }

        let kernel_points_len = plan.kernel_points.len();
        if kernel_points_len > u32::MAX as usize {
            return self
                .imfilter_exec_fallback(image, kernel, options, admitted_host)
                .await;
        }

        let mut kernel_offsets = Vec::with_capacity(kernel_points_len * plan.rank);
        let mut kernel_values_f64 = Vec::with_capacity(kernel_points_len);
        for point in &plan.kernel_points {
            if point.offsets.len() != plan.rank {
                return self
                    .imfilter_exec_fallback(image, kernel, options, admitted_host)
                    .await;
            }
            for &offset in &point.offsets {
                if offset < i32::MIN as isize || offset > i32::MAX as isize {
                    return self
                        .imfilter_exec_fallback(image, kernel, options, admitted_host)
                        .await;
                }
                kernel_offsets.push(offset as i32);
            }
            kernel_values_f64.push(point.value);
        }

        if kernel_offsets.len() > u32::MAX as usize {
            return self
                .imfilter_exec_fallback(image, kernel, options, admitted_host)
                .await;
        }

        let element_size = image_entry.element_type.element_size();
        let allocated_bytes = u64::try_from(output_len)?
            .checked_mul(u64::try_from(element_size)?)
            .ok_or_else(|| anyhow!("imfilter output byte extent overflow"))?;
        let pipeline = match (storage_precision, accumulation_precision) {
            (NumericPrecision::F32, NumericPrecision::F32) => &self.pipelines.imfilter_single,
            (NumericPrecision::F32, NumericPrecision::F64) => {
                &self.pipelines.imfilter_single_double
            }
            (NumericPrecision::F64, _) => &self.pipelines.imfilter_double,
        };
        pipeline.preflight()?;
        // Only materialize a borrowed lazy view once the direct recipe has
        // passed geometry admission. Never mutate the caller's view identity.
        let materialized = if runmat_accelerate_api::handle_is_transposed(image) {
            Some(self.reshape_exec(image, image.shape())?)
        } else {
            None
        };
        let image_entry = match materialized.as_ref() {
            Some(handle) => self.get_entry_raw(handle)?,
            None => image_entry,
        };
        let out_buffer = self.with_emission(|emission, pending| -> Result<_> {
            let kernel_offsets_buffer =
                emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                    label: Some("runmat-imfilter-kernel-offsets"),
                    contents: bytemuck::cast_slice(&kernel_offsets),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let kernel_values_buffer = match accumulation_precision {
                NumericPrecision::F64 => {
                    emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-imfilter-kernel-values-f64"),
                        contents: bytemuck::cast_slice(&kernel_values_f64),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
                }
                NumericPrecision::F32 => {
                    let kernel_values_f32: Vec<f32> =
                        kernel_values_f64.iter().map(|&v| v as f32).collect();
                    emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                        label: Some("runmat-imfilter-kernel-values-f32"),
                        contents: bytemuck::cast_slice(&kernel_values_f32),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
                }
            };

            let out_buffer = self.create_storage_buffer_checked_with_element_size(
                emission,
                output_len,
                "runmat-imfilter-out",
                BufferUsageClass::Generic,
                element_size,
            )?;
            if output_len == 0 {
                return Ok(out_buffer);
            }

            let mut image_shape_arr = [crate::backend::wgpu::params::AlignedU32::default();
                crate::backend::wgpu::params::IMFILTER_MAX_RANK];
            let mut image_strides_arr = [crate::backend::wgpu::params::AlignedU32::default();
                crate::backend::wgpu::params::IMFILTER_MAX_RANK];
            let mut output_shape_arr = [crate::backend::wgpu::params::AlignedU32::default();
                crate::backend::wgpu::params::IMFILTER_MAX_RANK];
            let mut base_offset_arr = [crate::backend::wgpu::params::PackedI32::default();
                crate::backend::wgpu::params::IMFILTER_MAX_RANK];

            for i in 0..plan.rank {
                let dim = plan.image_shape_ext[i];
                ensure!(
                    dim <= u32::MAX as usize,
                    "imfilter: image dimension exceeds GPU limits"
                );
                image_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(dim as u32);

                let stride = plan.image_strides[i];
                ensure!(
                    stride <= u32::MAX as usize,
                    "imfilter: image stride exceeds GPU limits"
                );
                image_strides_arr[i] = crate::backend::wgpu::params::AlignedU32::new(stride as u32);

                let out_dim = plan.output_shape_ext[i];
                ensure!(
                    out_dim <= u32::MAX as usize,
                    "imfilter: output dimension exceeds GPU limits"
                );
                output_shape_arr[i] = crate::backend::wgpu::params::AlignedU32::new(out_dim as u32);

                let offset = plan.base_offset[i];
                ensure!(
                    offset >= i32::MIN as isize && offset <= i32::MAX as isize,
                    "imfilter: base offset exceeds GPU limits"
                );
                base_offset_arr[i] =
                    crate::backend::wgpu::params::PackedI32::from_scalar(offset as i32);
            }

            let padding_mode = match options.padding {
                ImfilterPadding::Constant => 0u32,
                ImfilterPadding::Replicate => 1u32,
                ImfilterPadding::Symmetric => 2u32,
                ImfilterPadding::Circular => 3u32,
            };

            let kernel_points_u32 = kernel_points_len as u32;
            let image_len_u32 = image_entry.len as u32;

            let chunk_capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
                * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;

            let mut offset = 0usize;
            while offset < output_len {
                let remaining = output_len - offset;
                let chunk_len = remaining.min(chunk_capacity);
                let offset_u32 = offset as u32;
                let chunk_u32 = chunk_len as u32;

                let params_buffer = match accumulation_precision {
                    NumericPrecision::F64 => {
                        let params = crate::backend::wgpu::params::ImfilterParamsF64 {
                            len: chunk_u32,
                            offset: offset_u32,
                            rank: plan.rank as u32,
                            padding: padding_mode,
                            kernel_points: kernel_points_u32,
                            image_len: image_len_u32,
                            _pad0: 0,
                            _pad1: 0,
                            constant_value: options.constant_value,
                            _pad_const: 0.0,
                            image_shape: image_shape_arr,
                            image_strides: image_strides_arr,
                            output_shape: output_shape_arr,
                            base_offset: base_offset_arr,
                            _pad_tail: crate::backend::wgpu::params::AlignedU32::default(),
                        };

                        self.uniform_buffer(emission, &params, "runmat-imfilter-params-f64")?
                    }
                    NumericPrecision::F32 => {
                        let params = crate::backend::wgpu::params::ImfilterParamsF32 {
                            len: chunk_u32,
                            offset: offset_u32,
                            rank: plan.rank as u32,
                            padding: padding_mode,
                            kernel_points: kernel_points_u32,
                            image_len: image_len_u32,
                            _pad0: 0,
                            _pad1: 0,
                            constant_value: options.constant_value as f32,
                            _pad_const: [0.0; 3],
                            image_shape: image_shape_arr,
                            image_strides: image_strides_arr,
                            output_shape: output_shape_arr,
                            base_offset: base_offset_arr,
                            _pad_tail: crate::backend::wgpu::params::AlignedU32::default(),
                        };

                        self.uniform_buffer(emission, &params, "runmat-imfilter-params-f32")?
                    }
                };

                let bind_group = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-imfilter-bind"),
                        layout: &pipeline.get(emission)?.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: image_entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: kernel_offsets_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: kernel_values_buffer.as_entire_binding(),
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
                    chunk_u32,
                    crate::backend::wgpu::config::WORKGROUP_SIZE,
                );

                crate::backend::wgpu::dispatch::imfilter::run(
                    emission,
                    pending,
                    &pipeline.get(emission)?.pipeline,
                    &bind_group,
                    workgroups,
                )?;

                offset += chunk_len;
            }

            Ok(out_buffer)
        })??;
        let output = self.register_numeric_buffer(
            out_buffer,
            NumericBufferRegistration {
                shape: plan.final_shape,
                len: output_len,
                physical_element_type: image_entry.element_type,
                storage: GpuTensorStorage::Real,
                allocated_bytes,
                usage: BufferUsageClass::Generic,
            },
        )?;
        if admitted_direct {
            if let Some(recorder) = observed {
                if output_len == 0 {
                    recorder.confirm_provider_execution(self.device_id(), &operation);
                } else if let Some(checkpoint) = after_kernel_readback {
                    recorder.confirm_submitted_provider_execution_after(
                        self.device_id(),
                        &operation,
                        checkpoint,
                    );
                }
            }
        }
        Ok(output)
    }

    async fn imfilter_exec_fallback(
        &self,
        image: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: &ImfilterOptions,
        admitted_host: bool,
    ) -> Result<GpuTensorHandle> {
        let operation = ProviderOperationIdentity::new("imfilter");
        let observed = runmat_accelerate_api::observation::current_provider_observation()
            .filter(|recorder| recorder.expects_operation(&operation));
        if observed.is_some() && !admitted_host {
            return Err(unsupported_provider_operation(
                "imfilter host-mediated route was not physically admitted",
            ));
        }
        let image_tensor = native_tensor(self.download_numeric_exec(image).await?)?;
        let kernel_tensor = native_tensor(self.download_numeric_exec(kernel).await?)?;
        let result = apply_imfilter_native(
            &image_tensor,
            &kernel_tensor,
            options,
            runmat_accelerate_api::handle_is_explicit(image),
            "imfilter",
        )?;
        let data = if let Some(values) = result.as_f32_slice() {
            runmat_accelerate_api::HostNumericDataView::F32(values)
        } else {
            runmat_accelerate_api::HostNumericDataView::F64(
                result
                    .as_f64_slice()
                    .ok_or_else(|| anyhow!("imfilter returned nonfloating native storage"))?,
            )
        };
        let output = self.upload_numeric_exec(&HostNumericTensorView {
            data,
            shape: result.shape(),
            storage: GpuTensorStorage::Real,
        })?;
        if let Some(recorder) = observed {
            // The mapped-at-creation upload has no queue submission. Its own
            // completion fence is the last stage of the admitted host route.
            let checkpoint = recorder.submission_checkpoint();
            self.with_emission(|emission, pending| -> Result<()> {
                let encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-imfilter-host-mediated-completion"),
                    },
                );
                self.submit(emission, pending, encoder)?;
                Ok(())
            })??;
            recorder.confirm_provider_host_mediated_after(self.device_id(), &operation, checkpoint);
        }
        Ok(output)
    }
}

fn native_tensor(host: runmat_accelerate_api::HostNumericTensorOwned) -> Result<Tensor> {
    ensure!(
        host.storage == GpuTensorStorage::Real,
        "imfilter requires real native payload"
    );
    match host.data {
        HostNumericDataOwned::F32(values) => Tensor::from_f32(values, host.shape),
        HostNumericDataOwned::F64(values) => Tensor::new(values, host.shape),
        _ => {
            return Err(unsupported_provider_operation(
                "imfilter requires floating native payload",
            ))
        }
    }
    .map_err(|error| anyhow!(error))
}
