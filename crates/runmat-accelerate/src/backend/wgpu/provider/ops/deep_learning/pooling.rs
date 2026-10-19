use super::*;
use runmat_accelerate_api::{GpuTensorLayout, GpuTensorView};

#[path = "pooling/adjoint.rs"]
mod adjoint;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Pool1DParams {
    pool_size: u32,
    chunk_output_count: u32,
    inner_stride: u32,
    output_offset: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PoolDispatchChunk {
    offset: u32,
    output_count: u32,
    workgroups: u32,
}

fn next_pool_chunk(
    total: u32,
    offset: u32,
    max_workgroups: u32,
) -> Result<Option<PoolDispatchChunk>> {
    ensure!(
        max_workgroups > 0,
        "pool dispatch requires workgroup capacity"
    );
    ensure!(offset <= total, "pool dispatch offset exceeds output count");
    if offset == total {
        return Ok(None);
    }
    let lanes = crate::backend::wgpu::shaders::deep_learning::POOL_WORKGROUP_SIZE;
    let capacity = max_workgroups.saturating_mul(lanes);
    let output_count = (total - offset).min(capacity);
    let workgroups = output_count.div_ceil(lanes);
    debug_assert!(workgroups > 0 && workgroups <= max_workgroups);
    Ok(Some(PoolDispatchChunk {
        offset,
        output_count,
        workgroups,
    }))
}

impl WgpuProvider {
    /// Pool output and every per-chunk uniform, plus the exact physical
    /// permutation used to present a non-identity target axis order.
    pub(crate) fn pool_1d_buffer_demand(
        &self,
        input: &GpuTensorView,
        route: &DeepLearningPool1DRoute,
    ) -> Option<u64> {
        let input_shape = input.shape();
        let pool_size = *input_shape.get(route.reduction.axis_zero_based)?;
        if pool_size == 0 {
            return None;
        }
        let output_count =
            u64::try_from(checked_shape_count(input_shape).ok()? / pool_size).ok()?;
        let output = output_count.max(1).checked_mul(self.element_size as u64)?;
        let mut total = output;
        if output_count > 0 {
            // The reduction owner materializes a lazy transpose through
            // reshape before dispatch. Current forward-route admission
            // requires a contiguous pool input; retain the complete owner
            // formula if that admission is extended later.
            if matches!(input.layout(), GpuTensorLayout::Transpose(_)) {
                let input_count = u64::try_from(checked_shape_count(input_shape).ok()?).ok()?;
                total = total.checked_add(self.real_permute_buffer_demand(input_count)?)?;
            }
            let capacity = u64::from(crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS)
                .checked_mul(u64::from(
                    crate::backend::wgpu::shaders::deep_learning::POOL_WORKGROUP_SIZE,
                ))?;
            total = total.checked_add(
                output_count
                    .div_ceil(capacity)
                    .checked_mul(std::mem::size_of::<Pool1DParams>() as u64)?,
            )?;
        }
        if route
            .target_to_source
            .iter()
            .enumerate()
            .any(|(target, &source)| target != source)
        {
            total = total.checked_add(self.real_permute_buffer_demand(output_count)?)?;
        }
        Some(total)
    }

    pub(crate) fn deep_learning_global_average_pool_1d_capability_exec(
        &self,
        input: &GpuTensorHandle,
        route: &DeepLearningPool1DRoute,
    ) -> Result<DeepLearningPool1DCapability> {
        let entry = self.get_entry_raw(input)?;
        ensure!(
            input.shape() == entry.shape
                && input.physical().element_type() == entry.element_type
                && input.physical().storage() == entry.storage
                && entry.logical == runmat_accelerate_api::handle_is_logical(input)
                && entry.integer_type() == runmat_accelerate_api::handle_integer_type(input),
            "deep-learning pooling input descriptor does not match its provider-owned buffer"
        );
        ensure!(
            checked_shape_count(input.shape())? == entry.len,
            "pooling input shape/storage mismatch"
        );
        self.deep_learning_global_average_pool_1d_planned_capability_exec(
            input,
            input.shape(),
            route,
        )
    }

    pub(crate) fn deep_learning_global_average_pool_1d_planned_capability_exec(
        &self,
        anchor: &GpuTensorHandle,
        shape: &[usize],
        route: &DeepLearningPool1DRoute,
    ) -> Result<DeepLearningPool1DCapability> {
        let entry = self.get_entry_raw(anchor)?;
        ensure!(
            anchor.physical().element_type() == entry.element_type
                && anchor.physical().storage() == entry.storage
                && checked_shape_count(anchor.shape())? == entry.len
                && entry.logical == runmat_accelerate_api::handle_is_logical(anchor)
                && entry.integer_type() == runmat_accelerate_api::handle_integer_type(anchor),
            "deep-learning pooling anchor differs from provider-owned buffer"
        );
        if entry.storage != GpuTensorStorage::Real
            || entry.logical
            || entry.integer_type().is_some()
        {
            return Ok(DeepLearningPool1DCapability::UnsupportedOperation);
        }
        self.deep_learning_global_average_pool_1d_capability_for_type(
            entry.element_type,
            shape,
            route,
        )
    }

    pub(crate) fn deep_learning_global_average_pool_1d_capability_for_type(
        &self,
        element_type: runmat_accelerate_api::NumericElementType,
        shape: &[usize],
        route: &DeepLearningPool1DRoute,
    ) -> Result<DeepLearningPool1DCapability> {
        let axis = route.reduction.axis_zero_based;
        ensure!(axis < shape.len(), "pool axis exceeds input rank");
        let pool_size = shape[axis];
        ensure!(pool_size > 0, "cannot pool an empty axis");
        let total = checked_shape_count(shape)?;
        let output_count = total / pool_size;
        let mut keep_shape = shape.to_vec();
        keep_shape[axis] = 1;
        let expected_source = match route.result_axis {
            DeepLearningPool1DResultAxis::KeepSingleton => keep_shape,
            DeepLearningPool1DResultAxis::RemoveReduced => {
                ensure!(shape.len() > 1, "cannot remove the sole pooling axis");
                keep_shape.remove(axis);
                keep_shape
            }
        };
        ensure!(
            route.source_shape == expected_source,
            "pooling source shape does not match the requested reduced-axis view"
        );
        let rank = route.source_shape.len();
        ensure!(
            route.target_to_source.len() == rank && route.target_shape.len() == rank,
            "pooling presentation rank mismatch"
        );
        let mut seen = vec![false; rank];
        for (target_axis, &source_axis) in route.target_to_source.iter().enumerate() {
            ensure!(source_axis < rank, "pooling presentation axis out of range");
            ensure!(!seen[source_axis], "pooling presentation repeats an axis");
            seen[source_axis] = true;
            ensure!(
                route.target_shape[target_axis] == route.source_shape[source_axis],
                "pooling presentation shape does not match its permutation"
            );
        }
        let needs_permutation = route
            .target_to_source
            .iter()
            .enumerate()
            .any(|(target, &source)| target != source);
        let stride = checked_shape_count(&shape[..axis])?;
        if !self.supports_native_real_element_type(element_type) {
            return Ok(DeepLearningPool1DCapability::UnsupportedOperation);
        }
        if output_count != 0 && !self.ieee_words_available(self.precision) {
            return Ok(DeepLearningPool1DCapability::UnsupportedOperation);
        }
        if total > u32::MAX as usize
            || pool_size > u32::MAX as usize
            || stride > u32::MAX as usize
            || output_count > u32::MAX as usize
            || (needs_permutation
                && (rank > crate::backend::wgpu::params::PERMUTE_MAX_RANK
                    || route
                        .source_shape
                        .iter()
                        .any(|&size| size > u32::MAX as usize)
                    || route
                        .target_shape
                        .iter()
                        .any(|&size| size > u32::MAX as usize)))
        {
            return Ok(DeepLearningPool1DCapability::ResourceLimit);
        }
        Ok(DeepLearningPool1DCapability::Ready)
    }

    pub(crate) fn deep_learning_global_average_pool_1d_exec(
        &self,
        input: &GpuTensorHandle,
        request: DeepLearningPool1DRequest,
    ) -> Result<GpuTensorHandle> {
        let logical = self.get_entry(input)?;
        ensure!(
            logical.len <= u32::MAX as usize,
            "pool input exceeds GPU indexing limits"
        );
        ensure!(
            logical.storage == GpuTensorStorage::Real
                && runmat_accelerate_api::handle_storage(input) == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(input)
                && runmat_accelerate_api::handle_integer_type(input).is_none()
                && logical.precision == self.precision,
            "deep-learning global-average-pool-1d requires real floating input"
        );
        let shape = input.shape();
        let axis = request.axis_zero_based;
        ensure!(axis < shape.len(), "pool axis exceeds input rank");
        let pool_size = shape[axis];
        ensure!(pool_size > 0, "cannot pool an empty axis");
        let inner_stride = shape[..axis].iter().try_fold(1usize, |n, &extent| {
            n.checked_mul(extent)
                .ok_or_else(|| anyhow!("pool stride overflows"))
        })?;
        let output_count = logical.len / pool_size;
        let pool_size =
            u32::try_from(pool_size).map_err(|_| anyhow!("pool axis exceeds GPU limits"))?;
        let inner_stride =
            u32::try_from(inner_stride).map_err(|_| anyhow!("pool stride exceeds GPU limits"))?;
        let output_count =
            u32::try_from(output_count).map_err(|_| anyhow!("pool output exceeds GPU limits"))?;
        let mut output_shape = shape.to_vec();
        output_shape[axis] = 1;
        if output_count == 0 {
            let output = self.with_emission(|emission, _| {
                self.create_storage_buffer_checked(emission, 0, "runmat-deep-learning-empty-pool")
            })??;
            return self.register_existing_buffer_with_storage(
                output,
                output_shape,
                0,
                GpuTensorStorage::Real,
            );
        }

        self.require_ieee_words(self.precision)?;

        // The kernel uses physical column-major offsets. A lazy transpose
        // must be materialized by its owner before reading the backing buffer.
        let materialized = self.materialize_nd_reduction_input(input)?;
        let source = materialized.as_ref().unwrap_or(input);
        let entry = self.get_entry(source)?;
        let output = self.with_emission(|emission, pending| -> Result<_> {
            let output = self.create_storage_buffer_checked(
                emission,
                output_count as usize,
                "runmat-deep-learning-pool-1d-out",
            )?;
            let pipeline = self.pipelines.deep_learning_stable_pool_1d.get(emission)?;
            let mut offset = 0u32;
            while let Some(chunk) = next_pool_chunk(
                output_count,
                offset,
                crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS,
            )? {
                let params = self.uniform_buffer(
                    emission,
                    &Pool1DParams {
                        pool_size,
                        chunk_output_count: chunk.output_count,
                        inner_stride,
                        output_offset: chunk.offset,
                    },
                    "runmat-deep-learning-pool-1d-params",
                )?;
                let bind = emission
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("runmat-deep-learning-pool-1d-bind"),
                        layout: &pipeline.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: entry.buffer.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: output.as_ref().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params.as_entire_binding(),
                            },
                        ],
                    });
                crate::backend::wgpu::dispatch::elementwise::run(
                    emission,
                    pending,
                    &pipeline.pipeline,
                    &bind,
                    chunk.workgroups,
                )?;
                offset += chunk.output_count;
            }
            Ok(output)
        })??;
        self.register_existing_buffer(output, output_shape, output_count as usize)
    }
}

fn checked_shape_count(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| anyhow!("pooling shape product overflows"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires actual WGPU device; run explicitly for pure future-stage admission"]
    fn planned_pooling_admits_future_shape_without_effects() {
        use runmat_accelerate_api::{AccelProvider, HostTensorView};

        let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
        let anchor = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[2, 1],
            })
            .unwrap();
        let route = DeepLearningPool1DRoute {
            reduction: DeepLearningPool1DRequest { axis_zero_based: 1 },
            result_axis: DeepLearningPool1DResultAxis::KeepSingleton,
            source_shape: vec![3, 1],
            target_shape: vec![3, 1],
            target_to_source: vec![0, 1],
        };
        provider.reset_telemetry();
        let before = provider.telemetry_snapshot();
        assert_eq!(
            provider
                .deep_learning_global_average_pool_1d_planned_capability_exec(
                    &anchor,
                    &[3, 4],
                    &route,
                )
                .unwrap(),
            DeepLearningPool1DCapability::Ready
        );
        assert!(provider
            .deep_learning_global_average_pool_1d_capability_exec(&anchor, &route)
            .is_err());
        let telemetry = provider.telemetry_snapshot();
        assert_eq!(
            telemetry, before,
            "capability queries must have no provider effects"
        );
        anchor.release().unwrap();
    }

    #[test]
    fn pool_dispatch_chunks_cover_each_output_once_without_overlap() {
        let lanes = crate::backend::wgpu::shaders::deep_learning::POOL_WORKGROUP_SIZE;
        for max_groups in [1, 2] {
            for total in [0, 1, lanes - 1, lanes, lanes + 1, 1025] {
                let mut visits = vec![0u8; total as usize];
                let mut offset = 0u32;
                while let Some(chunk) = next_pool_chunk(total, offset, max_groups).unwrap() {
                    assert_eq!(chunk.offset, offset);
                    assert!(chunk.workgroups <= max_groups);
                    assert_eq!(chunk.workgroups, chunk.output_count.div_ceil(lanes));
                    for local in 0..chunk.workgroups * lanes {
                        // Exactly the shader's chunk-local guard.
                        if local < chunk.output_count {
                            visits[(chunk.offset + local) as usize] += 1;
                        }
                    }
                    offset += chunk.output_count;
                }
                assert_eq!(offset, total);
                assert!(visits.iter().all(|&count| count == 1));
            }
        }
    }

    #[test]
    fn pool_dispatch_chunks_cover_u32_limit_without_allocating_outputs() {
        let total = u32::MAX;
        let max_groups = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS;
        let lanes = crate::backend::wgpu::shaders::deep_learning::POOL_WORKGROUP_SIZE;
        let mut offset = 0u32;
        let mut chunks = 0usize;
        while let Some(chunk) = next_pool_chunk(total, offset, max_groups).unwrap() {
            assert_eq!(chunk.offset, offset);
            assert!(chunk.workgroups > 0 && chunk.workgroups <= max_groups);
            assert_eq!(chunk.workgroups, chunk.output_count.div_ceil(lanes));
            assert!(chunk.output_count <= total - offset);
            offset += chunk.output_count;
            chunks += 1;
        }
        assert_eq!(offset, total);
        assert!(chunks > 1);
    }

    #[test]
    fn chunk_local_column_indices_preserve_numerical_order_across_boundary() {
        let total = 1025u32;
        let lanes = crate::backend::wgpu::shaders::deep_learning::POOL_WORKGROUP_SIZE;
        let mut values = vec![f64::NAN; total as usize];
        let mut offset = 0u32;
        while let Some(chunk) = next_pool_chunk(total, offset, 1).unwrap() {
            for local in 0..chunk.workgroups * lanes {
                if local < chunk.output_count {
                    let column = chunk.offset + local;
                    let sample_a = column as f64;
                    let sample_b = column as f64 + 10.0;
                    values[column as usize] = (sample_a + sample_b) / 2.0;
                }
            }
            offset += chunk.output_count;
        }
        assert_eq!(offset, total);
        for (column, value) in values.into_iter().enumerate() {
            assert_eq!(value, column as f64 + 5.0);
        }
    }
}
