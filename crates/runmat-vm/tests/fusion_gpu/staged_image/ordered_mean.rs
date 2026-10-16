use super::*;
use runmat_accelerate_api::{
    GpuTensorLayout, NumericElementType, ProviderElementType, ProviderExecutedOrderedRealMean,
    ProviderExecutionCompletionError, ProviderExecutionError, ProviderExecutionResult,
    ProviderLayout, ProviderOrderedRealMean, ProviderOrderedRealMeanControl,
    ProviderOrderedRealMeanLimits, ProviderOrderedRealMeanRecipe, ProviderOrderedRealMeanResources,
    ProviderOrderedRealMeanSource, ProviderOrderedRealMeanStep, ProviderPrecision,
    ProviderRealMeanNaNMode, ProviderRepresentation, ProviderResidency, ProviderStorage,
};

fn real_value(input: &BufferPayload, index: usize) -> Option<f64> {
    match input {
        BufferPayload::F32(values) => values.get(index).copied().map(f64::from),
        BufferPayload::F64(values) => values.get(index).copied(),
        BufferPayload::ComplexF32(_) | BufferPayload::ComplexF64(_) | BufferPayload::U8(_) => None,
    }
}

fn requested_bytes(shape: &[usize], precision: ProviderPrecision) -> Option<u64> {
    shape
        .iter()
        .try_fold(1_u64, |count, &extent| {
            count.checked_mul(u64::try_from(extent).ok()?)
        })?
        .checked_mul(if precision == ProviderPrecision::F32 {
            4
        } else {
            8
        })
}

fn reduce_axis(
    input: &BufferPayload,
    shape: &[usize],
    axis: usize,
    output_precision: ProviderPrecision,
) -> Option<(Vec<f64>, Vec<usize>)> {
    let extent = *shape.get(axis)?;
    if extent == 0 {
        return None;
    }
    let mut output_shape = shape.to_vec();
    output_shape[axis] = 1;
    let count = output_shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))?;
    let axis_stride = shape
        .iter()
        .take(axis)
        .try_fold(1usize, |product, &size| product.checked_mul(size))?;
    let mut output = Vec::new();
    output.try_reserve_exact(count).ok()?;
    for linear in 0..count {
        let mut remaining = linear;
        let mut base = 0usize;
        let mut stride = 1usize;
        for (dimension, &size) in output_shape.iter().enumerate() {
            let coordinate = remaining % size;
            remaining /= size;
            base = base.checked_add(coordinate.checked_mul(stride)?)?;
            stride = stride.checked_mul(shape[dimension])?;
        }
        let mut total = 0.0_f64;
        for position in 0..extent {
            let offset = base.checked_add(position.checked_mul(axis_stride)?)?;
            total += real_value(input, offset)?;
        }
        let mean = total / extent as f64;
        output.push(if output_precision == ProviderPrecision::F32 {
            f64::from(mean as f32)
        } else {
            mean
        });
    }
    Some((output, output_shape))
}

fn failed_after_selected_output(
    retained: Vec<GpuTensorHandle>,
    primary: ProviderExecutionError,
) -> ProviderExecutionError {
    if retained.is_empty() {
        return primary;
    }
    let cleanup = retained
        .into_iter()
        .filter_map(|intermediate| intermediate.release().err())
        .collect::<Vec<_>>();
    if cleanup.is_empty() {
        primary.after_effect()
    } else {
        ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
            Some(primary),
            cleanup,
        ))
        .after_effect()
    }
}

impl TestProvider {
    pub(in super::super) fn staged_mean_plan(
        &self,
        source: &ProviderRepresentation,
        source_buffer_bytes: u64,
        axes_zero_based: &[usize],
        output_precision: ProviderPrecision,
        nan_mode: ProviderRealMeanNaNMode,
    ) -> anyhow::Result<Option<ProviderOrderedRealMeanRecipe>> {
        if nan_mode != ProviderRealMeanNaNMode::Include
            || axes_zero_based.is_empty()
            || source.storage != ProviderStorage::DenseReal
            || source.layout != ProviderLayout::ColumnMajorContiguous
            || source.residency != ProviderResidency::Device
            || source.checked_byte_len() != Some(source_buffer_bytes)
            || !matches!(
                source.element_type,
                ProviderElementType::F32 | ProviderElementType::F64
            )
        {
            return Ok(None);
        }
        let mut shape = source
            .shape
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        if shape.contains(&0)
            || axes_zero_based.iter().any(|&axis| axis >= shape.len())
            || axes_zero_based.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Ok(None);
        }
        let mut input_precision = match source.element_type {
            ProviderElementType::F32 => ProviderPrecision::F32,
            ProviderElementType::F64 => ProviderPrecision::F64,
            _ => return Ok(None),
        };
        let mut input_bytes = source_buffer_bytes;
        let mut peak = 0_u64;
        let mut steps = Vec::new();
        steps.try_reserve_exact(axes_zero_based.len())?;
        for (index, &axis) in axes_zero_based.iter().enumerate() {
            let previous = shape.clone();
            shape[axis] = 1;
            let precision = if index + 1 == axes_zero_based.len() {
                output_precision
            } else {
                ProviderPrecision::F64
            };
            let Some(output_bytes) = requested_bytes(&shape, precision) else {
                return Ok(None);
            };
            peak = peak
                .checked_add(output_bytes)
                .ok_or_else(|| anyhow!("selected mean resource size overflow"))?;
            steps.push(
                ProviderOrderedRealMeanStep::new(
                    axis,
                    previous,
                    shape.clone(),
                    input_precision,
                    precision,
                )
                .with_physical_resources(input_bytes, output_bytes, 0, 1),
            );
            input_precision = precision;
            input_bytes = output_bytes;
        }
        Ok(Some(ProviderOrderedRealMeanRecipe::new(
            ProviderOrderedRealMeanSource {
                representation: source.clone(),
                buffer_bytes: source_buffer_bytes,
            },
            ProviderOrderedRealMeanControl {
                axes: axes_zero_based.to_vec(),
                output_precision,
                nan_mode,
            },
            ProviderOrderedRealMeanResources {
                limits: ProviderOrderedRealMeanLimits {
                    max_buffer_size: u64::MAX,
                    max_storage_buffer_binding_size: u32::MAX,
                    max_uniform_buffer_binding_size: u32::MAX,
                    max_compute_invocations_per_workgroup: 256,
                    max_compute_workgroup_size_x: 256,
                    max_compute_workgroups_per_dimension: u32::MAX,
                },
                steps,
                incremental_peak_bytes: peak,
                queue_submissions: u32::try_from(axes_zero_based.len())?,
            },
        )))
    }

    pub(in super::super) fn staged_mean_prepare(
        &self,
        source: &GpuTensorHandle,
        recipe: &ProviderOrderedRealMeanRecipe,
    ) -> anyhow::Result<Option<ProviderOrderedRealMean>> {
        if source.view().layout() != GpuTensorLayout::Contiguous {
            return Ok(None);
        }
        let live_real = {
            let buffers = self.buffers.lock().unwrap();
            buffers.get(&source.buffer_id()).is_some_and(|record| {
                record.owns(source)
                    && record.shape == source.shape()
                    && matches!(
                        (&record.data, source.physical().element_type()),
                        (BufferPayload::F32(_), NumericElementType::F32)
                            | (BufferPayload::F64(_), NumericElementType::F64)
                    )
            })
        };
        if !live_real {
            return Ok(None);
        }
        let element_type = match source.physical().element_type() {
            NumericElementType::F32 => ProviderElementType::F32,
            NumericElementType::F64 => ProviderElementType::F64,
            _ => return Ok(None),
        };
        let representation = ProviderRepresentation {
            element_type,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: source
                .shape()
                .iter()
                .copied()
                .map(u64::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            residency: ProviderResidency::Device,
        };
        let current = self.staged_mean_plan(
            &representation,
            source.physical().requested_bytes(),
            recipe.axes(),
            recipe.output_precision(),
            recipe.nan_mode(),
        )?;
        Ok(current
            .filter(|current| current == recipe)
            .map(|current| ProviderOrderedRealMean::new(source.clone(), current)))
    }

    pub(in super::super) fn staged_mean_execute(
        &self,
        selected: ProviderOrderedRealMean,
    ) -> ProviderExecutionResult<ProviderExecutedOrderedRealMean> {
        selected.steps().last().ok_or_else(|| {
            ProviderExecutionError::from(anyhow!("selected mean has no reduction step"))
        })?;
        let mut retained = Vec::new();
        retained
            .try_reserve_exact(selected.steps().len())
            .map_err(ProviderExecutionError::from_source)?;
        for step in selected.steps() {
            let reduced = {
                let source = retained.last().unwrap_or(selected.source());
                let buffers = self.buffers.lock().unwrap();
                buffers
                    .get(&source.buffer_id())
                    .filter(|record| record.owns(source) && record.shape == step.input_shape())
                    .and_then(|record| {
                        reduce_axis(
                            &record.data,
                            &record.shape,
                            step.axis(),
                            step.output_precision(),
                        )
                    })
            };
            let Some((values, shape)) = reduced else {
                return Err(failed_after_selected_output(
                    retained,
                    ProviderExecutionError::from(anyhow!(
                        "selected mean source or reduction geometry is invalid"
                    )),
                ));
            };
            let class = if step.output_precision() == ProviderPrecision::F32 {
                NumericElementType::F32
            } else {
                NumericElementType::F64
            };
            retained.push(self.push_typed(values, shape, class));
        }
        let output = retained.pop().ok_or_else(|| {
            ProviderExecutionError::from(anyhow!("selected mean has no output allocation"))
        })?;
        let mut cleanup = retained
            .into_iter()
            .filter_map(|intermediate| intermediate.release().err())
            .collect::<Vec<_>>();
        if !cleanup.is_empty() {
            cleanup.extend(output.release().err());
            return Err(ProviderExecutionError::from_source(
                ProviderExecutionCompletionError::new(None, cleanup),
            )
            .after_effect());
        }
        Ok(ProviderExecutedOrderedRealMean {
            observed_output_bytes: output.physical().requested_bytes(),
            output,
            requested_incremental_peak_bytes: selected.requested_incremental_peak_bytes(),
            observed_incremental_peak_bytes: selected.requested_incremental_peak_bytes(),
            selected_submissions: selected.recipe().queue_submissions(),
        })
    }
}
