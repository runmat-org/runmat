use super::*;

pub(super) async fn apply_resample(
    input: SampleInput,
    p: usize,
    q: usize,
    options: &HostResampleOptions,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<ResampleEval> {
    match input {
        SampleInput::Real { storage, shape } => {
            let dtype = storage.numeric_dtype();
            let data = match storage {
                NumericStorage::F64(values) => values,
                NumericStorage::F32(values) => {
                    values.into_iter().map(f64::from).collect::<Vec<_>>()
                }
                NumericStorage::I8(_)
                | NumericStorage::I16(_)
                | NumericStorage::I32(_)
                | NumericStorage::I64(_)
                | NumericStorage::U8(_)
                | NumericStorage::U16(_)
                | NumericStorage::U32(_)
                | NumericStorage::U64(_) => {
                    return Err(sample_error_with_detail(
                        RESAMPLE_NAME,
                        &SAMPLE_ERROR_INVALID_INPUT,
                        "resample input must be single or double",
                    ))
                }
            };
            let (output, shape) =
                resample_rational_column_major(data, shape, p, q, options, cooperation).await?;
            let storage = match dtype {
                runmat_value::NumericDType::F64 => NumericStorage::F64(output),
                runmat_value::NumericDType::F32 => {
                    NumericStorage::F32(output.into_iter().map(|value| value as f32).collect())
                }
                _ => unreachable!("resample input dtype was validated"),
            };
            let tensor = Tensor::from_numeric_storage(storage, shape).map_err(|err| {
                sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_BUILD_OUTPUT, err)
            })?;
            Ok(ResampleEval {
                y: tensor::tensor_into_value(tensor),
                filter: ResampleFilter::Host(options.filter.clone()),
            })
        }
        SampleInput::Complex { data, shape, dtype } => {
            let (output, shape) =
                resample_rational_column_major(data, shape, p, q, options, cooperation).await?;
            let tensor =
                ComplexTensor::from_f64_values_with_dtype(output, shape, dtype).map_err(|err| {
                    sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_BUILD_OUTPUT, err)
                })?;
            let y = if dtype == runmat_value::NumericDType::F64
                && tensor.materialize_f64().len() == 1
            {
                let (re, im) = tensor.materialize_f64()[0];
                Value::Complex(re, im)
            } else {
                Value::ComplexTensor(tensor)
            };
            Ok(ResampleEval {
                y,
                filter: ResampleFilter::Host(options.filter.clone()),
            })
        }
    }
}

pub(super) async fn apply_sample_rate(
    input: SampleInput,
    factor: usize,
    phase: usize,
    op: SampleOp,
    builtin: &'static str,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    match input {
        SampleInput::Real { storage, shape } => {
            let (storage, shape) =
                resample_numeric_storage(storage, shape, factor, phase, op, builtin, cooperation)
                    .await?;
            let tensor = Tensor::from_numeric_storage(storage, shape).map_err(|err| {
                sample_error_with_detail(builtin, &SAMPLE_ERROR_BUILD_OUTPUT, err)
            })?;
            Ok(tensor::tensor_into_value(tensor))
        }
        SampleInput::Complex { data, shape, dtype } => {
            let (output, shape) = resample_column_major(
                data,
                shape,
                factor,
                phase,
                op,
                (0.0, 0.0),
                builtin,
                cooperation,
            )
            .await?;
            let tensor =
                ComplexTensor::from_f64_values_with_dtype(output, shape, dtype).map_err(|err| {
                    sample_error_with_detail(builtin, &SAMPLE_ERROR_BUILD_OUTPUT, err)
                })?;
            if dtype == runmat_value::NumericDType::F64 && tensor.materialize_f64().len() == 1 {
                let (re, im) = tensor.materialize_f64()[0];
                Ok(Value::Complex(re, im))
            } else {
                Ok(Value::ComplexTensor(tensor))
            }
        }
    }
}

pub(super) async fn resample_numeric_storage(
    storage: NumericStorage,
    shape: Vec<usize>,
    factor: usize,
    phase: usize,
    op: SampleOp,
    builtin: &'static str,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<(NumericStorage, Vec<usize>)> {
    macro_rules! resample {
        ($values:expr, $zero:expr, $variant:ident) => {{
            let (values, shape) = resample_column_major(
                $values,
                shape,
                factor,
                phase,
                op,
                $zero,
                builtin,
                cooperation,
            )
            .await?;
            (NumericStorage::$variant(values), shape)
        }};
    }
    Ok(match storage {
        NumericStorage::F64(values) => resample!(values, 0.0, F64),
        NumericStorage::F32(values) => resample!(values, 0.0_f32, F32),
        NumericStorage::I8(values) => resample!(values, 0_i8, I8),
        NumericStorage::I16(values) => resample!(values, 0_i16, I16),
        NumericStorage::I32(values) => resample!(values, 0_i32, I32),
        NumericStorage::I64(values) => resample!(values, 0_i64, I64),
        NumericStorage::U8(values) => resample!(values, 0_u8, U8),
        NumericStorage::U16(values) => resample!(values, 0_u16, U16),
        NumericStorage::U32(values) => resample!(values, 0_u32, U32),
        NumericStorage::U64(values) => resample!(values, 0_u64, U64),
    })
}

pub(super) fn upsample_linear_indices(
    shape: &[usize],
    dim: usize,
    input_len: usize,
    output_len: usize,
    factor: usize,
    phase: usize,
    builtin: &'static str,
) -> BuiltinResult<Option<Vec<u32>>> {
    let input_count = checked_element_count(shape)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    if input_count > u32::MAX as usize {
        return Ok(None);
    }

    let leading = checked_product(&shape[..dim])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let trailing = checked_product(&shape[dim + 1..])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(input_count)
        .map_err(|_| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;

    for trail in 0..trailing {
        for input_pos in 0..input_len {
            let output_pos = input_pos
                .checked_mul(factor)
                .and_then(|offset| offset.checked_add(phase))
                .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
            let trail_offset = output_len
                .checked_mul(trail)
                .and_then(|offset| offset.checked_add(output_pos))
                .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
            for before in 0..leading {
                let Some(dst) = before.checked_add(
                    leading
                        .checked_mul(trail_offset)
                        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?,
                ) else {
                    return Err(sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW));
                };
                let Ok(dst) = u32::try_from(dst) else {
                    return Ok(None);
                };
                indices.push(dst);
            }
        }
    }

    Ok(Some(indices))
}

pub(super) fn downsample_linear_indices(
    shape: &[usize],
    dim: usize,
    input_len: usize,
    output_len: usize,
    factor: usize,
    phase: usize,
    builtin: &'static str,
) -> BuiltinResult<Option<Vec<u32>>> {
    let output_shape_count = {
        let mut output_shape = shape.to_vec();
        output_shape[dim] = output_len;
        checked_element_count(&output_shape)
            .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?
    };
    if output_shape_count > u32::MAX as usize {
        return Ok(None);
    }

    let leading = checked_product(&shape[..dim])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let trailing = checked_product(&shape[dim + 1..])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(output_shape_count)
        .map_err(|_| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;

    for trail in 0..trailing {
        for output_pos in 0..output_len {
            let input_pos = output_pos
                .checked_mul(factor)
                .and_then(|offset| offset.checked_add(phase))
                .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
            let trail_offset = input_len
                .checked_mul(trail)
                .and_then(|offset| offset.checked_add(input_pos))
                .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
            for before in 0..leading {
                let Some(src) = before.checked_add(
                    leading
                        .checked_mul(trail_offset)
                        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?,
                ) else {
                    return Err(sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW));
                };
                let Ok(src) = u32::try_from(src) else {
                    return Ok(None);
                };
                indices.push(src);
            }
        }
    }

    Ok(Some(indices))
}

pub(super) async fn resample_column_major<T: Copy>(
    data: Vec<T>,
    shape: Vec<usize>,
    factor: usize,
    phase: usize,
    op: SampleOp,
    zero: T,
    builtin: &'static str,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<(Vec<T>, Vec<usize>)> {
    let shape = canonical_shape(shape, data.len());
    let dim = first_non_singleton_dim(&shape);
    let input_len = shape[dim];
    let output_len = output_len(input_len, factor, phase, op)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output_shape = shape.clone();
    output_shape[dim] = output_len;
    let output_count = checked_element_count(&output_shape)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_count)
        .map_err(|_| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    output.resize(output_count, zero);
    if input_len == 0 || output_len == 0 {
        return Ok((output, output_shape));
    }

    let leading = checked_product(&shape[..dim])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let trailing = checked_product(&shape[dim + 1..])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;

    let mut work = 0u64;
    for trail in 0..trailing {
        for before in 0..leading {
            match op {
                SampleOp::Up => {
                    for input_pos in 0..input_len {
                        let output_pos = phase + input_pos * factor;
                        let src = before + leading * (input_pos + input_len * trail);
                        let dst = before + leading * (output_pos + output_len * trail);
                        output[dst] = data[src];
                        work += 1;
                        if work == SIGNAL_WORK_CHUNK {
                            cooperation.completed(work).await?;
                            work = 0;
                        }
                    }
                }
                SampleOp::Down => {
                    for output_pos in 0..output_len {
                        let input_pos = phase + output_pos * factor;
                        let src = before + leading * (input_pos + input_len * trail);
                        let dst = before + leading * (output_pos + output_len * trail);
                        output[dst] = data[src];
                        work += 1;
                        if work == SIGNAL_WORK_CHUNK {
                            cooperation.completed(work).await?;
                            work = 0;
                        }
                    }
                }
            }
        }
    }

    cooperation.completed(work).await?;
    Ok((output, output_shape))
}

trait ResampleSample: Copy {
    fn zero() -> Self;
    fn add_scaled(self, sample: Self, coeff: f64) -> Self;
}

impl ResampleSample for f64 {
    fn zero() -> Self {
        0.0
    }

    fn add_scaled(self, sample: Self, coeff: f64) -> Self {
        self + sample * coeff
    }
}

impl ResampleSample for (f64, f64) {
    fn zero() -> Self {
        (0.0, 0.0)
    }

    fn add_scaled(self, sample: Self, coeff: f64) -> Self {
        (self.0 + sample.0 * coeff, self.1 + sample.1 * coeff)
    }
}

async fn resample_rational_column_major<T: ResampleSample>(
    data: Vec<T>,
    shape: Vec<usize>,
    p: usize,
    q: usize,
    options: &HostResampleOptions,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<(Vec<T>, Vec<usize>)> {
    let mut shape = canonical_shape(shape, data.len());
    let dim = options
        .dim
        .unwrap_or_else(|| first_non_singleton_dim(&shape));
    if dim >= shape.len() {
        shape.resize(dim + 1, 1);
    }
    let input_len = shape[dim];
    let output_len = resample_output_len(input_len, p, q)
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output_shape = shape.clone();
    output_shape[dim] = output_len;
    let output_count = checked_element_count(&output_shape)
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_count)
        .map_err(|_| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    output.resize(output_count, T::zero());
    if input_len == 0 || output_len == 0 {
        return Ok((output, output_shape));
    }

    let leading = checked_product(&shape[..dim])
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let trailing = checked_product(&shape[dim + 1..])
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let delay = options.filter.len() / 2;

    let mut work = 0u64;
    for trail in 0..trailing {
        for before in 0..leading {
            for output_pos in 0..output_len {
                let Some(center) = output_pos
                    .checked_mul(q)
                    .and_then(|value| value.checked_add(delay))
                else {
                    return Err(sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW));
                };
                let mut acc = T::zero();
                for (tap, coeff) in options.filter.iter().enumerate() {
                    work += 1;
                    if work == SIGNAL_WORK_CHUNK {
                        cooperation.completed(work).await?;
                        work = 0;
                    }
                    if tap > center {
                        break;
                    }
                    let upsampled_pos = center - tap;
                    if upsampled_pos % p != 0 {
                        continue;
                    }
                    let input_pos = upsampled_pos / p;
                    if input_pos >= input_len {
                        continue;
                    }
                    let src = before + leading * (input_pos + input_len * trail);
                    acc = acc.add_scaled(data[src], *coeff);
                }
                let dst = before + leading * (output_pos + output_len * trail);
                output[dst] = acc;
            }
        }
    }

    cooperation.completed(work).await?;
    Ok((output, output_shape))
}

pub(super) fn resample_output_len(input_len: usize, p: usize, q: usize) -> Option<usize> {
    input_len
        .checked_mul(p)?
        .checked_add(q.checked_sub(1)?)?
        .checked_div(q)
}

pub(super) fn design_resample_filter(
    p: usize,
    q: usize,
    n: usize,
    beta: f64,
) -> BuiltinResult<Vec<f64>> {
    let len = resample_filter_len(p, q, n)?;
    let max_factor = p.max(q);
    let half_len = (len - 1) / 2;
    let cutoff = 1.0 / max_factor as f64;
    let mut filter = Vec::new();
    filter
        .try_reserve_exact(len)
        .map_err(|_| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let denom = modified_bessel_i0(beta);
    for idx in 0..len {
        let centered = idx as isize - half_len as isize;
        let sinc = if centered == 0 {
            cutoff
        } else {
            let x = std::f64::consts::PI * cutoff * centered as f64;
            x.sin() / (std::f64::consts::PI * centered as f64)
        };
        let window = if half_len == 0 {
            1.0
        } else {
            let ratio = centered as f64 / half_len as f64;
            modified_bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) / denom
        };
        filter.push(sinc * window);
    }
    let sum = filter.iter().sum::<f64>();
    if !sum.is_finite() || sum == 0.0 {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "designed filter has zero or nonfinite gain",
        ));
    }
    let gain = p as f64 / sum;
    for coeff in &mut filter {
        *coeff *= gain;
    }
    Ok(filter)
}

pub(super) fn resample_filter_len(p: usize, q: usize, n: usize) -> BuiltinResult<usize> {
    n.checked_mul(p.max(q))
        .and_then(|half_len| half_len.checked_mul(2))
        .and_then(|twice| twice.checked_add(1))
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))
}

pub(super) fn modified_bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.515_622_9
                + y * (3.089_942_4
                    + y * (1.206_749_2 + y * (0.265_973_2 + y * (0.036_076_8 + y * 0.004_581_3)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.398_942_28
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77))))))))
    }
}

pub(super) fn filter_tensor(filter: Vec<f64>) -> BuiltinResult<Tensor> {
    let len = filter.len();
    Tensor::new(filter, vec![1, len])
        .map_err(|err| sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_BUILD_OUTPUT, err))
}

pub(super) fn canonical_shape(shape: Vec<usize>, len: usize) -> Vec<usize> {
    if shape.is_empty() {
        if len == 0 {
            vec![0, 1]
        } else {
            vec![1, 1]
        }
    } else {
        shape
    }
}

pub(super) fn first_non_singleton_dim(shape: &[usize]) -> usize {
    shape.iter().position(|dim| *dim != 1).unwrap_or(0)
}

pub(super) fn output_len(
    input_len: usize,
    factor: usize,
    phase: usize,
    op: SampleOp,
) -> Option<usize> {
    match op {
        SampleOp::Up => input_len.checked_mul(factor),
        SampleOp::Down => {
            if input_len <= phase {
                Some(0)
            } else {
                Some(((input_len - 1 - phase) / factor) + 1)
            }
        }
    }
}

pub(super) fn checked_element_count(shape: &[usize]) -> Option<usize> {
    checked_product(shape)
}

pub(super) fn checked_product(values: &[usize]) -> Option<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
}
