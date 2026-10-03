use runmat_accelerate_api::{
    ImfilterMode, ImfilterOptions, ImfilterPadding, ImfilterShape, ProviderPrecision,
};
use runmat_value::{LogicalArray, NumericDType, NumericStorage, Tensor, Value};

use crate::builtins::common::tensor;

pub use super::super::geometry::ImfilterKernelPoint;
use super::super::geometry::{extend_shape, normalize_shape, validate_kernel_shape};
use super::{
    filter_error, BuiltinResult, ImageOutputClass, ImfilterKernelError, ImfilterKernelResult,
};

/// Core host implementation of `imfilter`, shared with the in-process acceleration provider.

#[derive(Clone, Debug)]
pub struct ImfilterPlan {
    pub rank: usize,
    pub output_shape_ext: Vec<usize>,
    pub final_shape: Vec<usize>,
    pub image_shape_ext: Vec<usize>,
    pub image_strides: Vec<usize>,
    pub base_offset: Vec<isize>,
    pub kernel_points: Vec<ImfilterKernelPoint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterAccumulation {
    HostDouble,
    GpuSingle,
}

/// Floating provider arithmetic follows placement intent, independently of
/// physical input/output storage or a provider's preferred precision.
pub fn floating_accumulation(precision: ProviderPrecision, explicit: bool) -> FilterAccumulation {
    fallback_accumulation(
        match precision {
            ProviderPrecision::F32 => ImageOutputClass::Single,
            ProviderPrecision::F64 => ImageOutputClass::Double,
        },
        explicit,
    )
}

pub(super) fn fallback_accumulation(
    output_class: ImageOutputClass,
    explicit_gpu_array: bool,
) -> FilterAccumulation {
    if !explicit_gpu_array {
        FilterAccumulation::HostDouble
    } else {
        match output_class {
            ImageOutputClass::Integer(NumericDType::U32) | ImageOutputClass::Double => {
                FilterAccumulation::HostDouble
            }
            ImageOutputClass::Integer(_) | ImageOutputClass::Logical | ImageOutputClass::Single => {
                FilterAccumulation::GpuSingle
            }
        }
    }
}

impl ImfilterPlan {
    #[inline]
    pub fn evaluate(&self, image_data: &[f64], options: &ImfilterOptions) -> Vec<f64> {
        self.evaluate_with_accumulation(image_data, options, FilterAccumulation::HostDouble)
    }

    fn evaluate_with_accumulation(
        &self,
        image_data: &[f64],
        options: &ImfilterOptions,
        accumulation: FilterAccumulation,
    ) -> Vec<f64> {
        evaluate_filter(
            &self.output_shape_ext,
            &self.base_offset,
            &self.image_shape_ext,
            &self.image_strides,
            &self.kernel_points,
            options,
            image_data,
            accumulation,
        )
    }
}

pub fn build_imfilter_plan(
    image_shape: &[usize],
    kernel: &Tensor,
    options: &ImfilterOptions,
    builtin: &str,
) -> ImfilterKernelResult<ImfilterPlan> {
    if kernel.is_empty() || kernel.shape.contains(&0) {
        return Err(ImfilterKernelError::new(
            builtin,
            format!("{builtin}: filter must be non-empty along every dimension"),
        ));
    }

    let final_shape = filter_output_shape(image_shape, &kernel.shape, options.shape, builtin)?;
    let image_shape_norm = normalize_shape(image_shape);
    let kernel_shape_norm = normalize_shape(&kernel.shape);
    let rank = image_shape_norm.len().max(kernel_shape_norm.len());
    let image_ext = extend_shape(&image_shape_norm, rank);
    let kernel_ext = extend_shape(&kernel_shape_norm, rank);

    validate_kernel_shape(&image_shape_norm, &kernel_ext, builtin)
        .map_err(|message| ImfilterKernelError::new(builtin, message))?;

    // MATLAB defines the one-based center as floor((size(h) + 1) / 2).
    // In zero-based coordinates that is (size(h) - 1) / 2, which matters for
    // every even-sized kernel.
    let origin: Vec<usize> = kernel_ext
        .iter()
        .map(|&dim| dim.saturating_sub(1) / 2)
        .collect();
    let full_shape: Vec<usize> = image_ext
        .iter()
        .zip(kernel_ext.iter())
        .map(|(&img, &ker)| {
            img.checked_add(ker.saturating_sub(1)).ok_or_else(|| {
                ImfilterKernelError::new(builtin, format!("{builtin}: output shape overflows"))
            })
        })
        .collect::<ImfilterKernelResult<_>>()?;

    let kernel_points = build_kernel_points(kernel, &kernel_ext, &origin, options.mode);
    let zero_offset = vec![0isize; rank];
    let origin_signed: Vec<isize> = origin.iter().map(|&o| o as isize).collect();
    let full_offset: Vec<isize> = origin
        .iter()
        .zip(kernel_ext.iter())
        .map(|(&orig, &dim)| orig as isize - (dim as isize - 1))
        .collect();

    let (target_shape_ext, base_offset) = match options.shape {
        ImfilterShape::Full => (full_shape.clone(), full_offset),
        ImfilterShape::Same => (image_ext.clone(), zero_offset),
        ImfilterShape::Valid => {
            let target: Vec<usize> = image_ext
                .iter()
                .zip(kernel_ext.iter())
                .map(|(&img, &ker)| if img >= ker { img - ker + 1 } else { 0 })
                .collect();
            (target, origin_signed)
        }
    };

    let image_strides = compute_strides(&image_ext);

    Ok(ImfilterPlan {
        rank,
        output_shape_ext: target_shape_ext,
        final_shape,
        image_shape_ext: image_ext,
        image_strides,
        base_offset,
        kernel_points,
    })
}

/// Output geometry is needed before a provider result can be adopted. This
/// projection performs the same image/kernel rank and shape admission as the
/// host plan without materializing a dummy kernel or touching resident data.
pub fn filter_output_shape(
    image_shape: &[usize],
    kernel_shape: &[usize],
    shape_mode: ImfilterShape,
    builtin: &str,
) -> ImfilterKernelResult<Vec<usize>> {
    super::super::geometry::filter_output_shape(image_shape, kernel_shape, shape_mode, builtin)
        .map_err(|message| ImfilterKernelError::new(builtin, message))
}

#[cfg(test)]
mod output_shape_tests {
    use super::*;

    #[test]
    fn shape_only_provider_admission_matches_host_plan_for_all_modes() {
        let kernel = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
        for shape in [
            ImfilterShape::Same,
            ImfilterShape::Full,
            ImfilterShape::Valid,
        ] {
            let options = ImfilterOptions {
                shape,
                ..Default::default()
            };
            let plan = build_imfilter_plan(&[4, 5], &kernel, &options, "imfilter").unwrap();
            assert_eq!(
                filter_output_shape(&[4, 5], &kernel.shape, shape, "imfilter").unwrap(),
                plan.final_shape,
            );
        }
        assert!(filter_output_shape(&[4, 5], &[0, 3], ImfilterShape::Same, "imfilter").is_err());
    }
}

pub fn apply_imfilter_tensor(
    image: &Tensor,
    kernel: &Tensor,
    options: &ImfilterOptions,
    builtin: &str,
) -> ImfilterKernelResult<Tensor> {
    let plan = build_imfilter_plan(&image.shape, kernel, options, builtin)?;
    let image_values = tensor::tensor_values_f64_cow(image);
    let data = plan.evaluate(&image_values, options);
    Tensor::new(data, plan.final_shape.clone())
        .map_err(|e| ImfilterKernelError::new(builtin, format!("{builtin}: {e}")))
}

/// Native floating adapter over the same operation-owned plan and evaluator.
/// Prepared H may use the accumulation encoding; public H admission remains
/// Double at the builtin boundary. Output storage always follows A.
pub fn apply_imfilter_native(
    image: &Tensor,
    kernel: &Tensor,
    options: &ImfilterOptions,
    explicit: bool,
    builtin: &str,
) -> ImfilterKernelResult<Tensor> {
    let precision = match image.numeric_dtype() {
        NumericDType::F32 => ProviderPrecision::F32,
        NumericDType::F64 => ProviderPrecision::F64,
        _ => {
            return Err(ImfilterKernelError::new(
                builtin,
                "native imfilter requires floating A",
            ))
        }
    };
    let plan = build_imfilter_plan(&image.shape, kernel, options, builtin)?;
    let values = tensor::tensor_values_f64_cow(image);
    let data = plan.evaluate_with_accumulation(
        &values,
        options,
        floating_accumulation(precision, explicit),
    );
    match precision {
        ProviderPrecision::F32 => Tensor::from_f32(
            data.into_iter().map(|value| value as f32).collect(),
            plan.final_shape,
        ),
        ProviderPrecision::F64 => Tensor::new(data, plan.final_shape),
    }
    .map_err(|error| ImfilterKernelError::new(builtin, error))
}

pub(super) fn apply_imfilter_host(
    image: &Tensor,
    kernel: &Tensor,
    options: &ImfilterOptions,
    output_class: ImageOutputClass,
    accumulation: FilterAccumulation,
    builtin: &str,
) -> BuiltinResult<Value> {
    let plan = build_imfilter_plan(&image.shape, kernel, options, builtin)?;
    let image_values = tensor::tensor_values_f64_cow(image);
    let data = plan.evaluate_with_accumulation(&image_values, options, accumulation);
    let shape = plan.final_shape;

    let storage = match output_class {
        ImageOutputClass::Double => NumericStorage::F64(data),
        ImageOutputClass::Single => {
            NumericStorage::F32(data.into_iter().map(|value| value as f32).collect())
        }
        ImageOutputClass::Integer(dtype) => integer_filter_storage(data, dtype),
        ImageOutputClass::Logical => {
            let bits = data
                .into_iter()
                .map(|value| u8::from(round_and_clamp(value, 0.0, 1.0) != 0.0))
                .collect();
            let logical = LogicalArray::new(bits, shape)
                .map_err(|error| filter_error(builtin, format!("{builtin}: {error}")))?;
            return Ok(if logical.data.len() == 1 {
                Value::Bool(logical.data[0] != 0)
            } else {
                Value::LogicalArray(logical)
            });
        }
    };
    let tensor = Tensor::from_numeric_storage(storage, shape)
        .map_err(|error| filter_error(builtin, format!("{builtin}: {error}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn integer_filter_storage(data: Vec<f64>, dtype: NumericDType) -> NumericStorage {
    macro_rules! signed {
        ($variant:ident, $ty:ty) => {
            NumericStorage::$variant(
                data.into_iter()
                    .map(|value| {
                        round_and_clamp(value, <$ty>::MIN as f64, <$ty>::MAX as f64) as $ty
                    })
                    .collect(),
            )
        };
    }
    macro_rules! unsigned {
        ($variant:ident, $ty:ty) => {
            NumericStorage::$variant(
                data.into_iter()
                    .map(|value| round_and_clamp(value, 0.0, <$ty>::MAX as f64) as $ty)
                    .collect(),
            )
        };
    }
    match dtype {
        NumericDType::I8 => signed!(I8, i8),
        NumericDType::I16 => signed!(I16, i16),
        NumericDType::I32 => signed!(I32, i32),
        NumericDType::U8 => unsigned!(U8, u8),
        NumericDType::U16 => unsigned!(U16, u16),
        NumericDType::U32 => unsigned!(U32, u32),
        NumericDType::I64 | NumericDType::U64 | NumericDType::F32 | NumericDType::F64 => {
            unreachable!("unsupported imfilter output class rejected before finalization")
        }
    }
}

fn round_and_clamp(value: f64, minimum: f64, maximum: f64) -> f64 {
    if value.is_nan() {
        0.0_f64.clamp(minimum, maximum)
    } else {
        value.round().clamp(minimum, maximum)
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(dim);
    }
    strides
}

fn advance_index(index: &mut [usize], shape: &[usize]) {
    for (idx, &dimension) in index.iter_mut().zip(shape.iter()) {
        *idx += 1;
        if *idx < dimension {
            return;
        }
        *idx = 0;
    }
}

fn build_kernel_points(
    kernel: &Tensor,
    kernel_shape: &[usize],
    origin: &[usize],
    mode: ImfilterMode,
) -> Vec<ImfilterKernelPoint> {
    let rank = kernel_shape.len();
    let strides = compute_strides(kernel_shape);
    let values = tensor::tensor_values_f64_cow(kernel);
    let total = values.len();
    let mut points = Vec::with_capacity(total);
    if total == 0 {
        return points;
    }

    let mut index = vec![0usize; rank];
    for _ in 0..total {
        let linear = index_to_linear(&index, &strides);
        let value = match mode {
            ImfilterMode::Correlation => values[linear],
            ImfilterMode::Convolution => {
                values[flipped_linear_index(&index, kernel_shape, &strides)]
            }
        };
        let offsets = index
            .iter()
            .zip(origin.iter())
            .map(|(&idx, &orig)| idx as isize - orig as isize)
            .collect();
        points.push(ImfilterKernelPoint { offsets, value });
        advance_index(&mut index, kernel_shape);
    }

    points
}

fn flipped_linear_index(index: &[usize], shape: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .enumerate()
        .map(|(dim, &coord)| (shape[dim] - 1 - coord) * strides[dim])
        .sum()
}

fn index_to_linear(index: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .zip(strides.iter())
        .map(|(&coord, &stride)| coord * stride)
        .sum()
}

fn sample_with_padding(
    image: &[f64],
    image_shape: &[usize],
    image_strides: &[usize],
    base_index: &[usize],
    base_offset: &[isize],
    offsets: &[isize],
    options: &ImfilterOptions,
) -> f64 {
    if image.is_empty() {
        return options.constant_value;
    }

    let mut final_indices = Vec::with_capacity(image_shape.len());
    for (dim, (&base, &offset)) in base_index.iter().zip(offsets.iter()).enumerate() {
        let coord = base as isize + base_offset[dim] + offset;
        let len = image_shape[dim] as isize;
        if coord >= 0 && coord < len {
            final_indices.push(coord as usize);
            continue;
        }
        match options.padding {
            ImfilterPadding::Constant => return options.constant_value,
            ImfilterPadding::Replicate => final_indices.push(clamp_index(coord, len)),
            ImfilterPadding::Circular => final_indices.push(wrap_index(coord, len)),
            ImfilterPadding::Symmetric => final_indices.push(reflect_index(coord, len)),
        }
    }

    let linear: usize = final_indices
        .iter()
        .zip(image_strides.iter())
        .map(|(&coord, &stride)| coord * stride)
        .sum();
    image.get(linear).copied().unwrap_or(options.constant_value)
}

fn evaluate_filter(
    output_shape: &[usize],
    base_offset: &[isize],
    image_shape: &[usize],
    image_strides: &[usize],
    kernel_points: &[ImfilterKernelPoint],
    options: &ImfilterOptions,
    image_data: &[f64],
    accumulation: FilterAccumulation,
) -> Vec<f64> {
    let total = tensor::element_count(output_shape);
    let mut out = vec![0.0; total];
    if total == 0 || kernel_points.is_empty() {
        return out;
    }

    let mut out_index = vec![0usize; output_shape.len()];
    for out_value in out.iter_mut() {
        *out_value = match accumulation {
            FilterAccumulation::HostDouble => {
                let mut sum = 0.0;
                for point in kernel_points {
                    let value = sample_with_padding(
                        image_data,
                        image_shape,
                        image_strides,
                        &out_index,
                        base_offset,
                        &point.offsets,
                        options,
                    );
                    sum += point.value * value;
                }
                sum
            }
            FilterAccumulation::GpuSingle => {
                let mut sum = 0.0_f32;
                for point in kernel_points {
                    let value = sample_with_padding(
                        image_data,
                        image_shape,
                        image_strides,
                        &out_index,
                        base_offset,
                        &point.offsets,
                        options,
                    ) as f32;
                    sum += (point.value as f32) * value;
                }
                f64::from(sum)
            }
        };
        advance_index(&mut out_index, output_shape);
    }

    out
}

fn clamp_index(coord: isize, len: isize) -> usize {
    if len <= 0 || coord <= 0 {
        0
    } else if coord >= len {
        (len - 1) as usize
    } else {
        coord as usize
    }
}

fn wrap_index(mut coord: isize, len: isize) -> usize {
    if len <= 0 {
        return 0;
    }
    coord %= len;
    if coord < 0 {
        coord += len;
    }
    coord as usize
}

pub(super) fn reflect_index(coord: isize, len: isize) -> usize {
    if len <= 0 {
        return 0;
    }
    if len == 1 {
        return 0;
    }
    // MATLAB's symmetric boundary repeats the border sample. For [a b c],
    // the conceptual extension is ... c b a a b c c b a ... .
    let period = 2 * len;
    let mut value = coord % period;
    if value < 0 {
        value += period;
    }
    if value >= len {
        value = period - 1 - value;
    }
    value as usize
}
