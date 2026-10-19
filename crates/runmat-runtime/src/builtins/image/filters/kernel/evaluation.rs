//! Indexed kernel-point construction and padded host evaluation.

use super::{tensor, FilterAccumulation, ImfilterKernelPoint};
use runmat_accelerate_api::{ImfilterMode, ImfilterOptions, ImfilterPadding};
use runmat_value::Tensor;

pub(super) fn compute_strides(shape: &[usize]) -> Vec<usize> {
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

pub(super) fn build_kernel_points(
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

pub(super) fn evaluate_filter(
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

pub(in crate::builtins::image::filters) fn reflect_index(coord: isize, len: isize) -> usize {
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
