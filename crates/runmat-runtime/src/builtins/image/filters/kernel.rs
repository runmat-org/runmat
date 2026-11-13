use runmat_accelerate_api::{ImfilterOptions, ImfilterShape, ProviderPrecision};
use runmat_value::{NumericDType, Tensor};

use crate::builtins::common::tensor;

pub use super::geometry::ImfilterKernelPoint;
use super::geometry::{extend_shape, normalize_shape, validate_kernel_shape};
mod error;
pub use error::{ImfilterKernelError, ImfilterKernelResult};
mod evaluation;
#[cfg(test)]
pub(super) use evaluation::reflect_index;
use evaluation::{build_kernel_points, compute_strides, evaluate_filter};

/// Pure host filter plan shared by filter2, imfilter and provider adapters.

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
    if explicit && precision == ProviderPrecision::F32 {
        FilterAccumulation::GpuSingle
    } else {
        FilterAccumulation::HostDouble
    }
}

impl ImfilterPlan {
    #[inline]
    pub fn evaluate(&self, image_data: &[f64], options: &ImfilterOptions) -> Vec<f64> {
        self.evaluate_with_accumulation(image_data, options, FilterAccumulation::HostDouble)
    }

    pub(in crate::builtins::image::filters) fn evaluate_with_accumulation(
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
    if kernel.is_empty() || kernel.shape().contains(&0) {
        return Err(ImfilterKernelError::new(
            builtin,
            format!("{builtin}: filter must be non-empty along every dimension"),
        ));
    }

    let final_shape = filter_output_shape(image_shape, kernel.shape(), options.shape, builtin)?;
    let image_shape_norm = normalize_shape(image_shape);
    let kernel_shape_norm = normalize_shape(kernel.shape());
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
    super::geometry::filter_output_shape(image_shape, kernel_shape, shape_mode, builtin)
        .map_err(|message| ImfilterKernelError::new(builtin, message))
}

#[cfg(test)]
mod output_shape_tests;

pub fn apply_imfilter_tensor(
    image: &Tensor,
    kernel: &Tensor,
    options: &ImfilterOptions,
    builtin: &str,
) -> ImfilterKernelResult<Tensor> {
    let plan = build_imfilter_plan(image.shape(), kernel, options, builtin)?;
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
    let plan = build_imfilter_plan(image.shape(), kernel, options, builtin)?;
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
