//! MATLAB-compatible `fspecial` builtin for generating 2-D image filters.

use std::env;

use log::warn;
use runmat_accelerate_api::{self, FspecialFilter, FspecialRequest};
#[cfg(test)]
use runmat_builtins::FSPECIAL_INTEGER_CAPABILITIES;
use runmat_builtins::{
    BuiltinErrorDescriptor, FspecialFilterKind as FilterKind, FSPECIAL_ERROR_INTERNAL,
    FSPECIAL_ERROR_INVALID_ARGUMENT, FSPECIAL_ERROR_INVALID_INPUT,
    FSPECIAL_NONDOUBLE_PARAMETER_EXTENSION, FSPECIAL_NONDOUBLE_SIZE_EXTENSION,
    FSPECIAL_RESIDENT_OUTPUT_EXTENSION, FSPECIAL_UNSHARP_EXTENSION,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::IntValue;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "fspecial";

mod generation;
use generation::*;
mod parsing;
use parsing::*;
mod placement;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::filters::fspecial")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fspecial",
    op_kind: GpuOpKind::Custom("kernel-generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("fspecial")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Average, gaussian, laplacian, prewitt, sobel, and unsharp execute on the device when supported; disk/log/motion currently gather to host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::filters::fspecial")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fspecial",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Generates constant kernels; fusion is not applicable.",
};

#[path = "fspecial/kernel_error.rs"]
pub mod kernel_error;
use kernel_error::{FspecialKernelError, FspecialKernelResult};

fn fspecial_descriptor_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let message = kernel_error::message(error, detail.as_ref());
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fspecial_argument_error(detail: impl AsRef<str>) -> RuntimeError {
    fspecial_descriptor_error(&FSPECIAL_ERROR_INVALID_ARGUMENT, detail)
}

fn fspecial_error(detail: impl AsRef<str>) -> RuntimeError {
    fspecial_descriptor_error(&FSPECIAL_ERROR_INVALID_INPUT, detail)
}

/// Shared specification of an `fspecial` kernel used by both the runtime and acceleration providers.
#[derive(Debug, Clone)]
pub enum FspecialFilterSpec {
    Average {
        rows: usize,
        cols: usize,
    },
    Disk {
        radius: f64,
        size: usize,
    },
    Gaussian {
        rows: usize,
        cols: usize,
        sigma: f64,
    },
    Laplacian {
        alpha: f64,
    },
    Log {
        rows: usize,
        cols: usize,
        sigma: f64,
    },
    Motion {
        length: usize,
        kernel_size: usize,
        angle_degrees: f64,
        oversample: usize,
    },
    Prewitt,
    Sobel,
    Unsharp {
        alpha: f64,
    },
}

impl FspecialFilterSpec {
    pub fn generate_tensor(&self) -> FspecialKernelResult<Tensor> {
        match self {
            FspecialFilterSpec::Average { rows, cols } => generate_average(*rows, *cols),
            FspecialFilterSpec::Disk { radius, size } => generate_disk(*radius, *size),
            FspecialFilterSpec::Gaussian { rows, cols, sigma } => {
                generate_gaussian(*rows, *cols, *sigma)
            }
            FspecialFilterSpec::Laplacian { alpha } => generate_laplacian(*alpha),
            FspecialFilterSpec::Log { rows, cols, sigma } => generate_log(*rows, *cols, *sigma),
            FspecialFilterSpec::Motion {
                length,
                kernel_size,
                angle_degrees,
                oversample,
            } => generate_motion(*length, *kernel_size, *angle_degrees, *oversample),
            FspecialFilterSpec::Prewitt => generate_prewitt(),
            FspecialFilterSpec::Sobel => generate_sobel(),
            FspecialFilterSpec::Unsharp { alpha } => generate_unsharp(*alpha),
        }
    }

    pub fn to_request(&self) -> BuiltinResult<FspecialRequest> {
        use std::convert::TryFrom;
        let filter = match self {
            FspecialFilterSpec::Average { rows, cols } => FspecialFilter::Average {
                rows: u32::try_from(*rows)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
                cols: u32::try_from(*cols)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
            },
            FspecialFilterSpec::Disk { radius, size } => FspecialFilter::Disk {
                radius: *radius,
                size: u32::try_from(*size)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
            },
            FspecialFilterSpec::Gaussian { rows, cols, sigma } => FspecialFilter::Gaussian {
                rows: u32::try_from(*rows)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
                cols: u32::try_from(*cols)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
                sigma: *sigma,
            },
            FspecialFilterSpec::Laplacian { alpha } => FspecialFilter::Laplacian { alpha: *alpha },
            FspecialFilterSpec::Log { rows, cols, sigma } => FspecialFilter::Log {
                rows: u32::try_from(*rows)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
                cols: u32::try_from(*cols)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
                sigma: *sigma,
            },
            FspecialFilterSpec::Motion {
                length,
                kernel_size,
                angle_degrees,
                oversample,
            } => FspecialFilter::Motion {
                length: u32::try_from(*length)
                    .map_err(|_| fspecial_error("fspecial: LENGTH exceeds GPU limits"))?,
                kernel_size: u32::try_from(*kernel_size)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
                angle_degrees: *angle_degrees,
                oversample: u32::try_from(*oversample)
                    .map_err(|_| fspecial_error("fspecial: oversample exceeds GPU limits"))?,
            },
            FspecialFilterSpec::Prewitt => FspecialFilter::Prewitt,
            FspecialFilterSpec::Sobel => FspecialFilter::Sobel,
            FspecialFilterSpec::Unsharp { alpha } => FspecialFilter::Unsharp { alpha: *alpha },
        };
        Ok(FspecialRequest { filter })
    }

    fn is_gpu_supported(&self) -> bool {
        matches!(
            self,
            FspecialFilterSpec::Average { .. }
                | FspecialFilterSpec::Gaussian { .. }
                | FspecialFilterSpec::Laplacian { .. }
                | FspecialFilterSpec::Prewitt
                | FspecialFilterSpec::Sobel
                | FspecialFilterSpec::Unsharp { .. }
        )
    }
}

/// Convert an API request into a runtime specification.
#[allow(dead_code)]
pub fn spec_from_request(filter: &FspecialFilter) -> FspecialFilterSpec {
    match filter {
        FspecialFilter::Average { rows, cols } => FspecialFilterSpec::Average {
            rows: *rows as usize,
            cols: *cols as usize,
        },
        FspecialFilter::Disk { radius, size } => FspecialFilterSpec::Disk {
            radius: *radius,
            size: *size as usize,
        },
        FspecialFilter::Gaussian { rows, cols, sigma } => FspecialFilterSpec::Gaussian {
            rows: *rows as usize,
            cols: *cols as usize,
            sigma: *sigma,
        },
        FspecialFilter::Laplacian { alpha } => FspecialFilterSpec::Laplacian { alpha: *alpha },
        FspecialFilter::Log { rows, cols, sigma } => FspecialFilterSpec::Log {
            rows: *rows as usize,
            cols: *cols as usize,
            sigma: *sigma,
        },
        FspecialFilter::Motion {
            length,
            kernel_size,
            angle_degrees,
            oversample,
        } => FspecialFilterSpec::Motion {
            length: *length as usize,
            kernel_size: *kernel_size as usize,
            angle_degrees: *angle_degrees,
            oversample: *oversample as usize,
        },
        FspecialFilter::Prewitt => FspecialFilterSpec::Prewitt,
        FspecialFilter::Sobel => FspecialFilterSpec::Sobel,
        FspecialFilter::Unsharp { alpha } => FspecialFilterSpec::Unsharp { alpha: *alpha },
    }
}

#[runtime_builtin(
    name = "fspecial",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::image::filters::fspecial"
)]
async fn fspecial_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    kind: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    ensure_fspecial_extensions_enabled(&kind, rest.view())?;
    if should_materialize_on_gpu() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FSPECIAL_RESIDENT_OUTPUT_EXTENSION,
            NAME,
        )?;
    }
    let spec = build_filter_spec(&kind, rest.view())?;
    let materialize_on_gpu = should_materialize_on_gpu();
    let mut selected = placement::select(runtime.as_ref(), &spec, materialize_on_gpu)?;
    let tensor = spec.generate_tensor()?;
    let recorder = selected.as_ref().map(|selection| selection.recorder());
    let provider = selected.as_ref().map(|selection| selection.provider());
    let publication =
        finalize_output_with_observation(&spec, tensor, materialize_on_gpu, provider, recorder);
    let publication_is_unsupported_fallback = publication
        .as_ref()
        .is_ok_and(|published| published.unsupported_fallback);
    let result = publication.map(|published| published.value);
    let result = match result {
        Ok(Value::GpuTensor(output))
            if selected
                .as_mut()
                .is_some_and(|selection| !selection.validate_direct_provider_output(&output)) =>
        {
            let error = fspecial_descriptor_error(
                &FSPECIAL_ERROR_INTERNAL,
                "provider result exceeded its admitted physical publication",
            );
            Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                output, error,
            ))
        }
        other => other,
    };
    if let Some(selection) = selected {
        match &result {
            Ok(Value::GpuTensor(_)) => selection.finish(Ok(true)),
            Ok(Value::Tensor(_)) if publication_is_unsupported_fallback => {
                selection.finish_conditional_host_fallback()
            }
            Ok(_) => selection.finish(Ok(false)),
            Err(error) if error.identifier() == Some("RunMat:Cancelled") => selection.finish(Err(
                runmat_accelerate_api::observation::ProviderObservationFailure::Cancelled,
            )),
            Err(_) => selection.finish(Err(
                runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
            )),
        }
    }
    result
}

fn ensure_fspecial_extensions_enabled(
    kind: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let filter_kind = parse_filter_kind(kind)?;
    if matches!(filter_kind, FilterKind::Unsharp) {
        crate::compatibility::ensure_builtin_extension_enabled(&FSPECIAL_UNSHARP_EXTENSION, NAME)?;
    }
    let (size_indices, parameter_indices): (&[usize], &[usize]) = match filter_kind {
        FilterKind::Average => (&[0], &[]),
        FilterKind::Disk | FilterKind::Laplacian | FilterKind::Unsharp => (&[], &[0]),
        FilterKind::Gaussian | FilterKind::Log => (&[0], &[1]),
        FilterKind::Motion => (&[0], &[1]),
        FilterKind::Prewitt | FilterKind::Sobel => (&[], &[]),
    };
    for &index in size_indices {
        if rest.get(index).is_some_and(is_nondouble_numeric_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FSPECIAL_NONDOUBLE_SIZE_EXTENSION,
                NAME,
            )?;
        }
    }
    for &index in parameter_indices {
        if rest.get(index).is_some_and(is_nondouble_numeric_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FSPECIAL_NONDOUBLE_PARAMETER_EXTENSION,
                NAME,
            )?;
        }
    }
    Ok(())
}

fn is_nondouble_numeric_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_)
    ) || matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() != NumericDType::F64)
}

fn build_filter_spec(
    kind: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<FspecialFilterSpec> {
    let filter_kind = parse_filter_kind(kind)?;
    match filter_kind {
        FilterKind::Average => {
            ensure_arg_count("average", rest, 0, 1)?;
            let (rows, cols) = parse_average_dims(rest.first())?;
            Ok(FspecialFilterSpec::Average { rows, cols })
        }
        FilterKind::Disk => {
            ensure_arg_count("disk", rest, 0, 1)?;
            let (radius, size) = parse_disk_params(rest.first())?;
            Ok(FspecialFilterSpec::Disk { radius, size })
        }
        FilterKind::Gaussian => {
            ensure_arg_count("gaussian", rest, 0, 2)?;
            let (rows, cols, sigma) = parse_gaussian_params(rest.first(), rest.get(1))?;
            Ok(FspecialFilterSpec::Gaussian { rows, cols, sigma })
        }
        FilterKind::Laplacian => {
            ensure_arg_count("laplacian", rest, 0, 1)?;
            let alpha = parse_laplacian_alpha(rest.first())?;
            Ok(FspecialFilterSpec::Laplacian { alpha })
        }
        FilterKind::Log => {
            ensure_arg_count("log", rest, 0, 2)?;
            let (rows, cols, sigma) = parse_log_params(rest.first(), rest.get(1))?;
            Ok(FspecialFilterSpec::Log { rows, cols, sigma })
        }
        FilterKind::Motion => {
            ensure_arg_count("motion", rest, 0, 2)?;
            let (length, kernel_size, angle, oversample) =
                parse_motion_params(rest.first(), rest.get(1))?;
            Ok(FspecialFilterSpec::Motion {
                length,
                kernel_size,
                angle_degrees: angle,
                oversample,
            })
        }
        FilterKind::Prewitt => {
            ensure_arg_count("prewitt", rest, 0, 0)?;
            Ok(FspecialFilterSpec::Prewitt)
        }
        FilterKind::Sobel => {
            ensure_arg_count("sobel", rest, 0, 0)?;
            Ok(FspecialFilterSpec::Sobel)
        }
        FilterKind::Unsharp => {
            ensure_arg_count("unsharp", rest, 0, 1)?;
            let alpha = parse_unsharp_alpha(rest.first())?;
            Ok(FspecialFilterSpec::Unsharp { alpha })
        }
    }
}

#[cfg(test)]
fn finalize_output_with_gpu_policy(
    spec: &FspecialFilterSpec,
    tensor: Tensor,
    materialize_on_gpu: bool,
) -> BuiltinResult<Value> {
    finalize_output_with_observation(spec, tensor, materialize_on_gpu, None, None)
        .map(|published| published.value)
}

struct FspecialPublication {
    value: Value,
    unsupported_fallback: bool,
}

fn finalize_output_with_observation(
    spec: &FspecialFilterSpec,
    tensor: Tensor,
    materialize_on_gpu: bool,
    admitted_provider: Option<&'static dyn runmat_accelerate_api::AccelProvider>,
    recorder: Option<runmat_accelerate_api::observation::ProviderOperationRecorder>,
) -> BuiltinResult<FspecialPublication> {
    if !materialize_on_gpu || !spec.is_gpu_supported() {
        return Ok(FspecialPublication {
            value: Value::Tensor(tensor),
            unsupported_fallback: false,
        });
    }

    let mut unsupported_fallback = false;
    if let Some(provider) = admitted_provider.or_else(runmat_accelerate_api::provider) {
        match spec.to_request() {
            Ok(request) => match runmat_accelerate_api::observation::with_provider_observation(
                recorder,
                || provider.fspecial(&request),
            ) {
                Ok(handle) => {
                    let expected_precision = provider.precision();
                    let valid = handle.shape() == tensor.shape()
                        && handle.device_id() == provider.device_id()
                        && handle.allocation_lease().validate_live().is_ok()
                        && runmat_accelerate_api::provider_for_handle(&handle).is_some_and(
                            |owner| runmat_accelerate_api::same_provider_instance(owner, provider),
                        )
                        && runmat_accelerate_api::handle_storage(&handle)
                            == runmat_accelerate_api::GpuTensorStorage::Real
                        && runmat_accelerate_api::handle_precision(&handle)
                            == Some(expected_precision)
                        && runmat_accelerate_api::handle_integer_type(&handle).is_none()
                        && !runmat_accelerate_api::handle_is_logical(&handle)
                        && crate::builtins::common::gpu_helpers::gpu_class_metadata_matches(
                            &handle,
                            Some(expected_precision),
                            None,
                            false,
                        );
                    if !valid {
                        let error = fspecial_descriptor_error(
                            &FSPECIAL_ERROR_INTERNAL,
                            "provider returned an invalid fspecial result",
                        );
                        return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                            handle, error,
                        ));
                    }
                    return Ok(FspecialPublication {
                        value: Value::GpuTensor(handle),
                        unsupported_fallback: false,
                    });
                }
                Err(err)
                    if crate::builtins::common::gpu_helpers::provider_hook_allows_host_fallback(
                        &err,
                    ) =>
                {
                    warn!("fspecial: provider hook unavailable, falling back to host path: {err}");
                    unsupported_fallback = true;
                }
                Err(err) => {
                    return Err(crate::runtime_error::with_provider_execution_source(
                        fspecial_descriptor_error(
                            &FSPECIAL_ERROR_INTERNAL,
                            format!("provider generation failed: {err}"),
                        ),
                        err.into(),
                    ))
                }
            },
            Err(error) => {
                warn!(
                    "fspecial: provider hook unavailable, falling back to host path: {}",
                    error.message()
                );
            }
        }
    }

    Ok(FspecialPublication {
        value: Value::Tensor(tensor),
        unsupported_fallback,
    })
}

fn should_materialize_on_gpu() -> bool {
    match env::var("RUNMAT_ACCEL_FSPECIAL_DEVICE") {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

#[cfg(test)]
#[path = "fspecial/tests.rs"]
pub(crate) mod tests;
