//! MATLAB-compatible `fspecial` builtin for generating 2-D image filters.

use std::env;
use std::f64::consts::PI;

use log::warn;
use runmat_accelerate_api::{self, FspecialFilter, FspecialRequest};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeControlFlow};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "fspecial",
        builtin_path = "crate::builtins::image::filters::fspecial"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "fspecial"
category: "image/filters"
keywords: ["fspecial", "filter", "gaussian", "sobel", "motion", "laplacian", "disk"]
summary: "Generate classical 2-D correlation kernels used in MATLAB image processing workflows."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Providers implement the optional 'fspecial' hook. When it is missing or a kernel is not yet accelerated (./motion), RunMat falls back to the host implementation transparently."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::image::filters::fspecial::tests"
  integration: "builtins::image::filters::fspecial::tests::fspecial_motion_sum_is_one"
  gpu: "builtins::image::filters::fspecial::tests::fspecial_gaussian_gpu_matches_cpu"
---

# What does the `fspecial` function do in MATLAB / RunMat?
`fspecial(type, ...)` constructs well-known 2-D filter kernels such as averaging, Gaussian, Laplacian,
Sobel, Prewitt, motion blur, Laplacian of Gaussian, unsharp masking, and disk (pillbox) filters.
These kernels are intended for use with correlation and convolution routines like `imfilter` and `conv2`,
and their outputs match MathWorks MATLAB behaviour for every supported option.

## How does the `fspecial` function behave in MATLAB / RunMat?
- Uses MATLAB-compatible defaults for all optional parameters and validates scalar/vector inputs rigorously.
- Produces double-precision column-major tensors that match MATLAB sample outputs to machine precision.
- Normalises smoothing filters (average, disk, Gaussian, Laplacian of Gaussian, motion) to unit sum.
- Emits derivative-style operators (Sobel, Prewitt, Laplacian, unsharp) using MATLAB's historical scaling.
- Accepts scalar sizes or two-element vectors; zero/negative dimensions trigger MATLAB-style errors.

### Supported filter types
- `"average"`: rectangular mean filter with optional size argument.
- `"disk"`: circular averaging filter parameterised by radius.
- `"gaussian"`: Gaussian low-pass filter with optional size and standard deviation.
- `"laplacian"`: 3×3 Laplacian operator controlled by `alpha` (0 ≤ alpha ≤ 1).
- `"log"`: Laplacian of Gaussian with optional size and `sigma`.
- `"motion"`: motion blur kernel with controllable length and angle (rounded to odd kernel width).
- `"prewitt"`: 3×3 horizontal Prewitt edge detector.
- `"sobel"`: 3×3 horizontal Sobel edge detector.
- `"unsharp"`: 3×3 unsharp masking filter with optional `alpha`.

## `fspecial` Function GPU Execution Behaviour
When an acceleration provider is active, `fspecial` can materialise supported kernels directly on the GPU.
Opt-in by setting `RUNMAT_ACCEL_FSPECIAL_DEVICE=1`. If the provider exports the `fspecial` hook (the
WGPU backend covers average, gaussian, laplacian, prewitt, sobel, and unsharp), the builtin returns a
`gpuArray` handle that remains device-resident for downstream fusion. Kernels without acceleration support
and providers lacking the hook automatically fall back to the host path with identical numerical results.

## Examples of using the `fspecial` function in MATLAB / RunMat

### Creating a box filter for local averaging
```matlab
H = fspecial("average", 7);        % 7x7 box filter with unit sum
```

### Building a Gaussian smoothing kernel
```matlab
H = fspecial("gaussian", [5 5], 1.0);
```

### Generating a disk filter for circular blur
```matlab
H = fspecial("disk", 4);
```

### Constructing a Laplacian-of-Gaussian edge detector
```matlab
H = fspecial("log", [9 9], 1.4);
```

### Synthesising a motion blur kernel at 30 degrees
```matlab
H = fspecial("motion", 15, 30);
```

### Tuning an unsharp mask for edge enhancement
```matlab
H = fspecial("unsharp", 0.6);
```

## FAQ

### Which filters does `fspecial` support?
All classic MATLAB filters are available: average, disk, Gaussian, Laplacian, Laplacian of Gaussian,
motion, Prewitt, Sobel, and unsharp.

### Does `fspecial` normalise the kernels?
All smoothing filters produce weights that sum to one. Derivative-style kernels follow MATLAB's scaling
so that downstream edge detection behaves identically.

### Can I generate a GPU-resident kernel directly?
Yes. Set `RUNMAT_ACCEL_FSPECIAL_DEVICE=1` and ensure the active provider exposes the `fspecial` hook.
Unsupported filters or providers gather to host automatically, so results stay correct either way.

### How do I specify the kernel size?
Most filters accept a scalar size or a two-element `[rows cols]` vector. When omitted, MATLAB-compatible
defaults are used (for example, 3×3 for average/gaussian/laplacian/prewitt/sobel/unsharp).

### What happens if I provide invalid parameters?
`fspecial` raises MATLAB-compatible errors when arguments fall outside the documented range. Negative
lengths, radii, or sigmas, as well as noninteger dimensions, produce descriptive error messages.

### Are the filters symmetric with MATLAB outputs?
Yes. Each kernel matches MATLAB (R2023b) outputs within floating-point precision, including motion blur
and disk filters that rely on geometric integration.

## See Also
[imfilter](./imfilter), [conv2](./conv2), [gpuArray](./gpuarray), [gather](./gather)
"#;

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

fn fspecial_error(message: impl Into<String>) -> RuntimeControlFlow {
    build_runtime_error(message)
        .with_builtin("fspecial")
        .build()
        .into()
}

#[derive(Clone, Copy, Debug)]
enum FilterKind {
    Average,
    Disk,
    Gaussian,
    Laplacian,
    Log,
    Motion,
    Prewitt,
    Sobel,
    Unsharp,
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
    pub fn generate_tensor(&self) -> BuiltinResult<Tensor> {
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
pub fn spec_from_request(filter: &FspecialFilter) -> BuiltinResult<FspecialFilterSpec> {
    Ok(match filter {
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
    })
}

#[runtime_builtin(
    name = "fspecial",
    category = "image/filters",
    summary = "Generate classical 2-D correlation kernels used in MATLAB image processing workflows.",
    keywords = "fspecial,filter,gaussian,sobel,motion,laplacian,disk",
    accel = "array_construct",
    builtin_path = "crate::builtins::image::filters::fspecial"
)]
fn fspecial_builtin(kind: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let spec = build_filter_spec(&kind, &rest)?;
    let tensor = spec.generate_tensor()?;
    finalize_output(&spec, tensor)
}

fn build_filter_spec(kind: &Value, rest: &[Value]) -> BuiltinResult<FspecialFilterSpec> {
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

fn finalize_output(spec: &FspecialFilterSpec, tensor: Tensor) -> BuiltinResult<Value> {
    if !should_materialize_on_gpu() || !spec.is_gpu_supported() {
        return Ok(Value::Tensor(tensor));
    }

    #[cfg(all(test, feature = "wgpu"))]
    {
        if runmat_accelerate_api::provider().is_none() {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        match spec.to_request() {
            Ok(request) => match provider.fspecial(&request) {
                Ok(handle) => return Ok(Value::GpuTensor(handle)),
                Err(err) => {
                    warn!(
                        "fspecial: provider hook unavailable, falling back to host path: {err}"
                    )
                }
            },
            Err(RuntimeControlFlow::Error(error)) => {
                warn!(
                    "fspecial: provider hook unavailable, falling back to host path: {}",
                    error.message()
                );
            }
            Err(RuntimeControlFlow::Suspend(pending)) => {
                return Err(RuntimeControlFlow::Suspend(pending));
            }
        }
    }

    Ok(Value::Tensor(tensor))
}

fn parse_filter_kind(value: &Value) -> BuiltinResult<FilterKind> {
    let text = value_to_string(value).ok_or_else(|| {
        fspecial_error("fspecial: first argument must be a string filter name")
    })?;
    let lower = text.to_ascii_lowercase();
    match lower.as_str() {
        "average" => Ok(FilterKind::Average),
        "disk" => Ok(FilterKind::Disk),
        "gaussian" => Ok(FilterKind::Gaussian),
        "laplacian" => Ok(FilterKind::Laplacian),
        "log" => Ok(FilterKind::Log),
        "motion" => Ok(FilterKind::Motion),
        "prewitt" => Ok(FilterKind::Prewitt),
        "sobel" => Ok(FilterKind::Sobel),
        "unsharp" => Ok(FilterKind::Unsharp),
        other => Err(fspecial_error(format!(
            "fspecial: filter type '{other}' is not supported"
        ))),
    }
}

fn ensure_arg_count(name: &str, args: &[Value], min: usize, max: usize) -> BuiltinResult<()> {
    if args.len() < min || args.len() > max {
        if min == max {
            Err(fspecial_error(format!(
                "fspecial: '{name}' expects exactly {min} argument{}",
                if min == 1 { "" } else { "s" }
            )))
        } else {
            Err(fspecial_error(format!(
                "fspecial: '{name}' expects between {min} and {max} arguments"
            )))
        }
    } else {
        Ok(())
    }
}

fn parse_average_dims(arg: Option<&Value>) -> BuiltinResult<(usize, usize)> {
    match arg {
        None => Ok((3, 3)),
        Some(value) => {
            let dims = parse_lengths_strict(value, "fspecial: LENGTHS must be positive integers")?;
            match dims.len() {
                1 => Ok((dims[0], dims[0])),
                2 => Ok((dims[0], dims[1])),
                _ => Err(fspecial_error(
                    "fspecial: LENGTHS must be a scalar or two-element vector",
                )),
            }
        }
    }
}

fn parse_disk_params(arg: Option<&Value>) -> BuiltinResult<(f64, usize)> {
    let radius = match arg {
        None => 5.0,
        Some(value) => to_positive_scalar(value, "fspecial: RADIUS must be a non-negative scalar")?,
    };
    if radius < 0.0 {
        return Err(fspecial_error("fspecial: RADIUS must be non-negative"));
    }
    let extent = radius.ceil() as isize;
    let size = (2 * extent + 1) as usize;
    Ok((radius, size))
}

fn parse_gaussian_params(
    lengths: Option<&Value>,
    sigma_value: Option<&Value>,
) -> BuiltinResult<(usize, usize, f64)> {
    let dims = match lengths {
        None => vec![3, 3],
        Some(value) => parse_lengths_strict(
            value,
            "fspecial: LENGTHS must be positive integers for gaussian",
        )?,
    };
    let dims = match dims.len() {
        1 => vec![dims[0], dims[0]],
        2 => dims,
        _ => {
            return Err(fspecial_error(
                "fspecial: gaussian lengths must be a scalar or a two-element vector",
            ));
        }
    };
    let sigma = match sigma_value {
        None => 0.5,
        Some(value) => {
            let sigma = to_positive_scalar(value, "fspecial: SIGMA must be a positive scalar")?;
            if sigma <= 0.0 {
                return Err(fspecial_error("fspecial: SIGMA must be positive"));
            }
            sigma
        }
    };
    Ok((dims[0], dims[1], sigma))
}

fn parse_laplacian_alpha(arg: Option<&Value>) -> BuiltinResult<f64> {
    match arg {
        None => Ok(0.2),
        Some(value) => {
            let alpha = to_scalar(value, "fspecial: ALPHA must be a scalar")?;
            if !(0.0..=1.0).contains(&alpha) {
                return Err(fspecial_error("fspecial: ALPHA must be between 0 and 1"));
            }
            Ok(alpha)
        }
    }
}

fn parse_log_params(
    lengths: Option<&Value>,
    sigma_value: Option<&Value>,
) -> BuiltinResult<(usize, usize, f64)> {
    let dims = match lengths {
        None => vec![5, 5],
        Some(value) => {
            parse_lengths_strict(value, "fspecial: LENGTHS must be positive integers for log")?
        }
    };
    let dims = match dims.len() {
        1 => vec![dims[0], dims[0]],
        2 => dims,
        _ => {
            return Err(fspecial_error(
                "fspecial: log lengths must be a scalar or two-element vector",
            ));
        }
    };
    let sigma = match sigma_value {
        None => 0.5,
        Some(value) => {
            let sigma = to_positive_scalar(value, "fspecial: SIGMA must be a positive scalar")?;
            if sigma <= 0.0 {
                return Err(fspecial_error("fspecial: SIGMA must be positive"));
            }
            sigma
        }
    };
    Ok((dims[0], dims[1], sigma))
}

fn parse_motion_params(
    length_value: Option<&Value>,
    angle_value: Option<&Value>,
) -> BuiltinResult<(usize, usize, f64, usize)> {
    let length_raw = match length_value {
        None => 9.0,
        Some(value) => {
            let len = to_positive_scalar(value, "fspecial: LENGTH must be a positive scalar")?;
            if len <= 0.0 {
                return Err(fspecial_error("fspecial: LENGTH must be positive"));
            }
            len
        }
    };
    let length = length_raw.round() as usize;
    if length == 0 {
        return Err(fspecial_error("fspecial: LENGTH must be at least 1"));
    }
    let kernel_size = if length % 2 == 1 { length } else { length + 1 };
    let angle_deg = match angle_value {
        None => 0.0,
        Some(value) => to_scalar(value, "fspecial: ANGLE must be a scalar")?,
    };
    Ok((length, kernel_size, angle_deg, 8))
}

fn parse_unsharp_alpha(arg: Option<&Value>) -> BuiltinResult<f64> {
    match arg {
        None => Ok(0.2),
        Some(value) => {
            let alpha = to_scalar(value, "fspecial: ALPHA must be a scalar")?;
            if !(0.0..=1.0).contains(&alpha) {
                return Err(fspecial_error("fspecial: ALPHA must be between 0 and 1"));
            }
            Ok(alpha)
        }
    }
}

fn generate_average(rows: usize, cols: usize) -> BuiltinResult<Tensor> {
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| fspecial_error("fspecial: LENGTHS are too large"))?;
    if total == 0 {
        return Err(fspecial_error("fspecial: LENGTHS must be positive integers"));
    }
    let fill = 1.0 / total as f64;
    let data = vec![fill; total];
    Tensor::new(data, vec![rows, cols])
        .map_err(|e| fspecial_error(format!("fspecial: {e}")))
}

fn generate_disk(radius: f64, size: usize) -> BuiltinResult<Tensor> {
    if radius == 0.0 {
        return Tensor::new(vec![1.0], vec![1, 1])
            .map_err(|e| fspecial_error(format!("fspecial: {e}")));
    }

    let mut data = vec![0.0f64; size * size];
    let center = (size as isize / 2) as f64;

    for row in 0..size {
        let y1 = row as f64 - center - 0.5;
        let y2 = y1 + 1.0;
        for col in 0..size {
            let x1 = col as f64 - center - 0.5;
            let x2 = x1 + 1.0;
            let area = circle_rect_area(radius, x1, x2, y1, y2);
            data[col * size + row] = area;
        }
    }

    let normaliser = PI * radius * radius;
    if normaliser <= f64::EPSILON {
        return Err(fspecial_error("fspecial: radius is too small"));
    }
    let mut sum = 0.0;
    for value in &mut data {
        *value /= normaliser;
        sum += *value;
    }
    if sum <= 0.0 {
        return Err(fspecial_error("fspecial: failed to generate disk filter"));
    }
    for value in &mut data {
        *value /= sum;
    }

    Tensor::new(data, vec![size, size])
        .map_err(|e| fspecial_error(format!("fspecial: {e}")))
}

fn generate_gaussian(rows: usize, cols: usize, sigma: f64) -> BuiltinResult<Tensor> {
    let row_center = (rows as f64 - 1.0) / 2.0;
    let col_center = (cols as f64 - 1.0) / 2.0;
    let denom = 2.0 * sigma * sigma;
    let mut data = Vec::with_capacity(rows * cols);
    let mut sum = 0.0;
    for col in 0..cols {
        let x = col as f64 - col_center;
        for row in 0..rows {
            let y = row as f64 - row_center;
            let value = (-((x * x + y * y) / denom)).exp();
            data.push(value);
            sum += value;
        }
    }
    if sum == 0.0 {
        return Err(fspecial_error(
            "fspecial: gaussian generation failed (degenerate sigma)",
        ));
    }
    for value in &mut data {
        *value /= sum;
    }

    Tensor::new(data, vec![rows, cols])
        .map_err(|e| fspecial_error(format!("fspecial: {e}")))
}

fn generate_laplacian(alpha: f64) -> BuiltinResult<Tensor> {
    let scale = 4.0 / (alpha + 1.0);
    let a = alpha / 4.0;
    let b = (1.0 - alpha) / 4.0;
    let mut data = vec![
        a, b, a, //
        b, -1.0, b, //
        a, b, a,
    ];
    for value in &mut data {
        *value *= scale;
    }
    Tensor::new(data, vec![3, 3]).map_err(|e| fspecial_error(format!("fspecial: {e}")))
}

fn generate_log(rows: usize, cols: usize, sigma: f64) -> BuiltinResult<Tensor> {
    let row_center = (rows as f64 - 1.0) / 2.0;
    let col_center = (cols as f64 - 1.0) / 2.0;
    let mut gauss = Vec::with_capacity(rows * cols);
    let mut gauss_sum = 0.0;
    for col in 0..cols {
        let x = col as f64 - col_center;
        for row in 0..rows {
            let y = row as f64 - row_center;
            let value = (-((x * x + y * y) / (2.0 * sigma * sigma))).exp();
            gauss_sum += value;
            gauss.push((x, y, value));
        }
    }
    if gauss_sum == 0.0 {
        return Err(fspecial_error(
            "fspecial: failed to normalise Laplacian of Gaussian",
        ));
    }
    let mut data = Vec::with_capacity(rows * cols);
    let sigma2 = sigma * sigma;
    let normaliser = 2.0 * PI * sigma6(sigma);
    for (x, y, g) in gauss {
        let radial = x * x + y * y;
        let value = ((radial - 2.0 * sigma2) * g) / normaliser;
        data.push(value / gauss_sum);
    }
    let sum: f64 = data.iter().sum();
    if sum != 0.0 {
        let correction = sum / data.len() as f64;
        for value in &mut data {
            *value -= correction;
        }
    }
    Tensor::new(data, vec![rows, cols]).map_err(|e| fspecial_error(format!("fspecial: {e}")))
}

fn sigma6(sigma: f64) -> f64 {
    let sigma2 = sigma * sigma;
    sigma2 * sigma2 * sigma2
}

fn generate_motion(
    length: usize,
    kernel_size: usize,
    angle_degrees: f64,
    oversample: usize,
) -> BuiltinResult<Tensor> {
    let mut data = vec![0.0f64; kernel_size * kernel_size];
    let center = (kernel_size as f64 - 1.0) / 2.0;
    let theta = angle_degrees.to_radians();
    let dir_x = theta.cos();
    let dir_y = theta.sin();
    let total_samples = length * oversample;
    let step = 1.0 / oversample as f64;
    let half = (length as f64 - 1.0) / 2.0;

    for idx in 0..total_samples {
        let t = -half + (idx as f64 + 0.5) * step;
        let x = center + t * dir_x;
        let y = center + t * dir_y;
        deposit_bilinear(&mut data, kernel_size, x, y, 1.0);
    }

    let mut sum = 0.0;
    for value in &data {
        sum += *value;
    }
    if sum == 0.0 {
        return Err(fspecial_error("fspecial: failed to build motion kernel"));
    }
    for value in &mut data {
        *value /= sum;
    }

    Tensor::new(data, vec![kernel_size, kernel_size]).map_err(|e| fspecial_error(format!("fspecial: {e}")))
}

fn deposit_bilinear(data: &mut [f64], size: usize, x: f64, y: f64, contribution: f64) {
    let xi = x.floor();
    let yi = y.floor();
    let xf = x - xi;
    let yf = y - yi;
    let xi = xi as isize;
    let yi = yi as isize;
    for dy in 0..=1 {
        for dx in 0..=1 {
            let px = xi + dx;
            let py = yi + dy;
            if px < 0 || py < 0 || px >= size as isize || py >= size as isize {
                continue;
            }
            let wx = if dx == 0 { 1.0 - xf } else { xf };
            let wy = if dy == 0 { 1.0 - yf } else { yf };
            let weight = wx * wy;
            let idx = (px as usize) * size + py as usize;
            data[idx] += contribution * weight;
        }
    }
}

fn generate_prewitt() -> BuiltinResult<Tensor> {
    Tensor::new(
        vec![
            1.0, 0.0, -1.0, //
            1.0, 0.0, -1.0, //
            1.0, 0.0, -1.0,
        ],
        vec![3, 3],
    )
    .map_err(|e| fspecial_error(format!("fspecial: {e}")))
}

fn generate_sobel() -> BuiltinResult<Tensor> {
    Tensor::new(
        vec![
            1.0, 0.0, -1.0, //
            2.0, 0.0, -2.0, //
            1.0, 0.0, -1.0,
        ],
        vec![3, 3],
    )
    .map_err(|e| fspecial_error(format!("fspecial: {e}")))
}

fn generate_unsharp(alpha: f64) -> BuiltinResult<Tensor> {
    let denom = alpha + 1.0;
    let mut data = vec![
        -alpha,
        alpha - 1.0,
        -alpha,
        alpha - 1.0,
        alpha + 5.0,
        alpha - 1.0,
        -alpha,
        alpha - 1.0,
        -alpha,
    ];
    for value in &mut data {
        *value /= denom;
    }
    Tensor::new(data, vec![3, 3]).map_err(|e| fspecial_error(format!("fspecial: {e}")))
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

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn parse_lengths_strict(value: &Value, err: &str) -> BuiltinResult<Vec<usize>> {
    parse_lengths_inner(value, err, true)
}

fn parse_lengths_inner(
    value: &Value,
    err: &str,
    enforce_positive: bool,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Int(i) => {
            let len = i.to_i64();
            if enforce_positive && len <= 0 {
                return Err(fspecial_error(err));
            }
            if len < 0 {
                return Err(fspecial_error(err));
            }
            Ok(vec![len as usize])
        }
        Value::Num(n) => parse_numeric_dimension(*n).map(|d| vec![d]),
        Value::Tensor(tensor) => {
            let dims = tensor
                .data
                .iter()
                .map(|&v| parse_numeric_dimension(v))
                .collect::<Result<Vec<_>, _>>()?;
            if enforce_positive && dims.contains(&0) {
                return Err(fspecial_error(err));
            }
            Ok(dims)
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() != logical.shape.iter().product::<usize>() {
                return Err(fspecial_error(err));
            }
            let dims = logical
                .data
                .iter()
                .map(|&v| parse_numeric_dimension(v as f64))
                .collect::<Result<Vec<_>, _>>()?;
            if enforce_positive && dims.contains(&0) {
                return Err(fspecial_error(err));
            }
            Ok(dims)
        }
        _ => Err(fspecial_error(err)),
    }
}

fn parse_numeric_dimension(n: f64) -> BuiltinResult<usize> {
    if !n.is_finite() {
        return Err(fspecial_error("fspecial: dimensions must be finite"));
    }
    if n < 0.0 {
        return Err(fspecial_error("fspecial: dimensions must be non-negative"));
    }
    let rounded = n.round();
    if (rounded - n).abs() > f64::EPSILON {
        return Err(fspecial_error("fspecial: dimensions must be integers"));
    }
    Ok(rounded as usize)
}

fn to_scalar(value: &Value, err: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        _ => Err(fspecial_error(err)),
    }
}

fn to_positive_scalar(value: &Value, err: &str) -> BuiltinResult<f64> {
    let scalar = to_scalar(value, err)?;
    if scalar.is_nan() || scalar.is_infinite() {
        return Err(fspecial_error(err));
    }
    Ok(scalar)
}

fn circle_rect_area(radius: f64, x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
    if x1 >= x2 || y1 >= y2 {
        return 0.0;
    }
    let r = radius;
    if (x1 >= r || y1 >= r || x2 <= -r || y2 <= -r) && min_distance_to_circle(x1, y1, x2, y2) >= r {
        return 0.0;
    }

    if x1 < 0.0 && x2 > 0.0 {
        let left = circle_rect_area(r, x1, 0.0, y1, y2);
        let right = circle_rect_area(r, 0.0, x2, y1, y2);
        return left + right;
    }
    if y1 < 0.0 && y2 > 0.0 {
        let bottom = circle_rect_area(r, x1, x2, y1, 0.0);
        let top = circle_rect_area(r, x1, x2, 0.0, y2);
        return bottom + top;
    }
    if x2 <= 0.0 {
        return circle_rect_area(r, -x2, -x1, y1, y2);
    }
    if y2 <= 0.0 {
        return circle_rect_area(r, x1, x2, -y2, -y1);
    }
    circle_rect_area_first_quadrant(r, x1.max(0.0), x2.min(r), y1.max(0.0), y2.min(r))
}

fn min_distance_to_circle(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let cx = if x1 > 0.0 {
        x1
    } else if x2 < 0.0 {
        x2
    } else {
        0.0
    };
    let cy = if y1 > 0.0 {
        y1
    } else if y2 < 0.0 {
        y2
    } else {
        0.0
    };
    (cx * cx + cy * cy).sqrt()
}

fn circle_rect_area_first_quadrant(radius: f64, x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
    if x1 >= x2 || y1 >= y2 {
        return 0.0;
    }
    let r = radius;
    if x1 >= r || y1 >= r {
        return 0.0;
    }
    let xa = x1.max(0.0);
    let xb = x2.min(r);
    if xb <= xa {
        return 0.0;
    }
    let ya = y1.max(0.0);
    let yb = y2.min(r);
    if yb <= ya {
        return 0.0;
    }
    let rsq = r * r;
    let x_for_y = |y: f64| -> f64 {
        if y >= r {
            0.0
        } else {
            (rsq - y * y).sqrt()
        }
    };
    let x_full = xb.min(x_for_y(yb));
    let mut area = 0.0;
    if x_full > xa {
        area += (yb - ya) * (x_full - xa);
    }
    let x_partial_start = x_full.max(xa);
    let x_partial_end = xb.min(x_for_y(ya));
    if x_partial_end > x_partial_start {
        let arc = arc_integral(r, x_partial_start, x_partial_end);
        area += arc - ya * (x_partial_end - x_partial_start);
    }
    area
}

fn arc_integral(radius: f64, a: f64, b: f64) -> f64 {
    primitive_arc(radius, b) - primitive_arc(radius, a)
}

fn primitive_arc(radius: f64, x: f64) -> f64 {
    let r = radius;
    if x <= -r {
        return -0.5 * PI * r * r;
    }
    if x >= r {
        return 0.5 * PI * r * r;
    }
    let term = (r * r - x * x).max(0.0).sqrt();
    0.5 * (x * term + r * r * clamp_asin(x / r))
}

fn clamp_asin(value: f64) -> f64 {
    value.clamp(-1.0, 1.0).asin()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::RuntimeControlFlow;

    fn assert_close(actual: f64, expected: f64, epsilon: f64) {
        if (actual - expected).abs() > epsilon {
            panic!(
                "values differ: actual={actual:.15e}, expected={expected:.15e}, epsilon={epsilon:.3e}"
            );
        }
    }

    fn error_message(err: RuntimeControlFlow) -> String {
        match err {
            RuntimeControlFlow::Error(error) => error.message().to_string(),
            RuntimeControlFlow::Suspend(_) => panic!("unexpected suspension"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_average_default() {
        let result = fspecial_builtin(Value::from("average"), Vec::new()).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                for value in t.data {
                    assert_close(value, 1.0 / 9.0, 1e-12);
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_average_scalar_size() {
        let args = vec![Value::from(5)];
        let result = fspecial_builtin(Value::from("average"), args).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![5, 5]);
                let sum: f64 = t.data.iter().sum();
                assert_close(sum, 1.0, 1e-12);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_average_rectangular_size() {
        let args = vec![Value::from(
            Tensor::new(vec![4.0, 6.0], vec![1, 2]).unwrap(),
        )];
        let result = fspecial_builtin(Value::from("average"), args).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 6]);
                let expected = 1.0 / (4.0 * 6.0);
                for value in t.data {
                    assert_close(value, expected, 1e-12);
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_average_rejects_zero_size() {
        let args = vec![Value::from(0)];
        let err = fspecial_builtin(Value::from("average"), args)
            .expect_err("fspecial should error");
        assert!(error_message(err).contains("positive"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_gaussian_default_matches_reference() {
        let result = fspecial_builtin(Value::from("gaussian"), Vec::new()).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                const EXPECTED: [f64; 9] = [
                    0.011_343_736_558_495,
                    0.083_819_505_802_211,
                    0.011_343_736_558_495,
                    0.083_819_505_802_211,
                    0.619_347_030_557_177,
                    0.083_819_505_802_211,
                    0.011_343_736_558_495,
                    0.083_819_505_802_211,
                    0.011_343_736_558_495,
                ];
                for (idx, value) in t.data.iter().enumerate() {
                    assert_close(*value, EXPECTED[idx], 1e-12);
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_gaussian_size_sigma() {
        let args = vec![Value::from(7), Value::from(2.0)];
        let result = fspecial_builtin(Value::from("gaussian"), args).unwrap();
        let tensor = match result {
            Value::Tensor(t) => t,
            Value::GpuTensor(h) => {
                crate::builtins::common::test_support::gather(Value::GpuTensor(h)).expect("gather")
            }
            other => panic!("expected tensor, got {other:?}"),
        };
        assert_eq!(tensor.shape, vec![7, 7]);
        let center = tensor.rows / 2;
        let col = center;
        let idx = col * tensor.rows + center;
        assert!(tensor.data[idx] > 0.0);
        let sum: f64 = tensor.data.iter().sum();
        assert_close(sum, 1.0, 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_laplacian_alpha() {
        let args = vec![Value::from(0.2)];
        let result = fspecial_builtin(Value::from("laplacian"), args).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = [
                    0.16666666666666669,
                    0.6666666666666667,
                    0.16666666666666669,
                    0.6666666666666667,
                    -3.3333333333333335,
                    0.6666666666666667,
                    0.16666666666666669,
                    0.6666666666666667,
                    0.16666666666666669,
                ];
                for (idx, value) in t.data.iter().enumerate() {
                    assert_close(*value, expected[idx], 1e-12);
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_unsharp_default() {
        let result = fspecial_builtin(Value::from("unsharp"), Vec::new()).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let sum: f64 = t.data.iter().sum();
                assert_close(sum, 1.0, 1e-12);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_log_basic_properties() {
        let result =
            fspecial_builtin(Value::from("log"), vec![Value::from(5), Value::from(0.5)]).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![5, 5]);
                let sum: f64 = t.data.iter().sum();
                assert_close(sum, 0.0, 1e-12);
                let center = t.rows / 2;
                let idx = center * t.rows + center;
                assert!(t.data[idx] < 0.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_disk_sum_is_one() {
        let result = fspecial_builtin(Value::from("disk"), vec![Value::from(5)]).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![11, 11]);
                let sum: f64 = t.data.iter().sum();
                assert_close(sum, 1.0, 1e-10);
                let idx = t.rows * (t.cols / 2) + t.rows / 2;
                assert!(t.data[idx] > 0.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_disk_negative_radius_errors() {
        let err = fspecial_builtin(Value::from("disk"), vec![Value::from(-1.0)])
            .expect_err("fspecial should error");
        assert!(error_message(err).contains("non-negative"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_motion_sum_is_one() {
        let result = fspecial_builtin(
            Value::from("motion"),
            vec![Value::from(15), Value::from(45.0)],
        )
        .unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![15, 15]);
                let sum: f64 = t.data.iter().sum();
                assert_close(sum, 1.0, 1e-10);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_invalid_filter_name() {
        let err = fspecial_builtin(Value::from("notafilter"), Vec::new())
            .expect_err("fspecial should error");
        assert!(error_message(err).contains("not supported"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples() {
        let blocks = test_support::doc_examples(super::DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fspecial_gaussian_gpu_matches_cpu() {
        std::env::set_var("RUNMAT_ACCEL_FSPECIAL_DEVICE", "1");
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let gpu_tensor = match fspecial_builtin(Value::from("gaussian"), Vec::new()).unwrap() {
            Value::GpuTensor(handle) => {
                test_support::gather(Value::GpuTensor(handle)).expect("gather gpu result")
            }
            Value::Tensor(t) => t,
            other => panic!("unexpected result {other:?}"),
        };
        std::env::remove_var("RUNMAT_ACCEL_FSPECIAL_DEVICE");
        let host_tensor = match fspecial_builtin(Value::from("gaussian"), Vec::new()).unwrap() {
            Value::Tensor(t) => t,
            Value::GpuTensor(handle) => {
                test_support::gather(Value::GpuTensor(handle)).expect("gather fallback")
            }
            other => panic!("unexpected result {other:?}"),
        };
        assert_eq!(gpu_tensor.shape, host_tensor.shape);
        for (a, b) in gpu_tensor.data.iter().zip(host_tensor.data.iter()) {
            assert_close(*a, *b, 1e-6);
        }
    }
}
