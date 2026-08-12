use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use log::warn;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, NumericScalar, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode, SurfacePlot};

use super::common::{tensor_to_surface_grid, SurfaceDataInput};
use super::op_common::surface_inputs::{
    axis_sources_to_host, image_axis_sources_from_xy_values, parse_image_call_args, AxisSource,
};
use super::state::{color_limits_snapshot, render_active_plot, PlotRenderOptions};
use super::style::{parse_surface_style_args, value_as_string, SurfaceStyleDefaults};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "image";

const IMAGE_FOUR_CHANNEL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "image-four-channel-cdata",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "image with M-by-N-by-4 CData is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ImageFourChannelCDataExtension"),
};

pub const IMAGE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [IMAGE_FOUR_CHANNEL_EXTENSION];

const IMAGE_INTEGER_CDATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "CData",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Indexed and truecolor CData accept every built-in integer class; exact CData remains authoritative on the image object.",
}];
pub const IMAGE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "h = image(integer_CData, ...)",
        inputs: &IMAGE_INTEGER_CDATA,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Indexed values retain direct-mapping provenance, truecolor values normalize across their native class range, and renderer conversion is explicit.",
    }];

const IMAGE_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the rendered image object.",
}];

const IMAGE_INPUTS_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Image data array (indexed matrix or truecolor MxNx3/MxNx4).",
}];

const IMAGE_INPUTS_X_Y_C: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates or extent vector.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates or extent vector.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Image data array (indexed matrix or truecolor MxNx3/MxNx4).",
    },
];

const IMAGE_INPUTS_C_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Image data array (indexed matrix or truecolor MxNx3/MxNx4).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value surface style options.",
    },
];

const IMAGE_INPUTS_X_Y_C_PROPS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates or extent vector.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates or extent vector.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Image data array (indexed matrix or truecolor MxNx3/MxNx4).",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value surface style options.",
    },
];

const IMAGE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = image(C)",
        inputs: &IMAGE_INPUTS_C,
        outputs: &IMAGE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = image(X, Y, C)",
        inputs: &IMAGE_INPUTS_X_Y_C,
        outputs: &IMAGE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = image(C, Name, Value, ...)",
        inputs: &IMAGE_INPUTS_C_PROPS,
        outputs: &IMAGE_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = image(X, Y, C, Name, Value, ...)",
        inputs: &IMAGE_INPUTS_X_Y_C_PROPS,
        outputs: &IMAGE_OUTPUT_HANDLE,
    },
];

pub const IMAGE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMAGE.INVALID_ARGUMENT",
    identifier: Some("RunMat:image:InvalidArgument"),
    when: "Image data, axis inputs, or name/value style arguments are invalid.",
    message: "image: invalid argument",
};

pub const IMAGE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMAGE.INTERNAL",
    identifier: Some("RunMat:image:Internal"),
    when: "Internal image/surface construction or rendering fails unexpectedly.",
    message: "image: internal operation failed",
};

const IMAGE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [IMAGE_ERROR_INVALID_ARGUMENT, IMAGE_ERROR_INTERNAL];

pub const IMAGE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMAGE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMAGE_ERRORS,
};

fn image_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_image_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    image_error_with_detail(&IMAGE_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_image_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    image_error_with_detail(&IMAGE_ERROR_INTERNAL, err.message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::image")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "image",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "image is a plotting sink; indexed and truecolor gpuArray inputs may remain on device when a shared WGPU context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::image")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "image",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "image terminates fusion graphs and performs rendering.",
};

#[runtime_builtin(
    name = "image",
    category = "plotting",
    summary = "Display indexed or truecolor images.",
    keywords = "image,plotting,imshow,colormap",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::image::IMAGE_DESCRIPTOR),
    extensions(crate::builtins::plotting::image::IMAGE_EXTENSIONS),
    integer_capabilities(crate::builtins::plotting::image::IMAGE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::image"
)]
pub async fn image_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (x, y, c, rest) =
        parse_image_call_args(args, BUILTIN_NAME).map_err(map_image_invalid_argument)?;
    let (c_data_mapping, rest) =
        extract_cdata_mapping(rest, "direct", BUILTIN_NAME).map_err(map_image_invalid_argument)?;
    if image_channel_count(&c) == Some(4) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IMAGE_FOUR_CHANNEL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let (rows, cols, kind, retained_c_data) = classify_image_input(&c, BUILTIN_NAME)
        .await
        .map_err(map_image_invalid_argument)?;
    let (x_axis, y_axis) = image_axis_sources_from_xy_values(x, y, rows, cols, BUILTIN_NAME)
        .await
        .map_err(map_image_invalid_argument)?;
    let defaults =
        SurfaceStyleDefaults::new(ColorMap::Parula, ShadingMode::None, false, 1.0, true, false);
    let style = Arc::new(
        parse_surface_style_args(BUILTIN_NAME, &rest, defaults)
            .map_err(map_image_invalid_argument)?,
    );
    let color_limits = color_limits_snapshot();

    let mut surface = match kind {
        ImageInputKind::TrueColorHost(tensor) => {
            let (x_host, y_host) = axis_sources_to_host(&x_axis, &y_axis, BUILTIN_NAME)
                .await
                .map_err(map_image_invalid_argument)?;
            build_truecolor_image_surface(tensor, x_host, y_host)
                .map_err(map_image_invalid_argument)?
        }
        ImageInputKind::TrueColorGpu(handle, channels) => {
            build_truecolor_image_surface_gpu(&handle, &x_axis, &y_axis, rows, cols, channels)
                .map_err(map_image_invalid_argument)?
        }
        ImageInputKind::Indexed(input) => build_indexed_image_surface(
            &input,
            &x_axis,
            &y_axis,
            style.colormap.clone(),
            color_limits,
            &c_data_mapping,
        )
        .await
        .map_err(map_image_invalid_argument)?,
    };

    surface = surface.with_flatten_z(true).with_image_mode(true);
    let mut surface = Some(surface);
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Image",
            x_label: "X",
            y_label: "Y",
            axis_equal: true,
            ..Default::default()
        },
        move |figure, axes| {
            let plot_index = figure
                .add_surface_plot_on_axes(surface.take().expect("image plot consumed once"), axes);
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_image_handle(
        figure_handle,
        axes,
        plot_index,
        Some(retained_c_data),
        &c_data_mapping,
    );
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(map_image_internal(err));
    }
    Ok(handle)
}

pub(crate) fn extract_cdata_mapping(
    rest: Vec<Value>,
    default: &str,
    builtin: &'static str,
) -> crate::BuiltinResult<(String, Vec<Value>)> {
    let contains_mapping = rest.iter().any(|value| {
        value_as_string(value).is_some_and(|name| name.trim().eq_ignore_ascii_case("CDataMapping"))
    });
    if !contains_mapping {
        return Ok((default.to_string(), rest));
    }
    if !rest.len().is_multiple_of(2) {
        return Err(build_runtime_error(format!(
            "{builtin}: name-value arguments must come in pairs"
        ))
        .build());
    }
    let mut mapping = default.to_string();
    let mut retained = Vec::with_capacity(rest.len());
    let mut seen = false;
    for pair in rest.chunks_exact(2) {
        let is_mapping = value_as_string(&pair[0])
            .is_some_and(|name| name.trim().eq_ignore_ascii_case("CDataMapping"));
        if !is_mapping {
            retained.extend_from_slice(pair);
            continue;
        }
        if seen {
            return Err(build_runtime_error(format!(
                "{builtin}: CDataMapping may be specified only once"
            ))
            .build());
        }
        let value = value_as_string(&pair[1]).ok_or_else(|| {
            build_runtime_error(format!(
                "{builtin}: CDataMapping must be 'direct' or 'scaled'"
            ))
            .build()
        })?;
        let value = value.trim().to_ascii_lowercase();
        if value != "direct" && value != "scaled" {
            return Err(build_runtime_error(format!(
                "{builtin}: CDataMapping must be 'direct' or 'scaled'"
            ))
            .build());
        }
        mapping = value;
        seen = true;
    }
    Ok((mapping, retained))
}

pub(crate) fn image_channel_count(value: &Value) -> Option<usize> {
    match value {
        Value::GpuTensor(handle) => handle.shape.get(2).copied(),
        other => Tensor::try_from(other)
            .ok()
            .and_then(|tensor| tensor.shape.get(2).copied()),
    }
}

pub(crate) enum ImageInputKind {
    Indexed(SurfaceDataInput),
    TrueColorHost(Tensor),
    TrueColorGpu(runmat_accelerate_api::GpuTensorHandle, u32),
}

pub(crate) async fn classify_image_input(
    value: &Value,
    builtin: &'static str,
) -> crate::BuiltinResult<(usize, usize, ImageInputKind, Tensor)> {
    match value {
        Value::GpuTensor(handle)
            if handle.shape.len() >= 3
                && runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle) =>
        {
            let (rows, cols, channels) = truecolor_gpu_shape(handle, builtin)?;
            let retained = download_retained_cdata(handle, builtin).await?;
            Ok((
                rows,
                cols,
                ImageInputKind::TrueColorGpu(handle.clone(), channels),
                retained,
            ))
        }
        _ => {
            let tensor = match value {
                Value::GpuTensor(handle) => download_retained_cdata(handle, builtin).await?,
                other => {
                    tensor_utils::value_into_tensor_for(builtin, other.clone()).map_err(|e| {
                        crate::builtins::plotting::plotting_error(
                            builtin,
                            format!("{builtin}: {e}"),
                        )
                    })?
                }
            };
            if tensor.shape.len() >= 3 {
                let (rows, cols) = truecolor_shape(&tensor, builtin)?;
                Ok((
                    rows,
                    cols,
                    ImageInputKind::TrueColorHost(tensor.clone()),
                    tensor,
                ))
            } else {
                let input = match value {
                    Value::GpuTensor(handle)
                        if runmat_accelerate_api::handle_integer_type(handle).is_some()
                            || runmat_accelerate_api::handle_is_logical(handle) =>
                    {
                        SurfaceDataInput::Host(tensor.clone())
                    }
                    Value::GpuTensor(_) => SurfaceDataInput::from_value(value.clone(), builtin)?,
                    _ => SurfaceDataInput::Host(tensor.clone()),
                };
                let (rows, cols) = input.grid_shape(builtin)?;
                Ok((rows, cols, ImageInputKind::Indexed(input), tensor))
            }
        }
    }
}

async fn download_retained_cdata(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    builtin: &'static str,
) -> crate::BuiltinResult<Tensor> {
    let provider = runmat_accelerate_api::provider_for_handle(handle).ok_or_else(|| {
        crate::builtins::plotting::plotting_error(
            builtin,
            format!("{builtin}: no provider owns resident CData"),
        )
    })?;
    let value = crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
        provider, handle,
    )
    .await?;
    tensor_utils::value_into_tensor_for(builtin, value)
        .map_err(|error| crate::builtins::plotting::plotting_error(builtin, error))
}

pub(crate) fn build_truecolor_image_surface_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    x_axis: &AxisSource,
    y_axis: &AxisSource,
    rows: usize,
    cols: usize,
    channels: u32,
) -> crate::BuiltinResult<SurfacePlot> {
    let context = super::gpu_helpers::ensure_shared_wgpu_context(BUILTIN_NAME)?;
    let image_ref = runmat_accelerate_api::export_wgpu_buffer(handle).ok_or_else(|| {
        crate::builtins::plotting::plotting_error(
            BUILTIN_NAME,
            "image: unable to export truecolor GPU image",
        )
    })?;
    let scalar = runmat_plot::gpu::ScalarType::from_is_f64(
        image_ref.precision == runmat_accelerate_api::ProviderPrecision::F64,
    );
    // Axis buffers can have a different native class, precision, provider, or storage
    // layout than CData. Materialize only the small axes and encode them with the
    // image scalar type; CData itself remains resident and is consumed zero-copy.
    let (x_host, y_host) =
        futures::executor::block_on(axis_sources_to_host(x_axis, y_axis, BUILTIN_NAME))?;
    let x_f32;
    let y_f32;
    let x_f64;
    let y_f64;
    let (x_data, y_data) = match scalar {
        runmat_plot::gpu::ScalarType::F32 => {
            x_f32 = x_host.iter().map(|value| *value as f32).collect::<Vec<_>>();
            y_f32 = y_host.iter().map(|value| *value as f32).collect::<Vec<_>>();
            (
                runmat_plot::gpu::axis::AxisData::F32(&x_f32),
                runmat_plot::gpu::axis::AxisData::F32(&y_f32),
            )
        }
        runmat_plot::gpu::ScalarType::F64 => {
            x_f64 = x_host.clone();
            y_f64 = y_host.clone();
            (
                runmat_plot::gpu::axis::AxisData::F64(&x_f64),
                runmat_plot::gpu::axis::AxisData::F64(&y_f64),
            )
        }
    };
    let gpu_vertices = runmat_plot::gpu::image::pack_truecolor_vertices(
        &context.device,
        &context.queue,
        &runmat_plot::gpu::image::TrueColorImageGpuInputs {
            x_axis: x_data,
            y_axis: y_data,
            image_buffer: image_ref.buffer.clone(),
            rows: rows as u32,
            cols: cols as u32,
            channels,
            scalar,
        },
    )
    .map_err(|e| {
        crate::builtins::plotting::plotting_error(
            BUILTIN_NAME,
            format!("image: failed to build GPU truecolor vertices: {e}"),
        )
    })?;
    let bounds = runmat_plot::core::BoundingBox::new(
        glam::Vec3::new(
            x_host.first().copied().unwrap_or(0.0) as f32,
            y_host.first().copied().unwrap_or(0.0) as f32,
            0.0,
        ),
        glam::Vec3::new(
            x_host.last().copied().unwrap_or(0.0) as f32,
            y_host.last().copied().unwrap_or(0.0) as f32,
            0.0,
        ),
    );
    let mut surface = SurfacePlot::from_gpu_buffer(cols, rows, gpu_vertices, rows * cols, bounds)
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_shading(ShadingMode::None)
        .with_gpu_color_grid_source(runmat_plot::plots::SurfaceGpuColorGridSource {
            image_buffer: image_ref.buffer.clone(),
            rows,
            cols,
            channels: channels as usize,
            scalar,
        });
    surface.x_data = x_host;
    surface.y_data = y_host;
    surface.z_data = Some(vec![vec![0.0; rows]; cols]);
    Ok(surface)
}

fn truecolor_gpu_shape(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    builtin: &'static str,
) -> crate::BuiltinResult<(usize, usize, u32)> {
    let rows = handle.shape.first().copied().unwrap_or(0);
    let cols = handle.shape.get(1).copied().unwrap_or(0);
    let channels = handle.shape.get(2).copied().unwrap_or(1);
    let trailing_singleton = handle.shape.iter().skip(3).all(|&dim| dim == 1);
    if rows == 0 || cols == 0 || !matches!(channels, 3 | 4) || !trailing_singleton {
        return Err(crate::builtins::plotting::plotting_error(
            builtin,
            format!("{builtin}: truecolor image data must be MxNx3 or MxNx4"),
        ));
    }
    let expected = rows
        .checked_mul(cols)
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| {
            crate::builtins::plotting::plotting_error(
                builtin,
                format!("{builtin}: truecolor image dimensions overflow"),
            )
        })?;
    let actual = handle
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
    if actual != Some(expected) {
        return Err(crate::builtins::plotting::plotting_error(
            builtin,
            format!("{builtin}: truecolor image data length mismatch"),
        ));
    }
    Ok((rows, cols, channels as u32))
}

fn truecolor_shape(tensor: &Tensor, builtin: &'static str) -> crate::BuiltinResult<(usize, usize)> {
    let rows = tensor.shape.first().copied().unwrap_or(tensor.rows);
    let cols = tensor.shape.get(1).copied().unwrap_or(tensor.cols);
    let channels = tensor.shape.get(2).copied().unwrap_or(1);
    if rows == 0 || cols == 0 || (channels != 3 && channels != 4) {
        return Err(crate::builtins::plotting::plotting_error(
            builtin,
            format!("{builtin}: truecolor image data must be MxNx3 or MxNx4"),
        ));
    }
    let expected_len = rows * cols * channels;
    if tensor_utils::tensor_element_len(tensor) != expected_len {
        return Err(crate::builtins::plotting::plotting_error(
            builtin,
            format!("{builtin}: truecolor image data length mismatch"),
        ));
    }
    Ok((rows, cols))
}

pub(crate) async fn build_indexed_image_surface(
    c_input: &SurfaceDataInput,
    x_axis: &super::op_common::surface_inputs::AxisSource,
    y_axis: &super::op_common::surface_inputs::AxisSource,
    colormap: ColorMap,
    color_limits: Option<(f64, f64)>,
    mapping: &str,
) -> crate::BuiltinResult<SurfacePlot> {
    let direct_limits = (mapping == "direct").then(|| {
        let integer = match c_input {
            SurfaceDataInput::Host(tensor) => tensor.integer_storage().is_some(),
            SurfaceDataInput::Gpu(handle) => {
                runmat_accelerate_api::handle_integer_type(handle).is_some()
            }
        };
        direct_colormap_limits(integer, super::state::current_colormap_length())
    });
    let effective_limits = direct_limits.or(color_limits);
    if let Some(c_gpu) = c_input.gpu_handle().cloned() {
        match super::gpu_helpers::axis_bounds_async(&c_gpu, BUILTIN_NAME).await {
            Ok((min_z, max_z)) => match super::surf::build_surface_gpu_plot_with_bounds_async(
                BUILTIN_NAME,
                x_axis,
                y_axis,
                &c_gpu,
                min_z,
                max_z,
                colormap.clone(),
                1.0,
                true,
            )
            .await
            {
                Ok(surface) => {
                    return Ok(surface
                        .with_flatten_z(true)
                        .with_image_mode(true)
                        .with_color_limits(effective_limits));
                }
                Err(err) => warn!("image GPU path unavailable: {err}"),
            },
            Err(err) => warn!("image GPU bounds unavailable: {err}"),
        }
    }

    let (x_host, y_host) = axis_sources_to_host(x_axis, y_axis, BUILTIN_NAME).await?;
    let tensor = match c_input.clone() {
        SurfaceDataInput::Host(tensor) => tensor,
        SurfaceDataInput::Gpu(handle) => {
            super::common::gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await?
        }
    };
    let grid = tensor_to_surface_grid(tensor, x_host.len(), y_host.len(), BUILTIN_NAME)?;
    Ok(super::surf::build_surface(x_host, y_host, grid)?
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_colormap(colormap)
        .with_shading(ShadingMode::None)
        .with_color_limits(effective_limits))
}

pub(crate) fn direct_colormap_limits(integer: bool, colormap_len: usize) -> (f64, f64) {
    let offset = if integer { 0.0 } else { 1.0 };
    let upper = offset + colormap_len.saturating_sub(1) as f64;
    (offset, upper.max(offset + f64::MIN_POSITIVE))
}

pub(crate) fn build_truecolor_image_surface(
    tensor: Tensor,
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
) -> crate::BuiltinResult<SurfacePlot> {
    let rows = y_axis.len();
    let cols = x_axis.len();
    let channels = tensor.shape.get(2).copied().unwrap_or(3);
    let dtype = tensor.numeric_dtype();
    let mut grid = vec![vec![glam::Vec4::ZERO; rows]; cols];
    for row in 0..rows {
        for col in 0..cols {
            let base = row + rows * col;
            let r = normalized_truecolor_sample(&tensor, base, dtype)?;
            let g = normalized_truecolor_sample(&tensor, base + rows * cols, dtype)?;
            let b = normalized_truecolor_sample(&tensor, base + 2 * rows * cols, dtype)?;
            let a = if channels == 4 {
                normalized_truecolor_sample(&tensor, base + 3 * rows * cols, dtype)?
            } else {
                1.0
            };
            grid[col][row] = glam::Vec4::new(r, g, b, a);
        }
    }
    let z = vec![vec![0.0; rows]; cols];
    Ok(SurfacePlot::new(x_axis, y_axis, z)
        .map_err(|e| {
            crate::builtins::plotting::plotting_error(BUILTIN_NAME, format!("image: {e}"))
        })?
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_color_grid(grid)
        .with_shading(ShadingMode::None))
}

fn normalized_truecolor_sample(
    tensor: &Tensor,
    index: usize,
    dtype: NumericDType,
) -> crate::BuiltinResult<f32> {
    let sample = tensor.numeric_value_at(index).ok_or_else(|| {
        image_error_with_detail(&IMAGE_ERROR_INVALID_ARGUMENT, "truecolor sample is missing")
    })?;
    let normalized = match sample {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        NumericScalar::I8(value) => (f64::from(value) - f64::from(i8::MIN)) / u8::MAX as f64,
        NumericScalar::I16(value) => (f64::from(value) - f64::from(i16::MIN)) / u16::MAX as f64,
        NumericScalar::I32(value) => (f64::from(value) - f64::from(i32::MIN)) / u32::MAX as f64,
        NumericScalar::I64(value) => (value as i128 - i64::MIN as i128) as f64 / u64::MAX as f64,
        NumericScalar::U8(value) => f64::from(value) / u8::MAX as f64,
        NumericScalar::U16(value) => f64::from(value) / u16::MAX as f64,
        NumericScalar::U32(value) => f64::from(value) / u32::MAX as f64,
        NumericScalar::U64(value) => value as f64 / u64::MAX as f64,
    };
    if !normalized.is_finite() {
        return Err(image_error_with_detail(
            &IMAGE_ERROR_INVALID_ARGUMENT,
            format!("truecolor {} sample must be finite", dtype.class_name()),
        ));
    }
    Ok(normalized.clamp(0.0, 1.0) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_builtins::IntegerStorage;
    use runmat_plot::plots::PlotElement;

    fn truecolor_tensor() -> Tensor {
        Tensor::new(
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            vec![2, 2, 3],
        )
        .expect("truecolor image")
    }

    fn typed_truecolor_tensor() -> Tensor {
        let tensor = Tensor::new_integer(
            IntegerStorage::U8(vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]),
            vec![2, 2, 3],
        )
        .expect("typed truecolor");
        tensor
    }

    #[test]
    fn image_truecolor_builds_image_surface_and_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle =
            futures::executor::block_on(image_builtin(vec![Value::Tensor(truecolor_tensor())]))
                .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert!(surface.image_mode);
        assert!(surface.color_grid.is_some());
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap(),
            Value::String("image".into())
        );
    }

    #[test]
    fn image_truecolor_reads_typed_integer_storage_without_mirror() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        futures::executor::block_on(image_builtin(vec![Value::Tensor(typed_truecolor_tensor())]))
            .unwrap();

        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        let grid = surface.color_grid.as_ref().expect("color grid");
        assert_eq!(grid[0][0], glam::Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(grid[0][1], glam::Vec4::new(0.0, 1.0, 0.0, 1.0));
        assert_eq!(grid[1][0], glam::Vec4::new(0.0, 0.0, 1.0, 1.0));
        assert_eq!(grid[1][1], glam::Vec4::new(1.0, 1.0, 1.0, 1.0));
    }

    #[test]
    fn image_accepts_two_element_extent_vectors() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let c =
            Tensor::new((1..=12).map(|v| v as f64).collect(), vec![3, 4]).expect("indexed image");
        let _ = futures::executor::block_on(image_builtin(vec![
            Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2]).expect("image x extent")),
            Value::Tensor(Tensor::new(vec![1.0, 5.0], vec![2]).expect("image y extent")),
            Value::Tensor(c),
        ]))
        .expect("image with extent vectors should succeed");
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface")
        };
        assert_eq!(
            surface.x_data,
            vec![10.0, 13.333333333333332, 16.666666666666664, 20.0]
        );
        assert_eq!(surface.y_data, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn image_non_square_truecolor_preserves_axes_shape_and_integer_cdata() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let data = IntegerStorage::I8(vec![i8::MIN; 18]);
        let input = Tensor::new_integer(data, vec![2, 3, 3]).unwrap();
        let handle = futures::executor::block_on(image_builtin(vec![Value::Tensor(input)]))
            .expect("non-square integer truecolor image");
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface")
        };
        assert_eq!(surface.x_data, vec![1.0, 2.0, 3.0]);
        assert_eq!(surface.y_data, vec![1.0, 2.0]);
        assert_eq!(surface.color_grid.as_ref().unwrap().len(), 3);
        assert_eq!(surface.color_grid.as_ref().unwrap()[0].len(), 2);
        let retained = get_builtin(vec![Value::Num(handle), Value::String("CData".into())])
            .expect("retained CData");
        let retained = Tensor::try_from(&retained).unwrap();
        assert_eq!(retained.numeric_dtype(), NumericDType::I8);
        assert_eq!(retained.shape, vec![2, 3, 3]);
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("CDataMapping".into())
            ])
            .unwrap(),
            Value::String("direct".into())
        );
    }

    #[test]
    fn image_cdata_set_rejects_instead_of_silently_succeeding() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(image_builtin(vec![Value::Tensor(
            Tensor::new_integer(IntegerStorage::U8(vec![1, 2, 3, 4]), vec![2, 2]).unwrap(),
        )]))
        .unwrap();
        let error = set_builtin(vec![
            Value::Num(handle),
            Value::String("CData".into()),
            Value::Tensor(Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap()),
        ])
        .expect_err("unsupported CData mutation must be explicit");
        assert!(error
            .message
            .contains("setting image CData is not implemented"));
    }

    #[test]
    fn image_direct_mapping_uses_colormap_indices_and_mapping_updates_renderer_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(image_builtin(vec![Value::Tensor(
            Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![1, 2]).unwrap(),
        )]))
        .unwrap();
        let figure = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = figure.plots().next().unwrap() else {
            panic!("expected image surface");
        };
        assert_eq!(surface.color_limits, Some((0.0, 255.0)));

        set_builtin(vec![
            Value::Num(handle),
            Value::String("CDataMapping".into()),
            Value::String("scaled".into()),
        ])
        .unwrap();
        let figure = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = figure.plots().next().unwrap() else {
            panic!("expected image surface");
        };
        assert_eq!(surface.color_limits, None);
    }

    #[test]
    fn image_constructor_accepts_cdata_mapping_and_colormap_resize_refreshes_limits() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = futures::executor::block_on(image_builtin(vec![
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![1, 2]).unwrap()),
            Value::String("CDataMapping".into()),
            Value::String("direct".into()),
        ]))
        .expect("constructor CDataMapping");
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("CDataMapping".into())
            ])
            .unwrap(),
            Value::String("direct".into())
        );

        crate::builtins::plotting::state::set_colormap_with_length(ColorMap::Parula, 2);
        let figure = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = figure.plots().next().unwrap() else {
            panic!("expected image surface");
        };
        assert_eq!(surface.color_limits, Some((0.0, 1.0)));
    }

    #[test]
    fn image_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = IMAGE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = image(C)"));
        assert!(labels.contains(&"h = image(X, Y, C)"));
        assert!(labels.contains(&"h = image(X, Y, C, Name, Value, ...)"));
    }

    #[test]
    fn image_missing_input_uses_stable_identifier() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let err = futures::executor::block_on(image_builtin(vec![]))
            .expect_err("missing args should fail");
        assert_eq!(err.identifier(), IMAGE_ERROR_INVALID_ARGUMENT.identifier);
    }
}
