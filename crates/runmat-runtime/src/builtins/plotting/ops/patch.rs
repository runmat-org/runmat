//! MATLAB-compatible `patch` builtin.

use glam::{Vec3, Vec4};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{PatchData, PatchEdgeColorMode, PatchFaceColorMode, PatchPlot};
use runmat_value::{IntegerStorage, NumericStorage, StructValue, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_helpers;
use crate::builtins::plotting::plotting_error;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::common::gather_tensor_from_gpu;
use super::op_common::{apply_axes_target, split_leading_axes_handle};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_color_value, value_as_f64, value_as_string, LineStyleParseOptions};

const BUILTIN_NAME: &str = "patch";

const PATCH_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created patch object.",
}];

const PATCH_INPUTS_XY_C: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Patch X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Patch Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Face color specification.",
    },
];

const PATCH_INPUTS_XYZ_C: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Patch X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Patch Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Patch Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Face color specification.",
    },
];

const PATCH_INPUTS_XYZ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Patch X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Patch Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Patch Z coordinates.",
    },
];

const PATCH_INPUTS_STRUCT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Struct containing patch properties.",
}];

const PATCH_INPUTS_FACEVERT_PROPS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "facevert",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Faces/Vertices property/value pairs.",
    },
    BuiltinParamDescriptor {
        name: "props",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional patch property/value pairs.",
    },
];

const PATCH_INPUTS_AX_DATA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Patch positional/property arguments.",
    },
];

const PATCH_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "h = patch(X, Y, C)",
        inputs: &PATCH_INPUTS_XY_C,
        outputs: &PATCH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = patch(X, Y, Z)",
        inputs: &PATCH_INPUTS_XYZ,
        outputs: &PATCH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = patch(X, Y, Z, C)",
        inputs: &PATCH_INPUTS_XYZ_C,
        outputs: &PATCH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = patch(S)",
        inputs: &PATCH_INPUTS_STRUCT,
        outputs: &PATCH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = patch('Faces', F, 'Vertices', V, ...)",
        inputs: &PATCH_INPUTS_FACEVERT_PROPS,
        outputs: &PATCH_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = patch(ax, ...)",
        inputs: &PATCH_INPUTS_AX_DATA,
        outputs: &PATCH_OUTPUT_HANDLE,
    },
];

const PATCH_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PATCH.INVALID_ARGUMENT",
    identifier: Some("RunMat:patch:InvalidArgument"),
    when: "Patch coordinates, faces/vertices, or property/value arguments are malformed.",
    message: "patch: invalid argument",
};

const PATCH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PATCH.INTERNAL",
    identifier: Some("RunMat:patch:Internal"),
    when: "Internal patch triangulation/render setup fails unexpectedly.",
    message: "patch: internal operation failed",
};

const PATCH_ERRORS: [BuiltinErrorDescriptor; 2] =
    [PATCH_ERROR_INVALID_ARGUMENT, PATCH_ERROR_INTERNAL];

pub const PATCH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PATCH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PATCH_ERRORS,
};

fn patch_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let raw = detail.as_ref().trim();
    let normalized = raw.strip_prefix("patch:").map(str::trim).unwrap_or(raw);
    let message = if normalized.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {}", error.message, normalized)
    };
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_patch_invalid(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    patch_error_with_detail(&PATCH_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_patch_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    patch_error_with_detail(&PATCH_ERROR_INTERNAL, err.message)
}

fn patch_invalid(detail: impl AsRef<str>) -> RuntimeError {
    patch_error_with_detail(&PATCH_ERROR_INVALID_ARGUMENT, detail)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::patch")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "patch",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "patch is a plotting sink. Initial implementation gathers gpuArray coordinate inputs, triangulates on the host, then renders through the shared GPU renderer.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::patch")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "patch",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "patch performs rendering and terminates fusion graphs.",
};

#[derive(Clone, Debug)]
struct PatchOptions {
    x_data: Option<Tensor>,
    y_data: Option<Tensor>,
    z_data: Option<Tensor>,
    faces: Option<Tensor>,
    vertices: Option<Tensor>,
    c_data: Option<Tensor>,
    face_color: Vec4,
    edge_color: Vec4,
    face_color_mode: PatchFaceColorMode,
    edge_color_mode: PatchEdgeColorMode,
    face_alpha: f32,
    edge_alpha: f32,
    line_width: f32,
    label: Option<String>,
    visible: bool,
}

impl Default for PatchOptions {
    fn default() -> Self {
        Self {
            x_data: None,
            y_data: None,
            z_data: None,
            faces: None,
            vertices: None,
            c_data: None,
            face_color: Vec4::new(0.0, 0.447, 0.741, 1.0),
            edge_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            face_color_mode: PatchFaceColorMode::Color,
            edge_color_mode: PatchEdgeColorMode::Color,
            face_alpha: 1.0,
            edge_alpha: 1.0,
            line_width: 0.5,
            label: None,
            visible: true,
        }
    }
}

#[runtime_builtin(
    name = "patch",
    category = "plotting",
    summary = "Create filled polygon patch graphics from coordinate or faces/vertices data.",
    keywords = "patch,plotting,polygon,faces,vertices",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::patch::PATCH_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::patch"
)]
pub fn patch_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (axes_target, args) =
        split_leading_axes_handle(args, BUILTIN_NAME).map_err(map_patch_invalid)?;
    let mut plot = Some(parse_patch_plot(args).map_err(map_patch_invalid)?);
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_patch_invalid)?;
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Patch",
            x_label: "X",
            y_label: "Y",
            ..Default::default()
        },
        move |figure, axes| {
            let patch = plot.take().expect("patch plot consumed once");
            let plot_index = figure.add_patch_plot_on_axes(patch, axes);
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_patch_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(map_patch_internal(err));
    }
    Ok(handle)
}

pub(super) fn parse_patch_plot(args: Vec<Value>) -> BuiltinResult<PatchPlot> {
    if args.is_empty() {
        return Err(patch_invalid("patch: expected input data"));
    }
    let mut opts = PatchOptions::default();
    let mut remaining = if let Some(Value::Struct(st)) = args.first() {
        apply_struct_options(&mut opts, st)?;
        args[1..].to_vec()
    } else {
        args
    };

    if remaining.first().and_then(value_as_string).is_some() {
        apply_property_pairs(&mut opts, &remaining)?;
    } else {
        apply_positional_data(&mut opts, &mut remaining)?;
        apply_property_pairs(&mut opts, &remaining)?;
    }

    let (vertices, faces) = if let (Some(faces), Some(vertices)) = (&opts.faces, &opts.vertices) {
        (vertices_from_tensor(vertices)?, faces_from_tensor(faces)?)
    } else {
        vertices_faces_from_xyz(&opts)?
    };

    let mut plot =
        PatchPlot::new(vertices, faces).map_err(|err| patch_invalid(format!("patch: {err}")))?;
    plot.set_source_data(
        opts.x_data.as_ref().map(patch_data_from_tensor),
        opts.y_data.as_ref().map(patch_data_from_tensor),
        opts.z_data.as_ref().map(patch_data_from_tensor),
        opts.c_data.as_ref().map(patch_data_from_tensor),
    );
    plot.set_face_color(opts.face_color);
    plot.set_edge_color(opts.edge_color);
    plot.set_face_color_mode(opts.face_color_mode);
    plot.set_edge_color_mode(opts.edge_color_mode);
    plot.set_face_alpha(opts.face_alpha);
    plot.set_edge_alpha(opts.edge_alpha);
    plot.set_line_width(opts.line_width);
    plot.set_label(opts.label);
    plot.set_visible(opts.visible);
    Ok(plot)
}

fn apply_positional_data(opts: &mut PatchOptions, args: &mut Vec<Value>) -> BuiltinResult<()> {
    if args.len() < 2 {
        apply_property_pairs(opts, args)?;
        args.clear();
        return Ok(());
    }
    if args.len() >= 3
        && !is_property_name(&args[2])
        && value_matches_coordinate_values(&args[0], &args[1], &args[2])
        && !is_color_literal(&args[2])
    {
        opts.z_data = Some(tensor_from_value(args.remove(2))?);
    }
    opts.x_data = Some(tensor_from_value(args.remove(0))?);
    opts.y_data = Some(tensor_from_value(args.remove(0))?);

    if args.first().is_none_or(is_property_name) {
        return Ok(());
    }

    if opts.z_data.is_some() {
        apply_color_argument(opts, &args.remove(0));
        return Ok(());
    }

    if args.len() >= 2 && !is_property_name(&args[1]) && !is_color_literal(&args[0]) {
        opts.z_data = Some(tensor_from_value(args.remove(0))?);
        apply_color_argument(opts, &args.remove(0));
    } else {
        let value = args.remove(0);
        if value_matches_coordinate_shape(opts, &value) && !is_color_literal(&value)
            || !apply_color_argument(opts, &value)
        {
            opts.z_data = Some(tensor_from_value(value)?);
        }
    }
    Ok(())
}

fn value_matches_coordinate_values(x: &Value, y: &Value, value: &Value) -> bool {
    let (Ok(x), Ok(y), Ok(tensor)) = (
        Tensor::try_from(x),
        Tensor::try_from(y),
        Tensor::try_from(value),
    ) else {
        return false;
    };
    tensor.rows == x.rows && tensor.cols == x.cols && tensor.rows == y.rows && tensor.cols == y.cols
}

fn value_matches_coordinate_shape(opts: &PatchOptions, value: &Value) -> bool {
    let (Some(x), Some(y)) = (&opts.x_data, &opts.y_data) else {
        return false;
    };
    let Ok(tensor) = Tensor::try_from(value) else {
        return false;
    };
    tensor.rows == x.rows && tensor.cols == x.cols && tensor.rows == y.rows && tensor.cols == y.cols
}

fn is_color_literal(value: &Value) -> bool {
    if value_as_string(value).is_some() {
        return parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value).is_ok();
    }
    let Ok(tensor) = Tensor::try_from(value) else {
        return false;
    };
    tensor.rows == 1
        && tensor.cols == 3
        && parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value).is_ok()
}

fn apply_struct_options(opts: &mut PatchOptions, st: &StructValue) -> BuiltinResult<()> {
    for (key, value) in &st.fields {
        apply_property(opts, key, value)?;
    }
    Ok(())
}

fn apply_property_pairs(opts: &mut PatchOptions, args: &[Value]) -> BuiltinResult<()> {
    if args.is_empty() {
        return Ok(());
    }
    if !args.len().is_multiple_of(2) {
        return Err(patch_invalid(
            "patch: property/value arguments must come in pairs",
        ));
    }
    for pair in args.chunks_exact(2) {
        let key = value_as_string(&pair[0])
            .ok_or_else(|| patch_invalid("patch: property names must be strings"))?;
        apply_property(opts, &key, &pair[1])?;
    }
    Ok(())
}

fn apply_property(opts: &mut PatchOptions, key: &str, value: &Value) -> BuiltinResult<()> {
    match key.trim().to_ascii_lowercase().as_str() {
        "xdata" => opts.x_data = Some(tensor_from_value(value.clone())?),
        "ydata" => opts.y_data = Some(tensor_from_value(value.clone())?),
        "zdata" => opts.z_data = Some(tensor_from_value(value.clone())?),
        "faces" => opts.faces = Some(tensor_from_value(value.clone())?),
        "vertices" => opts.vertices = Some(tensor_from_value(value.clone())?),
        "cdata" => apply_c_data(opts, value)?,
        "facecolor" | "color" => apply_face_color(opts, value)?,
        "edgecolor" => apply_edge_color(opts, value)?,
        "facealpha" => {
            let alpha = property_scalar_f64(value)
                .ok_or_else(|| patch_invalid("patch: FaceAlpha must be numeric"))?;
            if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
                return Err(patch_invalid(
                    "patch: FaceAlpha must be in the range [0, 1]",
                ));
            }
            opts.face_alpha = alpha as f32;
        }
        "edgealpha" => {
            let alpha = property_scalar_f64(value)
                .ok_or_else(|| patch_invalid("patch: EdgeAlpha must be numeric"))?;
            if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
                return Err(patch_invalid(
                    "patch: EdgeAlpha must be in the range [0, 1]",
                ));
            }
            opts.edge_alpha = alpha as f32;
        }
        "linewidth" => {
            let width = property_scalar_f64(value)
                .ok_or_else(|| patch_invalid("patch: LineWidth must be numeric"))?;
            if !width.is_finite() || width <= 0.0 {
                return Err(patch_invalid(
                    "patch: LineWidth must be a positive finite value",
                ));
            }
            opts.line_width = width as f32;
        }
        "displayname" => opts.label = value_as_string(value),
        "visible" => {
            opts.visible = visible_value(value)?;
        }
        _ => {
            return Err(patch_invalid(format!(
                "patch: unsupported property `{key}`"
            )))
        }
    }
    Ok(())
}

fn property_scalar_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Tensor(tensor) if tensor.len() != 1 => None,
        _ => value_as_f64(value),
    }
}

fn visible_value(value: &Value) -> BuiltinResult<bool> {
    if let Some(text) = value_as_string(value) {
        return match text.trim().to_ascii_lowercase().as_str() {
            "on" => Ok(true),
            "off" => Ok(false),
            _ => Err(patch_invalid(
                "patch: Visible must be on/off or numeric/logical 0 or 1",
            )),
        };
    }
    let numeric = match value {
        Value::Bool(value) => return Ok(*value),
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor_helpers::tensor_value_f64(tensor, 0),
        _ => {
            return Err(patch_invalid(
                "patch: Visible must be on/off or numeric/logical 0 or 1",
            ))
        }
    };
    match numeric {
        0.0 => Ok(false),
        1.0 => Ok(true),
        _ => Err(patch_invalid(
            "patch: Visible numeric values must be exactly 0 or 1",
        )),
    }
}

fn apply_c_data(opts: &mut PatchOptions, value: &Value) -> BuiltinResult<()> {
    let tensor = tensor_from_value(value.clone())?;
    if tensor.len() == 3 {
        if let Ok(color) = parse_color_value(
            &LineStyleParseOptions::generic(BUILTIN_NAME),
            &Value::Tensor(tensor.clone()),
        ) {
            opts.face_color = color;
            opts.face_color_mode = PatchFaceColorMode::Color;
        } else {
            opts.face_color_mode = PatchFaceColorMode::Flat;
        }
    } else {
        opts.face_color_mode = PatchFaceColorMode::Flat;
    }
    opts.c_data = Some(tensor);
    Ok(())
}

fn patch_data_from_tensor(tensor: &Tensor) -> PatchData {
    let storage = tensor
        .clone()
        .into_numeric_storage()
        .expect("numeric tensor storage");
    PatchData {
        storage,
        shape: tensor.shape.clone(),
    }
}

pub(super) fn split_coordinate_group_columns(
    coordinates: &[Value],
    color: &Value,
    context: &'static str,
) -> BuiltinResult<Vec<(Vec<Value>, Value)>> {
    let coordinates = coordinates
        .iter()
        .cloned()
        .map(|value| numeric_tensor_for_plot(value, context))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let Some(first) = coordinates.first() else {
        return Err(plotting_error(
            context,
            format!("{context}: missing coordinates"),
        ));
    };
    // MATLAB treats row and column vectors as the same logical polygon axis.
    // Once any true matrix is present, its rows define polygon length and its
    // columns define independent Patch objects; a matching-length vector is
    // shared without changing its authoritative orientation. This also makes
    // square matrices deterministically column-grouped.
    let matrix = coordinates.iter().find(|tensor| !is_vector_tensor(tensor));
    let (polygon_length, polygon_count) = if let Some(matrix) = matrix {
        for tensor in &coordinates {
            if is_vector_tensor(tensor) {
                if tensor.len() != matrix.rows {
                    return Err(plotting_error(
                        context,
                        format!(
                            "{context}: shared coordinate vectors must match the matrix row count"
                        ),
                    ));
                }
            } else if tensor.rows != matrix.rows || tensor.cols != matrix.cols {
                return Err(plotting_error(
                    context,
                    format!("{context}: coordinate matrices must have the same size"),
                ));
            }
        }
        (matrix.rows, matrix.cols)
    } else {
        if coordinates.iter().any(|tensor| tensor.len() != first.len()) {
            return Err(plotting_error(
                context,
                format!("{context}: coordinate vectors must have the same length"),
            ));
        }
        (first.len(), 1)
    };

    let literal_color =
        matches!(color, Value::String(_) | Value::CharArray(_)).then(|| color.clone());
    let color = if literal_color.is_some() {
        None
    } else {
        Some(numeric_tensor_for_plot(color.clone(), context)?)
    };
    let split_color = color.as_ref().is_some_and(|tensor| {
        let rgb_triplet = is_rgb_triplet_tensor(tensor);
        polygon_count > 1
            && tensor.cols == polygon_count
            && (tensor.rows == polygon_length || tensor.rows == 1)
            && !rgb_triplet
    });

    (0..polygon_count)
        .map(|column| {
            let coordinates = coordinates
                .iter()
                .map(|tensor| {
                    if polygon_count == 1 || is_vector_tensor(tensor) {
                        Ok(Value::Tensor(tensor.clone()))
                    } else {
                        tensor_column_value(tensor, column, context)
                    }
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            let color = match &color {
                Some(tensor) if split_color => tensor_column_value(tensor, column, context)?,
                Some(tensor) => Value::Tensor(tensor.clone()),
                None => literal_color
                    .clone()
                    .expect("nonnumeric color is retained exactly"),
            };
            Ok((coordinates, color))
        })
        .collect()
}

fn is_rgb_triplet_tensor(tensor: &Tensor) -> bool {
    if tensor.len() != 3 {
        return false;
    }
    match tensor.clone().into_numeric_storage() {
        Ok(NumericStorage::F64(values)) => values
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
        Ok(NumericStorage::F32(values)) => values
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
        Ok(NumericStorage::I8(values)) => values.iter().all(|value| (0..=1).contains(value)),
        Ok(NumericStorage::I16(values)) => values.iter().all(|value| (0..=1).contains(value)),
        Ok(NumericStorage::I32(values)) => values.iter().all(|value| (0..=1).contains(value)),
        Ok(NumericStorage::I64(values)) => values.iter().all(|value| (0..=1).contains(value)),
        Ok(NumericStorage::U8(values)) => values.iter().all(|value| *value <= 1),
        Ok(NumericStorage::U16(values)) => values.iter().all(|value| *value <= 1),
        Ok(NumericStorage::U32(values)) => values.iter().all(|value| *value <= 1),
        Ok(NumericStorage::U64(values)) => values.iter().all(|value| *value <= 1),
        Err(_) => false,
    }
}

fn numeric_tensor_for_plot(value: Value, context: &'static str) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => gather_tensor_from_gpu(handle, context),
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|error| plotting_error(context, format!("{context}: {error}"))),
        Value::Int(value) => Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])
            .map_err(|error| plotting_error(context, format!("{context}: {error}"))),
        Value::Tensor(tensor) => Ok(tensor),
        other => Err(plotting_error(
            context,
            format!("{context}: expected numeric data, got {other:?}"),
        )),
    }
}

fn tensor_column_value(
    tensor: &Tensor,
    column: usize,
    context: &'static str,
) -> BuiltinResult<Value> {
    let start = column
        .checked_mul(tensor.rows)
        .ok_or_else(|| plotting_error(context, format!("{context}: column offset overflow")))?;
    let end = start + tensor.rows;
    macro_rules! column_storage {
        ($storage:expr, $variant:ident) => {
            NumericStorage::$variant($storage[start..end].to_vec())
        };
    }
    let storage = match tensor
        .clone()
        .into_numeric_storage()
        .map_err(|error| plotting_error(context, format!("{context}: {error}")))?
    {
        NumericStorage::F64(values) => column_storage!(values, F64),
        NumericStorage::F32(values) => column_storage!(values, F32),
        NumericStorage::I8(values) => column_storage!(values, I8),
        NumericStorage::I16(values) => column_storage!(values, I16),
        NumericStorage::I32(values) => column_storage!(values, I32),
        NumericStorage::I64(values) => column_storage!(values, I64),
        NumericStorage::U8(values) => column_storage!(values, U8),
        NumericStorage::U16(values) => column_storage!(values, U16),
        NumericStorage::U32(values) => column_storage!(values, U32),
        NumericStorage::U64(values) => column_storage!(values, U64),
    };
    Tensor::from_numeric_storage(storage, vec![tensor.rows, 1])
        .map(Value::Tensor)
        .map_err(|error| plotting_error(context, format!("{context}: {error}")))
}

fn tensor_from_value(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => gather_tensor_from_gpu(handle, BUILTIN_NAME),
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| patch_invalid(format!("patch: {err}"))),
        Value::Int(value) => {
            Tensor::new_integer(runmat_value::IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|err| patch_invalid(format!("patch: {err}")))
        }
        other => Tensor::try_from(&other).map_err(|err| patch_invalid(format!("patch: {err}"))),
    }
}

fn apply_color_argument(opts: &mut PatchOptions, value: &Value) -> bool {
    if let Ok(color) = parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value) {
        opts.face_color = color;
        opts.face_color_mode = PatchFaceColorMode::Color;
        return true;
    }
    false
}

fn apply_face_color(opts: &mut PatchOptions, value: &Value) -> BuiltinResult<()> {
    if let Some(text) = value_as_string(value) {
        match text.trim().to_ascii_lowercase().as_str() {
            "none" => {
                opts.face_color_mode = PatchFaceColorMode::None;
                return Ok(());
            }
            "flat" | "interp" => {
                opts.face_color_mode = PatchFaceColorMode::Flat;
                return Ok(());
            }
            _ => {}
        }
    }
    opts.face_color = parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value)?;
    opts.face_color_mode = PatchFaceColorMode::Color;
    Ok(())
}

fn apply_edge_color(opts: &mut PatchOptions, value: &Value) -> BuiltinResult<()> {
    if let Some(text) = value_as_string(value) {
        if text.trim().eq_ignore_ascii_case("none") {
            opts.edge_color_mode = PatchEdgeColorMode::None;
            return Ok(());
        }
    }
    opts.edge_color = parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value)?;
    opts.edge_color_mode = PatchEdgeColorMode::Color;
    Ok(())
}

fn vertices_from_tensor(tensor: &Tensor) -> BuiltinResult<Vec<Vec3>> {
    if tensor.cols != 2 && tensor.cols != 3 {
        return Err(patch_invalid(
            "patch: Vertices must be an N-by-2 or N-by-3 matrix",
        ));
    }
    let values = tensor_helpers::tensor_values_f64(tensor);
    let mut out = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let x = values[row];
        let y = values[row + tensor.rows];
        let z = if tensor.cols >= 3 {
            values[row + 2 * tensor.rows]
        } else {
            0.0
        };
        out.push(Vec3::new(x as f32, y as f32, z as f32));
    }
    Ok(out)
}

fn faces_from_tensor(tensor: &Tensor) -> BuiltinResult<Vec<Vec<usize>>> {
    if tensor.rows == 0 || tensor.cols == 0 {
        return Err(patch_invalid("patch: Faces must not be empty"));
    }
    let values = tensor_helpers::tensor_values_f64(tensor);
    let mut faces = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let mut face = Vec::new();
        for col in 0..tensor.cols {
            let value = values[row + col * tensor.rows];
            if value.is_nan() {
                continue;
            }
            if value < 1.0 || value.fract() != 0.0 {
                return Err(patch_invalid(
                    "patch: Faces must contain positive integer vertex indices",
                ));
            }
            face.push(value as usize - 1);
        }
        if face.len() >= 3 {
            faces.push(face);
        }
    }
    Ok(faces)
}

fn vertices_faces_from_xyz(opts: &PatchOptions) -> BuiltinResult<(Vec<Vec3>, Vec<Vec<usize>>)> {
    let x = opts
        .x_data
        .as_ref()
        .ok_or_else(|| patch_invalid("patch: missing XData"))?;
    let y = opts
        .y_data
        .as_ref()
        .ok_or_else(|| patch_invalid("patch: missing YData"))?;
    let all_vectors = is_vector_tensor(x)
        && is_vector_tensor(y)
        && opts.z_data.as_ref().is_none_or(is_vector_tensor);
    if all_vectors {
        let x_values = tensor_helpers::tensor_values_f64(x);
        let y_values = tensor_helpers::tensor_values_f64(y);
        let z_values = opts.z_data.as_ref().map(tensor_helpers::tensor_values_f64);
        if x_values.len() != y_values.len()
            || z_values
                .as_ref()
                .map(|z| z.len() != x_values.len())
                .unwrap_or(false)
        {
            return Err(patch_invalid(
                "patch: vector XData, YData, and ZData must have the same length",
            ));
        }
        let mut vertices = Vec::new();
        let mut face = Vec::new();
        for idx in 0..x_values.len() {
            let xv = x_values[idx];
            let yv = y_values[idx];
            let zv = z_values.as_ref().map(|z| z[idx]).unwrap_or(0.0);
            if xv.is_nan() || yv.is_nan() || zv.is_nan() {
                continue;
            }
            face.push(vertices.len());
            vertices.push(Vec3::new(xv as f32, yv as f32, zv as f32));
        }
        return Ok((vertices, vec![face]));
    }
    if x.rows != y.rows || x.cols != y.cols {
        return Err(patch_invalid(
            "patch: XData and YData must have the same size",
        ));
    }
    if let Some(z) = &opts.z_data {
        if z.rows != x.rows || z.cols != x.cols {
            return Err(patch_invalid(
                "patch: ZData must have the same size as XData and YData",
            ));
        }
    }
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let x_values = tensor_helpers::tensor_values_f64(x);
    let y_values = tensor_helpers::tensor_values_f64(y);
    let z_values = opts.z_data.as_ref().map(tensor_helpers::tensor_values_f64);
    for col in 0..x.cols {
        let mut face = Vec::new();
        for row in 0..x.rows {
            let idx = row + col * x.rows;
            let xv = x_values[idx];
            let yv = y_values[idx];
            let zv = z_values.as_ref().map(|z| z[idx]).unwrap_or(0.0);
            if xv.is_nan() || yv.is_nan() || zv.is_nan() {
                continue;
            }
            face.push(vertices.len());
            vertices.push(Vec3::new(xv as f32, yv as f32, zv as f32));
        }
        if face.len() >= 3 {
            faces.push(face);
        }
    }
    Ok((vertices, faces))
}

fn is_vector_tensor(tensor: &Tensor) -> bool {
    tensor.rows == 1 || tensor.cols == 1
}

pub(super) fn is_property_name(value: &Value) -> bool {
    value_as_string(value)
        .map(|name| {
            matches!(
                name.trim().to_ascii_lowercase().as_str(),
                "xdata"
                    | "ydata"
                    | "zdata"
                    | "faces"
                    | "vertices"
                    | "cdata"
                    | "facecolor"
                    | "edgecolor"
                    | "facealpha"
                    | "edgealpha"
                    | "linewidth"
                    | "displayname"
                    | "visible"
                    | "color"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_value::IntegerStorage;

    fn tensor(rows: usize, cols: usize, data: &[f64]) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), vec![rows, cols]).expect("patch test tensor"))
    }

    fn int_tensor(rows: usize, cols: usize, storage: IntegerStorage) -> Value {
        let tensor = Tensor::new_integer(storage, vec![rows, cols]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    fn setup_plot_test() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn patch_xyc_vector_builds_single_polygon() {
        let plot = parse_patch_plot(vec![
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            Value::String("r".into()),
        ])
        .unwrap();
        assert_eq!(plot.faces().len(), 1);
        assert_eq!(plot.vertices().len(), 3);
        assert_eq!(plot.face_color(), Vec4::new(1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn patch_xy_rgb_row_vector_treats_third_argument_as_color() {
        let plot = parse_patch_plot(vec![
            tensor(1, 3, &[0.0, 1.0, 0.0]),
            tensor(1, 3, &[0.0, 0.0, 1.0]),
            tensor(1, 3, &[1.0, 0.0, 0.0]),
        ])
        .unwrap();
        assert_eq!(plot.vertices().len(), 3);
        assert!(plot.vertices().iter().all(|vertex| vertex.z == 0.0));
        assert_eq!(plot.face_color(), Vec4::new(1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn patch_matrix_columns_build_multiple_polygons() {
        let plot = parse_patch_plot(vec![
            tensor(4, 2, &[0.0, 1.0, 1.0, 0.0, 2.0, 3.0, 3.0, 2.0]),
            tensor(4, 2, &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
            Value::String("g".into()),
        ])
        .unwrap();
        assert_eq!(plot.faces().len(), 2);
        assert_eq!(plot.vertices().len(), 8);
    }

    #[test]
    fn patch_xyz_without_color_preserves_z_data() {
        let plot = parse_patch_plot(vec![
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            tensor(3, 1, &[0.25, 0.5, 0.75]),
        ])
        .unwrap();
        assert_eq!(plot.vertices().len(), 3);
        assert_eq!(plot.vertices()[0].z, 0.25);
        assert_eq!(plot.vertices()[1].z, 0.5);
        assert_eq!(plot.vertices()[2].z, 0.75);
    }

    #[test]
    fn patch_xyz_vectors_read_typed_integer_storage_exactly() {
        let plot = parse_patch_plot(vec![
            int_tensor(3, 1, IntegerStorage::I16(vec![0, 1, 0])),
            int_tensor(3, 1, IntegerStorage::I16(vec![0, 0, 1])),
            int_tensor(3, 1, IntegerStorage::I16(vec![2, 3, 4])),
        ])
        .unwrap();
        assert_eq!(plot.faces(), &[vec![0, 1, 2]]);
        assert_eq!(plot.vertices()[0], Vec3::new(0.0, 0.0, 2.0));
        assert_eq!(plot.vertices()[1], Vec3::new(1.0, 0.0, 3.0));
        assert_eq!(plot.vertices()[2], Vec3::new(0.0, 1.0, 4.0));
    }

    #[test]
    fn patch_xyz_accepts_trailing_name_value_pairs() {
        let plot = parse_patch_plot(vec![
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            tensor(3, 1, &[0.25, 0.5, 0.75]),
            Value::String("FaceColor".into()),
            Value::String("r".into()),
        ])
        .unwrap();
        assert_eq!(plot.vertices()[0].z, 0.25);
        assert_eq!(plot.vertices()[1].z, 0.5);
        assert_eq!(plot.vertices()[2].z, 0.75);
        assert_eq!(plot.face_color(), Vec4::new(1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn patch_xy_accepts_trailing_name_value_pairs() {
        let plot = parse_patch_plot(vec![
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            Value::String("FaceColor".into()),
            Value::String("r".into()),
            Value::String("EdgeColor".into()),
            Value::String("none".into()),
        ])
        .unwrap();
        assert_eq!(plot.face_color(), Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(plot.edge_color_mode(), PatchEdgeColorMode::None);
    }

    #[test]
    fn patch_faces_vertices_uses_one_based_faces() {
        let plot = parse_patch_plot(vec![
            Value::String("Faces".into()),
            tensor(1, 3, &[1.0, 2.0, 3.0]),
            Value::String("Vertices".into()),
            tensor(3, 2, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            Value::String("EdgeColor".into()),
            Value::String("none".into()),
        ])
        .unwrap();
        assert_eq!(plot.faces(), &[vec![0, 1, 2]]);
        assert_eq!(plot.edge_color_mode(), PatchEdgeColorMode::None);
    }

    #[test]
    fn patch_faces_vertices_read_typed_integer_storage_exactly() {
        let plot = parse_patch_plot(vec![
            Value::String("Faces".into()),
            int_tensor(1, 3, IntegerStorage::I16(vec![1, 2, 3])),
            Value::String("Vertices".into()),
            int_tensor(3, 2, IntegerStorage::I16(vec![0, 1, 0, 0, 0, 1])),
        ])
        .unwrap();
        assert_eq!(plot.faces(), &[vec![0, 1, 2]]);
        assert_eq!(plot.vertices()[0], Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(plot.vertices()[1], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(plot.vertices()[2], Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn patch_rejects_invalid_integer_relevant_properties_without_clamping() {
        let base = || {
            vec![
                Value::String("XData".into()),
                tensor(3, 1, &[0.0, 1.0, 0.0]),
                Value::String("YData".into()),
                tensor(3, 1, &[0.0, 0.0, 1.0]),
            ]
        };
        for (name, value) in [
            ("FaceAlpha", Value::Num(2.0)),
            ("EdgeAlpha", Value::Num(-0.5)),
            ("LineWidth", Value::Int(runmat_value::IntValue::I16(0))),
            ("Visible", Value::Int(runmat_value::IntValue::U8(2))),
            ("FutureProperty", Value::Num(1.0)),
        ] {
            let mut args = base();
            args.push(Value::String(name.into()));
            args.push(value);
            assert!(parse_patch_plot(args).is_err(), "{name} must be rejected");
        }
    }

    #[test]
    fn patch_registers_as_dispatch_builtin_and_returns_handle() {
        let _guard = setup_plot_test();
        let handle = crate::call_builtin(
            "patch",
            &[
                tensor(3, 1, &[0.0, 1.0, 0.0]),
                tensor(3, 1, &[0.0, 0.0, 1.0]),
                Value::String("b".into()),
            ],
        )
        .expect("patch builtin should dispatch");
        let Value::Num(handle) = handle else {
            panic!("expected numeric graphics handle");
        };
        let ty = crate::call_builtin("get", &[Value::Num(handle), Value::String("Type".into())])
            .expect("get patch type");
        assert_eq!(ty, Value::String("patch".into()));
    }

    #[test]
    fn patch_get_visible_tracks_set_visible() {
        let _guard = setup_plot_test();
        let handle = crate::call_builtin(
            "patch",
            &[
                tensor(3, 1, &[0.0, 1.0, 0.0]),
                tensor(3, 1, &[0.0, 0.0, 1.0]),
                Value::String("b".into()),
            ],
        )
        .expect("patch builtin should dispatch");
        let Value::Num(handle) = handle else {
            panic!("expected numeric graphics handle");
        };
        crate::call_builtin(
            "set",
            &[
                Value::Num(handle),
                Value::String("Visible".into()),
                Value::Bool(false),
            ],
        )
        .expect("set patch visible");

        let visible = crate::call_builtin(
            "get",
            &[Value::Num(handle), Value::String("Visible".into())],
        )
        .expect("get patch visible");
        assert_eq!(visible, Value::Bool(false));

        let all = crate::call_builtin("get", &[Value::Num(handle)]).expect("get patch struct");
        let Value::Struct(st) = all else {
            panic!("expected patch property struct");
        };
        assert_eq!(st.fields.get("Visible"), Some(&Value::Bool(false)));
    }

    #[test]
    fn patch_descriptor_includes_core_signatures() {
        let labels: Vec<&str> = PATCH_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = patch(X, Y, C)"));
        assert!(labels.contains(&"h = patch(S)"));
        assert!(labels.contains(&"h = patch(ax, ...)"));
    }

    #[test]
    fn patch_missing_input_uses_stable_identifier() {
        let err = patch_builtin(Vec::new()).expect_err("expected patch argument validation error");
        assert_eq!(err.identifier(), PATCH_ERROR_INVALID_ARGUMENT.identifier);
    }
}
