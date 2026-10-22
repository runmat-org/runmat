//! MATLAB-compatible `patch` builtin.

use glam::{Vec3, Vec4};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{NumericPlotData, PatchEdgeColorMode, PatchFaceColorMode, PatchPlot};
use runmat_value::{IntegerStorage, NumericStorage, StructValue, Tensor, Value};

use crate::builtins::common::tensor as tensor_helpers;
use crate::builtins::plotting::plotting_error;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::common::gather_tensor_from_gpu;
use super::op_common::{apply_axes_target, split_leading_axes_arguments};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_color_value, value_as_f64, value_as_string, LineStyleParseOptions};

const BUILTIN_NAME: &str = "patch";
#[cfg(test)]
use runmat_builtins::PATCH_DESCRIPTOR;
use runmat_builtins::{
    PATCH_ERROR_INTERNAL, PATCH_ERROR_INVALID_ARGUMENT, PATCH_INTEGER_AXES_EXTENSION,
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

#[derive(Clone, Debug)]
struct PatchOptions {
    x_data: Option<Tensor>,
    y_data: Option<Tensor>,
    z_data: Option<Tensor>,
    faces: Option<Tensor>,
    vertices: Option<Tensor>,
    c_data: Option<Tensor>,
    c_data_kind: runmat_plot::plots::PatchColorSourceKind,
    face_color: Vec4,
    edge_color: Vec4,
    face_color_mode: PatchFaceColorMode,
    face_color_explicit: bool,
    color_mapping: runmat_plot::plots::surface::SurfaceColorMapping,
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
            c_data_kind: runmat_plot::plots::PatchColorSourceKind::FaceVertex,
            face_color: Vec4::new(0.0, 0.447, 0.741, 1.0),
            edge_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            face_color_mode: PatchFaceColorMode::Color,
            face_color_explicit: false,
            color_mapping: runmat_plot::plots::surface::SurfaceColorMapping::Scaled,
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
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::patch"
)]
pub fn patch_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<f64> {
    gate_typed_integer_axes_alias(args.view())?;
    let (axes_target, args) =
        split_leading_axes_arguments(args, BUILTIN_NAME).map_err(map_patch_invalid)?;
    let mut plot = Some(parse_patch_plot_view(args.view()).map_err(map_patch_invalid)?);
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_patch_invalid)?;
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions::default(),
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
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(map_patch_internal(err));
    }
    Ok(handle)
}

fn gate_typed_integer_axes_alias(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    let Some(first) = args.first() else {
        return Ok(());
    };
    let typed_integer = matches!(first, Value::Int(_))
        || matches!(first, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(first, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
    if typed_integer && super::properties::resolve_plot_handle(first, BUILTIN_NAME).is_ok() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PATCH_INTEGER_AXES_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

pub(super) fn parse_patch_plot(args: Vec<Value>) -> BuiltinResult<PatchPlot> {
    parse_patch_plot_view(crate::call::FunctionArgumentView::from_slice(&args))
}

fn parse_patch_plot_view(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<PatchPlot> {
    if args.is_empty() {
        return Err(patch_invalid("patch: expected input data"));
    }
    let mut opts = PatchOptions::default();
    let mut remaining = if let Some(Value::Struct(st)) = args.first() {
        apply_struct_options(&mut opts, st)?;
        args.range(1..).expect("validated patch argument suffix")
    } else {
        args
    };

    if remaining
        .first()
        .is_some_and(|value| matches!(value, Value::String(_) | Value::CharArray(_)))
    {
        apply_property_pairs(&mut opts, remaining)?;
    } else {
        apply_positional_data(&mut opts, &mut remaining)?;
        apply_property_pairs(&mut opts, remaining)?;
    }

    if opts.c_data_kind == runmat_plot::plots::PatchColorSourceKind::CoordinateGrid
        && (opts.x_data.is_none()
            || opts.y_data.is_none()
            || opts.c_data.as_ref().is_some_and(|data| data.len() == 1))
    {
        opts.c_data_kind = runmat_plot::plots::PatchColorSourceKind::FaceVertex;
    }

    let (vertices, faces) = if let (Some(faces), Some(vertices)) = (&opts.faces, &opts.vertices) {
        (vertices_from_tensor(vertices)?, faces_from_tensor(faces)?)
    } else {
        vertices_faces_from_xyz(&opts)?
    };

    let mut plot =
        PatchPlot::new(vertices, faces).map_err(|err| patch_invalid(format!("patch: {err}")))?;
    plot.set_source_data(runmat_plot::plots::PatchSourceDataOwned {
        x: opts.x_data.as_ref().map(patch_data_from_tensor),
        y: opts.y_data.as_ref().map(patch_data_from_tensor),
        z: opts.z_data.as_ref().map(patch_data_from_tensor),
        colors: opts.c_data.as_ref().map(patch_data_from_tensor),
        color_kind: opts.c_data_kind,
        faces: opts.faces.as_ref().map(patch_data_from_tensor),
        vertices: opts.vertices.as_ref().map(patch_data_from_tensor),
    })
    .map_err(patch_invalid)?;
    plot.set_face_color(opts.face_color);
    plot.set_edge_color(opts.edge_color);
    plot.set_face_color_mode(opts.face_color_mode);
    plot.validate_color_mode().map_err(patch_invalid)?;
    plot.set_color_mapping(opts.color_mapping);
    plot.set_edge_color_mode(opts.edge_color_mode);
    plot.set_face_alpha(opts.face_alpha);
    plot.set_edge_alpha(opts.edge_alpha);
    plot.set_line_width(opts.line_width);
    plot.set_label(opts.label);
    plot.set_visible(opts.visible);
    Ok(plot)
}

fn apply_positional_data(
    opts: &mut PatchOptions,
    args: &mut crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if args.len() < 2 {
        apply_property_pairs(opts, *args)?;
        *args = args
            .range(args.len()..)
            .expect("bounded patch argument tail");
        return Ok(());
    }
    opts.x_data = Some(tensor_from_value(args[0].clone())?);
    opts.y_data = Some(tensor_from_value(args[1].clone())?);
    *args = args.range(2..).expect("validated patch coordinate prefix");
    if args.first().is_none_or(is_property_name) {
        return Ok(());
    }
    // The three-data form is patch(X,Y,C). Z is positional only in the
    // four-data form patch(X,Y,Z,C); a three-dimensional uncolored patch
    // uses the explicit XData/YData/ZData properties.
    if args.len() >= 2 && !is_property_name(&args[1]) {
        opts.z_data = Some(tensor_from_value(args[0].clone())?);
        *args = args.range(1..).expect("validated patch z-data prefix");
    }
    apply_color_argument(opts, &args[0])?;
    *args = args.range(1..).expect("validated patch color prefix");
    Ok(())
}

pub(super) fn is_color_literal(value: &Value) -> crate::BuiltinResult<bool> {
    Ok({
        if value_as_string(value)
            .map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "patch",
                    source.to_string(),
                    source,
                )
            })?
            .is_some()
        {
            return Ok(
                parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value).is_ok(),
            );
        }
        let Ok(tensor) = Tensor::try_from(value) else {
            return Ok(false);
        };
        patch_tensor_dimensions(tensor.shape()) == (1, 3)
            && parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value).is_ok()
    })
}

fn apply_struct_options(opts: &mut PatchOptions, st: &StructValue) -> BuiltinResult<()> {
    for (key, value) in &st.fields {
        apply_property(opts, key, value)?;
    }
    Ok(())
}

fn apply_property_pairs(
    opts: &mut PatchOptions,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
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
            .map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    BUILTIN_NAME,
                    source.to_string(),
                    source,
                )
            })?
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
        "cdata" => apply_c_data(
            opts,
            value,
            runmat_plot::plots::PatchColorSourceKind::CoordinateGrid,
        )?,
        "facevertexcdata" => apply_c_data(
            opts,
            value,
            runmat_plot::plots::PatchColorSourceKind::FaceVertex,
        )?,
        "cdatamapping" => {
            opts.color_mapping = super::properties::parse_color_data_mapping(value, BUILTIN_NAME)?;
        }
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
        "displayname" => {
            opts.label = value_as_string(value).map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    BUILTIN_NAME,
                    source.to_string(),
                    source,
                )
            })?
        }
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
    if let Some(text) = value_as_string(value).map_err(|source| {
        crate::builtins::plotting::plotting_error_with_source("patch", source.to_string(), source)
    })? {
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

fn apply_c_data(
    opts: &mut PatchOptions,
    value: &Value,
    kind: runmat_plot::plots::PatchColorSourceKind,
) -> BuiltinResult<()> {
    let tensor = tensor_from_value(value.clone())?;
    if tensor.is_empty() {
        opts.c_data = None;
        return Ok(());
    }
    if !opts.face_color_explicit {
        opts.face_color_mode = PatchFaceColorMode::Flat;
    }
    opts.c_data = Some(tensor);
    opts.c_data_kind = kind;
    Ok(())
}

fn patch_data_from_tensor(tensor: &Tensor) -> NumericPlotData {
    let storage = tensor
        .clone()
        .into_numeric_storage()
        .expect("numeric tensor storage");
    NumericPlotData::new(storage, tensor.shape().to_vec()).expect("validated patch source data")
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
        let matrix_dimensions = patch_tensor_dimensions(matrix.shape());
        for tensor in &coordinates {
            if is_vector_tensor(tensor) {
                if tensor.len() != matrix_dimensions.0 {
                    return Err(plotting_error(
                        context,
                        format!(
                            "{context}: shared coordinate vectors must match the matrix row count"
                        ),
                    ));
                }
            } else if patch_tensor_dimensions(tensor.shape()) != matrix_dimensions {
                return Err(plotting_error(
                    context,
                    format!("{context}: coordinate matrices must have the same size"),
                ));
            }
        }
        matrix_dimensions
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
        let (rows, cols) = patch_tensor_dimensions(tensor.shape());
        let rgb_triplet = is_rgb_triplet_tensor(tensor);
        polygon_count > 1
            && cols == polygon_count
            && (rows == polygon_length || rows == 1)
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
    let rows = patch_tensor_dimensions(tensor.shape()).0;
    let start = column
        .checked_mul(rows)
        .ok_or_else(|| plotting_error(context, format!("{context}: column offset overflow")))?;
    let end = start + rows;
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
    Tensor::from_numeric_storage(storage, vec![rows, 1])
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

fn apply_color_argument(opts: &mut PatchOptions, value: &Value) -> BuiltinResult<()> {
    if is_color_literal(value)? {
        let color = parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value)?;
        opts.face_color = color;
        opts.face_color_mode = PatchFaceColorMode::Color;
        return Ok(());
    }
    apply_c_data(
        opts,
        value,
        runmat_plot::plots::PatchColorSourceKind::CoordinateGrid,
    )
}

fn apply_face_color(opts: &mut PatchOptions, value: &Value) -> BuiltinResult<()> {
    if let Some(text) = value_as_string(value).map_err(|source| {
        crate::builtins::plotting::plotting_error_with_source(
            BUILTIN_NAME,
            source.to_string(),
            source,
        )
    })? {
        match text.trim().to_ascii_lowercase().as_str() {
            "none" => {
                opts.face_color_mode = PatchFaceColorMode::None;
                opts.face_color_explicit = true;
                return Ok(());
            }
            "flat" => {
                opts.face_color_mode = PatchFaceColorMode::Flat;
                opts.face_color_explicit = true;
                return Ok(());
            }
            "interp" => {
                opts.face_color_mode = PatchFaceColorMode::Interp;
                opts.face_color_explicit = true;
                return Ok(());
            }
            _ => {}
        }
    }
    opts.face_color = parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value)?;
    opts.face_color_mode = PatchFaceColorMode::Color;
    opts.face_color_explicit = true;
    Ok(())
}

fn apply_edge_color(opts: &mut PatchOptions, value: &Value) -> BuiltinResult<()> {
    if let Some(text) = value_as_string(value).map_err(|source| {
        crate::builtins::plotting::plotting_error_with_source(
            BUILTIN_NAME,
            source.to_string(),
            source,
        )
    })? {
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
    let (rows, cols) = patch_tensor_dimensions(tensor.shape());
    if cols != 2 && cols != 3 {
        return Err(patch_invalid(
            "patch: Vertices must be an N-by-2 or N-by-3 matrix",
        ));
    }
    let values = tensor_helpers::tensor_values_f64(tensor);
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let x = values[row];
        let y = values[row + rows];
        let z = if cols >= 3 {
            values[row + 2 * rows]
        } else {
            0.0
        };
        out.push(Vec3::new(x as f32, y as f32, z as f32));
    }
    Ok(out)
}

fn faces_from_tensor(tensor: &Tensor) -> BuiltinResult<Vec<Vec<usize>>> {
    let (rows, cols) = patch_tensor_dimensions(tensor.shape());
    if rows == 0 || cols == 0 {
        return Err(patch_invalid("patch: Faces must not be empty"));
    }
    let values = tensor_helpers::tensor_values_f64(tensor);
    let mut faces = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut face = Vec::new();
        for col in 0..cols {
            let value = values[row + col * rows];
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
    let (rows, cols) = patch_tensor_dimensions(x.shape());
    if patch_tensor_dimensions(y.shape()) != (rows, cols) {
        return Err(patch_invalid(
            "patch: XData and YData must have the same size",
        ));
    }
    if let Some(z) = &opts.z_data {
        if patch_tensor_dimensions(z.shape()) != (rows, cols) {
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
    for col in 0..cols {
        let mut face = Vec::new();
        for row in 0..rows {
            let idx = row + col * rows;
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
    let (rows, cols) = patch_tensor_dimensions(tensor.shape());
    rows == 1 || cols == 1
}

/// Matrix geometry used by patch coordinates and indexed face data.
///
/// Patch treats rank-one numeric inputs as row vectors and rank-zero inputs as
/// empty geometry; this projection is distinct from Tensor's general matrix
/// accessors and is part of the plotting admission contract.
fn patch_tensor_dimensions(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (0, 0),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    }
}

pub(super) fn is_property_name(value: &Value) -> bool {
    [
        "xdata",
        "ydata",
        "zdata",
        "faces",
        "vertices",
        "cdata",
        "facevertexcdata",
        "cdatamapping",
        "facecolor",
        "edgecolor",
        "facealpha",
        "edgealpha",
        "linewidth",
        "displayname",
        "visible",
        "color",
    ]
    .into_iter()
    .any(|key| crate::builtins::plotting::style::value_text_equals_ascii(value, key, true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
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
    fn explicit_interpolated_face_color_survives_face_vertex_data_order() {
        let data = tensor(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let geometry = [
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
        ];
        for data_first in [false, true] {
            let mut args = geometry.to_vec();
            let colors = [Value::String("FaceVertexCData".into()), data.clone()];
            let mode = [
                Value::String("FaceColor".into()),
                Value::String("interp".into()),
            ];
            if data_first {
                args.extend(colors);
                args.extend(mode);
            } else {
                args.extend(mode);
                args.extend(colors);
            }
            let mut plot = parse_patch_plot(args).expect("interpolated vertex colors");
            assert_eq!(plot.face_color_mode(), PatchFaceColorMode::Interp);
            let vertices = plot.render_data().vertices;
            assert_ne!(vertices[0].color, vertices[1].color);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn patch_rank_geometry_keeps_row_vectors_and_empty_rank_zero_data() {
        assert_eq!(patch_tensor_dimensions(&[]), (0, 0));
        assert_eq!(patch_tensor_dimensions(&[3]), (1, 3));

        let x = Tensor::new(vec![0.0, 1.0, 0.0], vec![3]).expect("rank-one x");
        let y = Tensor::new(vec![0.0, 0.0, 1.0], vec![3]).expect("rank-one y");
        let color = Tensor::new(vec![1.0, 0.0, 0.0], vec![3]).expect("rank-one color");
        let plot = parse_patch_plot(vec![
            Value::Tensor(x),
            Value::Tensor(y),
            Value::Tensor(color),
        ])
        .expect("rank-one coordinates and RGB row");
        assert_eq!(plot.vertices().len(), 3);
        assert!(plot.vertices().iter().all(|vertex| vertex.z == 0.0));
        assert_eq!(plot.face_color(), Vec4::new(1.0, 0.0, 0.0, 1.0));

        let rank_zero = Tensor::new(vec![1.0], vec![]).expect("rank-zero faces");
        let error = faces_from_tensor(&rank_zero).expect_err("rank-zero faces are empty geometry");
        assert_eq!(error.identifier(), PATCH_ERROR_INVALID_ARGUMENT.identifier);
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
    fn patch_three_data_arguments_are_coordinate_colors_not_z() {
        let plot = parse_patch_plot(vec![
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            tensor(3, 1, &[1.0, 2.0, 3.0]),
            Value::String("FaceColor".into()),
            Value::String("interp".into()),
        ])
        .expect("coordinate indexed colors");
        assert!(plot.vertices().iter().all(|vertex| vertex.z == 0.0));
        assert_eq!(
            plot.color_source_kind(),
            runmat_plot::plots::PatchColorSourceKind::CoordinateGrid
        );
        let (_, _, z, c, _, _) = plot.source_data();
        assert!(z.is_none());
        assert_eq!(c.expect("coordinate colors").shape(), &[3, 1]);

        let mixed_orientation = parse_patch_plot(vec![
            tensor(1, 3, &[0.0, 1.0, 0.0]),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            tensor(3, 1, &[1.0, 2.0, 3.0]),
        ])
        .expect("vector coordinate orientations preserve color pairing");
        assert_eq!(mixed_orientation.vertices().len(), 3);
        assert_eq!(
            mixed_orientation.color_source_kind(),
            plot.color_source_kind()
        );
    }

    #[test]
    fn patch_xyz_properties_preserve_z_data() {
        let plot = parse_patch_plot(vec![
            Value::String("XData".into()),
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            Value::String("YData".into()),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            Value::String("ZData".into()),
            tensor(3, 1, &[0.25, 0.5, 0.75]),
        ])
        .unwrap();
        assert_eq!(plot.vertices().len(), 3);
        assert_eq!(plot.vertices()[0].z, 0.25);
        assert_eq!(plot.vertices()[1].z, 0.5);
        assert_eq!(plot.vertices()[2].z, 0.75);
    }

    #[test]
    fn patch_xyz_properties_read_typed_integer_storage_exactly() {
        let plot = parse_patch_plot(vec![
            Value::String("XData".into()),
            int_tensor(3, 1, IntegerStorage::I16(vec![0, 1, 0])),
            Value::String("YData".into()),
            int_tensor(3, 1, IntegerStorage::I16(vec![0, 0, 1])),
            Value::String("ZData".into()),
            int_tensor(3, 1, IntegerStorage::I16(vec![2, 3, 4])),
        ])
        .unwrap();
        assert_eq!(plot.faces(), &[vec![0, 1, 2]]);
        assert_eq!(plot.vertices()[0], Vec3::new(0.0, 0.0, 2.0));
        assert_eq!(plot.vertices()[1], Vec3::new(1.0, 0.0, 3.0));
        assert_eq!(plot.vertices()[2], Vec3::new(0.0, 1.0, 4.0));
    }

    #[test]
    fn patch_xyz_properties_accept_trailing_name_value_pairs() {
        let plot = parse_patch_plot(vec![
            Value::String("XData".into()),
            tensor(3, 1, &[0.0, 1.0, 0.0]),
            Value::String("YData".into()),
            tensor(3, 1, &[0.0, 0.0, 1.0]),
            Value::String("ZData".into()),
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
    fn patch_faces_vertices_properties_retain_integer_storage() {
        let _guard = setup_plot_test();
        let faces = IntegerStorage::I16(vec![1, 2, 3]);
        let vertices = IntegerStorage::I16(vec![0, 1, 0, 0, 0, 1]);
        let handle = crate::call_builtin(
            "patch",
            &[
                Value::String("Faces".into()),
                int_tensor(1, 3, faces.clone()),
                Value::String("Vertices".into()),
                int_tensor(3, 2, vertices.clone()),
            ],
        )
        .expect("integer patch");
        let Value::Num(handle) = handle.value() else {
            panic!("expected patch handle");
        };
        let handle = *handle;
        let retained_faces = get_builtin(vec![Value::Num(handle), Value::String("Faces".into())])
            .expect("patch Faces");
        let retained_vertices =
            get_builtin(vec![Value::Num(handle), Value::String("Vertices".into())])
                .expect("patch Vertices");

        assert!(
            matches!(retained_faces, Value::Tensor(tensor) if tensor.shape() == [1, 3] && tensor.integer_storage() == Some(&faces))
        );
        assert!(
            matches!(retained_vertices, Value::Tensor(tensor) if tensor.shape() == [3, 2] && tensor.integer_storage() == Some(&vertices))
        );
    }

    #[test]
    fn patch_typed_axes_alias_is_gated_before_plotting() {
        let _guard = setup_plot_test();
        let axes = crate::builtins::plotting::gca::gca_builtin(Vec::new()).expect("axes handle");
        let Value::Num(axes) = axes else {
            panic!("expected numeric axes handle");
        };
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = patch_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::Int(runmat_value::IntValue::U64(axes as u64)),
                tensor(3, 1, &[0.0, 1.0, 0.0]),
                tensor(3, 1, &[0.0, 0.0, 1.0]),
            ],
        ))
        .expect_err("typed axes alias must be gated");

        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:PatchIntegerAxesHandleExtension")
        );
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
        let Value::Num(handle) = handle.value() else {
            panic!("expected numeric graphics handle");
        };
        let handle = *handle;
        let ty = crate::call_builtin("get", &[Value::Num(handle), Value::String("Type".into())])
            .expect("get patch type");
        assert_eq!(ty.value(), &Value::String("patch".into()));
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
        let Value::Num(handle) = handle.value() else {
            panic!("expected numeric graphics handle");
        };
        let handle = *handle;
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
        assert_eq!(visible.value(), &Value::Bool(false));

        let all = crate::call_builtin("get", &[Value::Num(handle)]).expect("get patch struct");
        let Value::Struct(st) = all.value() else {
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
        let err = patch_builtin(crate::builtins::common::test_support::builtin_arguments(
            Vec::new(),
        ))
        .expect_err("expected patch argument validation error");
        assert_eq!(err.identifier(), PATCH_ERROR_INVALID_ARGUMENT.identifier);
    }
}
