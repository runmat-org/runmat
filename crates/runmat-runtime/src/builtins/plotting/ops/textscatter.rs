//! MATLAB-compatible `textscatter` and `textscatter3` builtins.

use glam::{Vec3, Vec4};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    StringArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::figure::PlotElement;
use runmat_plot::plots::scatter::MarkerStyle;
use runmat_plot::plots::{Figure, Scatter3Plot, ScatterPlot, TextStyle};

use crate::builtins::plotting::op_common::{apply_axes_target, split_leading_axes_handle};
use crate::builtins::plotting::state::{
    register_textscatter_handle, update_textscatter_figure, update_textscatter_handle_state,
    FigureError, PlotRenderOptions, TextScatterHandleState, TextScatterMarkerColor,
};
use crate::builtins::plotting::style::{
    parse_color_value, value_as_f64, value_as_string, LineStyleParseOptions,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const TEXTSCATTER_NAME: &str = "textscatter";
const TEXTSCATTER3_NAME: &str = "textscatter3";
const DEFAULT_TEXT_DENSITY: f64 = 60.0;
const DEFAULT_MAX_TEXT_LENGTH: usize = 40;
const DEFAULT_MARKER_SIZE: f64 = 6.0;
const DEFAULT_TEXT_COLOR: Vec4 = Vec4::new(0.0, 0.0, 0.0, 1.0);

const OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ts",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the TextScatter chart object.",
}];

const INPUTS_TEXTSCATTER_XY: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String vector or cell array of character vectors.",
    },
];

const INPUTS_TEXTSCATTER_MATRIX: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "xy",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "N-by-2 coordinate matrix.",
    },
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String vector or cell array of character vectors.",
    },
];

const INPUTS_TEXTSCATTER3_XYZ: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String vector or cell array of character vectors.",
    },
];

const INPUTS_TEXTSCATTER3_MATRIX: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "xyz",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "N-by-3 coordinate matrix.",
    },
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String vector or cell array of character vectors.",
    },
];

const INPUTS_PROPS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "props",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name/value TextScatter properties.",
}];

const TEXTSCATTER_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "ts = textscatter(x, y, str)",
        inputs: &INPUTS_TEXTSCATTER_XY,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ts = textscatter(xy, str)",
        inputs: &INPUTS_TEXTSCATTER_MATRIX,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ts = textscatter(ax, ...)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ts = textscatter(..., Name, Value)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT_HANDLE,
    },
];

const TEXTSCATTER3_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "ts = textscatter3(x, y, z, str)",
        inputs: &INPUTS_TEXTSCATTER3_XYZ,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ts = textscatter3(xyz, str)",
        inputs: &INPUTS_TEXTSCATTER3_MATRIX,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ts = textscatter3(ax, ...)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ts = textscatter3(..., Name, Value)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT_HANDLE,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCATTER.INVALID_ARGUMENT",
    identifier: Some("RunMat:textscatter:InvalidArgument"),
    when: "Coordinates, text labels, axes target, or TextScatter property values are invalid.",
    message: "textscatter: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCATTER.INTERNAL",
    identifier: Some("RunMat:textscatter:Internal"),
    when: "Internal plotting state update fails while creating or mutating TextScatter.",
    message: "textscatter: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const TEXTSCATTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TEXTSCATTER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const TEXTSCATTER3_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TEXTSCATTER3_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[derive(Clone)]
struct ParsedTextScatter {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Option<Vec<f64>>,
    text: Vec<String>,
    options: TextScatterOptions,
}

#[derive(Clone)]
struct TextScatterOptions {
    text_density_percentage: f64,
    max_text_length: usize,
    marker_color: TextScatterMarkerColor,
    marker_size: f64,
    color_data: Option<Vec<Vec4>>,
    colors: Vec<Vec4>,
    visible: bool,
    base_style: TextStyle,
    text_data_override: Option<Vec<String>>,
}

impl Default for TextScatterOptions {
    fn default() -> Self {
        Self {
            text_density_percentage: DEFAULT_TEXT_DENSITY,
            max_text_length: DEFAULT_MAX_TEXT_LENGTH,
            marker_color: TextScatterMarkerColor::Auto,
            marker_size: DEFAULT_MARKER_SIZE,
            color_data: None,
            colors: default_colors(),
            visible: true,
            base_style: TextStyle::default(),
            text_data_override: None,
        }
    }
}

#[runtime_builtin(
    name = "textscatter",
    category = "plotting",
    summary = "Create a 2-D text scatter chart.",
    keywords = "textscatter,text,scatter,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(crate::builtins::plotting::type_resolvers::handle_scalar_type),
    descriptor(crate::builtins::plotting::textscatter::TEXTSCATTER_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::textscatter"
)]
pub fn textscatter_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
    textscatter_impl(args, false, TEXTSCATTER_NAME)
}

#[runtime_builtin(
    name = "textscatter3",
    category = "plotting",
    summary = "Create a 3-D text scatter chart.",
    keywords = "textscatter3,text,scatter,plotting,3d",
    sink = true,
    suppress_auto_output = true,
    type_resolver(crate::builtins::plotting::type_resolvers::handle_scalar_type),
    descriptor(crate::builtins::plotting::textscatter::TEXTSCATTER3_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::textscatter"
)]
pub fn textscatter3_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
    textscatter_impl(args, true, TEXTSCATTER3_NAME)
}

fn textscatter_impl(args: Vec<Value>, is_3d: bool, builtin: &'static str) -> BuiltinResult<f64> {
    let (axes_target, rest) = split_leading_axes_handle(args, builtin)?;
    apply_axes_target(axes_target, builtin)?;
    let parsed = parse_textscatter_args(rest, is_3d, builtin)?;
    render_textscatter_chart(parsed, is_3d, builtin)
}

fn parse_textscatter_args(
    args: Vec<Value>,
    is_3d: bool,
    builtin: &'static str,
) -> BuiltinResult<ParsedTextScatter> {
    if args.len() < 2 {
        return Err(textscatter_error(
            builtin,
            "expected coordinate data and text labels",
        ));
    }

    let matrix_width = if is_3d { 3 } else { 2 };
    let (x, y, z, text_value, option_args) =
        if let Ok(matrix) = tensor_from_value_local(&args[0], builtin) {
            if matrix.cols == matrix_width && args.len() >= 2 && looks_like_text_labels(&args[1]) {
                let (x, y, z) = coordinates_from_matrix(&matrix, matrix_width, builtin)?;
                (x, y, z, &args[1], &args[2..])
            } else if is_3d && args.len() >= 4 {
                explicit_coordinates(&args, true, builtin)?
            } else if !is_3d && args.len() >= 3 {
                explicit_coordinates(&args, false, builtin)?
            } else {
                return Err(textscatter_error(
                    builtin,
                    "coordinate matrix must be N-by-2 for textscatter or N-by-3 for textscatter3",
                ));
            }
        } else if is_3d && args.len() >= 4 {
            explicit_coordinates(&args, true, builtin)?
        } else if !is_3d && args.len() >= 3 {
            explicit_coordinates(&args, false, builtin)?
        } else {
            return Err(textscatter_error(
                builtin,
                "expected vector coordinates or coordinate matrix",
            ));
        };

    let mut text = labels_from_value(text_value, builtin)?;
    validate_lengths(&x, &y, z.as_deref(), &text, builtin)?;
    let mut options = parse_options(option_args, text.len(), builtin)?;
    if let Some(override_labels) = options.text_data_override.take() {
        text = override_labels;
    }

    Ok(ParsedTextScatter {
        x,
        y,
        z,
        text,
        options,
    })
}

type CoordinateParse<'a> = (Vec<f64>, Vec<f64>, Option<Vec<f64>>, &'a Value, &'a [Value]);
type CoordinateVectors = (Vec<f64>, Vec<f64>, Option<Vec<f64>>);

fn explicit_coordinates<'a>(
    args: &'a [Value],
    is_3d: bool,
    builtin: &'static str,
) -> BuiltinResult<CoordinateParse<'a>> {
    let x = tensor_vector(&args[0], "XData", builtin)?;
    let y = tensor_vector(&args[1], "YData", builtin)?;
    if is_3d {
        let z = tensor_vector(&args[2], "ZData", builtin)?;
        Ok((x, y, Some(z), &args[3], &args[4..]))
    } else {
        Ok((x, y, None, &args[2], &args[3..]))
    }
}

fn coordinates_from_matrix(
    matrix: &Tensor,
    width: usize,
    builtin: &'static str,
) -> BuiltinResult<CoordinateVectors> {
    if matrix.cols != width {
        return Err(textscatter_error(
            builtin,
            format!("coordinate matrix must have {width} columns"),
        ));
    }
    let mut x = Vec::with_capacity(matrix.rows);
    let mut y = Vec::with_capacity(matrix.rows);
    let mut z = if width == 3 {
        Some(Vec::with_capacity(matrix.rows))
    } else {
        None
    };
    for row in 0..matrix.rows {
        x.push(matrix.data[row]);
        y.push(matrix.data[row + matrix.rows]);
        if let Some(z_values) = z.as_mut() {
            z_values.push(matrix.data[row + 2 * matrix.rows]);
        }
    }
    validate_finite(&x, "XData", builtin)?;
    validate_finite(&y, "YData", builtin)?;
    if let Some(z_values) = z.as_ref() {
        validate_finite(z_values, "ZData", builtin)?;
    }
    Ok((x, y, z))
}

fn tensor_vector(value: &Value, name: &str, builtin: &'static str) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor_from_value_local(value, builtin)?;
    if tensor.data.is_empty() {
        return Err(textscatter_error(
            builtin,
            format!("{name} must be a nonempty numeric vector"),
        ));
    }
    let values = tensor.data;
    validate_finite(&values, name, builtin)?;
    Ok(values)
}

fn tensor_from_value_local(value: &Value, builtin: &'static str) -> BuiltinResult<Tensor> {
    if matches!(value, Value::GpuTensor(_)) {
        return Err(textscatter_error(
            builtin,
            "gpuArray coordinate and color inputs are not supported for textscatter annotation construction",
        ));
    }
    Tensor::try_from(value).map_err(|err| textscatter_error(builtin, err))
}

fn validate_lengths(
    x: &[f64],
    y: &[f64],
    z: Option<&[f64]>,
    text: &[String],
    builtin: &'static str,
) -> BuiltinResult<()> {
    if x.len() != y.len() || z.is_some_and(|z| z.len() != x.len()) || text.len() != x.len() {
        return Err(textscatter_error(
            builtin,
            "coordinate vectors and text labels must have equal lengths",
        ));
    }
    Ok(())
}

fn validate_finite(values: &[f64], name: &str, builtin: &'static str) -> BuiltinResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(textscatter_error(
            builtin,
            format!("{name} values must be finite"),
        ))
    }
}

fn labels_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(vec![chars.data.iter().collect()]),
        Value::CharArray(chars) => {
            let mut labels = Vec::with_capacity(chars.rows);
            for row in 0..chars.rows {
                let mut text = String::new();
                for col in 0..chars.cols {
                    text.push(chars.data[row * chars.cols + col]);
                }
                labels.push(text.trim_end().to_string());
            }
            Ok(labels)
        }
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data.len());
            for value in &cell.data {
                let text = value_as_string(value).ok_or_else(|| {
                    textscatter_error(builtin, "cell labels must contain text scalars")
                })?;
                labels.push(text);
            }
            Ok(labels)
        }
        _ => Err(textscatter_error(
            builtin,
            "labels must be a string vector or cell array of character vectors",
        )),
    }
}

fn looks_like_text_labels(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_)
    )
}

fn parse_options(
    args: &[Value],
    point_count: usize,
    builtin: &'static str,
) -> BuiltinResult<TextScatterOptions> {
    if !args.len().is_multiple_of(2) {
        return Err(textscatter_error(
            builtin,
            "property/value arguments must come in pairs",
        ));
    }
    let mut options = TextScatterOptions::default();
    for pair in args.chunks_exact(2) {
        apply_option(
            &mut options,
            property_key(&pair[0], builtin)?,
            &pair[1],
            point_count,
            builtin,
        )?;
    }
    Ok(options)
}

fn apply_option(
    options: &mut TextScatterOptions,
    key: String,
    value: &Value,
    point_count: usize,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match key.as_str() {
        "textdat" | "textdata" => {
            let labels = labels_from_value(value, builtin)?;
            if labels.len() != point_count {
                return Err(textscatter_error(
                    builtin,
                    "TextData length must match XData length",
                ));
            }
            options.text_data_override = Some(labels);
        }
        "textdensitypercentage" => {
            let density = numeric_scalar(value, "TextDensityPercentage", builtin)?;
            if !(0.0..=100.0).contains(&density) {
                return Err(textscatter_error(
                    builtin,
                    "TextDensityPercentage must be between 0 and 100",
                ));
            }
            options.text_density_percentage = density;
        }
        "maxtextlength" => {
            options.max_text_length = positive_integer(value, "MaxTextLength", builtin)?;
        }
        "markercolor" => {
            options.marker_color = marker_color(value, builtin)?;
        }
        "markersize" => {
            let size = numeric_scalar(value, "MarkerSize", builtin)?;
            if !size.is_finite() || size <= 0.0 {
                return Err(textscatter_error(builtin, "MarkerSize must be positive"));
            }
            options.marker_size = size;
        }
        "colordata" => {
            options.color_data = color_data(value, point_count, builtin)?;
        }
        "colors" => {
            options.colors = color_matrix(value, builtin)?;
        }
        "visible" => {
            options.visible = visible_value(value, builtin)?;
        }
        "color" => {
            options.base_style.color = Some(parse_color(value, builtin)?);
        }
        "fontsize" => {
            let size = numeric_scalar(value, "FontSize", builtin)?;
            if !size.is_finite() || size <= 0.0 {
                return Err(textscatter_error(builtin, "FontSize must be positive"));
            }
            options.base_style.font_size = Some(size as f32);
        }
        "fontweight" => {
            options.base_style.font_weight = Some(text_scalar(value, "FontWeight", builtin)?);
        }
        "fontangle" => {
            options.base_style.font_angle = Some(text_scalar(value, "FontAngle", builtin)?);
        }
        "interpreter" => {
            options.base_style.interpreter = Some(text_scalar(value, "Interpreter", builtin)?);
        }
        "backgroundcolor" | "edgecolor" | "margin" | "tag" | "userdata" | "displayname"
        | "annotation" | "handlevisibility" | "buttondownfcn" | "contextmenu" | "selected"
        | "selectionhighlight" | "datatiptemplate" | "pickableparts" | "hittest"
        | "interruptible" | "busyaction" | "createfcn" | "deletefcn" | "beingdeleted"
        | "xdatasource" | "ydatasource" | "zdatasource" => {
            return Err(textscatter_error(
                builtin,
                format!("TextScatter property `{key}` is not supported by RunMat yet"),
            ));
        }
        other => {
            return Err(textscatter_error(
                builtin,
                format!("unsupported TextScatter property `{other}`"),
            ));
        }
    }
    Ok(())
}

fn render_textscatter_chart(
    parsed: ParsedTextScatter,
    is_3d: bool,
    builtin: &'static str,
) -> BuiltinResult<f64> {
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let annotations_slot = std::rc::Rc::new(std::cell::RefCell::new(Vec::<usize>::new()));
    let marker_slot = std::rc::Rc::new(std::cell::RefCell::new(None::<usize>));
    let axes_slot = std::rc::Rc::new(std::cell::RefCell::new(None::<usize>));
    let annotations_out = std::rc::Rc::clone(&annotations_slot);
    let marker_out = std::rc::Rc::clone(&marker_slot);
    let axes_out = std::rc::Rc::clone(&axes_slot);
    let options = parsed.options.clone();
    let x = parsed.x.clone();
    let y = parsed.y.clone();
    let z = parsed.z.clone();
    let labels = parsed.text.clone();

    let render_result = crate::builtins::plotting::state::render_active_plot(
        builtin,
        PlotRenderOptions {
            title: if is_3d {
                "3-D Text Scatter"
            } else {
                "Text Scatter"
            },
            x_label: "X",
            y_label: "Y",
            ..Default::default()
        },
        move |figure, axes| {
            *axes_out.borrow_mut() = Some(axes);
            let visible_mask = density_mask(labels.len(), options.text_density_percentage);
            let text_colors = resolved_text_colors(labels.len(), &options);
            let mut annotation_indices = Vec::with_capacity(labels.len());
            for idx in 0..labels.len() {
                let mut style = options.base_style.clone();
                style.visible = options.visible && visible_mask[idx];
                if let Some(color) = text_colors.get(idx).copied() {
                    style.color = Some(color);
                }
                let position = Vec3::new(
                    x[idx] as f32,
                    y[idx] as f32,
                    z.as_ref().map(|z| z[idx]).unwrap_or(0.0) as f32,
                );
                let label = truncated_label(&labels[idx], options.max_text_length);
                let annotation = figure.add_axes_text_annotation(axes, position, label, style);
                annotation_indices.push(annotation);
            }
            if should_show_markers(&options) {
                let marker_colors = resolved_marker_colors(labels.len(), &options, &text_colors);
                let marker_index = if is_3d {
                    let points: Vec<Vec3> = (0..labels.len())
                        .map(|idx| {
                            Vec3::new(
                                x[idx] as f32,
                                y[idx] as f32,
                                z.as_ref().map(|z| z[idx]).unwrap_or(0.0) as f32,
                            )
                        })
                        .collect();
                    let mut plot = Scatter3Plot::new(points)
                        .map_err(|err| textscatter_error(builtin, err))?
                        .with_point_size(options.marker_size as f32)
                        .with_colors(marker_colors)
                        .map_err(|err| textscatter_error(builtin, err))?;
                    plot.set_marker_style(MarkerStyle::Circle);
                    plot.set_visible(options.visible);
                    figure.add_scatter3_plot_on_axes(plot, axes)
                } else {
                    let mut plot = ScatterPlot::new(x.clone(), y.clone())
                        .map_err(|err| textscatter_error(builtin, err))?
                        .with_style(
                            DEFAULT_TEXT_COLOR,
                            options.marker_size as f32,
                            MarkerStyle::Circle,
                        );
                    plot.set_colors(marker_colors);
                    plot.set_visible(options.visible);
                    figure.add_scatter_plot_on_axes(plot, axes)
                };
                *marker_out.borrow_mut() = Some(marker_index);
            }
            *annotations_out.borrow_mut() = annotation_indices;
            Ok(())
        },
    );

    let Some(axes_index) = *axes_slot.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let state = TextScatterHandleState {
        figure: figure_handle,
        axes_index,
        annotation_indices: annotations_slot.borrow().clone(),
        marker_plot_index: *marker_slot.borrow(),
        is_3d,
        text_data: parsed.text,
        text_density_percentage: parsed.options.text_density_percentage,
        max_text_length: parsed.options.max_text_length,
        marker_color: parsed.options.marker_color,
        marker_size: parsed.options.marker_size,
        color_data: parsed.options.color_data,
        colors: parsed.options.colors,
        visible: parsed.options.visible,
        base_style: parsed.options.base_style,
    };
    let handle = register_textscatter_handle(state);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(map_internal(builtin, err));
    }
    Ok(handle)
}

pub(crate) fn get_textscatter_property(
    state: &TextScatterHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match property.map(canonical_key).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("textscatter".into()));
            st.insert(
                "Parent",
                Value::Num(crate::builtins::plotting::state::encode_axes_handle(
                    state.figure,
                    state.axes_index,
                )),
            );
            st.insert("Children", handles_value(Vec::new()));
            st.insert("TextData", string_array_value(state.text_data.clone())?);
            st.insert("XData", vector_value(x_data(state, builtin)?));
            st.insert("YData", vector_value(y_data(state, builtin)?));
            st.insert("ZData", vector_value(z_data(state, builtin)?));
            st.insert(
                "TextDensityPercentage",
                Value::Num(state.text_density_percentage),
            );
            st.insert("MaxTextLength", Value::Num(state.max_text_length as f64));
            st.insert("MarkerColor", marker_color_value(&state.marker_color));
            st.insert("MarkerSize", Value::Num(state.marker_size));
            st.insert("ColorData", color_data_value(state.color_data.as_deref()));
            st.insert("Colors", color_matrix_value(&state.colors));
            st.insert(
                "Visible",
                Value::String(if state.visible { "on" } else { "off" }.into()),
            );
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("textscatter".into())),
        Some("parent") => Ok(Value::Num(
            crate::builtins::plotting::state::encode_axes_handle(state.figure, state.axes_index),
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("textdata") => string_array_value(state.text_data.clone()),
        Some("xdata") => Ok(vector_value(x_data(state, builtin)?)),
        Some("ydata") => Ok(vector_value(y_data(state, builtin)?)),
        Some("zdata") => Ok(vector_value(z_data(state, builtin)?)),
        Some("textdensitypercentage") => Ok(Value::Num(state.text_density_percentage)),
        Some("maxtextlength") => Ok(Value::Num(state.max_text_length as f64)),
        Some("markercolor") => Ok(marker_color_value(&state.marker_color)),
        Some("markersize") => Ok(Value::Num(state.marker_size)),
        Some("colordata") => Ok(color_data_value(state.color_data.as_deref())),
        Some("colors") => Ok(color_matrix_value(&state.colors)),
        Some("visible") => Ok(Value::String(
            if state.visible { "on" } else { "off" }.into(),
        )),
        Some("displayname") | Some("tag") => Ok(Value::String(String::new())),
        Some(other) => Err(textscatter_error(
            builtin,
            format!("unsupported TextScatter property `{other}`"),
        )),
    }
}

pub(crate) fn apply_textscatter_property(
    handle: f64,
    state: &TextScatterHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let mut next = match crate::builtins::plotting::state::plot_child_handle_snapshot(handle) {
        Ok(crate::builtins::plotting::state::PlotChildHandleState::TextScatter(current)) => current,
        _ => state.clone(),
    };
    let prop = canonical_key(key);
    match prop.as_str() {
        "textdata" => {
            let labels = labels_from_value(value, builtin)?;
            if labels.len() != next.text_data.len() {
                return Err(textscatter_error(
                    builtin,
                    "TextData length must match XData length",
                ));
            }
            next.text_data = labels;
        }
        "xdata" => {
            let values = tensor_vector(value, "XData", builtin)?;
            ensure_data_len(values.len(), next.text_data.len(), "XData", builtin)?;
            set_annotation_positions(&next, Some(values), None, None, builtin)?;
            return update_textscatter_handle_state(handle, next)
                .map_err(|err| textscatter_error(builtin, err.to_string()));
        }
        "ydata" => {
            let values = tensor_vector(value, "YData", builtin)?;
            ensure_data_len(values.len(), next.text_data.len(), "YData", builtin)?;
            set_annotation_positions(&next, None, Some(values), None, builtin)?;
            return update_textscatter_handle_state(handle, next)
                .map_err(|err| textscatter_error(builtin, err.to_string()));
        }
        "zdata" => {
            if !next.is_3d {
                return Err(textscatter_error(
                    builtin,
                    "ZData can only be set on textscatter3 charts",
                ));
            }
            let values = tensor_vector(value, "ZData", builtin)?;
            ensure_data_len(values.len(), next.text_data.len(), "ZData", builtin)?;
            set_annotation_positions(&next, None, None, Some(values), builtin)?;
            return update_textscatter_handle_state(handle, next)
                .map_err(|err| textscatter_error(builtin, err.to_string()));
        }
        "textdensitypercentage" => {
            let density = numeric_scalar(value, "TextDensityPercentage", builtin)?;
            if !(0.0..=100.0).contains(&density) {
                return Err(textscatter_error(
                    builtin,
                    "TextDensityPercentage must be between 0 and 100",
                ));
            }
            next.text_density_percentage = density;
        }
        "maxtextlength" => {
            next.max_text_length = positive_integer(value, "MaxTextLength", builtin)?;
        }
        "markercolor" => {
            next.marker_color = marker_color(value, builtin)?;
        }
        "markersize" => {
            let size = numeric_scalar(value, "MarkerSize", builtin)?;
            if !size.is_finite() || size <= 0.0 {
                return Err(textscatter_error(builtin, "MarkerSize must be positive"));
            }
            next.marker_size = size;
        }
        "colordata" => {
            next.color_data = color_data(value, next.text_data.len(), builtin)?;
        }
        "colors" => {
            next.colors = color_matrix(value, builtin)?;
        }
        "visible" => {
            next.visible = visible_value(value, builtin)?;
        }
        "color" => {
            next.base_style.color = Some(parse_color(value, builtin)?);
            next.color_data = None;
        }
        "fontsize" => {
            let size = numeric_scalar(value, "FontSize", builtin)?;
            if !size.is_finite() || size <= 0.0 {
                return Err(textscatter_error(builtin, "FontSize must be positive"));
            }
            next.base_style.font_size = Some(size as f32);
        }
        "fontweight" => {
            next.base_style.font_weight = Some(text_scalar(value, "FontWeight", builtin)?);
        }
        "fontangle" => {
            next.base_style.font_angle = Some(text_scalar(value, "FontAngle", builtin)?);
        }
        "interpreter" => {
            next.base_style.interpreter = Some(text_scalar(value, "Interpreter", builtin)?);
        }
        other => {
            return Err(textscatter_error(
                builtin,
                format!("unsupported TextScatter set property `{other}`"),
            ));
        }
    }
    refresh_textscatter_rendering(&mut next, builtin)?;
    update_textscatter_handle_state(handle, next)
        .map_err(|err| textscatter_error(builtin, err.to_string()))
}

fn add_marker_plot(
    figure: &mut Figure,
    axes_index: usize,
    is_3d: bool,
    x: &[f64],
    y: &[f64],
    z: Option<&[f64]>,
    marker_colors: Vec<Vec4>,
    marker_size: f64,
    visible: bool,
    builtin: &'static str,
) -> BuiltinResult<usize> {
    if is_3d {
        let z = z.ok_or_else(|| textscatter_error(builtin, "missing ZData for 3-D markers"))?;
        let points: Vec<Vec3> = (0..x.len())
            .map(|idx| Vec3::new(x[idx] as f32, y[idx] as f32, z[idx] as f32))
            .collect();
        let mut plot = Scatter3Plot::new(points)
            .map_err(|err| textscatter_error(builtin, err))?
            .with_point_size(marker_size as f32)
            .with_colors(marker_colors)
            .map_err(|err| textscatter_error(builtin, err))?;
        plot.set_marker_style(MarkerStyle::Circle);
        plot.set_visible(visible);
        Ok(figure.add_scatter3_plot_on_axes(plot, axes_index))
    } else {
        let mut plot = ScatterPlot::new(x.to_vec(), y.to_vec())
            .map_err(|err| textscatter_error(builtin, err))?
            .with_style(DEFAULT_TEXT_COLOR, marker_size as f32, MarkerStyle::Circle);
        plot.set_colors(marker_colors);
        plot.set_visible(visible);
        Ok(figure.add_scatter_plot_on_axes(plot, axes_index))
    }
}

fn refresh_textscatter_rendering(
    state: &mut TextScatterHandleState,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let visible_mask = density_mask(state.text_data.len(), state.text_density_percentage);
    let text_colors = resolved_text_colors(state.text_data.len(), &options_from_state(state));
    let x_values = x_data(state, builtin)?;
    let y_values = y_data(state, builtin)?;
    let z_values = z_data(state, builtin)?;
    let location = state.clone();
    update_textscatter_figure(&location, |figure| {
        for (idx, annotation_index) in state.annotation_indices.iter().copied().enumerate() {
            if figure
                .axes_text_annotation(state.axes_index, annotation_index)
                .is_none()
            {
                return Err(crate::builtins::plotting::state::FigureError::InvalidPlotObjectHandle);
            }
            let mut style = state.base_style.clone();
            style.visible = state.visible && visible_mask[idx];
            style.color = text_colors.get(idx).copied();
            figure.set_axes_text_annotation_text(
                state.axes_index,
                annotation_index,
                truncated_label(&state.text_data[idx], state.max_text_length),
            );
            figure.set_axes_text_annotation_style(state.axes_index, annotation_index, style);
        }
        let marker_colors = resolved_marker_colors(
            state.text_data.len(),
            &options_from_state(state),
            &text_colors,
        );
        if let Some(plot_index) = state.marker_plot_index {
            if let Some(plot) = figure.get_plot_mut(plot_index) {
                match plot {
                    PlotElement::Scatter(scatter) => {
                        if state.is_3d {
                            scatter.set_visible(false);
                            if should_show_markers_state(state) {
                                let new_index = add_marker_plot(
                                    figure,
                                    state.axes_index,
                                    state.is_3d,
                                    &x_values,
                                    &y_values,
                                    Some(&z_values),
                                    marker_colors,
                                    state.marker_size,
                                    state.visible,
                                    builtin,
                                )
                                .map_err(|_| FigureError::InvalidPlotObjectHandle)?;
                                state.marker_plot_index = Some(new_index);
                            }
                        } else {
                            scatter.set_marker_size(state.marker_size as f32);
                            scatter.set_visible(state.visible && should_show_markers_state(state));
                            scatter.set_colors(marker_colors);
                        }
                    }
                    PlotElement::Scatter3(scatter) => {
                        if state.is_3d {
                            let mut updated = scatter
                                .clone()
                                .with_colors(marker_colors)
                                .map_err(|_| FigureError::InvalidPlotObjectHandle)?
                                .with_point_size(state.marker_size as f32);
                            updated.set_visible(state.visible && should_show_markers_state(state));
                            *scatter = updated;
                        } else {
                            scatter.set_visible(false);
                            if should_show_markers_state(state) {
                                let new_index = add_marker_plot(
                                    figure,
                                    state.axes_index,
                                    state.is_3d,
                                    &x_values,
                                    &y_values,
                                    None,
                                    marker_colors,
                                    state.marker_size,
                                    state.visible,
                                    builtin,
                                )
                                .map_err(|_| FigureError::InvalidPlotObjectHandle)?;
                                state.marker_plot_index = Some(new_index);
                            }
                        }
                    }
                    _ => {}
                }
            } else if should_show_markers_state(state) {
                let new_index = add_marker_plot(
                    figure,
                    state.axes_index,
                    state.is_3d,
                    &x_values,
                    &y_values,
                    if state.is_3d { Some(&z_values) } else { None },
                    marker_colors,
                    state.marker_size,
                    state.visible,
                    builtin,
                )
                .map_err(|_| FigureError::InvalidPlotObjectHandle)?;
                state.marker_plot_index = Some(new_index);
            }
        } else if should_show_markers_state(state) {
            let new_index = add_marker_plot(
                figure,
                state.axes_index,
                state.is_3d,
                &x_values,
                &y_values,
                if state.is_3d { Some(&z_values) } else { None },
                marker_colors,
                state.marker_size,
                state.visible,
                builtin,
            )
            .map_err(|_| FigureError::InvalidPlotObjectHandle)?;
            state.marker_plot_index = Some(new_index);
        }
        Ok(())
    })
    .map_err(|err| textscatter_error(builtin, err.to_string()))
}

fn set_annotation_positions(
    state: &TextScatterHandleState,
    x: Option<Vec<f64>>,
    y: Option<Vec<f64>>,
    z: Option<Vec<f64>>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let current_x = positions_component(state, 0, builtin)?;
    let current_y = positions_component(state, 1, builtin)?;
    let current_z = positions_component(state, 2, builtin)?;
    update_textscatter_figure(state, |figure| {
        let x_values = x.as_deref().unwrap_or(&current_x);
        let y_values = y.as_deref().unwrap_or(&current_y);
        let z_values = z.as_deref().unwrap_or(&current_z);
        for (idx, annotation_index) in state.annotation_indices.iter().copied().enumerate() {
            figure.set_axes_text_annotation_position(
                state.axes_index,
                annotation_index,
                Vec3::new(
                    x_values[idx] as f32,
                    y_values[idx] as f32,
                    z_values[idx] as f32,
                ),
            );
        }
        if let Some(plot_index) = state.marker_plot_index {
            if let Some(plot) = figure.get_plot_mut(plot_index) {
                match plot {
                    PlotElement::Scatter(scatter) => {
                        scatter.x_data = x_values.to_vec();
                        scatter.y_data = y_values.to_vec();
                    }
                    PlotElement::Scatter3(scatter) => {
                        scatter.points = (0..x_values.len())
                            .map(|idx| {
                                Vec3::new(
                                    x_values[idx] as f32,
                                    y_values[idx] as f32,
                                    z_values[idx] as f32,
                                )
                            })
                            .collect();
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    })
    .map_err(|err| textscatter_error(builtin, err.to_string()))
}

fn x_data(state: &TextScatterHandleState, builtin: &'static str) -> BuiltinResult<Vec<f64>> {
    positions_component(state, 0, builtin)
}

fn y_data(state: &TextScatterHandleState, builtin: &'static str) -> BuiltinResult<Vec<f64>> {
    positions_component(state, 1, builtin)
}

fn z_data(state: &TextScatterHandleState, builtin: &'static str) -> BuiltinResult<Vec<f64>> {
    if state.is_3d {
        positions_component(state, 2, builtin)
    } else {
        Ok(Vec::new())
    }
}

fn positions_component(
    state: &TextScatterHandleState,
    component: usize,
    builtin: &'static str,
) -> BuiltinResult<Vec<f64>> {
    let figure = crate::builtins::plotting::clone_figure(state.figure)
        .ok_or_else(|| textscatter_error(builtin, "invalid TextScatter figure"))?;
    let mut values = Vec::with_capacity(state.annotation_indices.len());
    for annotation_index in &state.annotation_indices {
        let annotation = figure
            .axes_text_annotation(state.axes_index, *annotation_index)
            .ok_or_else(|| textscatter_error(builtin, "invalid TextScatter annotation"))?;
        values.push(annotation.position[component] as f64);
    }
    Ok(values)
}

fn options_from_state(state: &TextScatterHandleState) -> TextScatterOptions {
    TextScatterOptions {
        text_density_percentage: state.text_density_percentage,
        max_text_length: state.max_text_length,
        marker_color: state.marker_color.clone(),
        marker_size: state.marker_size,
        color_data: state.color_data.clone(),
        colors: state.colors.clone(),
        visible: state.visible,
        base_style: state.base_style.clone(),
        text_data_override: None,
    }
}

fn density_mask(len: usize, density: f64) -> Vec<bool> {
    if len == 0 {
        return Vec::new();
    }
    if density <= 0.0 {
        return vec![false; len];
    }
    if density >= 100.0 {
        return vec![true; len];
    }
    let target = ((len as f64) * density / 100.0)
        .ceil()
        .clamp(1.0, len as f64) as usize;
    let mut mask = vec![false; len];
    if target >= len {
        mask.fill(true);
        return mask;
    }
    for slot in 0..target {
        let idx = ((slot as f64) * (len.saturating_sub(1) as f64)
            / (target.saturating_sub(1).max(1) as f64))
            .round() as usize;
        mask[idx.min(len - 1)] = true;
    }
    mask
}

fn should_show_markers(options: &TextScatterOptions) -> bool {
    options.text_density_percentage < 100.0
        && !matches!(options.marker_color, TextScatterMarkerColor::None)
}

fn should_show_markers_state(state: &TextScatterHandleState) -> bool {
    state.text_density_percentage < 100.0
        && !matches!(state.marker_color, TextScatterMarkerColor::None)
}

fn resolved_text_colors(point_count: usize, options: &TextScatterOptions) -> Vec<Vec4> {
    match &options.color_data {
        Some(colors) if colors.len() == 1 => vec![colors[0]; point_count],
        Some(colors) => colors.clone(),
        None => vec![options.base_style.color.unwrap_or(DEFAULT_TEXT_COLOR); point_count],
    }
}

fn resolved_marker_colors(
    point_count: usize,
    options: &TextScatterOptions,
    text_colors: &[Vec4],
) -> Vec<Vec4> {
    match options.marker_color {
        TextScatterMarkerColor::Auto => text_colors.to_vec(),
        TextScatterMarkerColor::None => vec![Vec4::ZERO; point_count],
        TextScatterMarkerColor::Color(color) => vec![color; point_count],
    }
}

fn truncated_label(label: &str, max_len: usize) -> String {
    let count = label.chars().count();
    if count <= max_len {
        return label.to_string();
    }
    if max_len <= 3 {
        return ".".repeat(max_len);
    }
    let prefix: String = label.chars().take(max_len - 3).collect();
    format!("{prefix}...")
}

fn color_data(
    value: &Value,
    point_count: usize,
    builtin: &'static str,
) -> BuiltinResult<Option<Vec<Vec4>>> {
    if is_empty_numeric(value) {
        return Ok(None);
    }
    let colors = color_matrix(value, builtin)?;
    if colors.len() == 1 || colors.len() == point_count {
        Ok(Some(colors))
    } else {
        Err(textscatter_error(
            builtin,
            "ColorData must be an RGB triplet or an N-by-3 RGB matrix",
        ))
    }
}

fn color_matrix(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<Vec4>> {
    let tensor = tensor_from_value_local(value, builtin)?;
    if tensor.data.is_empty() {
        return Ok(Vec::new());
    }
    if tensor.data.len() == 3 && (tensor.rows == 1 || tensor.cols == 1) {
        let color = rgb_triplet(&tensor.data, builtin)?;
        return Ok(vec![color]);
    }
    if tensor.cols != 3 {
        return Err(textscatter_error(
            builtin,
            "RGB color matrices must have three columns",
        ));
    }
    let mut colors = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let rgb = [
            tensor.data[row],
            tensor.data[row + tensor.rows],
            tensor.data[row + 2 * tensor.rows],
        ];
        colors.push(rgb_triplet(&rgb, builtin)?);
    }
    Ok(colors)
}

fn marker_color(value: &Value, builtin: &'static str) -> BuiltinResult<TextScatterMarkerColor> {
    if let Some(text) = value_as_string(value) {
        match text.trim().to_ascii_lowercase().as_str() {
            "auto" => return Ok(TextScatterMarkerColor::Auto),
            "none" => return Ok(TextScatterMarkerColor::None),
            _ => {}
        }
    }
    Ok(TextScatterMarkerColor::Color(parse_color(value, builtin)?))
}

fn parse_color(value: &Value, builtin: &'static str) -> BuiltinResult<Vec4> {
    parse_color_value(&LineStyleParseOptions::generic(builtin), value)
}

fn rgb_triplet(values: &[f64], builtin: &'static str) -> BuiltinResult<Vec4> {
    if values.len() != 3
        || values
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(textscatter_error(
            builtin,
            "RGB triplets must contain three finite values in [0,1]",
        ));
    }
    Ok(Vec4::new(
        values[0] as f32,
        values[1] as f32,
        values[2] as f32,
        1.0,
    ))
}

fn numeric_scalar(value: &Value, name: &str, builtin: &'static str) -> BuiltinResult<f64> {
    let scalar = value_as_f64(value)
        .ok_or_else(|| textscatter_error(builtin, format!("{name} must be numeric")))?;
    if scalar.is_finite() {
        Ok(scalar)
    } else {
        Err(textscatter_error(builtin, format!("{name} must be finite")))
    }
}

fn positive_integer(value: &Value, name: &str, builtin: &'static str) -> BuiltinResult<usize> {
    let scalar = numeric_scalar(value, name, builtin)?;
    if scalar.fract() != 0.0 || scalar <= 0.0 {
        return Err(textscatter_error(
            builtin,
            format!("{name} must be a positive integer"),
        ));
    }
    Ok(scalar as usize)
}

fn text_scalar(value: &Value, name: &str, builtin: &'static str) -> BuiltinResult<String> {
    value_as_string(value).ok_or_else(|| textscatter_error(builtin, format!("{name} must be text")))
}

fn visible_value(value: &Value, builtin: &'static str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        _ => {
            let text = value_as_string(value)
                .ok_or_else(|| textscatter_error(builtin, "Visible must be 'on' or 'off'"))?;
            match text.trim().to_ascii_lowercase().as_str() {
                "on" => Ok(true),
                "off" => Ok(false),
                _ => Err(textscatter_error(builtin, "Visible must be 'on' or 'off'")),
            }
        }
    }
}

fn property_key(value: &Value, builtin: &'static str) -> BuiltinResult<String> {
    let key = value_as_string(value)
        .ok_or_else(|| textscatter_error(builtin, "property names must be text"))?;
    Ok(canonical_key(&key))
}

fn canonical_key(name: &str) -> String {
    name.trim()
        .chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn ensure_data_len(
    len: usize,
    expected: usize,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if len == expected {
        Ok(())
    } else {
        Err(textscatter_error(
            builtin,
            format!("{name} length must match TextData length"),
        ))
    }
}

fn vector_value(values: Vec<f64>) -> Value {
    let len = values.len();
    Value::Tensor(Tensor {
        rows: 1,
        cols: len,
        shape: vec![1, len],
        data: values,
        dtype: runmat_builtins::NumericDType::F64,
    })
}

fn color_data_value(colors: Option<&[Vec4]>) -> Value {
    match colors {
        Some(colors) => color_matrix_value(colors),
        None => Value::Tensor(Tensor {
            rows: 0,
            cols: 0,
            shape: vec![0, 0],
            data: Vec::new(),
            dtype: runmat_builtins::NumericDType::F64,
        }),
    }
}

fn color_matrix_value(colors: &[Vec4]) -> Value {
    let rows = colors.len();
    let mut data = Vec::with_capacity(rows * 3);
    for component in 0..3 {
        for color in colors {
            data.push(color[component] as f64);
        }
    }
    Value::Tensor(Tensor {
        rows,
        cols: 3,
        shape: vec![rows, 3],
        data,
        dtype: runmat_builtins::NumericDType::F64,
    })
}

fn marker_color_value(color: &TextScatterMarkerColor) -> Value {
    match color {
        TextScatterMarkerColor::Auto => Value::String("auto".into()),
        TextScatterMarkerColor::None => Value::String("none".into()),
        TextScatterMarkerColor::Color(color) => color_matrix_value(&[*color]),
    }
}

fn string_array_value(values: Vec<String>) -> BuiltinResult<Value> {
    let len = values.len();
    StringArray::new(values, vec![1, len])
        .map(Value::StringArray)
        .map_err(|err| textscatter_error(TEXTSCATTER_NAME, err))
}

fn handles_value(handles: Vec<f64>) -> Value {
    vector_value(handles)
}

fn is_empty_numeric(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.data.is_empty())
}

fn default_colors() -> Vec<Vec4> {
    vec![
        Vec4::new(0.0, 0.4470, 0.7410, 1.0),
        Vec4::new(0.8500, 0.3250, 0.0980, 1.0),
        Vec4::new(0.9290, 0.6940, 0.1250, 1.0),
        Vec4::new(0.4940, 0.1840, 0.5560, 1.0),
        Vec4::new(0.4660, 0.6740, 0.1880, 1.0),
        Vec4::new(0.3020, 0.7450, 0.9330, 1.0),
        Vec4::new(0.6350, 0.0780, 0.1840, 1.0),
    ]
}

fn textscatter_error(builtin: &'static str, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{builtin}: {}", detail.as_ref())).with_builtin(builtin);
    builder = builder.with_identifier("RunMat:textscatter:InvalidArgument");
    builder.build()
}

fn map_internal(builtin: &'static str, err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    let mut builder =
        build_runtime_error(format!("{builtin}: {}", err.message)).with_builtin(builtin);
    builder = builder.with_identifier("RunMat:textscatter:Internal");
    builder.build()
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

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new_2d(values.to_vec(), 1, values.len()).unwrap())
    }

    #[test]
    fn textscatter_descriptor_covers_core_forms() {
        let labels: Vec<&str> = TEXTSCATTER_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"ts = textscatter(x, y, str)"));
        assert!(labels.contains(&"ts = textscatter(xy, str)"));
        let labels3: Vec<&str> = TEXTSCATTER3_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels3.contains(&"ts = textscatter3(x, y, z, str)"));
        assert!(labels3.contains(&"ts = textscatter3(xyz, str)"));
    }

    #[test]
    fn textscatter_creates_textscatter_handle_and_annotations() {
        let _guard = setup();
        let labels = Value::StringArray(
            StringArray::new(
                vec!["alpha".into(), "beta".into(), "gamma".into()],
                vec![1, 3],
            )
            .unwrap(),
        );
        let handle = textscatter_builtin(vec![
            row(&[1.0, 2.0, 3.0]),
            row(&[4.0, 5.0, 6.0]),
            labels,
            Value::String("TextDensityPercentage".into()),
            Value::Num(100.0),
            Value::String("MaxTextLength".into()),
            Value::Num(5.0),
        ])
        .unwrap();

        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap(),
            Value::String("textscatter".into())
        );
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.axes_text_annotations(0).len(), 3);
        assert_eq!(fig.axes_text_annotations(0)[0].text, "alpha");
        let x = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("XData".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(x.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn textscatter_accepts_coordinate_matrix_and_colordata() {
        let _guard = setup();
        let xy = Value::Tensor(Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap());
        let labels = Value::StringArray(
            StringArray::new(vec!["one".into(), "two".into()], vec![1, 2]).unwrap(),
        );
        let colors =
            Value::Tensor(Tensor::new_2d(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3).unwrap());
        let handle = textscatter_builtin(vec![
            xy,
            labels,
            Value::String("ColorData".into()),
            colors,
            Value::String("MarkerColor".into()),
            Value::String("auto".into()),
        ])
        .unwrap();
        let color_data = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("ColorData".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(color_data.rows, 2);
        assert_eq!(color_data.cols, 3);
    }

    #[test]
    fn textscatter3_supports_xyz_matrix_and_zdata() {
        let _guard = setup();
        let xyz =
            Value::Tensor(Tensor::new_2d(vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0], 2, 3).unwrap());
        let labels = Value::StringArray(
            StringArray::new(vec!["one".into(), "two".into()], vec![1, 2]).unwrap(),
        );
        let handle = textscatter3_builtin(vec![xyz, labels]).unwrap();
        let z = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("ZData".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(z.data, vec![100.0, 200.0]);
    }

    #[test]
    fn textscatter_set_updates_chart_state_and_annotations() {
        let _guard = setup();
        let labels = Value::StringArray(
            StringArray::new(vec!["alphabet".into(), "betatron".into()], vec![1, 2]).unwrap(),
        );
        let handle = textscatter_builtin(vec![row(&[1.0, 2.0]), row(&[3.0, 4.0]), labels]).unwrap();
        set_builtin(vec![
            Value::Num(handle),
            Value::String("MaxTextLength".into()),
            Value::Num(5.0),
            Value::String("TextDensityPercentage".into()),
            Value::Num(0.0),
            Value::String("FontSize".into()),
            Value::Num(14.0),
        ])
        .unwrap();
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("MaxTextLength".into())
            ])
            .unwrap(),
            Value::Num(5.0)
        );
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.axes_text_annotations(0)[0].text, "al...");
        assert!(!fig.axes_text_annotations(0)[0].style.visible);
        assert_eq!(fig.axes_text_annotations(0)[0].style.font_size, Some(14.0));
    }

    #[test]
    fn textscatter_creation_textdata_overrides_positional_labels() {
        let _guard = setup();
        let original = Value::StringArray(
            StringArray::new(vec!["old-a".into(), "old-b".into()], vec![1, 2]).unwrap(),
        );
        let replacement = Value::StringArray(
            StringArray::new(vec!["new-a".into(), "new-b".into()], vec![1, 2]).unwrap(),
        );
        let handle = textscatter_builtin(vec![
            row(&[1.0, 2.0]),
            row(&[3.0, 4.0]),
            original,
            Value::String("TextData".into()),
            replacement,
            Value::String("TextDensityPercentage".into()),
            Value::Num(100.0),
        ])
        .unwrap();

        let labels =
            get_builtin(vec![Value::Num(handle), Value::String("TextData".into())]).unwrap();
        let Value::StringArray(labels) = labels else {
            panic!("expected string array labels");
        };
        assert_eq!(labels.data, vec!["new-a".to_string(), "new-b".to_string()]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.axes_text_annotations(0)[0].text, "new-a");
    }

    #[test]
    fn textscatter_set_density_creates_marker_plot_when_needed() {
        let _guard = setup();
        let labels = Value::StringArray(
            StringArray::new(vec!["one".into(), "two".into(), "three".into()], vec![1, 3]).unwrap(),
        );
        let handle = textscatter_builtin(vec![
            row(&[1.0, 2.0, 3.0]),
            row(&[4.0, 5.0, 6.0]),
            labels,
            Value::String("TextDensityPercentage".into()),
            Value::Num(100.0),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.len(), 0);

        set_builtin(vec![
            Value::Num(handle),
            Value::String("TextDensityPercentage".into()),
            Value::Num(50.0),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.len(), 1);
    }

    #[test]
    fn textscatter_rejects_zdata_set_on_2d_chart() {
        let _guard = setup();
        let labels = Value::StringArray(
            StringArray::new(vec!["one".into(), "two".into()], vec![1, 2]).unwrap(),
        );
        let handle = textscatter_builtin(vec![row(&[1.0, 2.0]), row(&[3.0, 4.0]), labels]).unwrap();
        let err = set_builtin(vec![
            Value::Num(handle),
            Value::String("ZData".into()),
            row(&[5.0, 6.0]),
        ])
        .expect_err("2-D textscatter should reject zdata mutation");
        assert!(err
            .to_string()
            .contains("ZData can only be set on textscatter3"));
    }

    #[test]
    fn textscatter_rejects_length_mismatch() {
        let _guard = setup();
        let labels = Value::StringArray(
            StringArray::new(vec!["one".into(), "two".into()], vec![1, 2]).unwrap(),
        );
        let err = textscatter_builtin(vec![row(&[1.0]), row(&[2.0]), labels])
            .expect_err("length mismatch");
        assert_eq!(err.identifier(), Some("RunMat:textscatter:InvalidArgument"));
    }
}
