use std::cell::RefCell;
use std::rc::Rc;

use runmat_builtins::{
    BuiltinErrorDescriptor, GPU_CDATA_EXTENSION, HEATMAP_ERROR_INTERNAL,
    HEATMAP_ERROR_INVALID_ARGUMENT,
};
#[cfg(test)]
use runmat_builtins::{
    BuiltinIntegerInputAvailability, HEATMAP_DESCRIPTOR, HEATMAP_INTEGER_CAPABILITIES,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ColorMap, ShadingMode};
use runmat_value::{IntegerStorage, Tensor, Value};

use super::common::{surface_matrix_dimensions, SurfaceDataInput};
use super::op_common::surface_inputs::AxisSource;
use super::state::{color_limits_snapshot, render_active_plot, PlotRenderOptions};
#[cfg(test)]
use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "heatmap";

fn heatmap_error_with_detail(
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

fn map_heatmap_invalid(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    heatmap_error_with_detail(&HEATMAP_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_heatmap_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    heatmap_error_with_detail(&HEATMAP_ERROR_INTERNAL, err.message)
}

fn heatmap_invalid(detail: impl AsRef<str>) -> RuntimeError {
    heatmap_error_with_detail(&HEATMAP_ERROR_INVALID_ARGUMENT, detail)
}

#[runtime_builtin(
    name = "heatmap",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::heatmap"
)]
pub async fn heatmap_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<f64> {
    gate_gpu_cdata_extension(args.view())?;
    let ParsedHeatmap {
        x_labels,
        y_labels,
        color_data,
        rest,
    } = parse_heatmap_args(args)
        .await
        .map_err(map_heatmap_invalid)?;

    crate::builtins::plotting::properties::validate_heatmap_property_pairs(
        &rest,
        x_labels.len(),
        y_labels.len(),
        BUILTIN_NAME,
    )
    .map_err(map_heatmap_invalid)?;

    let (rows, cols) = surface_matrix_dimensions(&color_data);
    let integer_color_data = color_data.integer_storage().is_some();
    let public_color_limit_value = integer_public_color_limit_value(&color_data);
    let public_color_limits = integer_public_color_limits(&color_data);
    // MATLAB draws the first row of a heatmap at the top; the y axis runs upward, so the
    // rows are drawn in reverse order (and their labels with them).
    let render_data = flip_rows(&normalize_integer_color_data(&color_data));
    let x_axis = AxisSource::Host(default_axis(cols));
    let y_axis = AxisSource::Host(default_axis(rows));
    let color_limits = if integer_color_data {
        Some((0.0, 1.0))
    } else {
        color_limits_snapshot()
    };
    let mut surface = super::image::build_indexed_image_surface(
        &SurfaceDataInput::Host(render_data),
        &x_axis,
        &y_axis,
        ColorMap::Parula,
        color_limits,
        "scaled",
    )
    .await
    .map_err(map_heatmap_invalid)?;
    surface = surface
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_colormap(ColorMap::Parula)
        .with_shading(ShadingMode::None)
        .with_cell_labels(
            heatmap_cell_labels(&color_data)
                .into_iter()
                .map(|mut column| {
                    column.reverse();
                    column
                })
                .collect(),
        );
    if color_limits.is_some() {
        surface = surface.with_color_limits(color_limits);
    }

    let mut surface = Some(surface);
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let render_x_labels = x_labels.clone();
    let render_y_labels: Vec<String> = y_labels.iter().rev().cloned().collect();
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            // Heatmap cells fill the chart area, as in MATLAB; they are not kept square.
            axis_equal: false,
            ..Default::default()
        },
        move |figure, axes| {
            let plot_index = figure.add_surface_plot_on_axes(
                surface.take().expect("heatmap plot consumed once"),
                axes,
            );
            figure.set_axes_colorbar_enabled(axes, true);
            figure.set_axes_tick_labels(
                axes,
                Some(render_x_labels.clone()),
                Some(render_y_labels.clone()),
            );
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_heatmap_handle(
        figure_handle,
        axes,
        plot_index,
        x_labels,
        y_labels,
        color_data,
        public_color_limit_value,
    );
    if let Some(public_limits) = public_color_limits {
        crate::builtins::plotting::state::set_color_limits_for_axes(
            figure_handle,
            axes,
            Some(public_limits),
        )
        .map_err(|err| heatmap_invalid(err.to_string()))?;
        crate::builtins::plotting::state::update_plot_element(figure_handle, plot_index, |plot| {
            if let runmat_plot::plots::PlotElement::Surface(surface) = plot {
                surface.set_color_limits(Some((0.0, 1.0)));
            }
        })
        .map_err(|err| heatmap_invalid(err.to_string()))?;
    }
    if !rest.is_empty() {
        let plot_handle = crate::builtins::plotting::properties::resolve_plot_handle(
            &Value::Num(handle),
            BUILTIN_NAME,
        )?;
        crate::builtins::plotting::properties::set_properties(plot_handle, &rest, BUILTIN_NAME)
            .map_err(map_heatmap_invalid)?;
    }
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(map_heatmap_internal(err));
    }
    Ok(handle)
}

struct ParsedHeatmap {
    x_labels: Vec<String>,
    y_labels: Vec<String>,
    color_data: Tensor,
    rest: Vec<Value>,
}

async fn parse_heatmap_args(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<ParsedHeatmap> {
    if let Some(Value::Object(table)) = args.first() {
        if crate::builtins::table::is_tabular_object(table) {
            return parse_table_heatmap_args_view(args.view());
        }
    }
    let cdata_with_properties = args.len() >= 3
        && args.len() % 2 == 1
        && crate::builtins::plotting::properties::is_heatmap_property_name(&args[1]);
    if cdata_with_properties {
        let mut it = args.into_iter();
        let color_data = cdata_tensor(it.next().expect("CData")).await?;
        let (rows, cols) = surface_matrix_dimensions(&color_data);
        let x_labels = default_labels(cols);
        let y_labels = default_labels(rows);
        return Ok(ParsedHeatmap {
            x_labels,
            y_labels,
            color_data,
            rest: it.collect(),
        });
    }
    match args.len() {
        0 => Err(heatmap_invalid(
            "expected CData or XValues,YValues,CData input",
        )),
        1 => {
            let color_data = cdata_tensor(args.into_iter().next().expect("one arg")).await?;
            let (rows, cols) = surface_matrix_dimensions(&color_data);
            let x_labels = default_labels(cols);
            let y_labels = default_labels(rows);
            Ok(ParsedHeatmap {
                x_labels,
                y_labels,
                color_data,
                rest: Vec::new(),
            })
        }
        2 => Err(heatmap_invalid(
            "expected CData or XValues,YValues,CData input",
        )),
        _ => {
            let mut it = args.into_iter();
            let x = it.next().expect("x labels");
            let y = it.next().expect("y labels");
            let c = it.next().expect("cdata");
            let rest: Vec<Value> = it.collect();
            let color_data = cdata_tensor(c).await?;
            let (rows, cols) = surface_matrix_dimensions(&color_data);
            let x_labels = labels_from_value(&x, cols, "XValues")?;
            let y_labels = labels_from_value(&y, rows, "YValues")?;
            Ok(ParsedHeatmap {
                x_labels,
                y_labels,
                color_data,
                rest,
            })
        }
    }
}

#[cfg(test)]
fn parse_table_heatmap_args(args: &[Value]) -> crate::BuiltinResult<ParsedHeatmap> {
    parse_table_heatmap_args_view(crate::call::FunctionArgumentView::from_slice(args))
}

fn parse_table_heatmap_args_view(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<ParsedHeatmap> {
    let (Some(Value::Object(table)), Some(x), Some(y)) = (args.first(), args.get(1), args.get(2))
    else {
        return Err(heatmap_invalid(
            "table heatmap requires SourceTable, XVariable, and YVariable",
        ));
    };
    let options = args.range(3..).expect("validated table heatmap prefix");
    if !options.len().is_multiple_of(2) {
        return Err(heatmap_invalid(
            "table heatmap properties must be name/value pairs",
        ));
    }
    let mut color_variable = None;
    let mut method = None;
    let mut rest = Vec::new();
    for pair in options.chunks_exact(2) {
        let name =
            crate::builtins::plotting::properties::property_name_text(&pair[0], BUILTIN_NAME)
                .map_err(map_heatmap_invalid)?;
        if name.eq_ignore_ascii_case("ColorVariable") {
            color_variable = Some(pair.get(1).expect("heatmap option value"));
        } else if name.eq_ignore_ascii_case("ColorMethod") {
            let value =
                crate::builtins::plotting::properties::property_name_text(&pair[1], BUILTIN_NAME)
                    .map_err(map_heatmap_invalid)?;
            method = Some(
                crate::builtins::table::HeatmapAggregation::parse(&value)
                    .ok_or_else(|| heatmap_invalid("unsupported table ColorMethod"))?,
            );
        } else {
            rest.extend(pair.iter().cloned());
        }
    }
    let method = method.unwrap_or(if color_variable.is_some() {
        crate::builtins::table::HeatmapAggregation::Mean
    } else {
        crate::builtins::table::HeatmapAggregation::Count
    });
    let aggregated =
        crate::builtins::table::aggregate_heatmap_table(table, x, y, color_variable, method)
            .map_err(map_heatmap_invalid)?;
    Ok(ParsedHeatmap {
        x_labels: aggregated.x_labels,
        y_labels: aggregated.y_labels,
        color_data: aggregated.color_data,
        rest,
    })
}

async fn cdata_tensor(value: Value) -> crate::BuiltinResult<Tensor> {
    let tensor = match value {
        Value::GpuTensor(handle) => {
            if handle.shape().first().copied().unwrap_or(0) == 0
                || handle.shape().get(1).copied().unwrap_or(0) == 0
                || handle
                    .shape()
                    .iter()
                    .skip(2)
                    .any(|&dimension| dimension != 1)
            {
                return Err(heatmap_invalid(
                    "CData must be a nonempty 2-D numeric matrix",
                ));
            }
            let provider =
                runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
                    heatmap_invalid("no acceleration provider owns the GPU CData handle")
                })?;
            let downloaded =
                crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
                    provider, &handle,
                )
                .await
                .map_err(map_heatmap_invalid)?;
            Tensor::try_from(&downloaded).map_err(|e| heatmap_invalid(&e))?
        }
        other => Tensor::try_from(&other).map_err(|e| heatmap_invalid(&e))?,
    };
    let (rows, cols) = surface_matrix_dimensions(&tensor);
    if rows == 0 || cols == 0 {
        return Err(heatmap_invalid("CData must contain at least a 2-D grid"));
    }
    if tensor
        .shape()
        .iter()
        .skip(2)
        .any(|&dimension| dimension != 1)
    {
        return Err(heatmap_invalid("CData must be a 2-D numeric matrix"));
    }
    Ok(tensor)
}

fn gate_gpu_cdata_extension(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<()> {
    if matches!(args.first(), Some(Value::Object(table)) if crate::builtins::table::is_tabular_object(table))
    {
        return Ok(());
    }
    let cdata_with_properties = args.len() >= 3
        && args.len() % 2 == 1
        && crate::builtins::plotting::properties::is_heatmap_property_name(&args[1]);
    let cdata = match args.len() {
        1 => args.first(),
        _ if cdata_with_properties => args.first(),
        3.. => args.get(2),
        _ => None,
    };
    if matches!(cdata, Some(Value::GpuTensor(_))) {
        crate::compatibility::ensure_builtin_extension_enabled(&GPU_CDATA_EXTENSION, BUILTIN_NAME)?;
    }
    Ok(())
}

fn integer_public_color_limits(tensor: &Tensor) -> Option<(f64, f64)> {
    let limits = integer_public_color_limit_value(tensor)?;
    let values = limits.materialize_f64();
    let (minimum, maximum) = (values[0], values[1]);
    if minimum < maximum {
        Some((minimum, maximum))
    } else if minimum.is_finite() {
        Some((minimum, next_f64_up(minimum)))
    } else {
        None
    }
}

fn integer_public_color_limit_value(tensor: &Tensor) -> Option<Tensor> {
    let storage = tensor.integer_storage()?;
    macro_rules! limits {
        ($variant:ident, $values:expr) => {{
            let minimum = *$values.iter().min()?;
            let maximum = *$values.iter().max()?;
            if minimum == maximum {
                let (lower, upper) = match maximum.checked_add(1) {
                    Some(upper) => (minimum, upper),
                    None => (minimum.checked_sub(1)?, maximum),
                };
                IntegerStorage::$variant(vec![lower, upper])
            } else {
                IntegerStorage::$variant(vec![minimum, maximum])
            }
        }};
    }
    let limits = match storage {
        IntegerStorage::I8(values) => limits!(I8, values),
        IntegerStorage::I16(values) => limits!(I16, values),
        IntegerStorage::I32(values) => limits!(I32, values),
        IntegerStorage::I64(values) => limits!(I64, values),
        IntegerStorage::U8(values) => limits!(U8, values),
        IntegerStorage::U16(values) => limits!(U16, values),
        IntegerStorage::U32(values) => limits!(U32, values),
        IntegerStorage::U64(values) => limits!(U64, values),
    };
    Tensor::new_integer(limits, vec![1, 2]).ok()
}

pub(crate) fn public_color_limits_for_value(
    value: &Value,
    builtin: &'static str,
) -> crate::BuiltinResult<(f64, f64)> {
    if let Ok(tensor) = Tensor::try_from(value) {
        if integer_limits_are_ordered(&tensor) {
            let values = tensor.materialize_f64();
            return Ok((
                values[0],
                if values[0] < values[1] {
                    values[1]
                } else {
                    next_f64_up(values[0])
                },
            ));
        }
    }
    let (lo, hi) = crate::builtins::plotting::op_common::limits::limits_from_value(value, builtin)?;
    if lo < hi {
        return Ok((lo, hi));
    }
    Ok((lo, next_f64_up(lo)))
}

fn next_f64_up(value: f64) -> f64 {
    if value == 0.0 {
        f64::from_bits(1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

pub(crate) fn renderer_color_limits_for_value(
    color_data: &Tensor,
    value: &Value,
    builtin: &'static str,
) -> crate::BuiltinResult<(f64, f64)> {
    let limits =
        Tensor::try_from(value).map_err(|error| heatmap_invalid(format!("{builtin}: {error}")))?;
    if let Some(exact) = exact_integer_renderer_limits(color_data, &limits) {
        return Ok(exact);
    }
    let public = crate::builtins::plotting::op_common::limits::limits_from_value(value, builtin)?;
    let Some((source_lo, source_hi)) = integer_public_color_limits(color_data) else {
        return Ok(public);
    };
    let span = source_hi - source_lo;
    if !span.is_finite() || span <= 0.0 {
        return Ok((0.0, 1.0));
    }
    Ok(((public.0 - source_lo) / span, (public.1 - source_lo) / span))
}

fn integer_limits_are_ordered(tensor: &Tensor) -> bool {
    if tensor.len() != 2 {
        return false;
    }
    match tensor.integer_storage() {
        Some(IntegerStorage::I8(values)) => values[0] < values[1],
        Some(IntegerStorage::I16(values)) => values[0] < values[1],
        Some(IntegerStorage::I32(values)) => values[0] < values[1],
        Some(IntegerStorage::I64(values)) => values[0] < values[1],
        Some(IntegerStorage::U8(values)) => values[0] < values[1],
        Some(IntegerStorage::U16(values)) => values[0] < values[1],
        Some(IntegerStorage::U32(values)) => values[0] < values[1],
        Some(IntegerStorage::U64(values)) => values[0] < values[1],
        None => false,
    }
}

fn exact_integer_renderer_limits(color_data: &Tensor, limits: &Tensor) -> Option<(f64, f64)> {
    let source = color_data.integer_storage()?;
    let limit_storage = limits.integer_storage()?;
    macro_rules! signed_limits {
        ($source:expr, $limits:expr) => {{
            let minimum = $source.iter().copied().min()? as i128;
            let maximum = $source.iter().copied().max()? as i128;
            let lo = *$limits.first()? as i128;
            let hi = *$limits.get(1)? as i128;
            let span = maximum.checked_sub(minimum)?;
            if span == 0 {
                Some((0.0, 1.0))
            } else {
                Some((
                    (lo - minimum) as f64 / span as f64,
                    (hi - minimum) as f64 / span as f64,
                ))
            }
        }};
    }
    macro_rules! unsigned_limits {
        ($source:expr, $limits:expr) => {{
            let minimum = $source.iter().copied().min()? as u128;
            let maximum = $source.iter().copied().max()? as u128;
            let lo = *$limits.first()? as u128;
            let hi = *$limits.get(1)? as u128;
            let span = maximum.checked_sub(minimum)?;
            if span == 0 {
                Some((0.0, 1.0))
            } else {
                let normalized = |value: u128| {
                    if value >= minimum {
                        (value - minimum) as f64 / span as f64
                    } else {
                        -((minimum - value) as f64 / span as f64)
                    }
                };
                Some((normalized(lo), normalized(hi)))
            }
        }};
    }
    match (source, limit_storage) {
        (IntegerStorage::I8(source), IntegerStorage::I8(limits)) => signed_limits!(source, limits),
        (IntegerStorage::I16(source), IntegerStorage::I16(limits)) => {
            signed_limits!(source, limits)
        }
        (IntegerStorage::I32(source), IntegerStorage::I32(limits)) => {
            signed_limits!(source, limits)
        }
        (IntegerStorage::I64(source), IntegerStorage::I64(limits)) => {
            signed_limits!(source, limits)
        }
        (IntegerStorage::U8(source), IntegerStorage::U8(limits)) => {
            unsigned_limits!(source, limits)
        }
        (IntegerStorage::U16(source), IntegerStorage::U16(limits)) => {
            unsigned_limits!(source, limits)
        }
        (IntegerStorage::U32(source), IntegerStorage::U32(limits)) => {
            unsigned_limits!(source, limits)
        }
        (IntegerStorage::U64(source), IntegerStorage::U64(limits)) => {
            unsigned_limits!(source, limits)
        }
        _ => None,
    }
}

/// The text printed in each cell, indexed `[column][row]` like the image surface, formatted
/// as MATLAB's default `CellLabelFormat` (`%0.4g`); integer data prints exactly.
fn heatmap_cell_labels(tensor: &Tensor) -> Vec<Vec<String>> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let text = |index: usize| -> String {
        if let Some(storage) = tensor.integer_storage() {
            return match storage {
                IntegerStorage::I8(v) => v[index].to_string(),
                IntegerStorage::I16(v) => v[index].to_string(),
                IntegerStorage::I32(v) => v[index].to_string(),
                IntegerStorage::I64(v) => v[index].to_string(),
                IntegerStorage::U8(v) => v[index].to_string(),
                IntegerStorage::U16(v) => v[index].to_string(),
                IntegerStorage::U32(v) => v[index].to_string(),
                IntegerStorage::U64(v) => v[index].to_string(),
            };
        }
        tensor
            .as_f64_slice()
            .and_then(|data| data.get(index).copied())
            .map(|value| crate::builtins::strings::core::num2str::format_general(value, 4, false))
            .unwrap_or_default()
    };
    (0..cols)
        .map(|col| (0..rows).map(|row| text(row + col * rows)).collect())
        .collect()
}

/// The matrix with its rows in reverse order (column-major data).
fn flip_rows(tensor: &Tensor) -> Tensor {
    let (rows, cols) = (tensor.rows(), tensor.cols());
    let Some(data) = tensor.as_f64_slice() else {
        return tensor.clone();
    };
    let mut flipped = Vec::with_capacity(data.len());
    for col in 0..cols {
        for row in (0..rows).rev() {
            flipped.push(data[row + col * rows]);
        }
    }
    Tensor::new(flipped, tensor.shape().to_vec()).expect("flipped heatmap shape")
}

fn normalize_integer_color_data(tensor: &Tensor) -> Tensor {
    let Some(storage) = tensor.integer_storage() else {
        return tensor.clone();
    };
    let normalized = match storage {
        IntegerStorage::I8(values) => normalize_signed(values.iter().map(|&v| i128::from(v))),
        IntegerStorage::I16(values) => normalize_signed(values.iter().map(|&v| i128::from(v))),
        IntegerStorage::I32(values) => normalize_signed(values.iter().map(|&v| i128::from(v))),
        IntegerStorage::I64(values) => normalize_signed(values.iter().map(|&v| i128::from(v))),
        IntegerStorage::U8(values) => normalize_unsigned(values.iter().map(|&v| u128::from(v))),
        IntegerStorage::U16(values) => normalize_unsigned(values.iter().map(|&v| u128::from(v))),
        IntegerStorage::U32(values) => normalize_unsigned(values.iter().map(|&v| u128::from(v))),
        IntegerStorage::U64(values) => normalize_unsigned(values.iter().map(|&v| u128::from(v))),
    };
    Tensor::new(normalized, tensor.shape().to_vec()).expect("normalized heatmap shape")
}

fn normalize_signed(values: impl Iterator<Item = i128> + Clone) -> Vec<f64> {
    let Some(minimum) = values.clone().min() else {
        return Vec::new();
    };
    let maximum = values.clone().max().expect("nonempty integer color data");
    if minimum == maximum {
        return values.map(|_| 0.5).collect();
    }
    let span = (maximum - minimum) as f64;
    values
        .map(|value| (value - minimum) as f64 / span)
        .collect()
}

fn normalize_unsigned(values: impl Iterator<Item = u128> + Clone) -> Vec<f64> {
    let Some(minimum) = values.clone().min() else {
        return Vec::new();
    };
    let maximum = values.clone().max().expect("nonempty integer color data");
    if minimum == maximum {
        return values.map(|_| 0.5).collect();
    }
    let span = (maximum - minimum) as f64;
    values
        .map(|value| (value - minimum) as f64 / span)
        .collect()
}

fn labels_from_value(
    value: &Value,
    expected_len: usize,
    axis_name: &str,
) -> crate::BuiltinResult<Vec<String>> {
    let labels = crate::builtins::plotting::properties::label_strings_from_value(
        value,
        BUILTIN_NAME,
        axis_name,
    )
    .map_err(map_heatmap_invalid)?;
    if labels.len() != expected_len {
        return Err(heatmap_invalid(format!(
            "{axis_name} must have {expected_len} labels"
        )));
    }
    Ok(labels)
}

fn default_labels(len: usize) -> Vec<String> {
    (1..=len).map(|idx| idx.to_string()).collect()
}

fn default_axis(len: usize) -> Vec<f64> {
    (1..=len).map(|idx| idx as f64).collect()
}

#[cfg(test)]
fn transpose_for_surface(tensor: &Tensor) -> Tensor {
    let (rows, cols) = surface_matrix_dimensions(tensor);
    let mut indices = vec![0; tensor_utils::tensor_element_len(tensor)];
    for row in 0..rows {
        for col in 0..cols {
            let src = row + rows * col;
            let dst = col + cols * row;
            indices[dst] = src;
        }
    }
    let storage = tensor
        .clone()
        .into_numeric_storage()
        .and_then(|storage| storage.reorder(&indices))
        .expect("heatmap transpose permutation");
    Tensor::from_numeric_storage(storage, vec![cols, rows]).expect("heatmap transpose shape")
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
    use runmat_plot::plots::PlotElement;
    use runmat_value::{CellArray, IntegerStorage, StringArray, Value};

    async fn heatmap_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        super::heatmap_builtin(crate::call::BuiltinArguments::from_borrowed(&args, None)?).await
    }

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        Tensor::new(data, vec![rows, cols]).expect("heatmap test matrix")
    }

    fn int_tensor(data: Vec<i16>, rows: usize, cols: usize) -> Tensor {
        Tensor::new_integer(runmat_value::IntegerStorage::I16(data), vec![rows, cols])
            .expect("integer tensor")
    }

    #[test]
    fn heatmap_transpose_reads_typed_integer_storage_exactly() {
        let transposed = transpose_for_surface(&int_tensor(vec![1, 2, 3, 4, 5, 6], 2, 3));

        assert_eq!(transposed.shape(), [3, 2]);
        assert_eq!(
            transposed.materialize_f64(),
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
        );
        assert_eq!(transposed.numeric_dtype(), runmat_value::NumericDType::I16);
        assert_eq!(
            transposed.integer_storage(),
            Some(&runmat_value::IntegerStorage::I16(vec![1, 3, 5, 2, 4, 6]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn heatmap_preserves_dense_rank_one_grid_and_rank_zero_rejection() {
        let _guard = setup();
        let handle = futures::executor::block_on(heatmap_builtin(vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).expect("rank-one CData"),
        )]))
        .expect("rank-one CData is one heatmap row");
        assert!(handle.is_finite());
        let figure = clone_figure(current_figure_handle()).expect("heatmap figure");
        let PlotElement::Surface(surface) = figure.plots().next().expect("heatmap surface") else {
            panic!("expected heatmap surface")
        };
        assert_eq!(surface.x_data, vec![1.0, 2.0, 3.0]);
        assert_eq!(surface.y_data, vec![1.0]);

        let error = futures::executor::block_on(heatmap_builtin(vec![Value::Tensor(
            Tensor::new(vec![1.0], Vec::new()).expect("rank-zero CData"),
        )]))
        .expect_err("rank-zero CData has no grid extent");
        assert!(error.message().contains("at least a 2-D grid"));
    }

    #[test]
    fn heatmap_cdata_builds_heatmap_handle() {
        let _guard = setup();
        let handle = futures::executor::block_on(heatmap_builtin(vec![Value::Tensor(tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            2,
            3,
        ))]))
        .expect("heatmap should render");
        assert!(handle.is_finite());

        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = fig.plots().next().unwrap() else {
            panic!("expected surface");
        };
        assert!(surface.flatten_z);
        assert!(surface.image_mode);
        assert_eq!(surface.x_data, vec![1.0, 2.0, 3.0]);
        assert_eq!(surface.y_data, vec![1.0, 2.0]);
        // Rows are drawn in reverse so the first row is at the top, as in MATLAB.
        assert_eq!(
            surface.z_data.as_deref(),
            Some(&[vec![2.0, 1.0], vec![4.0, 3.0], vec![6.0, 5.0]][..])
        );
        assert!(fig.axes_metadata(0).unwrap().colorbar_enabled);
    }

    #[test]
    fn heatmap_accepts_labels_and_exposes_chart_properties() {
        let _guard = setup();
        let x = CellArray::new(
            vec![
                Value::String("Small".into()),
                Value::String("Medium".into()),
                Value::String("Large".into()),
            ],
            1,
            3,
        )
        .unwrap();
        let y = CellArray::new(
            vec![
                Value::String("Green".into()),
                Value::String("Red".into()),
                Value::String("Blue".into()),
                Value::String("Gray".into()),
            ],
            1,
            4,
        )
        .unwrap();
        let cdata = tensor(
            vec![
                45.0, 43.0, 32.0, 23.0, 60.0, 54.0, 94.0, 95.0, 32.0, 76.0, 68.0, 58.0,
            ],
            4,
            3,
        );
        let handle = futures::executor::block_on(heatmap_builtin(vec![
            Value::Cell(x),
            Value::Cell(y),
            Value::Tensor(cdata),
        ]))
        .expect("heatmap should render");

        set_builtin(vec![
            Value::Num(handle),
            Value::String("Title".into()),
            Value::String("T-Shirt Orders".into()),
            Value::String("XLabel".into()),
            Value::String("Sizes".into()),
            Value::String("YLabel".into()),
            Value::String("Colors".into()),
        ])
        .unwrap();

        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Title".into())]).unwrap(),
            Value::String("T-Shirt Orders".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("XLabel".into())]).unwrap(),
            Value::String("Sizes".into())
        );
        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(
            meta.x_tick_labels.as_ref().unwrap(),
            &vec![
                "Small".to_string(),
                "Medium".to_string(),
                "Large".to_string()
            ]
        );
        // The first row is drawn at the top, so the axis labels run bottom-up reversed.
        assert_eq!(
            meta.y_tick_labels.as_ref().unwrap(),
            &vec![
                "Gray".to_string(),
                "Blue".to_string(),
                "Red".to_string(),
                "Green".to_string()
            ]
        );
        let labels = get_builtin(vec![
            Value::Num(handle),
            Value::String("XDisplayLabels".into()),
        ])
        .unwrap();
        let Value::StringArray(labels) = labels else {
            panic!("expected string array");
        };
        assert_eq!(labels.data(), vec!["Small", "Medium", "Large"]);

        set_builtin(vec![
            Value::Num(handle),
            Value::String("XDisplayLabels".into()),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::String("S".into()),
                        Value::String("M".into()),
                        Value::String("L".into()),
                    ],
                    1,
                    3,
                )
                .unwrap(),
            ),
            Value::String("YDisplayLabels".into()),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::String("G".into()),
                        Value::String("R".into()),
                        Value::String("B".into()),
                        Value::String("Y".into()),
                    ],
                    1,
                    4,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let meta = fig.axes_metadata(0).unwrap();
        assert_eq!(
            meta.x_tick_labels.as_ref().unwrap(),
            &vec!["S".to_string(), "M".to_string(), "L".to_string()]
        );
        assert_eq!(
            meta.y_tick_labels.as_ref().unwrap(),
            &vec![
                "Y".to_string(),
                "B".to_string(),
                "R".to_string(),
                "G".to_string()
            ]
        );
    }

    #[test]
    fn heatmap_rejects_bad_property_pairs_before_mutating_figure() {
        let _guard = setup();
        let before = clone_figure(current_figure_handle())
            .map(|figure| figure.plots().count())
            .unwrap_or(0);

        let err = futures::executor::block_on(heatmap_builtin(vec![
            Value::Cell(
                CellArray::new(
                    vec![Value::String("A".into()), Value::String("B".into())],
                    1,
                    2,
                )
                .unwrap(),
            ),
            Value::Cell(
                CellArray::new(
                    vec![Value::String("C".into()), Value::String("D".into())],
                    1,
                    2,
                )
                .unwrap(),
            ),
            Value::Tensor(tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2)),
            Value::String("NotAHeatmapProperty".into()),
            Value::Num(1.0),
        ]))
        .expect_err("invalid property should fail");
        assert!(err.to_string().contains("unsupported heatmap property"));

        let after = clone_figure(current_figure_handle())
            .map(|figure| figure.plots().count())
            .unwrap_or(0);
        assert_eq!(after, before);
    }

    #[test]
    fn heatmap_descriptor_includes_core_signatures() {
        let labels: Vec<&str> = HEATMAP_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = heatmap(CData)"));
        assert!(labels.contains(&"h = heatmap(XValues, YValues, CData)"));
        assert!(labels.contains(&"h = heatmap(CData, Name, Value, ...)"));
    }

    #[test]
    fn heatmap_integer_capabilities_cover_documented_matrix_roles() {
        assert_eq!(HEATMAP_INTEGER_CAPABILITIES.len(), 8);
        for capability in HEATMAP_INTEGER_CAPABILITIES {
            for input in capability.inputs {
                assert_eq!(
                    input.classes,
                    crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES
                );
                assert_eq!(
                    input.availability,
                    BuiltinIntegerInputAvailability::Documented
                );
            }
        }
    }

    #[test]
    fn heatmap_wide_integer_labels_are_formatted_exactly() {
        let wide = u64::MAX;
        let labels = crate::builtins::plotting::properties::label_strings_from_value(
            &Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![wide - 1, wide]), vec![1, 2])
                    .expect("wide labels"),
            ),
            BUILTIN_NAME,
            "XValues",
        )
        .expect("integer labels");
        assert_eq!(labels, vec![(wide - 1).to_string(), wide.to_string()]);
    }

    #[test]
    fn heatmap_wide_integer_color_normalization_keeps_adjacent_values_distinct() {
        let wide = u64::MAX;
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![wide - 2, wide - 1, wide]),
            vec![1, 3],
        )
        .expect("wide ColorData");
        let normalized = normalize_integer_color_data(&input).materialize_f64();
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
        assert!(normalized.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn heatmap_gpu_cdata_extension_is_gated_before_provider_access() {
        let gpu = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![2, 2],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap()
                .handle,
        );
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = futures::executor::block_on(heatmap_builtin(vec![gpu]))
            .expect_err("strict mode must reject GPU CData");
        assert_eq!(err.identifier(), GPU_CDATA_EXTENSION.error_identifier);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn heatmap_wgpu_integer_cdata_download_is_exact_and_non_destructive() {
        let _guard = crate::builtins::common::test_support::accel_test_lock();
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("WGPU provider");
        let source = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .expect("wide CData");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &source)
            .expect("upload exact CData");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let downloaded =
            futures::executor::block_on(cdata_tensor(Value::GpuTensor(handle.clone())))
                .expect("download CData");
        assert_eq!(downloaded.integer_storage(), source.integer_storage());
        assert!(runmat_accelerate_api::provider_for_handle(&handle).is_some());
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&handle),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        handle.release().expect("release exact CData input");
    }

    #[test]
    fn heatmap_validates_documented_integer_on_off_and_font_size_properties() {
        let one = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![1]), vec![1, 1]).expect("integer scalar"),
        );
        crate::builtins::plotting::properties::validate_heatmap_property_pairs(
            &[
                Value::String("ColorbarVisible".into()),
                one.clone(),
                Value::String("GridVisible".into()),
                one,
                Value::String("FontSize".into()),
                Value::Int(runmat_value::IntValue::U64(12)),
            ],
            2,
            2,
            BUILTIN_NAME,
        )
        .expect("documented integer properties");

        let err = crate::builtins::plotting::properties::validate_heatmap_property_pairs(
            &[
                Value::String("ColorbarVisible".into()),
                Value::Int(runmat_value::IntValue::U64(2)),
            ],
            2,
            2,
            BUILTIN_NAME,
        )
        .expect_err("numeric on/off values must be exact zero or one");
        assert!(err.message.contains("0, or 1"));
    }

    #[test]
    fn heatmap_cdata_property_form_preserves_grammar_and_public_limits() {
        let _guard = setup();
        let source = Tensor::new_integer(IntegerStorage::U64(vec![10, 20, 30, 40]), vec![2, 2])
            .expect("integer CData");
        let handle = futures::executor::block_on(heatmap_builtin(vec![
            Value::Tensor(source.clone()),
            Value::String("FontSize".into()),
            Value::Int(runmat_value::IntValue::U8(12)),
            Value::String("ColorLimits".into()),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![15, 35]), vec![1, 2]).unwrap(),
            ),
        ]))
        .expect("CData plus properties");
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("ColorData".into())]).unwrap(),
            Value::Tensor(source)
        );
        let Value::Tensor(limits) = get_builtin(vec![
            Value::Num(handle),
            Value::String("ColorLimits".into()),
        ])
        .unwrap() else {
            panic!("expected public ColorLimits");
        };
        assert_eq!(limits.materialize_f64(), vec![15.0, 35.0]);
        let figure = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Surface(surface) = figure.plots().next().unwrap() else {
            panic!("expected heatmap surface");
        };
        assert_eq!(surface.color_limits, Some((1.0 / 6.0, 5.0 / 6.0)));
    }

    #[test]
    fn heatmap_rejects_nd_cdata_before_plot_mutation() {
        let _guard = setup();
        let before = clone_figure(current_figure_handle())
            .map(|figure| figure.plots().count())
            .unwrap_or(0);
        let cdata = Tensor::new_integer(IntegerStorage::U8(vec![1; 8]), vec![2, 2, 2]).unwrap();
        let error = futures::executor::block_on(heatmap_builtin(vec![Value::Tensor(cdata)]))
            .expect_err("N-D CData must reject");
        assert_eq!(
            error.identifier(),
            HEATMAP_ERROR_INVALID_ARGUMENT.identifier
        );
        assert_eq!(
            clone_figure(current_figure_handle())
                .map(|figure| figure.plots().count())
                .unwrap_or(0),
            before
        );
    }

    #[test]
    fn heatmap_rejects_resident_nd_cdata_before_provider_lookup() {
        let _guard = setup();
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![2, 2, 2],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap()
                .handle,
        );
        let error = futures::executor::block_on(heatmap_builtin(vec![resident]))
            .expect_err("N-D resident CData must reject from handle metadata");
        assert_eq!(
            error.identifier(),
            HEATMAP_ERROR_INVALID_ARGUMENT.identifier
        );
        assert!(!error.message.contains("provider"));
    }

    #[test]
    fn heatmap_wide_integer_color_limits_compare_before_f64_conversion() {
        let lower = 9_007_199_254_740_992_u64;
        let limits = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![lower, lower + 1]), vec![1, 2]).unwrap(),
        );
        crate::builtins::plotting::properties::validate_heatmap_property_pairs(
            &[Value::String("ColorLimits".into()), limits],
            2,
            2,
            BUILTIN_NAME,
        )
        .expect("adjacent wide integer limits remain ordered exactly");
        let source = Tensor::new_integer(
            IntegerStorage::U64(vec![lower - 1, lower, lower + 1]),
            vec![1, 3],
        )
        .unwrap();
        let limits = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![lower, lower + 1]), vec![1, 2]).unwrap(),
        );
        assert_eq!(
            renderer_color_limits_for_value(&source, &limits, BUILTIN_NAME).unwrap(),
            (0.5, 1.0)
        );
        let _guard = setup();
        let handle = futures::executor::block_on(heatmap_builtin(vec![
            Value::Tensor(source),
            Value::String("ColorLimits".into()),
            limits.clone(),
        ]))
        .expect("wide heatmap limits");
        let returned = get_builtin(vec![
            Value::Num(handle),
            Value::String("ColorLimits".into()),
        ])
        .unwrap();
        assert_eq!(returned, limits);
    }

    #[test]
    fn heatmap_auto_limits_cover_negative_adjacent_and_constant_wide_integers() {
        for storage in [
            IntegerStorage::I64(vec![-9_007_199_254_740_993, -9_007_199_254_740_992]),
            IntegerStorage::U64(vec![u64::MAX, u64::MAX]),
        ] {
            let _guard = setup();
            let handle = futures::executor::block_on(heatmap_builtin(vec![Value::Tensor(
                Tensor::new_integer(storage, vec![1, 2]).unwrap(),
            )]))
            .expect("wide integer heatmap");
            let Value::Tensor(returned) = get_builtin(vec![
                Value::Num(handle),
                Value::String("ColorLimits".into()),
            ])
            .unwrap() else {
                panic!("expected exact public ColorLimits")
            };
            let values = returned.materialize_f64();
            let figure = clone_figure(current_figure_handle()).unwrap();
            let axes = figure.axes_metadata(0).unwrap();
            let (lo, hi) = axes.color_limits.expect("client auto limits");
            assert!(lo < hi, "client limits must increase: {lo:?}, {hi:?}");
            assert_eq!(values.len(), 2);
            assert!(returned.integer_storage().is_some());
        }
    }

    #[test]
    fn heatmap_missing_input_uses_stable_identifier() {
        let err = futures::executor::block_on(heatmap_builtin(Vec::new()))
            .expect_err("expected heatmap argument validation error");
        assert_eq!(err.identifier(), HEATMAP_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn table_heatmap_aggregates_before_scene_publication_and_preserves_matrix_forms() {
        let _guard = setup();
        let table = crate::builtins::table::table_from_columns(
            vec!["Hour".into(), "Fixture".into(), "Force".into()],
            vec![
                Value::Tensor(tensor(vec![8.0, 8.0, 9.0], 3, 1)),
                Value::StringArray(
                    StringArray::new(vec!["A".into(), "A".into(), "B".into()], vec![3, 1]).unwrap(),
                ),
                Value::Tensor(tensor(vec![2.0, 4.0, 10.0], 3, 1)),
            ],
        )
        .unwrap();
        let inputs = vec![
            table.clone(),
            Value::String("Hour".into()),
            Value::String("Fixture".into()),
            Value::String("ColorVariable".into()),
            Value::String("Force".into()),
            Value::String("ColorMethod".into()),
            Value::String("mean".into()),
        ];
        let handle = futures::executor::block_on(heatmap_builtin(inputs)).unwrap();
        let Value::Tensor(colors) =
            get_builtin(vec![Value::Num(handle), Value::String("ColorData".into())]).unwrap()
        else {
            panic!("expected aggregated ColorData")
        };
        assert_eq!(colors.shape(), [2, 2]);
        let values = colors.materialize_f64();
        assert_eq!(values[0], 3.0);
        assert!(values[1].is_nan() && values[2].is_nan());
        assert_eq!(values[3], 10.0);
        let figure = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(figure.plots().count(), 1);
        let metadata = figure.axes_metadata(0).unwrap();
        assert_eq!(metadata.x_tick_labels.as_ref().unwrap(), &vec!["8", "9"]);
        // Rendering reverses rows and labels together; source table order stays A, B below.
        assert_eq!(metadata.y_tick_labels.as_ref().unwrap(), &vec!["B", "A"]);
        let indexed = parse_table_heatmap_args(&[
            table.clone(),
            Value::Num(1.0),
            Value::Num(2.0),
            Value::from("ColorVariable"),
            Value::Num(3.0),
        ])
        .expect("declared table order, not lexical field order");
        assert_eq!(indexed.x_labels, vec!["8".to_string(), "9".to_string()]);
        assert_eq!(indexed.y_labels, vec!["A".to_string(), "B".to_string()]);
        assert_eq!(indexed.color_data.materialize_f64()[0], 3.0);

        let table_with_missing_color = crate::builtins::table::table_from_columns(
            vec!["Hour".into(), "Fixture".into(), "Force".into()],
            vec![
                Value::Tensor(tensor(vec![8.0, 8.0, 9.0], 3, 1)),
                Value::StringArray(
                    StringArray::new(vec!["A".into(), "A".into(), "B".into()], vec![3, 1]).unwrap(),
                ),
                Value::Tensor(tensor(vec![f64::NAN, 4.0, 10.0], 3, 1)),
            ],
        )
        .unwrap();
        let count = parse_table_heatmap_args(&[
            table_with_missing_color,
            Value::String("Hour".into()),
            Value::String("Fixture".into()),
            Value::String("ColorVariable".into()),
            Value::String("Force".into()),
            Value::String("ColorMethod".into()),
            Value::String("count".into()),
        ])
        .unwrap();
        assert_eq!(count.color_data.materialize_f64(), vec![2.0, 0.0, 0.0, 1.0]);

        let before = clone_figure(current_figure_handle())
            .unwrap()
            .plots()
            .count();
        let error = futures::executor::block_on(heatmap_builtin(vec![
            table,
            Value::String("Hour".into()),
            Value::String("Fixture".into()),
            Value::String("ColorVariable".into()),
            Value::String("Force".into()),
            Value::String("ColorMethod".into()),
            Value::String("none".into()),
        ]))
        .expect_err("duplicate axis pairs cannot use no aggregation");
        assert_eq!(
            error.identifier(),
            HEATMAP_ERROR_INVALID_ARGUMENT.identifier
        );
        assert_eq!(
            clone_figure(current_figure_handle())
                .unwrap()
                .plots()
                .count(),
            before
        );

        let literal_missing = crate::builtins::table::table_from_columns(
            vec!["X".into(), "Y".into()],
            vec![
                Value::StringArray(
                    StringArray::new(vec!["".into(), "missing".into()], vec![2, 1]).unwrap(),
                ),
                Value::StringArray(
                    StringArray::new(vec!["one".into(), "one".into()], vec![2, 1]).unwrap(),
                ),
            ],
        )
        .unwrap();
        let categorized =
            parse_table_heatmap_args(&[literal_missing, Value::from("X"), Value::from("Y")])
                .expect("only observed nonmissing axis categories");
        assert_eq!(categorized.x_labels, vec!["missing".to_string()]);
        assert_eq!(categorized.color_data.materialize_f64(), vec![1.0]);
        let only_missing = crate::builtins::table::table_from_columns(
            vec!["X".into(), "Y".into()],
            vec![
                Value::StringArray(StringArray::new(vec!["".into()], vec![1, 1]).unwrap()),
                Value::StringArray(StringArray::new(vec!["A".into()], vec![1, 1]).unwrap()),
            ],
        )
        .unwrap();
        assert!(
            parse_table_heatmap_args(&[only_missing, Value::from("X"), Value::from("Y"),]).is_err()
        );

        let exact = crate::builtins::table::table_from_columns(
            vec!["X".into(), "Y".into(), "Color".into()],
            vec![
                Value::Tensor(tensor(vec![1.0, 2.0], 2, 1)),
                Value::StringArray(
                    StringArray::new(vec!["A".into(), "A".into()], vec![2, 1]).unwrap(),
                ),
                Value::Tensor(
                    Tensor::new_integer(
                        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                        vec![2, 1],
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        let exact_none = parse_table_heatmap_args(&[
            exact.clone(),
            Value::from("X"),
            Value::from("Y"),
            Value::from("ColorVariable"),
            Value::from("Color"),
            Value::from("ColorMethod"),
            Value::from("none"),
        ])
        .expect("complete unaggregated integer grid");
        assert_eq!(
            exact_none.color_data.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
        );
        let exact_handle = futures::executor::block_on(heatmap_builtin(vec![
            exact,
            Value::from("X"),
            Value::from("Y"),
            Value::from("ColorVariable"),
            Value::from("Color"),
            Value::from("ColorMethod"),
            Value::from("none"),
        ]))
        .expect("publish exact integer ColorData");
        let Value::Tensor(observed) =
            get_builtin(vec![Value::Num(exact_handle), Value::from("ColorData")])
                .expect("read retained ColorData")
        else {
            panic!("expected exact integer ColorData");
        };
        assert_eq!(
            observed.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
        );

        let incomplete = crate::builtins::table::table_from_columns(
            vec!["X".into(), "Y".into(), "Color".into()],
            vec![
                Value::Tensor(tensor(vec![1.0, 2.0], 2, 1)),
                Value::StringArray(
                    StringArray::new(vec!["A".into(), "B".into()], vec![2, 1]).unwrap(),
                ),
                Value::Tensor(
                    Tensor::new_integer(
                        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                        vec![2, 1],
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        let incomplete_result = parse_table_heatmap_args(&[
            incomplete,
            Value::from("X"),
            Value::from("Y"),
            Value::from("ColorVariable"),
            Value::from("Color"),
            Value::from("ColorMethod"),
            Value::from("none"),
        ]);
        let incomplete_error = match incomplete_result {
            Ok(_) => panic!("wide integer ColorData cannot be widened for holes"),
            Err(error) => error,
        };
        assert_eq!(
            incomplete_error.identifier(),
            runmat_builtins::HEATMAP_ERROR_UNSUPPORTED_COLOR_DATA.identifier
        );
        let single_holes = crate::builtins::table::table_from_columns(
            vec!["X".into(), "Y".into(), "Color".into()],
            vec![
                Value::Tensor(tensor(vec![1.0, 2.0], 2, 1)),
                Value::StringArray(
                    StringArray::new(vec!["A".into(), "B".into()], vec![2, 1]).unwrap(),
                ),
                Value::Tensor(Tensor::from_f32(vec![1.5, 2.5], vec![2, 1]).unwrap()),
            ],
        )
        .unwrap();
        let projected = parse_table_heatmap_args(&[
            single_holes,
            Value::from("X"),
            Value::from("Y"),
            Value::from("ColorVariable"),
            Value::from("Color"),
            Value::from("ColorMethod"),
            Value::from("none"),
        ])
        .expect("single class with missing grid cells");
        assert_eq!(
            projected.color_data.numeric_dtype(),
            runmat_value::NumericDType::F32
        );
        let data = projected.color_data.materialize_f64();
        assert_eq!(data[0], 1.5);
        assert!(data[1].is_nan() && data[2].is_nan());
        assert_eq!(data[3], 2.5);
    }
}
