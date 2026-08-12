//! MATLAB-compatible `stackedplot` builtin.

use glam::Vec4;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{LineStyle, PlotElement};
use runmat_value::{
    CellArray, NumericDType, ObjectInstance, StringArray, StructValue, Tensor, Value,
};

use super::plot::build_line_plot_for_builtin;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    plot_child_handle_snapshot, register_stackedplot_handle, render_stackedplot_chart,
    update_stackedplot_figure, update_stackedplot_handle_state, FigureHandle, PlotChildHandleState,
    StackedPlotHandleState, StackedSourceTableSnapshot,
};
use super::style::{
    marker_metadata_from_appearance, parse_line_style_args, value_as_string, LineAppearance,
    LineStyleParseOptions, MarkerKind,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::builtins::table::{
    is_tabular_object, parse_variable_selector_for_object, table_height,
    table_variable_names_from_object, table_variables, timetable_row_times,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "stackedplot";
const MAX_STACKED_VARIABLES: usize = 25;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the StackedLineChart object.",
}];

const INPUTS_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Vector or matrix data. Matrix columns are plotted in separate stacked axes.",
}];

const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Common x-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Vector or matrix y data.",
    },
];

const INPUTS_TABLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tbl",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Table or timetable data.",
}];

const INPUTS_PROPS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "props",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Variable selectors, line style, and StackedLineChart name-value options.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(Y)",
        inputs: &INPUTS_Y,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(tbl)",
        inputs: &INPUTS_TABLE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(tbl, vars)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(..., LineSpec)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(..., Name, Value)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "s = stackedplot(parent, ...)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STACKEDPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:stackedplot:InvalidArgument"),
    when: "Input data, table selectors, line style, parent, or StackedLineChart properties are invalid.",
    message: "stackedplot: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STACKEDPLOT.INTERNAL",
    identifier: Some("RunMat:stackedplot:Internal"),
    when: "Internal plotting state update fails.",
    message: "stackedplot: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const STACKEDPLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[derive(Clone, Debug)]
struct StackedLine {
    label: String,
    x: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Clone, Debug)]
struct StackedSeries {
    label: String,
    lines: Vec<StackedLine>,
}

#[derive(Clone, Debug)]
struct StackedOptions {
    vars: Option<Value>,
    x_variable: Vec<String>,
    combine_matching_names: bool,
    source_table: Option<StackedSourceTableSnapshot>,
    title: String,
    x_label: String,
    y_labels: Option<Vec<String>>,
    appearance: LineAppearance,
    visible: bool,
    grid_visible: bool,
    x_limits: Option<(f64, f64)>,
}

impl Default for StackedOptions {
    fn default() -> Self {
        Self {
            vars: None,
            x_variable: Vec::new(),
            combine_matching_names: true,
            source_table: None,
            title: String::new(),
            x_label: String::new(),
            y_labels: None,
            appearance: LineAppearance::default(),
            visible: true,
            grid_visible: true,
            x_limits: None,
        }
    }
}

#[derive(Clone, Debug)]
struct TabularInput {
    object: ObjectInstance,
}

#[runtime_builtin(
    name = "stackedplot",
    category = "plotting",
    summary = "Create a stacked plot of multiple variables with a common x-axis.",
    keywords = "stackedplot,stacked,line,table,timetable,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::stackedplot::STACKEDPLOT_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::stackedplot"
)]
pub fn stackedplot_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
    let (target, args) = split_parent(args)?;
    let (series, options) = parse_stackedplot_args(args)?;
    render_stackedplot(series, options, target)
}

fn split_parent(args: Vec<Value>) -> BuiltinResult<(Option<FigureHandle>, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    let rest = iter.collect::<Vec<_>>();
    if let Ok(handle) = resolve_plot_handle(&first, BUILTIN_NAME) {
        match handle {
            PlotHandle::Figure(handle) => {
                if !rest.is_empty()
                    && !looks_like_stacked_option(&rest[0])
                    && !looks_like_line_spec(&rest[0])
                {
                    return Ok((Some(handle), rest));
                }
            }
            PlotHandle::Axes(_, _) => {
                if !rest.is_empty()
                    && !looks_like_stacked_option(&rest[0])
                    && !looks_like_line_spec(&rest[0])
                {
                    return Err(stacked_err(
                        "parent must be a figure handle; axes handles are not valid stackedplot parents",
                    ));
                }
            }
            _ => {}
        }
    }
    let mut data_args = Vec::with_capacity(rest.len() + 1);
    data_args.push(first);
    data_args.extend(rest);
    Ok((None, data_args))
}

fn parse_stackedplot_args(args: Vec<Value>) -> BuiltinResult<(Vec<StackedSeries>, StackedOptions)> {
    if args.is_empty() {
        return Err(stacked_err(
            "expected table, timetable, vector, or matrix data",
        ));
    }

    if let Some((inputs, consumed)) = collect_tabular_inputs(&args)? {
        return parse_table_call(inputs, &args[consumed..]);
    }

    parse_matrix_call(&args)
}

fn collect_tabular_inputs(args: &[Value]) -> BuiltinResult<Option<(Vec<TabularInput>, usize)>> {
    match args.first() {
        Some(Value::Object(object)) if is_tabular_object(object) => {
            let mut inputs = Vec::new();
            for value in args {
                let Value::Object(object) = value else {
                    break;
                };
                if !is_tabular_object(object) {
                    break;
                }
                inputs.push(TabularInput {
                    object: object.clone(),
                });
            }
            let consumed = inputs.len();
            Ok(Some((inputs, consumed)))
        }
        Some(Value::Cell(cell)) => {
            if cell.data.is_empty() {
                return Ok(None);
            }
            let mut inputs = Vec::with_capacity(cell.data.len());
            for value in &cell.data {
                let Value::Object(object) = value else {
                    return Ok(None);
                };
                if !is_tabular_object(object) {
                    return Ok(None);
                }
                inputs.push(TabularInput {
                    object: object.clone(),
                });
            }
            Ok(Some((inputs, 1)))
        }
        _ => Ok(None),
    }
}

fn validate_tabular_kinds(inputs: &[TabularInput]) -> BuiltinResult<()> {
    if inputs.is_empty() {
        return Err(stacked_err("expected at least one table or timetable"));
    }
    let first_is_timetable = inputs[0].object.is_class("timetable");
    for input in inputs.iter().skip(1) {
        if input.object.is_class("timetable") != first_is_timetable {
            return Err(stacked_err(
                "multiple tabular inputs must be all tables or all timetables",
            ));
        }
    }
    Ok(())
}

fn source_table_snapshot(inputs: &[TabularInput]) -> BuiltinResult<StackedSourceTableSnapshot> {
    let mut classes = Vec::with_capacity(inputs.len());
    let mut variable_names = Vec::with_capacity(inputs.len());
    for input in inputs {
        classes.push(if input.object.is_class("timetable") {
            "timetable".into()
        } else {
            "table".into()
        });
        variable_names.push(
            table_variable_names_from_object(&input.object)
                .map_err(|err| stacked_err(format!("failed to read table variables: {err}")))?,
        );
    }
    Ok(StackedSourceTableSnapshot {
        classes,
        variable_names,
    })
}

fn parse_matrix_call(args: &[Value]) -> BuiltinResult<(Vec<StackedSeries>, StackedOptions)> {
    let mut data_end = if args.len() >= 2 && is_numeric_like(&args[0]) && is_numeric_like(&args[1])
    {
        2
    } else {
        1
    };
    if data_end > args.len() {
        data_end = args.len();
    }
    let (mut options, style_tokens) = parse_options(&args[data_end..], true)?;
    apply_style_tokens(&mut options, &style_tokens)?;

    let (x, y, explicit_x) = if data_end == 2 {
        (
            numeric_vector(&args[0], "X")?,
            numeric_tensor(&args[1], "Y")?,
            true,
        )
    } else {
        let y = numeric_tensor(&args[0], "Y")?;
        let rows = plotted_row_count(&y);
        (
            (1..=rows).map(|idx| idx as f64).collect::<Vec<_>>(),
            y,
            false,
        )
    };
    let rows = plotted_row_count(&y);
    if rows != x.len() {
        return Err(stacked_err(format!(
            "X length {} must match Y row count {rows}",
            x.len()
        )));
    }
    let columns = tensor_plot_columns(&y)?;
    if columns.len() > MAX_STACKED_VARIABLES {
        return Err(stacked_err(format!(
            "stackedplot supports at most {MAX_STACKED_VARIABLES} variables"
        )));
    }
    let labels = options
        .y_labels
        .clone()
        .unwrap_or_else(|| (1..=columns.len()).map(|idx| format!("Y{idx}")).collect());
    if labels.len() != columns.len() {
        return Err(stacked_err(
            "YLabels must match the number of plotted columns",
        ));
    }
    if options.x_label.is_empty() {
        options.x_label = if explicit_x { "X" } else { "Rows" }.into();
    }
    let mut series = Vec::with_capacity(columns.len());
    for (col, y) in columns.into_iter().enumerate() {
        series.push(StackedSeries {
            label: labels[col].clone(),
            lines: vec![StackedLine {
                label: labels[col].clone(),
                x: x.clone(),
                y,
            }],
        });
    }
    Ok((series, options))
}

fn parse_table_call(
    inputs: Vec<TabularInput>,
    rest: &[Value],
) -> BuiltinResult<(Vec<StackedSeries>, StackedOptions)> {
    let mut option_start = 0usize;
    let mut selector: Option<Value> = None;
    if let Some(first) = rest.first() {
        if !looks_like_stacked_option(first) && !looks_like_line_spec(first) {
            selector = Some(first.clone());
            option_start = 1;
        }
    }
    let (mut options, style_tokens) = parse_options(&rest[option_start..], false)?;
    apply_style_tokens(&mut options, &style_tokens)?;
    options.source_table = Some(source_table_snapshot(&inputs)?);
    validate_tabular_kinds(&inputs)?;
    let selected_value = options.vars.as_ref().or(selector.as_ref());

    let mut series = Vec::new();
    for (input_index, input) in inputs.iter().enumerate() {
        let object = &input.object;
        let names = table_variable_names_from_object(object)
            .map_err(|err| stacked_err(format!("failed to read table variables: {err}")))?;
        let variables = table_variables(object)
            .map_err(|err| stacked_err(format!("failed to read table variables: {err}")))?;
        let height = table_height(object)
            .map_err(|err| stacked_err(format!("failed to read table height: {err}")))?;
        let x = table_x_values(
            object,
            &variables,
            height,
            &options,
            input_index,
            inputs.len(),
        )?;
        let selected =
            parse_variable_selector_for_object(selected_value, object, &names).map_err(|err| {
                stacked_err(format!("failed to parse table variable selector: {err}"))
            })?;

        for name in selected {
            if table_x_variable_for_input(&options, input_index, inputs.len())?.as_deref()
                == Some(name.as_str())
            {
                continue;
            }
            let Some(value) = variables.fields.get(&name) else {
                return Err(stacked_err(format!(
                    "table variable `{name}` was not found"
                )));
            };
            let Ok(tensor) = numeric_tensor_from_value(value, &name) else {
                continue;
            };
            if plotted_row_count(&tensor) != height {
                return Err(stacked_err(format!(
                    "table variable `{name}` has {} rows but expected {height}",
                    plotted_row_count(&tensor)
                )));
            }
            let column_count = tensor_plot_column_count(&tensor);
            for (col, y) in tensor_plot_columns(&tensor)?.into_iter().enumerate() {
                let variable_label = if column_count == 1 {
                    name.clone()
                } else {
                    format!("{name}_{col_plus}", col_plus = col + 1)
                };
                let axis_label = if options.combine_matching_names || inputs.len() == 1 {
                    variable_label.clone()
                } else {
                    format!("{variable_label} (Input {input})", input = input_index + 1)
                };
                let line_label = if inputs.len() == 1 {
                    variable_label
                } else {
                    format!("{variable_label} (Input {input})", input = input_index + 1)
                };
                add_table_line(
                    &mut series,
                    axis_label,
                    StackedLine {
                        label: line_label,
                        x: x.clone(),
                        y,
                    },
                    options.combine_matching_names,
                );
            }
        }
    }
    if series.is_empty() {
        return Err(stacked_err(
            "table contains no selected plottable variables",
        ));
    }
    if series.len() > MAX_STACKED_VARIABLES {
        return Err(stacked_err(format!(
            "stackedplot supports at most {MAX_STACKED_VARIABLES} variables"
        )));
    }
    if let Some(labels) = options.y_labels.clone() {
        if labels.len() != series.len() {
            return Err(stacked_err(
                "YLabels must match the number of plotted variables",
            ));
        }
        for (series, label) in series.iter_mut().zip(labels) {
            series.label = label;
        }
    }
    if options.x_label.is_empty() {
        options.x_label = options.x_variable.first().cloned().unwrap_or_else(|| {
            if inputs[0].object.is_class("timetable") {
                "Time"
            } else {
                "Rows"
            }
            .into()
        });
    }
    Ok((series, options))
}

fn table_x_values(
    object: &ObjectInstance,
    variables: &StructValue,
    height: usize,
    options: &StackedOptions,
    input_index: usize,
    input_count: usize,
) -> BuiltinResult<Vec<f64>> {
    let x_variable = table_x_variable_for_input(options, input_index, input_count)?;
    if object.is_class("timetable") && x_variable.is_some() {
        return Err(stacked_err(
            "XVariable is only supported for table inputs; timetable row times are used automatically",
        ));
    }
    let x = if let Some(name) = x_variable {
        let value = variables
            .fields
            .get(&name)
            .ok_or_else(|| stacked_err(format!("XVariable `{name}` is not in the table")))?;
        numeric_vector_from_value(value, "XVariable")?
    } else if let Some(row_times) = timetable_row_times(object)
        .map_err(|err| stacked_err(format!("failed to read timetable RowTimes: {err}")))?
    {
        numeric_vector_from_value(&row_times, "RowTimes")?
    } else {
        (1..=height).map(|idx| idx as f64).collect()
    };
    if x.len() != height {
        return Err(stacked_err("x-axis values must match table height"));
    }
    Ok(x)
}

fn table_x_variable_for_input(
    options: &StackedOptions,
    input_index: usize,
    input_count: usize,
) -> BuiltinResult<Option<String>> {
    match options.x_variable.len() {
        0 => Ok(None),
        1 => Ok(Some(options.x_variable[0].clone())),
        len if len == input_count => Ok(Some(options.x_variable[input_index].clone())),
        _ => Err(stacked_err(
            "XVariable must be a single table variable or one variable per input table",
        )),
    }
}

fn add_table_line(
    series: &mut Vec<StackedSeries>,
    axis_label: String,
    line: StackedLine,
    combine_matching_names: bool,
) {
    if combine_matching_names {
        if let Some(existing) = series.iter_mut().find(|series| series.label == axis_label) {
            existing.lines.push(line);
            return;
        }
    }
    series.push(StackedSeries {
        label: axis_label,
        lines: vec![line],
    });
}

fn parse_options(
    args: &[Value],
    _allow_ylabels: bool,
) -> BuiltinResult<(StackedOptions, Vec<Value>)> {
    let mut options = StackedOptions::default();
    let mut style_tokens = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx == 0 && looks_like_line_spec(&args[idx]) {
            style_tokens.push(args[idx].clone());
            idx += 1;
            continue;
        }
        if idx + 1 >= args.len() {
            return Err(stacked_err("name-value options must come in pairs"));
        }
        let key = text_scalar(&args[idx], "property name")?;
        let lower = key.to_ascii_lowercase();
        match lower.as_str() {
            "xvariable" => options.x_variable = variable_selector(&args[idx + 1])?,
            "combinematchingnames" => {
                options.combine_matching_names =
                    bool_scalar(&args[idx + 1], "CombineMatchingNames")?;
            }
            "title" => options.title = text_scalar(&args[idx + 1], "Title")?,
            "xlabel" => options.x_label = text_scalar(&args[idx + 1], "XLabel")?,
            "displaylabels" | "ylabels" => {
                options.y_labels = Some(variable_selector(&args[idx + 1])?);
            }
            "displayvariables" => options.vars = Some(args[idx + 1].clone()),
            "visible" => options.visible = visible_bool(&args[idx + 1])?,
            "gridvisible" => options.grid_visible = visible_bool(&args[idx + 1])?,
            "xlimits" => options.x_limits = Some(parse_limits(&args[idx + 1], "XLimits")?),
            "eventsvisible" => {
                let visible = visible_bool(&args[idx + 1])?;
                if visible {
                    return Err(stacked_err(
                        "timetable event-table rendering is not supported yet",
                    ));
                }
            }
            "color" | "linestyle" | "linewidth" | "marker" | "markersize" => {
                style_tokens.push(Value::String(key));
                style_tokens.push(args[idx + 1].clone());
            }
            other => {
                return Err(stacked_err(format!(
                    "unsupported StackedLineChart option `{other}`"
                )));
            }
        }
        idx += 2;
    }
    Ok((options, style_tokens))
}

fn apply_style_tokens(options: &mut StackedOptions, tokens: &[Value]) -> BuiltinResult<()> {
    if tokens.is_empty() {
        return Ok(());
    }
    let parsed = parse_line_style_args(tokens, &LineStyleParseOptions::generic(BUILTIN_NAME))?;
    options.appearance = parsed.appearance;
    Ok(())
}

fn render_stackedplot(
    series: Vec<StackedSeries>,
    options: StackedOptions,
    target: Option<FigureHandle>,
) -> BuiltinResult<f64> {
    let axes_count = series.len();
    let line_count = series.iter().map(|series| series.lines.len()).sum();
    let mut line_indices = Vec::with_capacity(line_count);
    let line_group_counts = series
        .iter()
        .map(|series| series.lines.len())
        .collect::<Vec<_>>();
    let line_labels = series
        .iter()
        .flat_map(|series| series.lines.iter().map(|line| line.label.clone()))
        .collect::<Vec<_>>();
    let display_variables = series.iter().map(|s| s.label.clone()).collect::<Vec<_>>();
    let x_data = series
        .first()
        .and_then(|series| series.lines.first())
        .map(|line| line.x.clone())
        .unwrap_or_default();
    let y_data = series
        .iter()
        .flat_map(|series| series.lines.iter().map(|line| line.y.clone()))
        .collect::<Vec<_>>();
    let mut series_for_render = Some(series);
    let options_for_render = options.clone();
    let (figure, axes_indices, render_result) =
        render_stackedplot_chart(BUILTIN_NAME, target, axes_count, |figure, axes_indices| {
            let series = series_for_render
                .take()
                .expect("stackedplot series consumed exactly once");
            for (idx, series) in series.into_iter().enumerate() {
                let axes_index = axes_indices[idx];
                for line in series.lines {
                    let mut plot = build_line_plot_for_builtin(
                        BUILTIN_NAME,
                        line.x,
                        line.y,
                        &line.label,
                        &options_for_render.appearance,
                    )?;
                    plot.visible = options_for_render.visible;
                    let plot_index = figure.add_line_plot_on_axes(plot, axes_index);
                    line_indices.push(plot_index);
                }
                figure.set_axes_ylabel(axes_index, series.label);
                figure.set_axes_xlabel(
                    axes_index,
                    if idx + 1 == axes_count {
                        options_for_render.x_label.clone()
                    } else {
                        String::new()
                    },
                );
                if idx == 0 && !options_for_render.title.is_empty() {
                    figure.set_axes_title(axes_index, options_for_render.title.clone());
                }
                figure.set_axes_grid_enabled(axes_index, options_for_render.grid_visible);
                figure.set_axes_limits(axes_index, options_for_render.x_limits, None);
                let top = 0.08;
                let bottom = 0.10;
                let gap = 0.025;
                let usable =
                    (1.0 - top - bottom - gap * (axes_count.saturating_sub(1) as f64)).max(0.20);
                let height = usable / axes_count as f64;
                let y = bottom + (axes_count - 1 - idx) as f64 * (height + gap);
                figure.set_axes_position(axes_index, [0.13, y, 0.775, height]);
            }
            Ok(())
        })?;
    let _ = render_result;
    let state = StackedPlotHandleState {
        figure,
        axes_indices,
        line_plot_indices: line_indices,
        line_group_counts,
        line_labels,
        x_data,
        y_data,
        display_variables,
        source_table: options.source_table,
        x_variable: options.x_variable,
        combine_matching_names: options.combine_matching_names,
        x_label: options.x_label,
        title: options.title,
        appearance: options.appearance,
        visible: options.visible,
        grid_visible: options.grid_visible,
        x_limits: options.x_limits,
    };
    Ok(register_stackedplot_handle(state))
}

pub(crate) fn get_stackedplot_property(
    state: &StackedPlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match property.map(|p| p.to_ascii_lowercase()) {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("stackedplot".into()));
            st.insert("Parent", Value::Num(state.figure.as_u32() as f64));
            st.insert(
                "DisplayVariables",
                Value::StringArray(string_row(state.display_variables.clone())?),
            );
            st.insert(
                "SourceTable",
                source_table_value(state.source_table.as_ref())?,
            );
            st.insert("XVariable", string_or_row_value(&state.x_variable)?);
            st.insert(
                "CombineMatchingNames",
                Value::Bool(state.combine_matching_names),
            );
            st.insert("XData", vector_value(state.x_data.clone())?);
            st.insert("YData", matrix_from_columns(&state.y_data)?);
            st.insert("XLabel", Value::String(state.x_label.clone()));
            st.insert("Title", Value::String(state.title.clone()));
            st.insert(
                "Visible",
                Value::String(if state.visible { "on" } else { "off" }.into()),
            );
            st.insert("GridVisible", on_off_value(state.grid_visible));
            st.insert("XLimits", limits_output(state.x_limits)?);
            st.insert("LineWidth", Value::Num(state.appearance.line_width as f64));
            st.insert(
                "LineStyle",
                Value::String(line_style_name(state.appearance.line_style)),
            );
            st.insert("Color", color_value(state.appearance.color)?);
            Ok(Value::Struct(st))
        }
        Some(prop) => match prop.as_str() {
            "type" => Ok(Value::String("stackedplot".into())),
            "parent" => Ok(Value::Num(state.figure.as_u32() as f64)),
            "displayvariables" | "displaylabels" | "ylabels" => {
                string_row(state.display_variables.clone()).map(Value::StringArray)
            }
            "sourcetable" => source_table_value(state.source_table.as_ref()),
            "xvariable" => string_or_row_value(&state.x_variable),
            "combinematchingnames" => Ok(Value::Bool(state.combine_matching_names)),
            "xdata" => vector_value(state.x_data.clone()),
            "ydata" => matrix_from_columns(&state.y_data),
            "xlabel" => Ok(Value::String(state.x_label.clone())),
            "title" => Ok(Value::String(state.title.clone())),
            "visible" => Ok(Value::String(
                if state.visible { "on" } else { "off" }.into(),
            )),
            "gridvisible" => Ok(on_off_value(state.grid_visible)),
            "xlimits" => limits_output(state.x_limits),
            "linewidth" => Ok(Value::Num(state.appearance.line_width as f64)),
            "linestyle" => Ok(Value::String(line_style_name(state.appearance.line_style))),
            "color" => color_value(state.appearance.color),
            other => Err(stacked_runtime_error(
                builtin,
                format!("unsupported StackedLineChart property `{other}`"),
            )),
        },
    }
}

pub(crate) fn apply_stackedplot_property(
    handle: f64,
    _state: &StackedPlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let mut next = match plot_child_handle_snapshot(handle)
        .map_err(|err| stacked_runtime_error(builtin, err.to_string()))?
    {
        PlotChildHandleState::StackedPlot(state) => state,
        _ => return Err(stacked_runtime_error(builtin, "invalid stackedplot handle")),
    };
    match key.to_ascii_lowercase().as_str() {
        "title" => next.title = text_scalar(value, "Title")?,
        "xlabel" => next.x_label = text_scalar(value, "XLabel")?,
        "displaylabels" | "ylabels" => {
            let labels = variable_selector(value)?;
            if labels.len() != next.display_variables.len() {
                return Err(stacked_runtime_error(
                    builtin,
                    "DisplayLabels must match the number of plotted variables",
                ));
            }
            next.display_variables = labels;
        }
        "visible" => next.visible = visible_bool(value)?,
        "gridvisible" => next.grid_visible = visible_bool(value)?,
        "xlimits" => next.x_limits = Some(parse_limits(value, "XLimits")?),
        "color" | "linestyle" | "linewidth" | "marker" | "markersize" => {
            let tokens = [Value::String(key.to_string()), value.clone()];
            let parsed =
                parse_line_style_args(&tokens, &LineStyleParseOptions::generic(BUILTIN_NAME))?;
            next.appearance = merge_appearance(&next.appearance, parsed.appearance, key);
        }
        "sourcetable" | "xvariable" | "combinematchingnames" | "xdata" | "ydata" => {
            return Err(stacked_runtime_error(
                builtin,
                format!("StackedLineChart property `{key}` is read-only in this release"),
            ))
        }
        other => {
            return Err(stacked_runtime_error(
                builtin,
                format!("unsupported StackedLineChart set property `{other}`"),
            ))
        }
    }
    refresh_stackedplot_rendering(&next, builtin)?;
    update_stackedplot_handle_state(handle, next)
        .map_err(|err| stacked_runtime_error(builtin, err.to_string()))
}

fn refresh_stackedplot_rendering(
    state: &StackedPlotHandleState,
    builtin: &'static str,
) -> BuiltinResult<()> {
    update_stackedplot_figure(state, |figure| {
        for (idx, plot_index) in state.line_plot_indices.iter().copied().enumerate() {
            let Some(PlotElement::Line(line)) = figure.get_plot_mut(plot_index) else {
                return Err(super::state::FigureError::InvalidPlotObjectHandle);
            };
            line.visible = state.visible;
            line.color = state.appearance.color;
            line.line_width = state.appearance.line_width;
            line.line_style = state.appearance.line_style;
            line.marker =
                marker_metadata_from_appearance(&state.appearance).map(|marker| marker.into());
            if let Some(label) = state.line_labels.get(idx) {
                line.label = Some(label.clone());
            }
        }
        for (idx, axes_index) in state.axes_indices.iter().copied().enumerate() {
            if let Some(label) = state.display_variables.get(idx) {
                figure.set_axes_ylabel(axes_index, label.clone());
            }
            figure.set_axes_xlabel(
                axes_index,
                if idx + 1 == state.axes_indices.len() {
                    state.x_label.clone()
                } else {
                    String::new()
                },
            );
            if idx == 0 {
                figure.set_axes_title(axes_index, state.title.clone());
            }
            figure.set_axes_grid_enabled(axes_index, state.grid_visible);
            figure.set_axes_limits(axes_index, state.x_limits, None);
        }
        Ok(())
    })
    .map_err(|err| stacked_runtime_error(builtin, err.to_string()))
}

fn merge_appearance(
    previous: &LineAppearance,
    parsed: LineAppearance,
    key: &str,
) -> LineAppearance {
    let mut next = previous.clone();
    match key.to_ascii_lowercase().as_str() {
        "color" => next.color = parsed.color,
        "linestyle" => next.line_style = parsed.line_style,
        "linewidth" => next.line_width = parsed.line_width,
        "marker" => next.marker = parsed.marker,
        "markersize" => {
            let marker = next.marker.get_or_insert_with(default_marker);
            marker.size = parsed.marker.and_then(|m| m.size).or(marker.size);
        }
        _ => {}
    }
    next
}

fn default_marker() -> super::style::MarkerAppearance {
    super::style::MarkerAppearance {
        kind: MarkerKind::Circle,
        size: None,
        edge_color: super::style::MarkerColor::Auto,
        face_color: super::style::MarkerColor::Auto,
    }
}

fn is_numeric_like(value: &Value) -> bool {
    match value {
        Value::Tensor(_)
        | Value::GpuTensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_) => true,
        Value::Object(obj) => obj.is_class("datetime") || obj.is_class("duration"),
        _ => false,
    }
}

fn numeric_tensor(value: &Value, name: &str) -> BuiltinResult<Tensor> {
    numeric_tensor_from_value(value, name)
}

fn numeric_tensor_from_value(value: &Value, name: &str) -> BuiltinResult<Tensor> {
    match value {
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value)
                .map_err(|err| stacked_err(format!("{name}: {err}")))
        }
        Value::Object(obj) if obj.is_class("duration") => {
            crate::builtins::duration::duration_tensor_from_duration_value(value)
                .map_err(|err| stacked_err(format!("{name}: {err}")))
        }
        Value::Object(obj) if obj.is_class("categorical") => {
            let codes = obj
                .properties
                .get("Codes")
                .ok_or_else(|| stacked_err(format!("{name}: categorical Codes are missing")))?;
            numeric_tensor_from_value(codes, name)
        }
        Value::GpuTensor(_) => Err(stacked_err(
            "gpuArray inputs are not supported for stackedplot column splitting yet",
        )),
        Value::Num(value) => {
            Tensor::new_with_dtype(vec![*value], vec![1, 1], NumericDType::F64).map_err(stacked_err)
        }
        Value::Int(value) => {
            Tensor::new_with_dtype(vec![value.to_f64()], vec![1, 1], NumericDType::F64)
                .map_err(stacked_err)
        }
        Value::Bool(value) => Tensor::new_with_dtype(
            vec![if *value { 1.0 } else { 0.0 }],
            vec![1, 1],
            NumericDType::F64,
        )
        .map_err(stacked_err),
        Value::LogicalArray(array) => Tensor::new_with_dtype(
            array.data.iter().map(|value| *value as f64).collect(),
            array.shape.clone(),
            NumericDType::F64,
        )
        .map_err(stacked_err),
        other => Tensor::try_from(other).map_err(|err| stacked_err(format!("{name}: {err}"))),
    }
}

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    numeric_vector_from_value(value, name)
}

fn numeric_vector_from_value(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = numeric_tensor_from_value(value, name)?;
    let len = tensor_utils::tensor_element_len(&tensor);
    if tensor.rows() != len && tensor.cols() != len {
        return Err(stacked_err(format!("{name} must be a vector")));
    }
    Ok(tensor_utils::tensor_into_values_f64(tensor))
}

fn plotted_row_count(tensor: &Tensor) -> usize {
    if tensor.rows() == 1 || tensor.cols() == 1 {
        tensor_utils::tensor_element_len(tensor)
    } else {
        tensor.rows()
    }
}

fn tensor_plot_column_count(tensor: &Tensor) -> usize {
    if tensor.rows() == 1 || tensor.cols() == 1 {
        1
    } else {
        tensor.cols().max(1)
    }
}

fn tensor_plot_columns(tensor: &Tensor) -> BuiltinResult<Vec<Vec<f64>>> {
    if tensor.rows() == 1 || tensor.cols() == 1 {
        return Ok(vec![tensor_utils::tensor_values_f64(tensor)]);
    }
    let cols = tensor.cols().max(1);
    (0..cols).map(|col| tensor_column(tensor, col)).collect()
}

fn tensor_column(tensor: &Tensor, col: usize) -> BuiltinResult<Vec<f64>> {
    let rows = tensor.rows();
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let index = row + col * rows;
        out.push(tensor_utils::tensor_value_f64(tensor, index));
    }
    Ok(out)
}

fn variable_selector(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(array) => {
            if array.rows == 1 {
                Ok(vec![array.to_string().trim().to_string()])
            } else {
                Ok((0..array.rows)
                    .map(|row| {
                        let start = row * array.cols;
                        array.data[start..start + array.cols]
                            .iter()
                            .collect::<String>()
                            .trim()
                            .to_string()
                    })
                    .collect())
            }
        }
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|value| text_scalar(value, "variable selector"))
            .collect(),
        Value::Tensor(tensor) => Ok(tensor_utils::tensor_values_f64(tensor)
            .iter()
            .map(|v| {
                let index = *v as usize;
                format!("Var{index}")
            })
            .collect()),
        _ => Err(stacked_err(
            "variable selector must be text, string array, char array, or cellstr",
        )),
    }
}

fn looks_like_stacked_option(value: &Value) -> bool {
    let Some(key) = value_as_string(value) else {
        return false;
    };
    matches!(
        key.to_ascii_lowercase().as_str(),
        "xvariable"
            | "combinematchingnames"
            | "title"
            | "xlabel"
            | "displaylabels"
            | "displayvariables"
            | "ylabels"
            | "visible"
            | "gridvisible"
            | "xlimits"
            | "eventsvisible"
            | "color"
            | "linestyle"
            | "linewidth"
            | "marker"
            | "markersize"
    )
}

fn looks_like_line_spec(value: &Value) -> bool {
    let Some(token) = value_as_string(value) else {
        return false;
    };
    !token.is_empty() && token.chars().all(|c| "-:.ox+*^v<>sdphrgbcmykw".contains(c))
}

fn text_scalar(value: &Value, name: &str) -> BuiltinResult<String> {
    value_as_string(value).ok_or_else(|| stacked_err(format!("{name} must be text")))
}

fn bool_scalar(value: &Value, name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) => Ok(*value != 0.0),
        Value::Int(value) => Ok(value.to_f64() != 0.0),
        _ => Err(stacked_err(format!("{name} must be logical"))),
    }
}

fn visible_bool(value: &Value) -> BuiltinResult<bool> {
    if let Some(text) = value_as_string(value) {
        return match text.to_ascii_lowercase().as_str() {
            "on" => Ok(true),
            "off" => Ok(false),
            _ => Err(stacked_err("Visible must be 'on' or 'off'")),
        };
    }
    bool_scalar(value, "Visible")
}

fn parse_limits(value: &Value, name: &str) -> BuiltinResult<(f64, f64)> {
    let values = numeric_vector_from_value(value, name)?;
    if values.len() != 2 {
        return Err(stacked_err(format!("{name} must be a two-element vector")));
    }
    let lo = values[0];
    let hi = values[1];
    if lo.is_nan() || hi.is_nan() || lo >= hi {
        return Err(stacked_err(format!(
            "{name} must contain increasing numeric limits"
        )));
    }
    Ok((lo, hi))
}

fn vector_value(data: Vec<f64>) -> BuiltinResult<Value> {
    Tensor::new_with_dtype(data.clone(), vec![1, data.len()], NumericDType::F64)
        .map(Value::Tensor)
        .map_err(stacked_err)
}

fn empty_value() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("valid empty tensor"))
}

fn on_off_value(enabled: bool) -> Value {
    Value::String(if enabled { "on" } else { "off" }.into())
}

fn string_or_row_value(values: &[String]) -> BuiltinResult<Value> {
    match values.len() {
        0 => Ok(Value::String(String::new())),
        1 => Ok(Value::String(values[0].clone())),
        _ => string_row(values.to_vec()).map(Value::StringArray),
    }
}

fn limits_output(limits: Option<(f64, f64)>) -> BuiltinResult<Value> {
    limits.map_or_else(
        || Ok(empty_value()),
        |(lo, hi)| {
            Tensor::new_with_dtype(vec![lo, hi], vec![1, 2], NumericDType::F64)
                .map(Value::Tensor)
                .map_err(stacked_err)
        },
    )
}

fn source_table_value(snapshot: Option<&StackedSourceTableSnapshot>) -> BuiltinResult<Value> {
    let Some(snapshot) = snapshot else {
        return Ok(empty_value());
    };
    let mut st = StructValue::new();
    st.insert(
        "Class",
        Value::StringArray(string_row(snapshot.classes.clone())?),
    );
    let variables = snapshot
        .variable_names
        .iter()
        .map(|names| string_row(names.clone()).map(Value::StringArray))
        .collect::<BuiltinResult<Vec<_>>>()?;
    st.insert(
        "VariableNames",
        Value::Cell(
            CellArray::new(variables, 1, snapshot.variable_names.len())
                .map_err(|err| stacked_err(format!("SourceTable: {err}")))?,
        ),
    );
    Ok(Value::Struct(st))
}

fn matrix_from_columns(columns: &[Vec<f64>]) -> BuiltinResult<Value> {
    if columns.is_empty() {
        return Tensor::new_with_dtype(Vec::new(), vec![0, 0], NumericDType::F64)
            .map(Value::Tensor)
            .map_err(stacked_err);
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for col in columns {
        if col.len() != rows {
            let values = columns
                .iter()
                .map(|column| vector_value(column.clone()))
                .collect::<BuiltinResult<Vec<_>>>()?;
            return CellArray::new(values, 1, columns.len())
                .map(Value::Cell)
                .map_err(stacked_err);
        }
        data.extend_from_slice(col);
    }
    Tensor::new_with_dtype(data, vec![rows, cols], NumericDType::F64)
        .map(Value::Tensor)
        .map_err(stacked_err)
}

fn string_row(values: Vec<String>) -> BuiltinResult<StringArray> {
    StringArray::new(values.clone(), vec![1, values.len()]).map_err(stacked_err)
}

fn color_value(color: Vec4) -> BuiltinResult<Value> {
    Tensor::new_with_dtype(
        vec![color.x as f64, color.y as f64, color.z as f64],
        vec![1, 3],
        NumericDType::F64,
    )
    .map(Value::Tensor)
    .map_err(stacked_err)
}

fn line_style_name(style: LineStyle) -> String {
    match style {
        LineStyle::None => "none",
        LineStyle::Solid => "-",
        LineStyle::Dashed => "--",
        LineStyle::Dotted => ":",
        LineStyle::DashDot => "-.",
    }
    .into()
}

fn stacked_err(detail: impl AsRef<str>) -> RuntimeError {
    stacked_runtime_error(BUILTIN_NAME, detail)
}

fn stacked_runtime_error(builtin: &'static str, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{builtin}: {}", detail.as_ref())).with_builtin(builtin);
    if let Some(identifier) = ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::{
        clear_figure, clone_figure, PlotChildHandleState, PlotTestLockGuard,
    };
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::table::{table_from_columns, table_variable_names_from_object};

    fn setup() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        let _ = clear_figure(None);
        guard
    }

    fn tensor(data: &[f64], shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), shape).unwrap())
    }

    fn stacked_state(handle: f64) -> StackedPlotHandleState {
        match super::super::state::plot_child_handle_snapshot(handle).unwrap() {
            PlotChildHandleState::StackedPlot(state) => state,
            other => panic!("unexpected state {other:?}"),
        }
    }

    #[test]
    fn stackedplot_numeric_vector_reads_typed_integer_storage_exactly() {
        let x = Tensor::new_integer(runmat_value::IntegerStorage::U16(vec![1, 2, 3]), vec![1, 3])
            .expect("typed x vector");

        assert_eq!(
            numeric_vector(&Value::Tensor(x), "X").expect("numeric vector"),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn stackedplot_columns_reads_typed_integer_storage_exactly() {
        let y = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![10, 20, 30, 40]),
            vec![2, 2],
        )
        .expect("typed y matrix");

        assert_eq!(
            tensor_plot_columns(&y).expect("plot columns"),
            vec![vec![10.0, 20.0], vec![30.0, 40.0]]
        );
    }

    #[test]
    fn stackedplot_matrix_creates_one_axes_per_column() {
        let _guard = setup();
        let handle = stackedplot_builtin(vec![tensor(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])]).unwrap();
        let state = stacked_state(handle);
        assert_eq!(state.axes_indices, vec![0, 1]);
        assert_eq!(state.display_variables, vec!["Y1", "Y2"]);
        let fig = clone_figure(state.figure).unwrap();
        assert_eq!(fig.axes_count(), 2);
        assert_eq!(fig.plots().count(), 2);
    }

    #[test]
    fn stackedplot_row_vector_is_one_series() {
        let _guard = setup();
        let handle = stackedplot_builtin(vec![tensor(&[1.0, 2.0, 3.0], vec![1, 3])]).unwrap();
        let state = stacked_state(handle);
        assert_eq!(state.axes_indices.len(), 1);
        assert_eq!(state.line_plot_indices.len(), 1);
        assert_eq!(state.x_data, vec![1.0, 2.0, 3.0]);
        assert_eq!(state.y_data, vec![vec![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn stackedplot_scalar_input_is_data_not_parent() {
        let _guard = setup();
        let handle = stackedplot_builtin(vec![Value::Num(1.0)]).unwrap();
        let state = stacked_state(handle);
        assert_eq!(state.axes_indices.len(), 1);
        assert_eq!(state.x_data, vec![1.0]);
        assert_eq!(state.y_data, vec![vec![1.0]]);
    }

    #[test]
    fn stackedplot_explicit_x_y_vectors_stay_one_series() {
        let _guard = setup();
        let handle = stackedplot_builtin(vec![
            tensor(&[10.0, 20.0, 30.0], vec![3, 1]),
            tensor(&[1.0, 4.0, 9.0], vec![1, 3]),
        ])
        .unwrap();
        let state = stacked_state(handle);
        assert_eq!(state.axes_indices.len(), 1);
        assert_eq!(state.x_data, vec![10.0, 20.0, 30.0]);
        assert_eq!(state.y_data, vec![vec![1.0, 4.0, 9.0]]);
    }

    #[test]
    fn stackedplot_accepts_table_selector_and_xvariable() {
        let _guard = setup();
        let table = table_from_columns(
            vec!["T".into(), "A".into(), "B".into()],
            vec![
                Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1])
                    .map(Value::Tensor)
                    .unwrap(),
                Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1])
                    .map(Value::Tensor)
                    .unwrap(),
                Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1])
                    .map(Value::Tensor)
                    .unwrap(),
            ],
        )
        .unwrap();
        let names = table_variable_names_from_object(match &table {
            Value::Object(object) => object,
            _ => unreachable!(),
        })
        .unwrap();
        assert!(names.contains(&"A".into()));
        let handle = stackedplot_builtin(vec![
            table,
            Value::StringArray(StringArray::new(vec!["A".into(), "B".into()], vec![1, 2]).unwrap()),
            Value::String("XVariable".into()),
            Value::String("T".into()),
        ])
        .unwrap();
        let x = get_builtin(vec![Value::Num(handle), Value::String("XData".into())]).unwrap();
        match x {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![10.0, 20.0, 30.0]),
            other => panic!("unexpected XData {other:?}"),
        }
    }

    #[test]
    fn stackedplot_uses_table_numeric_and_logical_selectors() {
        let _guard = setup();
        let table = table_from_columns(
            vec!["T".into(), "A".into(), "B".into()],
            vec![
                Tensor::new(vec![10.0, 20.0], vec![2, 1])
                    .map(Value::Tensor)
                    .unwrap(),
                Tensor::new(vec![1.0, 2.0], vec![2, 1])
                    .map(Value::Tensor)
                    .unwrap(),
                Tensor::new(vec![3.0, 4.0], vec![2, 1])
                    .map(Value::Tensor)
                    .unwrap(),
            ],
        )
        .unwrap();
        let numeric = stackedplot_builtin(vec![
            table.clone(),
            tensor(&[2.0, 3.0], vec![1, 2]),
            Value::String("XVariable".into()),
            Value::String("T".into()),
        ])
        .unwrap();
        assert_eq!(stacked_state(numeric).display_variables, vec!["A", "B"]);

        let logical = stackedplot_builtin(vec![
            table,
            Value::LogicalArray(
                runmat_value::LogicalArray::new(vec![0, 1, 1], vec![1, 3]).unwrap(),
            ),
            Value::String("XVariable".into()),
            Value::String("T".into()),
        ])
        .unwrap();
        assert_eq!(stacked_state(logical).display_variables, vec!["A", "B"]);
    }

    #[test]
    fn stackedplot_plots_logical_table_variables() {
        let _guard = setup();
        let table = table_from_columns(
            vec!["Flag".into()],
            vec![Value::LogicalArray(
                runmat_value::LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap(),
            )],
        )
        .unwrap();
        let handle = stackedplot_builtin(vec![table]).unwrap();
        let state = stacked_state(handle);
        assert_eq!(state.display_variables, vec!["Flag"]);
        assert_eq!(state.y_data, vec![vec![1.0, 0.0, 1.0]]);
    }

    #[test]
    fn stackedplot_accepts_per_table_xvariable_names() {
        let _guard = setup();
        let table1 = table_from_columns(
            vec!["T1".into(), "A".into()],
            vec![
                Tensor::new(vec![1.0, 2.0], vec![2, 1])
                    .map(Value::Tensor)
                    .unwrap(),
                Tensor::new(vec![10.0, 20.0], vec![2, 1])
                    .map(Value::Tensor)
                    .unwrap(),
            ],
        )
        .unwrap();
        let table2 = table_from_columns(
            vec!["T2".into(), "A".into()],
            vec![
                Tensor::new(vec![3.0, 4.0], vec![2, 1])
                    .map(Value::Tensor)
                    .unwrap(),
                Tensor::new(vec![30.0, 40.0], vec![2, 1])
                    .map(Value::Tensor)
                    .unwrap(),
            ],
        )
        .unwrap();
        let handle = stackedplot_builtin(vec![
            table1,
            table2,
            Value::String("A".into()),
            Value::String("XVariable".into()),
            Value::StringArray(
                StringArray::new(vec!["T1".into(), "T2".into()], vec![1, 2]).unwrap(),
            ),
        ])
        .unwrap();
        let state = stacked_state(handle);
        assert_eq!(state.x_variable, vec!["T1", "T2"]);
        assert_eq!(state.line_group_counts, vec![2]);
    }

    #[test]
    fn stackedplot_rejects_timetable_xvariable() {
        let _guard = setup();
        let mut table = match table_from_columns(
            vec!["A".into()],
            vec![Tensor::new(vec![1.0, 2.0], vec![2, 1])
                .map(Value::Tensor)
                .unwrap()],
        )
        .unwrap()
        {
            Value::Object(object) => object,
            _ => unreachable!(),
        };
        table.class_name = "timetable".into();
        table.properties.insert(
            "RowTimes".into(),
            Tensor::new(vec![10.0, 20.0], vec![2, 1])
                .map(Value::Tensor)
                .unwrap(),
        );
        let err = stackedplot_builtin(vec![
            Value::Object(table),
            Value::String("A".into()),
            Value::String("XVariable".into()),
            Value::String("A".into()),
        ])
        .unwrap_err();
        assert!(err.to_string().contains("XVariable is only supported"));
    }

    #[test]
    fn stackedplot_ydata_handles_unequal_table_lengths() {
        let _guard = setup();
        let table1 = table_from_columns(
            vec!["A".into()],
            vec![Tensor::new(vec![1.0, 2.0], vec![2, 1])
                .map(Value::Tensor)
                .unwrap()],
        )
        .unwrap();
        let table2 = table_from_columns(
            vec!["A".into()],
            vec![Tensor::new(vec![3.0, 4.0, 5.0], vec![3, 1])
                .map(Value::Tensor)
                .unwrap()],
        )
        .unwrap();
        let handle = stackedplot_builtin(vec![table1, table2, Value::String("A".into())]).unwrap();
        let y = get_builtin(vec![Value::Num(handle), Value::String("YData".into())]).unwrap();
        match y {
            Value::Cell(cell) => assert_eq!(cell.data.len(), 2),
            other => panic!("expected cell YData for unequal lengths, got {other:?}"),
        }
    }

    #[test]
    fn stackedplot_multiple_tables_combine_matching_names() {
        let _guard = setup();
        let table1 = table_from_columns(
            vec!["A".into(), "B".into()],
            vec![
                Tensor::new(vec![1.0, 2.0], vec![2, 1])
                    .map(Value::Tensor)
                    .unwrap(),
                Tensor::new(vec![10.0, 20.0], vec![2, 1])
                    .map(Value::Tensor)
                    .unwrap(),
            ],
        )
        .unwrap();
        let table2 = table_from_columns(
            vec!["A".into(), "B".into()],
            vec![
                Tensor::new(vec![3.0, 4.0], vec![2, 1])
                    .map(Value::Tensor)
                    .unwrap(),
                Tensor::new(vec![30.0, 40.0], vec![2, 1])
                    .map(Value::Tensor)
                    .unwrap(),
            ],
        )
        .unwrap();
        let handle = stackedplot_builtin(vec![table1, table2, Value::String("A".into())]).unwrap();
        let state = stacked_state(handle);
        assert_eq!(state.display_variables, vec!["A"]);
        assert_eq!(state.line_group_counts, vec![2]);
        assert_eq!(state.line_plot_indices.len(), 2);
    }

    #[test]
    fn stackedplot_combine_matching_names_false_separates_tables() {
        let _guard = setup();
        let table1 = table_from_columns(
            vec!["A".into()],
            vec![Tensor::new(vec![1.0, 2.0], vec![2, 1])
                .map(Value::Tensor)
                .unwrap()],
        )
        .unwrap();
        let table2 = table_from_columns(
            vec!["A".into()],
            vec![Tensor::new(vec![3.0, 4.0], vec![2, 1])
                .map(Value::Tensor)
                .unwrap()],
        )
        .unwrap();
        let handle = stackedplot_builtin(vec![
            table1,
            table2,
            Value::String("A".into()),
            Value::String("CombineMatchingNames".into()),
            Value::Bool(false),
        ])
        .unwrap();
        let state = stacked_state(handle);
        assert_eq!(state.display_variables, vec!["A (Input 1)", "A (Input 2)"]);
        assert_eq!(state.line_group_counts, vec![1, 1]);
    }

    #[test]
    fn stackedplot_rejects_axes_parent() {
        let _guard = setup();
        let axes = super::super::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(1.0),
        )
        .unwrap();
        let err = stackedplot_builtin(vec![Value::Num(axes), tensor(&[1.0, 2.0], vec![1, 2])])
            .unwrap_err();
        assert!(err.to_string().contains("parent must be a figure handle"));
    }

    #[test]
    fn stackedplot_set_updates_labels_and_visibility() {
        let _guard = setup();
        let handle = stackedplot_builtin(vec![tensor(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])]).unwrap();
        set_builtin(vec![
            Value::Num(handle),
            Value::String("DisplayLabels".into()),
            Value::StringArray(StringArray::new(vec!["A".into(), "B".into()], vec![1, 2]).unwrap()),
            Value::String("Visible".into()),
            Value::String("off".into()),
            Value::String("GridVisible".into()),
            Value::String("off".into()),
            Value::String("XLimits".into()),
            tensor(&[f64::NEG_INFINITY, 10.0], vec![1, 2]),
        ])
        .unwrap();
        let state = stacked_state(handle);
        assert_eq!(state.display_variables, vec!["A", "B"]);
        assert!(!state.visible);
        assert!(!state.grid_visible);
        assert_eq!(state.x_limits, Some((f64::NEG_INFINITY, 10.0)));
    }
}
