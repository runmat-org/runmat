use super::*;
pub(super) struct ParsedAnimatedLine {
    pub(super) axes_target: AxesTarget,
    pub(super) object: AnimatedLineObject,
    pub(super) maximum_num_points: Option<usize>,
    pub(super) property_pairs: Vec<Value>,
}

pub(super) enum AnimatedLineObject {
    TwoD(LinePlot),
    ThreeD(Line3Plot),
}

pub(super) async fn parse_animatedline_args(
    mut axes_target: AxesTarget,
    args: Vec<Value>,
) -> BuiltinResult<ParsedAnimatedLine> {
    validate_pair_syntax(&args)?;
    let mut idx = 0usize;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut z = None;
    if args.len() >= 2 && is_numeric_value(&args[0]) && is_numeric_value(&args[1]) {
        x = numeric_vector(args[0].clone(), "X").await?;
        y = numeric_vector(args[1].clone(), "Y").await?;
        idx = 2;
        if args.get(idx).is_some_and(is_numeric_value) {
            z = Some(numeric_vector(args[idx].clone(), "Z").await?);
            idx += 1;
        }
    } else if args.first().is_some_and(is_numeric_value) {
        return Err(animatedline_err("expected both X and Y coordinate vectors"));
    }

    let mut property_pairs = Vec::new();
    let mut style_tokens = Vec::new();
    let mut maximum_num_points = Some(DEFAULT_MAXIMUM_NUM_POINTS);
    while idx < args.len() {
        let key = value_as_string(&args[idx])
            .map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "animatedline",
                    source.to_string(),
                    source,
                )
            })?
            .ok_or_else(|| animatedline_err("property names must be strings"))?;
        idx += 1;
        if idx >= args.len() {
            return Err(animatedline_err("name-value arguments must come in pairs"));
        }
        let value = args[idx].clone();
        idx += 1;
        match key.trim().to_ascii_lowercase().as_str() {
            "xdata" => {
                x = numeric_vector(value, "XData").await?;
            }
            "ydata" => {
                y = numeric_vector(value, "YData").await?;
            }
            "zdata" => {
                z = Some(numeric_vector(value, "ZData").await?);
            }
            "parent" => {
                let value = materialize_control(value).await?;
                let PlotHandle::Axes(handle, axes_index) =
                    resolve_plot_handle(&value, BUILTIN_NAME).map_err(|error| {
                        contextual_error(error, "RunMat:animatedline:InvalidArgument")
                    })?
                else {
                    return Err(animatedline_err("Parent must be an axes handle"));
                };
                axes_target = Some((handle, axes_index));
            }
            "maximumnumpoints" => {
                maximum_num_points = maximum_from_value(value).await?;
            }
            _ => {
                let value = if matches!(
                    key.trim().to_ascii_lowercase().as_str(),
                    "color" | "linewidth" | "markersize" | "markeredgecolor" | "markerfacecolor"
                ) {
                    materialize_control(value).await?
                } else {
                    value
                };
                style_tokens.push(Value::string(key.clone()));
                style_tokens.push(value.clone());
                property_pairs.push(Value::string(key));
                property_pairs.push(value);
            }
        }
    }

    if x.len() != y.len() || z.as_ref().is_some_and(|z| z.len() != x.len()) {
        return Err(animatedline_err(
            "coordinate vectors must have the same length",
        ));
    }

    trim_initial_points(maximum_num_points, &mut x, &mut y, z.as_mut());

    let parsed_style =
        parse_line_style_args(&style_tokens, &LineStyleParseOptions::generic(BUILTIN_NAME))
            .map_err(|error| contextual_error(error, "RunMat:animatedline:InvalidArgument"))?;
    let mut appearance = parsed_style.appearance;
    if !parsed_style.color_explicit {
        let target = axes_target.unwrap_or_else(|| {
            let axes = current_axes_state();
            (axes.handle, axes.active_index)
        });
        appearance.color = line_color_for_target_axes_series_index(target.0, target.1, 0);
    }
    let label = parsed_style.label.unwrap_or_default();
    let object = build_object(x, y, z, &label, &appearance)
        .map_err(|error| contextual_error(error, "RunMat:animatedline:InvalidArgument"))?;
    Ok(ParsedAnimatedLine {
        axes_target,
        object,
        maximum_num_points,
        property_pairs,
    })
}

// Only syntax is checked here. Property semantics remain owned by the existing
// style/property parsers; this prevents malformed pairs from initiating gather.
pub(super) fn validate_pair_syntax(args: &[Value]) -> BuiltinResult<()> {
    let mut start = 0;
    if args.len() >= 2 && is_numeric_value(&args[0]) && is_numeric_value(&args[1]) {
        start = 2;
        if args.get(start).is_some_and(is_numeric_value) {
            start += 1;
        }
    } else if args.first().is_some_and(is_numeric_value) {
        return Err(animatedline_err("expected both X and Y coordinate vectors"));
    }
    if !(args.len() - start).is_multiple_of(2) {
        return Err(animatedline_err("name-value arguments must come in pairs"));
    }
    for pair in args[start..].chunks_exact(2) {
        if value_as_string(&pair[0])
            .map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "animatedline",
                    source.to_string(),
                    source,
                )
            })?
            .is_none()
        {
            return Err(animatedline_err("property names must be strings"));
        }
    }
    Ok(())
}

pub(super) async fn materialize_control(value: Value) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::gather_if_needed_async(&value)
            .await
            .map_err(|error| contextual_error(error, "RunMat:animatedline:Internal"))
    } else {
        Ok(value)
    }
}

fn build_object(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Option<Vec<f64>>,
    label: &str,
    appearance: &LineAppearance,
) -> BuiltinResult<AnimatedLineObject> {
    if let Some(z) = z {
        if !x.is_empty() {
            if appearance.marker.is_some() {
                return Err(animatedline_err(
                    "3-D animated lines do not support marker properties yet",
                ));
            }
            return Ok(AnimatedLineObject::ThreeD(build_line3_plot_for_builtin(
                BUILTIN_NAME,
                x,
                y,
                z,
                label,
                appearance,
            )?));
        }
    }
    Ok(AnimatedLineObject::TwoD(build_line_plot_for_builtin(
        BUILTIN_NAME,
        x,
        y,
        label,
        appearance,
    )?))
}

async fn numeric_vector(value: Value, name: &str) -> BuiltinResult<Vec<f64>> {
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
        return Err(animatedline_err(format!("{name} must not be logical data")));
    }
    let tensor = NumericInput::from_value(value, BUILTIN_NAME)
        .map_err(|error| contextual_error(error, "RunMat:animatedline:InvalidArgument"))?
        .into_tensor_async(BUILTIN_NAME)
        .await?;
    if !is_vector_tensor(&tensor) {
        return Err(animatedline_err(format!(
            "{name} must be a scalar or vector"
        )));
    }
    Ok(tensor_utils::tensor_into_values_f64(tensor))
}

fn is_numeric_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    )
}

fn is_vector_tensor(tensor: &Tensor) -> bool {
    tensor.rows() == 1 || tensor.cols() == 1
}

pub(super) async fn maximum_from_value(value: Value) -> BuiltinResult<Option<usize>> {
    if matches!(value, Value::GpuTensor(_)) {
        let tensor = NumericInput::from_value(value, BUILTIN_NAME)?
            .into_tensor_async(BUILTIN_NAME)
            .await?;
        return animated_line_maximum_from_value(&Value::Tensor(tensor), BUILTIN_NAME)
            .map_err(|error| contextual_error(error, "RunMat:animatedline:InvalidArgument"));
    }
    animated_line_maximum_from_value(&value, BUILTIN_NAME)
        .map_err(|error| contextual_error(error, "RunMat:animatedline:InvalidArgument"))
}

fn trim_initial_points(
    maximum: Option<usize>,
    x: &mut Vec<f64>,
    y: &mut Vec<f64>,
    z: Option<&mut Vec<f64>>,
) {
    let Some(maximum) = maximum else {
        return;
    };
    if x.len() <= maximum {
        return;
    }
    let drop = x.len() - maximum;
    x.drain(0..drop);
    y.drain(0..drop);
    if let Some(z) = z {
        z.drain(0..drop);
    }
}
