//! Ordered, typed option admission before any resident observation transfer.
use super::*;
use runmat_accelerate_api::GpuTensorStorage;

pub(super) struct ScatterhistOptionPlan {
    steps: Vec<PlannedOption>,
}

enum PlannedOption {
    Host(DecodedOption),
    Resident(ScatterhistOptionKind, Value),
}

enum DecodedOption {
    Bins([usize; 2]),
    Group(Vec<Option<KeyAtom>>),
    Kernel(KernelMode),
    Legend(bool),
    PlotGroup(bool),
    Location(Location),
    Direction(Direction),
    Colors(Vec<Vec4>),
    Markers(Vec<MarkerStyle>),
    MarkerSizes(Vec<f32>),
    LineWidths(Vec<f32>),
    LineStyles(Vec<LineStyle>),
    Style(KernelMode, HistStyle),
    Bandwidths(Vec<[f64; 2]>),
    Parent(FigureHandle),
}

impl ScatterhistOptionPlan {
    pub(super) fn admit(
        args: crate::call::BuiltinArguments,
        observations: usize,
    ) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(invalid("scatterhist: name-value options must be paired"));
        }
        let mut steps = Vec::with_capacity(args.len() / 2);
        let mut values = args.into_iter();
        while let Some(name) = values.next() {
            let text = scalar_text(&name)?
                .ok_or_else(|| invalid("scatterhist: option names must be text"))?;
            let kind = ScatterhistOptionKind::parse(&text).ok_or_else(|| {
                invalid(format!(
                    "scatterhist: unsupported option '{}'",
                    text.trim().to_ascii_lowercase()
                ))
            })?;
            let value = values.next().expect("paired options checked above");
            steps.push(plan_step(kind, value, observations)?);
        }
        Ok(Self { steps })
    }

    pub(super) async fn evaluate(self, observations: usize) -> BuiltinResult<ScatterhistOptions> {
        let mut options = ScatterhistOptions::default();
        for step in self.steps {
            let decoded = match step {
                PlannedOption::Host(decoded) => decoded,
                PlannedOption::Resident(kind, value) => {
                    let gathered = gather_if_needed_async(&value)
                        .await
                        .map_err(|err| invalid_with_source(format!("scatterhist: {err}"), err))?;
                    if kind.integer_role() == ScatterhistIntegerRole::RendererFloat
                        && crate::builtins::common::validation::value_has_native_integer_class(
                            &gathered,
                        )
                        && !crate::builtins::common::validation::native_integer_value_is_exact_f64(
                            &gathered,
                        )
                    {
                        return Err(invalid("scatterhist: integer option value is not exactly representable as double"));
                    }
                    decode_option(kind, &gathered, observations)?
                }
            };
            decoded.apply(&mut options);
        }
        Ok(options)
    }
}

fn plan_step(
    kind: ScatterhistOptionKind,
    value: Value,
    observations: usize,
) -> BuiltinResult<PlannedOption> {
    let integer = if kind == ScatterhistOptionKind::Group {
        crate::builtins::common::validation::value_contains_native_integer_class(&value)
    } else {
        crate::builtins::common::validation::value_has_native_integer_class(&value)
    };
    if integer {
        match kind.integer_role() {
            ScatterhistIntegerRole::ParentHandle => {
                return Err(invalid(
                    "scatterhist: typed integer values are not graphics Parent handles",
                ))
            }
            ScatterhistIntegerRole::TextOnly => {
                return Err(invalid("scatterhist: this option requires text"))
            }
            ScatterhistIntegerRole::Structural => {
                let extension = if kind == ScatterhistOptionKind::Group {
                    &SCATTERHIST_INTEGER_GROUP_EXTENSION
                } else {
                    &SCATTERHIST_INTEGER_OPTION_EXTENSION
                };
                crate::compatibility::ensure_builtin_extension_enabled(extension, NAME)?;
            }
            ScatterhistIntegerRole::RendererFloat => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &SCATTERHIST_INTEGER_OPTION_EXTENSION,
                    NAME,
                )?;
                if !crate::value_contains_gpu(&value)
                    && !crate::builtins::common::validation::native_integer_value_is_exact_f64(
                        &value,
                    )
                {
                    return Err(invalid(
                        "scatterhist: integer option value is not exactly representable as double",
                    ));
                }
            }
        }
    }
    let resident = crate::value_contains_gpu(&value);
    if resident {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SCATTERHIST_GPU_INPUT_EXTENSION,
            NAME,
        )?;
        if kind == ScatterhistOptionKind::Group {
            preflight_group_labels(&value, observations)?;
        }
        if let Value::GpuTensor(handle) = &value {
            validate_resident_option(kind, handle, observations)?;
        } else if kind != ScatterhistOptionKind::Group {
            return Err(invalid("scatterhist: resident values inside option containers are only supported for Group"));
        }
        return Ok(PlannedOption::Resident(kind, value));
    }
    if kind == ScatterhistOptionKind::Group {
        preflight_group_labels(&value, observations)?;
    }
    Ok(PlannedOption::Host(decode_option(
        kind,
        &value,
        observations,
    )?))
}

fn validate_resident_option(
    kind: ScatterhistOptionKind,
    handle: &runmat_accelerate_api::GpuTensorHandle,
    observations: usize,
) -> BuiltinResult<()> {
    if handle.physical().storage() != GpuTensorStorage::Real {
        return Err(invalid("scatterhist: numeric option must be real"));
    }
    let count = handle
        .shape()
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(dim))
        .ok_or_else(|| invalid("scatterhist: option shape overflows"))?;
    let vector = handle.shape().iter().filter(|&&dim| dim > 1).count() <= 1;
    let valid = match kind {
        ScatterhistOptionKind::NBins => vector && (count == 1 || count == 2),
        ScatterhistOptionKind::Group => vector && count == observations,
        ScatterhistOptionKind::Kernel
        | ScatterhistOptionKind::Legend
        | ScatterhistOptionKind::PlotGroup => count == 1,
        ScatterhistOptionKind::MarkerSize | ScatterhistOptionKind::LineWidth => vector && count > 0,
        ScatterhistOptionKind::Bandwidth => {
            let shape = handle.shape();
            let rows = shape.first().copied().unwrap_or(1);
            let columns = shape.get(1).copied().unwrap_or(1);
            let matrix = shape.iter().skip(2).all(|&extent| extent <= 1);
            matrix && count > 0 && (count <= 2 || rows == 2 || columns == 2)
        }
        ScatterhistOptionKind::Color => count == 3 || (count > 0 && handle.shape().contains(&3)),
        ScatterhistOptionKind::Location
        | ScatterhistOptionKind::Direction
        | ScatterhistOptionKind::Marker
        | ScatterhistOptionKind::LineStyle
        | ScatterhistOptionKind::Style
        | ScatterhistOptionKind::Parent => false,
    };
    if !valid {
        return Err(invalid("scatterhist: option has invalid class or shape"));
    }
    Ok(())
}

fn decode_option(
    kind: ScatterhistOptionKind,
    value: &Value,
    observations: usize,
) -> BuiltinResult<DecodedOption> {
    use ScatterhistOptionKind as K;
    Ok(match kind {
        K::NBins => DecodedOption::Bins(parse_bins(value)?),
        K::Group => DecodedOption::Group(group_labels(value, observations)?),
        K::Kernel => DecodedOption::Kernel(parse_kernel(value)?),
        K::Legend => DecodedOption::Legend(option_bool(value, "Legend")?),
        K::PlotGroup => DecodedOption::PlotGroup(option_bool(value, "PlotGroup")?),
        K::Location => DecodedOption::Location(parse_location(value)?),
        K::Direction => DecodedOption::Direction(parse_direction(value)?),
        K::Color => DecodedOption::Colors(parse_color_cycle(value)?),
        K::Marker => DecodedOption::Markers(parse_marker_cycle(value)?),
        K::MarkerSize => DecodedOption::MarkerSizes(nonnegative_scalar_cycle(value, "MarkerSize")?),
        K::LineWidth => DecodedOption::LineWidths(nonnegative_scalar_cycle(value, "LineWidth")?),
        K::LineStyle => DecodedOption::LineStyles(parse_line_style_cycle(value)?),
        K::Style => {
            let (kernel, style) = parse_style(value)?;
            DecodedOption::Style(kernel, style)
        }
        K::Bandwidth => DecodedOption::Bandwidths(parse_bandwidth_cycle(value)?),
        K::Parent => DecodedOption::Parent(parse_parent(value)?),
    })
}

impl DecodedOption {
    fn apply(self, options: &mut ScatterhistOptions) {
        match self {
            Self::Bins(value) => options.bins = Some(value),
            Self::Group(value) => options.group = Some(value),
            Self::Kernel(value) => options.kernel = value,
            Self::Legend(value) => options.legend = Some(value),
            Self::PlotGroup(value) => options.plot_group = value,
            Self::Location(value) => options.location = value,
            Self::Direction(value) => options.direction = value,
            Self::Colors(value) => options.colors = Some(value),
            Self::Markers(value) => options.markers = value,
            Self::MarkerSizes(value) => options.marker_sizes = value,
            Self::LineWidths(value) => options.line_widths = value,
            Self::LineStyles(value) => options.line_styles = value,
            Self::Style(kernel, style) => {
                options.kernel = kernel;
                options.hist_style = Some(style);
            }
            Self::Bandwidths(value) => options.bandwidths = value,
            Self::Parent(value) => options.parent = Some(value),
        }
    }
}
