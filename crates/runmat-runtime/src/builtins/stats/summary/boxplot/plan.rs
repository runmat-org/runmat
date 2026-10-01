//! Ordered boxplot option plan: host controls decode before any provider work.
use super::options::*;
use super::*;
use runmat_accelerate_api::GpuTensorStorage;

pub(super) struct BoxplotOptionPlan {
    steps: Vec<PlannedOption>,
}

enum PlannedOption {
    Host(DecodedOption),
    Resident(BoxplotOptionKind, Value),
}

enum DecodedOption {
    Whisker(f64),
    Orientation(Orientation),
    Labels(Vec<String>),
    GroupOrder(Vec<String>),
    Positions(Vec<f64>),
    Widths(Vec<f64>),
    OutlierSize(f32),
    Symbol(Option<MarkerStyle>, Option<Vec4>),
    Colors(Vec<Vec4>),
    PlotStyle(bool),
    Jitter(f64),
    DataLim(Option<(f64, f64)>),
    ExtremeMode(ExtremeMode),
    ColorGroup(Value),
    Validated(BoxplotOptionKind),
}

impl BoxplotOptionPlan {
    pub(super) fn admit(
        args: Vec<Value>,
        first_kind: Option<BoxplotOptionKind>,
    ) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(invalid("boxplot: name-value options must be paired"));
        }
        let mut steps = Vec::with_capacity(args.len() / 2);
        let mut values = args.into_iter();
        let mut index = 0;
        while let Some(name_value) = values.next() {
            let kind = match (index, first_kind) {
                (0, Some(kind)) => kind,
                _ => decode_option_name(&name_value)?,
            };
            let value = values.next().expect("paired options checked above");
            if kind == BoxplotOptionKind::ColorGroup {
                if value_contains_integer(&value) {
                    enable_boxplot_extension(&BOXPLOT_INTEGER_GROUP_EXTENSION)?;
                }
                if value_contains_logical(&value) {
                    enable_boxplot_extension(&BOXPLOT_LOGICAL_GROUP_EXTENSION)?;
                }
            } else {
                if value_is_integer_numeric(&value) {
                    enable_boxplot_extension(&BOXPLOT_INTEGER_OPTION_EXTENSION)?;
                }
                if value_is_logical_numeric(&value) {
                    enable_boxplot_extension(&BOXPLOT_LOGICAL_OPTION_EXTENSION)?;
                }
            }
            if value_contains_gpu(&value) {
                enable_boxplot_extension(&BOXPLOT_GPU_INPUT_EXTENSION)?;
            }
            if let Value::GpuTensor(handle) = &value {
                validate_resident_option(kind, handle)?;
                steps.push(PlannedOption::Resident(kind, value));
            } else {
                steps.push(PlannedOption::Host(decode_option(kind, value)?));
            }
            index += 1;
        }
        Ok(Self { steps })
    }

    pub(super) async fn evaluate(self) -> BuiltinResult<BoxplotOptions> {
        let mut options = BoxplotOptions::default();
        for step in self.steps {
            let decoded = match step {
                PlannedOption::Host(decoded) => decoded,
                PlannedOption::Resident(kind, value) => {
                    let gathered = gather_if_needed_async(&value)
                        .await
                        .map_err(|err| invalid_with_source(format!("boxplot: {err}"), err))?;
                    decode_option(kind, gathered)?
                }
            };
            decoded.apply(&mut options);
        }
        Ok(options)
    }
}

fn decode_option_name(value: &Value) -> BuiltinResult<BoxplotOptionKind> {
    let name = keyword_of(value).ok_or_else(|| invalid("boxplot: option names must be text"))?;
    BoxplotOptionKind::parse(&name)
        .ok_or_else(|| invalid(format!("boxplot: unsupported option '{name}'")))
}

impl DecodedOption {
    fn apply(self, options: &mut BoxplotOptions) {
        match self {
            Self::Whisker(value) => options.whisker = value,
            Self::Orientation(value) => options.orientation = value,
            Self::Labels(value) => options.labels = Some(value),
            Self::GroupOrder(value) => options.group_order = Some(value),
            Self::Positions(value) => options.positions = Some(value),
            Self::Widths(value) => options.widths = value,
            Self::OutlierSize(value) => options.outlier_size = value,
            Self::Symbol(marker, color) => {
                options.outlier_marker = marker;
                options.outlier_color = color;
            }
            Self::Colors(value) => options.box_colors = value,
            Self::PlotStyle(compact) => {
                options.compact = compact;
                if compact {
                    options.widths = vec![0.25];
                    options.outlier_marker = Some(MarkerStyle::Circle);
                    options.outlier_color = None;
                    options.outlier_size = 4.0;
                    options.jitter = 0.5;
                } else {
                    if options.outlier_marker == Some(MarkerStyle::Circle) {
                        options.outlier_marker = Some(MarkerStyle::Plus);
                    }
                    options.outlier_size = 6.0;
                    options.jitter = 0.0;
                }
            }
            Self::Jitter(value) => options.jitter = value,
            Self::DataLim(value) => options.data_lim = value,
            Self::ExtremeMode(value) => options.extreme_mode = value,
            Self::ColorGroup(value) => options.color_group = Some(value),
            Self::Validated(kind) => {
                let _ = kind;
            }
        }
    }
}

fn validate_resident_option(
    kind: BoxplotOptionKind,
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<()> {
    if handle.physical().storage() != GpuTensorStorage::Real {
        return Err(invalid("boxplot: numeric options must be real"));
    }
    let len = handle
        .shape()
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| invalid("boxplot: option shape overflows"))?;
    use BoxplotOptionKind::*;
    match kind {
        Whisker | OutlierSize | Jitter | Notch | FullFactors if len != 1 => {
            Err(invalid("boxplot: scalar option has invalid shape"))
        }
        DataLim if len != 2 => Err(invalid("boxplot: DataLim must have two elements")),
        Orientation | Symbol | PlotStyle | BoxStyle | MedianStyle | ExtremeMode
        | LabelOrientation | LabelVerbosity | FactorDirection | FactorSeparator => {
            Err(invalid("boxplot: text options must be host values"))
        }
        _ => Ok(()),
    }
}

fn decode_option(kind: BoxplotOptionKind, value: Value) -> BuiltinResult<DecodedOption> {
    use BoxplotOptionKind::*;
    let decoded = match kind {
        Whisker => {
            let whisker = scalar(&value, "Whisker")?;
            if !whisker.is_finite() || whisker < 0.0 {
                return Err(invalid(
                    "boxplot: Whisker must be a nonnegative finite scalar",
                ));
            }
            DecodedOption::Whisker(whisker)
        }
        Orientation => {
            DecodedOption::Orientation(match text_scalar(&value, "Orientation")?.as_str() {
                "vertical" => super::Orientation::Vertical,
                "horizontal" => super::Orientation::Horizontal,
                _ => {
                    return Err(invalid(
                        "boxplot: Orientation must be 'vertical' or 'horizontal'",
                    ))
                }
            })
        }
        Labels => DecodedOption::Labels(text_vector(&value, "Labels")?),
        GroupOrder => DecodedOption::GroupOrder(text_vector(&value, "GroupOrder")?),
        Positions => {
            let values = numeric_vector(&value, "Positions")?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err(invalid("boxplot: Positions must be finite"));
            }
            DecodedOption::Positions(values)
        }
        Widths => {
            let values = numeric_vector(&value, "Widths")?;
            if values.is_empty()
                || values
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(invalid("boxplot: Widths must be positive finite values"));
            }
            DecodedOption::Widths(values)
        }
        OutlierSize => {
            let size = scalar(&value, "OutlierSize")?;
            if !size.is_finite() || size <= 0.0 {
                return Err(invalid(
                    "boxplot: OutlierSize must be a positive finite scalar",
                ));
            }
            let physical_size = size as f32;
            if !physical_size.is_finite() || physical_size <= 0.0 {
                return Err(invalid(
                    "boxplot: OutlierSize must be representable as a positive plot size",
                ));
            }
            DecodedOption::OutlierSize(physical_size)
        }
        Symbol => {
            let text = value_text_preserve_case(&value, "Symbol")?;
            let (marker, color) = parse_symbol(&text)?;
            DecodedOption::Symbol(marker, color)
        }
        Colors => DecodedOption::Colors(parse_colors(&value)?),
        PlotStyle => DecodedOption::PlotStyle(match text_scalar(&value, "PlotStyle")?.as_str() {
            "traditional" => false,
            "compact" => true,
            _ => {
                return Err(invalid(
                    "boxplot: PlotStyle must be 'traditional' or 'compact'",
                ))
            }
        }),
        BoxStyle => {
            let text = text_scalar(&value, "BoxStyle")?;
            if text != "outline" && text != "filled" {
                return Err(invalid("boxplot: BoxStyle must be 'outline' or 'filled'"));
            }
            DecodedOption::Validated(kind)
        }
        MedianStyle => {
            let text = text_scalar(&value, "MedianStyle")?;
            if text != "line" && text != "target" {
                return Err(invalid("boxplot: MedianStyle must be 'line' or 'target'"));
            }
            DecodedOption::Validated(kind)
        }
        Notch => {
            parse_on_off_marker(&value, "Notch")?;
            DecodedOption::Validated(kind)
        }
        Jitter => {
            let jitter = scalar(&value, "Jitter")?;
            if !jitter.is_finite() || jitter < 0.0 {
                return Err(invalid("boxplot: Jitter must be nonnegative and finite"));
            }
            DecodedOption::Jitter(jitter)
        }
        DataLim => {
            let limits = numeric_vector(&value, "DataLim")?;
            if limits.len() != 2
                || limits.iter().any(|value| value.is_nan())
                || limits[0] > limits[1]
            {
                return Err(invalid(
                    "boxplot: DataLim must be a two-element numeric range",
                ));
            }
            DecodedOption::DataLim(finite_data_lim(limits[0], limits[1]))
        }
        ExtremeMode => {
            DecodedOption::ExtremeMode(match text_scalar(&value, "ExtremeMode")?.as_str() {
                "clip" => super::ExtremeMode::Clip,
                "compress" => super::ExtremeMode::Compress,
                _ => return Err(invalid("boxplot: ExtremeMode must be 'compress' or 'clip'")),
            })
        }
        LabelOrientation => {
            let text = text_scalar(&value, "LabelOrientation")?;
            if text != "inline" && text != "horizontal" {
                return Err(invalid(
                    "boxplot: LabelOrientation must be 'inline' or 'horizontal'",
                ));
            }
            DecodedOption::Validated(kind)
        }
        LabelVerbosity => {
            let text = text_scalar(&value, "LabelVerbosity")?;
            if text != "all" && text != "minor" && text != "majorminor" {
                return Err(invalid(
                    "boxplot: LabelVerbosity must be 'all', 'minor', or 'majorminor'",
                ));
            }
            DecodedOption::Validated(kind)
        }
        ColorGroup => DecodedOption::ColorGroup(value),
        FactorDirection => {
            validate_choice_vector(&value, "FactorDirection", &["data", "list"])?;
            DecodedOption::Validated(kind)
        }
        FullFactors => {
            parse_on_off(&value, "FullFactors")?;
            DecodedOption::Validated(kind)
        }
        FactorGap => {
            let values = numeric_vector(&value, "FactorGap")?;
            if values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(invalid(
                    "boxplot: FactorGap must contain nonnegative finite values",
                ));
            }
            DecodedOption::Validated(kind)
        }
        FactorSeparator => {
            validate_factor_separator(&value)?;
            DecodedOption::Validated(kind)
        }
    };
    Ok(decoded)
}
