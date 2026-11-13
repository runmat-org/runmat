//! Ordered boxplot option plan: host controls decode before any provider work.
use super::options::*;
use super::*;
use runmat_accelerate_api::GpuTensorStorage;
mod decoding;
use decoding::decode_option;

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
    BoxStyle(BoxStyle),
    MedianStyle(MedianStyle),
    Notch(NotchStyle),
    LabelOrientation(LabelOrientation),
    LabelVerbosity(LabelVerbosity),
    Jitter(f64),
    DataLim(Option<(f64, f64)>),
    ExtremeMode(ExtremeMode),
    ColorGroup(Value),
    FactorDirection(Vec<FactorDirection>),
    FullFactors(bool),
    FactorGap(FactorGap),
    FactorSeparator(FactorSeparator),
}

impl BoxplotOptionPlan {
    pub(super) fn preflight_known_factor_count(
        &self,
        factor_count: Option<usize>,
    ) -> BuiltinResult<()> {
        let Some(factor_count) = factor_count else {
            return Ok(());
        };
        let mut direction_seen = false;
        let mut gap_seen = false;
        let mut separator_seen = false;
        for step in self.steps.iter().rev() {
            match step {
                PlannedOption::Host(DecodedOption::FactorDirection(values)) if !direction_seen => {
                    super::factor_layout::validate_direction_count(values, factor_count)?;
                    direction_seen = true;
                }
                PlannedOption::Resident(BoxplotOptionKind::FactorDirection, _) => {
                    direction_seen = true;
                }
                PlannedOption::Host(DecodedOption::FactorGap(gap)) if !gap_seen => {
                    super::factor_layout::validate_gap_count(gap, factor_count)?;
                    gap_seen = true;
                }
                PlannedOption::Resident(BoxplotOptionKind::FactorGap, _) => {
                    gap_seen = true;
                }
                PlannedOption::Host(DecodedOption::FactorSeparator(separator))
                    if !separator_seen =>
                {
                    super::factor_layout::validate_separator_count(separator, factor_count)?;
                    separator_seen = true;
                }
                PlannedOption::Resident(BoxplotOptionKind::FactorSeparator, _) => {
                    separator_seen = true;
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub(super) async fn admit_host_color_groups(
        &self,
        x: &Value,
        cooperation: &mut super::cooperation::BoxplotCooperation,
    ) -> BuiltinResult<()> {
        for step in &self.steps {
            match step {
                PlannedOption::Host(DecodedOption::ColorGroup(group))
                | PlannedOption::Resident(BoxplotOptionKind::ColorGroup, group) => {
                    super::admission::admit_host_visible(x, Some(group), cooperation).await?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub(super) fn admit(
        args: crate::call::BuiltinArguments,
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
                let roles = super::extension_scan::numeric_roles(&value)?;
                if roles.integer {
                    enable_boxplot_extension(&BOXPLOT_INTEGER_GROUP_EXTENSION)?;
                }
                if roles.logical {
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
            let resident = value_contains_gpu(&value)?;
            if resident {
                enable_boxplot_extension(&BOXPLOT_GPU_INPUT_EXTENSION)?;
            }
            if let Value::GpuTensor(handle) = &value {
                validate_resident_option(kind, handle)?;
                steps.push(PlannedOption::Resident(kind, value));
            } else if resident {
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
    let name = keyword_of(value)?.ok_or_else(|| invalid("boxplot: option names must be text"))?;
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
                    if !options.box_style_explicit {
                        options.box_style = BoxStyle::Filled;
                    }
                    if !options.median_style_explicit {
                        options.median_style = MedianStyle::Target;
                    }
                    if !options.label_orientation_explicit {
                        options.label_orientation = LabelOrientation::Inline;
                    }
                    if !options.label_verbosity_explicit {
                        options.label_verbosity = LabelVerbosity::MajorMinor;
                    }
                    if !options.factor_gap_explicit {
                        options.factor_gap = FactorGap::Auto;
                    }
                    if !options.factor_separator_explicit {
                        options.factor_separator = FactorSeparator::Auto;
                    }
                    options.widths = vec![0.25];
                    options.outlier_marker = Some(MarkerStyle::Circle);
                    options.outlier_color = None;
                    options.outlier_size = 4.0;
                    options.jitter = 0.5;
                } else {
                    if !options.box_style_explicit {
                        options.box_style = BoxStyle::Outline;
                    }
                    if !options.median_style_explicit {
                        options.median_style = MedianStyle::Line;
                    }
                    if !options.label_orientation_explicit {
                        options.label_orientation = LabelOrientation::Horizontal;
                    }
                    if !options.label_verbosity_explicit {
                        options.label_verbosity = LabelVerbosity::All;
                    }
                    if !options.factor_gap_explicit {
                        options.factor_gap = FactorGap::None;
                    }
                    if !options.factor_separator_explicit {
                        options.factor_separator = FactorSeparator::None;
                    }
                    if options.outlier_marker == Some(MarkerStyle::Circle) {
                        options.outlier_marker = Some(MarkerStyle::Plus);
                    }
                    options.outlier_size = 6.0;
                    options.jitter = 0.0;
                }
            }
            Self::BoxStyle(value) => {
                options.box_style = value;
                options.box_style_explicit = true;
            }
            Self::MedianStyle(value) => {
                options.median_style = value;
                options.median_style_explicit = true;
            }
            Self::Notch(value) => options.notch_style = value,
            Self::LabelOrientation(value) => {
                options.label_orientation = value;
                options.label_orientation_explicit = true;
            }
            Self::LabelVerbosity(value) => {
                options.label_verbosity = value;
                options.label_verbosity_explicit = true;
            }
            Self::Jitter(value) => options.jitter = value,
            Self::DataLim(value) => options.data_lim = value,
            Self::ExtremeMode(value) => options.extreme_mode = value,
            Self::ColorGroup(value) => options.color_group = Some(value),
            Self::FactorDirection(value) => options.factor_direction = value,
            Self::FullFactors(value) => options.full_factors = value,
            Self::FactorGap(value) => {
                options.factor_gap = value;
                options.factor_gap_explicit = true;
            }
            Self::FactorSeparator(value) => {
                options.factor_separator = value;
                options.factor_separator_explicit = true;
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
        Whisker | OutlierSize | Jitter if len != 1 => {
            Err(invalid("boxplot: scalar option has invalid shape"))
        }
        DataLim if len != 2 => Err(invalid("boxplot: DataLim must have two elements")),
        Orientation | Symbol | PlotStyle | BoxStyle | MedianStyle | Notch | FullFactors
        | ExtremeMode | LabelOrientation | LabelVerbosity | FactorDirection => {
            Err(invalid("boxplot: text options must be host values"))
        }
        _ => Ok(()),
    }
}
