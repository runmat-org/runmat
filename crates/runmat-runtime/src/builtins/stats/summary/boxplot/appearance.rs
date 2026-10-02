//! Typed box, median, and notch presentation controls.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BoxStyle {
    Outline,
    Filled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MedianStyle {
    Line,
    Target,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NotchStyle {
    Off,
    On,
    Marker,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LabelOrientation {
    Inline,
    Horizontal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LabelVerbosity {
    All,
    Minor,
    MajorMinor,
}

impl LabelVerbosity {
    pub(super) fn parse(text: &str) -> Option<Self> {
        match text {
            "all" => Some(Self::All),
            "minor" => Some(Self::Minor),
            "majorminor" => Some(Self::MajorMinor),
            _ => None,
        }
    }
}

#[cfg(test)]
pub(super) fn tick_labels(
    boxes: &[super::BoxStats],
    verbosity: LabelVerbosity,
    custom_labels: bool,
) -> Vec<String> {
    boxes
        .iter()
        .enumerate()
        .map(|(index, stats)| {
            let previous = index
                .checked_sub(1)
                .and_then(|previous| boxes.get(previous))
                .map(|previous| previous.group_key.as_slice());
            tick_label(
                &stats.label,
                &stats.group_key,
                previous,
                verbosity,
                custom_labels,
            )
        })
        .collect()
}

pub(super) fn tick_label(
    label: &str,
    key: &[crate::builtins::common::grouping_keys::KeyAtom],
    previous: Option<&[crate::builtins::common::grouping_keys::KeyAtom]>,
    verbosity: LabelVerbosity,
    custom_labels: bool,
) -> String {
    if custom_labels || verbosity == LabelVerbosity::All || key.is_empty() {
        return label.to_string();
    }
    let parts = key
        .iter()
        .enumerate()
        .map(|(factor, value)| {
            let show = previous.is_none_or(|previous| match verbosity {
                LabelVerbosity::All => true,
                LabelVerbosity::Minor => previous.get(factor) != Some(value),
                LabelVerbosity::MajorMinor => previous.get(..=factor) != Some(&key[..=factor]),
            });
            if show {
                value.label()
            } else {
                String::new()
            }
        })
        .collect::<Vec<_>>();
    if parts.iter().all(String::is_empty) {
        String::new()
    } else {
        parts.join(",")
    }
}

impl LabelOrientation {
    pub(super) fn parse(text: &str) -> Option<Self> {
        match text {
            "inline" => Some(Self::Inline),
            "horizontal" => Some(Self::Horizontal),
            _ => None,
        }
    }

    pub(super) fn angle(self, plot_orientation: super::Orientation) -> f64 {
        match (self, plot_orientation) {
            (Self::Inline, super::Orientation::Vertical) => 90.0,
            _ => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SegmentKind {
    Box,
    Median,
    LowerWhisker,
    UpperWhisker,
    LowerCap,
    UpperCap,
}

impl SegmentKind {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Box => "box",
            Self::Median => "median",
            Self::LowerWhisker => "lower whisker",
            Self::UpperWhisker => "upper whisker",
            Self::LowerCap => "lower cap",
            Self::UpperCap => "upper cap",
        }
    }

    pub(super) fn line_style(self, box_style: BoxStyle) -> runmat_plot::plots::LineStyle {
        if matches!(self, Self::LowerWhisker | Self::UpperWhisker) && box_style == BoxStyle::Outline
        {
            runmat_plot::plots::LineStyle::Dashed
        } else {
            runmat_plot::plots::LineStyle::Solid
        }
    }
}

impl BoxStyle {
    pub(super) fn parse(text: &str) -> Option<Self> {
        match text {
            "outline" => Some(Self::Outline),
            "filled" => Some(Self::Filled),
            _ => None,
        }
    }
}

impl MedianStyle {
    pub(super) fn parse(text: &str) -> Option<Self> {
        match text {
            "line" => Some(Self::Line),
            "target" => Some(Self::Target),
            _ => None,
        }
    }
}

impl NotchStyle {
    pub(super) fn parse(text: &str) -> Option<Self> {
        match text {
            "off" => Some(Self::Off),
            "on" => Some(Self::On),
            "marker" => Some(Self::Marker),
            _ => None,
        }
    }

    pub(super) fn effective(self, compact: bool) -> Self {
        if self == Self::On && compact {
            Self::Marker
        } else {
            self
        }
    }
}

pub(super) fn notch_interval(stats: &super::BoxStats) -> super::BuiltinResult<(f64, f64)> {
    let denominator = (stats.observation_count as f64).sqrt();
    let iqr = stats.q3 - stats.q1;
    let scaled_iqr = if iqr.is_finite() {
        iqr / denominator
    } else {
        stats.q3 / denominator - stats.q1 / denominator
    };
    let half_interval = 1.57 * scaled_iqr;
    let lower = stats.median - half_interval;
    let upper = stats.median + half_interval;
    if !lower.is_finite() || !upper.is_finite() {
        return Err(super::invalid(
            "boxplot: notch geometry is not representable",
        ));
    }
    Ok((lower, upper))
}

pub(super) fn box_outline(
    stats: &super::BoxStats,
    notch: NotchStyle,
) -> super::BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let half = stats.width * 0.5;
    let left = stats.position - half;
    let right = stats.position + half;
    if notch != NotchStyle::On {
        return Ok((
            vec![left, right, right, left, left],
            vec![stats.q1, stats.q1, stats.q3, stats.q3, stats.q1],
        ));
    }
    let (lower, upper) = notch_interval(stats)?;
    let indent = half * 0.5;
    Ok((
        vec![
            left,
            right,
            right,
            right - indent,
            right,
            right,
            left,
            left,
            left + indent,
            left,
            left,
        ],
        vec![
            stats.q1,
            stats.q1,
            lower,
            stats.median,
            upper,
            stats.q3,
            stats.q3,
            upper,
            stats.median,
            lower,
            stats.q1,
        ],
    ))
}
