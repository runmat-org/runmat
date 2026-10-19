//! Decode one admitted public option into a typed execution control.
use super::*;
pub(super) fn decode_option(kind: BoxplotOptionKind, value: Value) -> BuiltinResult<DecodedOption> {
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
            DecodedOption::BoxStyle(
                super::BoxStyle::parse(&text)
                    .ok_or_else(|| invalid("boxplot: BoxStyle must be 'outline' or 'filled'"))?,
            )
        }
        MedianStyle => {
            let text = text_scalar(&value, "MedianStyle")?;
            DecodedOption::MedianStyle(
                super::MedianStyle::parse(&text)
                    .ok_or_else(|| invalid("boxplot: MedianStyle must be 'line' or 'target'"))?,
            )
        }
        Notch => {
            let text = text_scalar(&value, "Notch")?;
            DecodedOption::Notch(
                super::NotchStyle::parse(&text)
                    .ok_or_else(|| invalid("boxplot: Notch must be 'on', 'off', or 'marker'"))?,
            )
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
            DecodedOption::LabelOrientation(super::LabelOrientation::parse(&text).ok_or_else(
                || invalid("boxplot: LabelOrientation must be 'inline' or 'horizontal'"),
            )?)
        }
        LabelVerbosity => {
            let text = text_scalar(&value, "LabelVerbosity")?;
            DecodedOption::LabelVerbosity(super::LabelVerbosity::parse(&text).ok_or_else(|| {
                invalid("boxplot: LabelVerbosity must be 'all', 'minor', or 'majorminor'")
            })?)
        }
        ColorGroup => DecodedOption::ColorGroup(value),
        FactorDirection => {
            DecodedOption::FactorDirection(super::super::factor_layout::parse_directions(&value)?)
        }
        FullFactors => {
            DecodedOption::FullFactors(super::super::factor_layout::parse_full_factors(&value)?)
        }
        FactorGap => DecodedOption::FactorGap(super::super::factor_layout::parse_gap(&value)?),
        FactorSeparator => {
            DecodedOption::FactorSeparator(super::super::factor_layout::parse_separator(&value)?)
        }
    };
    Ok(decoded)
}
