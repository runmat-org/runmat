//! Public option names decoded into one typed outlier-control vocabulary.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutlierOption {
    Median,
    Mean,
    Quartiles,
    Percentiles,
    MovingMedian,
    MovingMean,
    ThresholdFactor,
    OutlierLocations,
    SamplePoints,
    DataVariables,
    MaxNumOutliers,
    Grubbs,
    Gesd,
}

impl OutlierOption {
    /// Parse only at the external keyword/literal boundary. Execution and
    /// inference use the enum instead of comparing builtin or option names.
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "median" => Some(Self::Median),
            "mean" => Some(Self::Mean),
            "quartiles" => Some(Self::Quartiles),
            "percentiles" => Some(Self::Percentiles),
            "movmedian" => Some(Self::MovingMedian),
            "movmean" => Some(Self::MovingMean),
            "thresholdfactor" => Some(Self::ThresholdFactor),
            "outlierlocations" => Some(Self::OutlierLocations),
            "samplepoints" => Some(Self::SamplePoints),
            "data variables" | "datavariables" => Some(Self::DataVariables),
            "maxnumoutliers" => Some(Self::MaxNumOutliers),
            "grubbs" => Some(Self::Grubbs),
            "gesd" => Some(Self::Gesd),
            _ => None,
        }
    }

    pub const fn consumes_value(self) -> bool {
        matches!(
            self,
            Self::Percentiles
                | Self::MovingMedian
                | Self::MovingMean
                | Self::ThresholdFactor
                | Self::OutlierLocations
                | Self::SamplePoints
                | Self::DataVariables
                | Self::MaxNumOutliers
        )
    }

    pub const fn unsupported_array_form(self) -> bool {
        matches!(
            self,
            Self::SamplePoints
                | Self::DataVariables
                | Self::MaxNumOutliers
                | Self::Grubbs
                | Self::Gesd
        )
    }
}

#[cfg(test)]
mod tests {
    use super::OutlierOption;

    #[test]
    fn aliases_and_value_roles_are_typed() {
        assert_eq!(
            OutlierOption::parse("Data Variables"),
            Some(OutlierOption::DataVariables)
        );
        assert_eq!(
            OutlierOption::parse("movmedian"),
            Some(OutlierOption::MovingMedian)
        );
        assert!(OutlierOption::MovingMedian.consumes_value());
        assert!(!OutlierOption::Median.consumes_value());
        assert!(OutlierOption::Grubbs.unsupported_array_form());
    }
}
