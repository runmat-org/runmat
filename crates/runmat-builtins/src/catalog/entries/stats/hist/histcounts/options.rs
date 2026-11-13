/// Public name/value controls accepted by `histcounts`.
///
/// The spelling is decoded once at the source-language boundary; inference
/// and runtime preparation use this identity rather than separate name lists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistcountsOption {
    BinEdges,
    NumBins,
    BinWidth,
    BinLimits,
    Normalization,
    BinMethod,
}

impl HistcountsOption {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "binedges" => Some(Self::BinEdges),
            "numbins" => Some(Self::NumBins),
            "binwidth" => Some(Self::BinWidth),
            "binlimits" => Some(Self::BinLimits),
            "normalization" => Some(Self::Normalization),
            "binmethod" => Some(Self::BinMethod),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HistcountsOption;

    #[test]
    fn option_names_have_one_case_insensitive_decoder() {
        assert_eq!(
            HistcountsOption::parse(" BinEdges "),
            Some(HistcountsOption::BinEdges)
        );
        assert_eq!(
            HistcountsOption::parse("NUMBINS"),
            Some(HistcountsOption::NumBins)
        );
        assert_eq!(
            HistcountsOption::parse("BinWidth"),
            Some(HistcountsOption::BinWidth)
        );
        assert_eq!(
            HistcountsOption::parse("BinLimits"),
            Some(HistcountsOption::BinLimits)
        );
        assert_eq!(
            HistcountsOption::parse("Normalization"),
            Some(HistcountsOption::Normalization)
        );
        assert_eq!(
            HistcountsOption::parse("BinMethod"),
            Some(HistcountsOption::BinMethod)
        );
        assert_eq!(HistcountsOption::parse("unknown"), None);
    }
}
