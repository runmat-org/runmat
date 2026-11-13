/// Public name/value controls accepted by `histcounts2`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Histcounts2Option {
    NumBins,
    XBinWidth,
    YBinWidth,
    BinWidth,
    XBinLimits,
    YBinLimits,
    BinLimits,
    XBinEdges,
    YBinEdges,
    BinMethod,
    XBinMethod,
    YBinMethod,
    Normalization,
}

impl Histcounts2Option {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "numbins" => Some(Self::NumBins),
            "xbinwidth" => Some(Self::XBinWidth),
            "ybinwidth" => Some(Self::YBinWidth),
            "binwidth" => Some(Self::BinWidth),
            "xbinlimits" => Some(Self::XBinLimits),
            "ybinlimits" => Some(Self::YBinLimits),
            "binlimits" => Some(Self::BinLimits),
            "xbinedges" => Some(Self::XBinEdges),
            "ybinedges" => Some(Self::YBinEdges),
            "binmethod" => Some(Self::BinMethod),
            "xbinmethod" => Some(Self::XBinMethod),
            "ybinmethod" => Some(Self::YBinMethod),
            "normalization" => Some(Self::Normalization),
            _ => None,
        }
    }

    pub fn requires_runmat_compat(self) -> bool {
        matches!(
            self,
            Self::XBinWidth
                | Self::YBinWidth
                | Self::BinLimits
                | Self::XBinMethod
                | Self::YBinMethod
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Histcounts2Option;

    #[test]
    fn option_names_and_extension_identity_are_typed() {
        assert_eq!(
            Histcounts2Option::parse(" XBinEdges "),
            Some(Histcounts2Option::XBinEdges)
        );
        assert_eq!(
            Histcounts2Option::parse("NUMBINS"),
            Some(Histcounts2Option::NumBins)
        );
        assert_eq!(
            Histcounts2Option::parse("normalization"),
            Some(Histcounts2Option::Normalization)
        );
        assert_eq!(Histcounts2Option::parse("unknown"), None);
        assert!(Histcounts2Option::XBinMethod.requires_runmat_compat());
        assert!(!Histcounts2Option::XBinEdges.requires_runmat_compat());
    }
}
