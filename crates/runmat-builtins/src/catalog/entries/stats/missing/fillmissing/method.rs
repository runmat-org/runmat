/// Public method syntax decoded at the `fillmissing` call boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FillMissingMethodKind {
    Constant,
    Previous,
    Next,
    Nearest,
    Linear,
    Mean,
    Median,
}

impl FillMissingMethodKind {
    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "constant" => Some(Self::Constant),
            "previous" => Some(Self::Previous),
            "next" => Some(Self::Next),
            "nearest" => Some(Self::Nearest),
            "linear" => Some(Self::Linear),
            "mean" => Some(Self::Mean),
            "median" => Some(Self::Median),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FillMissingMethodKind;

    #[test]
    fn method_decoder_accepts_only_supported_public_spellings() {
        for (text, expected) in [
            ("constant", FillMissingMethodKind::Constant),
            ("previous", FillMissingMethodKind::Previous),
            ("next", FillMissingMethodKind::Next),
            ("nearest", FillMissingMethodKind::Nearest),
            ("linear", FillMissingMethodKind::Linear),
            ("mean", FillMissingMethodKind::Mean),
            ("median", FillMissingMethodKind::Median),
        ] {
            assert_eq!(FillMissingMethodKind::parse(text), Some(expected));
            assert_eq!(
                FillMissingMethodKind::parse(&text.to_uppercase()),
                Some(expected)
            );
        }
        for text in ["", " constant", "constant ", "spline", "CONSTANTLY"] {
            assert_eq!(FillMissingMethodKind::parse(text), None);
        }
    }
}
