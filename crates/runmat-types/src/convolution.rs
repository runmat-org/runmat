//! Public convolution shape selectors shared by static and runtime admission.

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvolutionMode {
    Full,
    Same,
    Valid,
}

impl ConvolutionMode {
    pub fn decode(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "full" => Some(Self::Full),
            "same" => Some(Self::Same),
            "valid" => Some(Self::Valid),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ConvolutionMode;

    #[test]
    fn shared_public_shape_spelling() {
        assert_eq!(
            ConvolutionMode::decode(" FULL "),
            Some(ConvolutionMode::Full)
        );
        assert_eq!(ConvolutionMode::decode("same"), Some(ConvolutionMode::Same));
        assert_eq!(
            ConvolutionMode::decode("VaLiD"),
            Some(ConvolutionMode::Valid)
        );
        assert_eq!(ConvolutionMode::decode("circular"), None);
    }
}
