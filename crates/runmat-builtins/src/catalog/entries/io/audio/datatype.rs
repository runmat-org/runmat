/// The public audioread sample representation, independent of source coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum AudioReadDataType {
    Double,
    Native,
}

impl AudioReadDataType {
    pub fn parse(text: &str) -> Option<Self> {
        if text.trim().eq_ignore_ascii_case("double") {
            Some(Self::Double)
        } else if text.trim().eq_ignore_ascii_case("native") {
            Some(Self::Native)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AudioReadDataType;

    #[test]
    fn public_spellings_are_case_insensitive_but_closed() {
        assert_eq!(
            AudioReadDataType::parse(" DOUBLE "),
            Some(AudioReadDataType::Double)
        );
        assert_eq!(
            AudioReadDataType::parse("Native"),
            Some(AudioReadDataType::Native)
        );
        assert_eq!(AudioReadDataType::parse("int16"), None);
    }
}
