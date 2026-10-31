//! Preserve conversion and allocation causes through string construction.
#[derive(Debug)]
pub(super) enum TextError {
    Format(std::fmt::Error),
    Utf16(runmat_value::Utf16DecodeError),
    Allocation(runmat_value::ExactHostNumericAllocationError),
}

impl From<std::fmt::Error> for TextError {
    fn from(error: std::fmt::Error) -> Self {
        Self::Format(error)
    }
}

impl std::fmt::Display for TextError {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Format(_) => output.write_str("string output exceeds its checked byte length"),
            Self::Utf16(error) => error.fmt(output),
            Self::Allocation(error) => error.fmt(output),
        }
    }
}

impl std::error::Error for TextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            Self::Utf16(error) => Some(error),
            Self::Allocation(error) => Some(error),
        }
    }
}
