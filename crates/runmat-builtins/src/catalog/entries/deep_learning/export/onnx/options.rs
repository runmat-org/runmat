//! Public ONNX-export option identities and their value roles.
//! Names are parsed at the call boundary; consumers use these typed identities.

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExportOnnxOptionKey {
    BatchSize,
    InputNames,
    OpsetVersion,
    OutputNames,
    Verbose,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExportOnnxOptionValueRole {
    PositiveInteger,
    SingleName,
    LogicalFlag,
}

impl ExportOnnxOptionKey {
    /// Lexical public-name order also preserves the runtime's existing
    /// first-error order after duplicate name-value pairs are folded.
    pub const ALL: [Self; 5] = [
        Self::BatchSize,
        Self::InputNames,
        Self::OpsetVersion,
        Self::OutputNames,
        Self::Verbose,
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|option| name.eq_ignore_ascii_case(option.canonical_name()))
    }

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::BatchSize => "BatchSize",
            Self::InputNames => "InputNames",
            Self::OpsetVersion => "OpsetVersion",
            Self::OutputNames => "OutputNames",
            Self::Verbose => "Verbose",
        }
    }

    pub const fn value_role(self) -> ExportOnnxOptionValueRole {
        match self {
            Self::BatchSize | Self::OpsetVersion => ExportOnnxOptionValueRole::PositiveInteger,
            Self::InputNames | Self::OutputNames => ExportOnnxOptionValueRole::SingleName,
            Self::Verbose => ExportOnnxOptionValueRole::LogicalFlag,
        }
    }

    pub const fn accepts_integer_extension(self) -> bool {
        !matches!(self.value_role(), ExportOnnxOptionValueRole::SingleName)
    }
}
