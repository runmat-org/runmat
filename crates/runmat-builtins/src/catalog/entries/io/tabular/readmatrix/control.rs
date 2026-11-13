/// Parsed public option identity for the numeric text reader. Unknown names
/// retain the reader's established forward-compatible ignore behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadMatrixControl {
    Delimiter,
    NumHeaderLines,
    TreatAsMissing,
    DecimalSeparator,
    ThousandsSeparator,
    EmptyValue,
    OutputType,
    Range,
    Encoding,
    Like,
    Unknown,
}

impl ReadMatrixControl {
    pub fn parse(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "delimiter" => Self::Delimiter,
            "numheaderlines" => Self::NumHeaderLines,
            "treatasmissing" => Self::TreatAsMissing,
            "decimalseparator" => Self::DecimalSeparator,
            "thousandsseparator" => Self::ThousandsSeparator,
            "emptyvalue" => Self::EmptyValue,
            "outputtype" => Self::OutputType,
            "range" => Self::Range,
            "encoding" => Self::Encoding,
            "like" => Self::Like,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadMatrixOutputClass {
    Numeric(runmat_types::NumericClass),
    Logical,
}

impl ReadMatrixOutputClass {
    pub fn parse(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("logical") {
            return Some(Self::Logical);
        }
        runmat_types::NumericClass::from_class_name(name).map(Self::Numeric)
    }
}
