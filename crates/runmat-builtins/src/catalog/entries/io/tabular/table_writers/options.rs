//! Canonical writer identities and public option spelling boundary.
use crate::*;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TabularWriteKind {
    Table,
    Timetable,
}
impl TabularWriteKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Table => "writetable",
            Self::Timetable => "writetimetable",
        }
    }
    pub const fn class(self) -> runmat_types::StaticClassIdentity {
        match self {
            Self::Table => runmat_types::standard::TABLE,
            Self::Timetable => runmat_types::standard::TIMETABLE,
        }
    }
    pub const fn argument_error(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Table => &super::WRITETABLE_ERROR_ARGUMENT,
            Self::Timetable => &super::WRITETIMETABLE_ERROR_ARGUMENT,
        }
    }
    pub const fn io_error(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Table => &super::WRITETABLE_ERROR_IO,
            Self::Timetable => &super::WRITETIMETABLE_ERROR_IO,
        }
    }
    pub const fn bytes_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            Self::Table => &super::WRITETABLE_BYTES_OUTPUT_EXTENSION,
            Self::Timetable => &super::WRITETIMETABLE_BYTES_OUTPUT_EXTENSION,
        }
    }
    pub const fn gpu_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            Self::Table => &super::WRITETABLE_EXPLICIT_GPU_EXTENSION,
            Self::Timetable => &super::WRITETIMETABLE_EXPLICIT_GPU_EXTENSION,
        }
    }
    pub const fn integer_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            Self::Table => &super::WRITETABLE_INTEGER_LOGICAL_EXTENSION,
            Self::Timetable => &super::WRITETIMETABLE_INTEGER_LOGICAL_EXTENSION,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TabularWriteOption {
    Delimiter,
    VariableNames,
}
impl TabularWriteOption {
    pub fn parse(value: &str) -> Option<Self> {
        if value.eq_ignore_ascii_case("Delimiter") {
            Some(Self::Delimiter)
        } else if value.eq_ignore_ascii_case("WriteVariableNames") {
            Some(Self::VariableNames)
        } else {
            None
        }
    }
}
