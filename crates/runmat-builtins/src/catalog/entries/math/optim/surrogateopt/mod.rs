//! Shared public contracts. The entry is registered with its complete
//! numerical/public lifecycle, not by declaring option metadata alone.
use crate::{BuiltinErrorDescriptor, BuiltinExtensionDescriptor, BuiltinExtensionMode};
mod descriptor;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;
pub use descriptor::SURROGATEOPT_DESCRIPTOR;
pub use entry::SURROGATEOPT_CATALOG_ENTRY;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&SURROGATEOPT_CATALOG_ENTRY];
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SurrogateSampleOrigin {
    Initial,
    Random,
    Adaptive,
}
impl SurrogateSampleOrigin {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::Random => "random",
            Self::Adaptive => "adaptive",
        }
    }
    pub fn parse(value: &str) -> Option<Self> {
        if value.eq_ignore_ascii_case("initial") {
            Some(Self::Initial)
        } else if value.eq_ignore_ascii_case("random") {
            Some(Self::Random)
        } else if value.eq_ignore_ascii_case("adaptive") {
            Some(Self::Adaptive)
        } else {
            None
        }
    }
}
pub const SURROGATEOPT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SURROGATEOPT.INVALID_ARGUMENT",
    identifier: Some("RunMat:surrogateopt:InvalidArgument"),
    when: "An input, selected option, callback result or numerical extent is invalid.",
    message: "surrogateopt: invalid argument",
};
pub const SURROGATEOPT_ERROR_NUMERICAL_FAILURE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SURROGATEOPT.NUMERICAL_FAILURE",
    identifier: Some("RunMat:surrogateopt:NumericalFailure"),
    when: "The selected original geometry or interpolant cannot be represented by the numerical solver.",
    message: "surrogateopt: numerical failure",
};
pub const SURROGATEOPT_ERROR_CHECKPOINT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SURROGATEOPT.CHECKPOINT",
    identifier: Some("RunMat:surrogateopt:Checkpoint"),
    when: "A checkpoint record, selected code capture or MAT-file publication is invalid or unavailable.",
    message: "surrogateopt: checkpoint operation failed",
};
pub const SURROGATEOPT_INTEGER_OPTIONS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "surrogateopt-integer-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Exact native-integer scalar controls before selected option provider access; decision data remains double.",
    error_identifier: Some("RunMat:compatibility:SurrogateoptIntegerOptionsExtension"),
};
pub const SURROGATEOPT_RESIDENT_OPTIONS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "surrogateopt-explicit-gpu-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Original-context host gather of selected resident numeric options, excluding callback captures.",
    error_identifier: Some("RunMat:compatibility:SurrogateoptResidentOptionsExtension"),
};
