macro_rules! documented_integer_input {
    ($name:literal, $notes:literal) => {
        BuiltinIntegerInputCapability {
            name: $name,
            classes: &ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::Documented,
            scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
            notes: $notes,
        }
    };
}
mod errors_output_handle;
mod errors_signatures;
mod integers;

pub use errors_signatures::{ERRORBAR_DESCRIPTOR, ERRORBAR_ERROR_INVALID_ARGUMENT};
pub use integers::ERRORBAR_INTEGER_CAPABILITIES;
