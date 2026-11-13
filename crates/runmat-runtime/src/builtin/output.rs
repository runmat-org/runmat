use runmat_value::{Value, ValueSequence};

use crate::execution::{RootedValue, RootedValueSequence};
use crate::RuntimeError;

/// Convert a canonical builtin handler's result into the executor-owned
/// sequence. This dispatches on the Rust result type, including named aliases,
/// rather than parsing a handler signature in the registration macro.
pub trait CanonicalBuiltinOutput {
    fn into_rooted_sequence(self) -> Result<RootedValueSequence, RuntimeError>;
}

impl CanonicalBuiltinOutput for RootedValueSequence {
    fn into_rooted_sequence(self) -> Result<RootedValueSequence, RuntimeError> {
        Ok(self)
    }
}

impl CanonicalBuiltinOutput for RootedValue {
    fn into_rooted_sequence(self) -> Result<RootedValueSequence, RuntimeError> {
        self.try_map_owned(
            crate::call::arguments::adapt_legacy_builtin_result,
            |error| error.into_runtime_error(),
        )
    }
}

impl CanonicalBuiltinOutput for ValueSequence {
    fn into_rooted_sequence(self) -> Result<RootedValueSequence, RuntimeError> {
        RootedValueSequence::new(self).map_err(|error| error.into_runtime_error())
    }
}

impl CanonicalBuiltinOutput for Value {
    fn into_rooted_sequence(self) -> Result<RootedValueSequence, RuntimeError> {
        RootedValueSequence::new(crate::call::arguments::adapt_legacy_builtin_result(self)?)
            .map_err(|error| error.into_runtime_error())
    }
}

macro_rules! scalar_output {
    ($($type:ty),+ $(,)?) => {
        $(impl CanonicalBuiltinOutput for $type {
            fn into_rooted_sequence(self) -> Result<RootedValueSequence, RuntimeError> {
                Value::from(self).into_rooted_sequence()
            }
        })+
    };
}

scalar_output!(
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    f64,
    bool,
    String,
    &'static str,
    runmat_value::Tensor
);
