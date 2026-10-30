//! Canonical owned text conversion for fixed builtin parameters. Existing
//! text retains its original bytes; formatting uses bounded stack control,
//! then admits only the actual final UTF-8 allocation.
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::RuntimeError;
use runmat_value::{IntValue, Utf16CodeUnit, Utf16TextCopyError, Utf8Text, Value};
use std::fmt::Write;

pub fn convert(value: &Value, runtime: Option<&RuntimeContext>) -> Result<Utf8Text, RuntimeError> {
    match value {
        Value::String(text) => retain(
            text.text_ref()
                .ok_or_else(|| invalid("cannot convert missing string to scalar text"))?,
            runtime,
        ),
        Value::StringArray(array) => {
            if array.data().len() == 1 {
                retain(
                    array.data()[0]
                        .text_ref()
                        .ok_or_else(|| invalid("cannot convert missing string to scalar text"))?,
                    runtime,
                )
            } else {
                Err(invalid("cannot convert string array to scalar string"))
            }
        }
        Value::CharArray(array) => {
            if crate::builtins::common::shape::effective_rank(&array.shape) > 2 || array.rows != 1 {
                return Err(invalid(
                    "cannot convert multi-row char array to scalar string",
                ));
            }
            array.validate_layout().map_err(invalid)?;
            copy_chars(array.data(), runtime)
        }
        Value::Num(value) => formatted(value, runtime),
        Value::Bool(value) => formatted(value, runtime),
        Value::Int(value) => match value {
            IntValue::I8(v) => formatted(v, runtime),
            IntValue::I16(v) => formatted(v, runtime),
            IntValue::I32(v) => formatted(v, runtime),
            IntValue::I64(v) => formatted(v, runtime),
            IntValue::U8(v) => formatted(v, runtime),
            IntValue::U16(v) => formatted(v, runtime),
            IntValue::U32(v) => formatted(v, runtime),
            IntValue::U64(v) => formatted(v, runtime),
        },
        _ => Err(invalid("cannot convert argument to scalar text")),
    }
}
/// Encode an already-validated domain character sequence directly in its
/// final UTF-8 owner. Shape policy remains with the owning domain.
pub(crate) fn copy_chars(
    characters: &[Utf16CodeUnit],
    runtime: Option<&RuntimeContext>,
) -> Result<Utf8Text, RuntimeError> {
    let owner =
        HostAllocationReceipt::capture_owner(runtime).map_err(|e| allocation(runtime, e))?;
    let check = |_| {
        if let Some(owner) = &owner {
            owner.check(runtime.expect("selected managed context"))?;
        }
        Ok::<_, HostAllocationError>(())
    };
    let (length, bytes) = Utf8Text::requested_bytes_for_utf16_with(characters, check)
        .map_err(|e| utf16_copy_error(runtime, e))?;
    let receipt = admit(runtime, bytes, usize::from(length != 0))?;
    if let Some(receipt) = &receipt {
        let context = runtime.expect("selected managed context");
        let permit = receipt
            .output_permit(context)
            .map_err(|e| allocation(runtime, e))?;
        let text = Utf8Text::try_copy_utf16_exact_with(characters, &permit, |_| {
            receipt.checkpoint(context)
        });
        receipt
            .checkpoint(context)
            .map_err(|e| allocation(runtime, e))?;
        text.map_err(|e| utf16_copy_error(runtime, e))
    } else {
        runmat_value::decode_utf16_code_units(characters)
            .map(Utf8Text::new)
            .map_err(|source| {
                crate::runtime_error::semantic_error_with_source(
                    "ArgumentConversion",
                    "argument text contains malformed UTF-16",
                    source,
                )
            })
    }
}
fn retain(text: &Utf8Text, runtime: Option<&RuntimeContext>) -> Result<Utf8Text, RuntimeError> {
    let bytes = if text.has_retained_claim() {
        0
    } else {
        Utf8Text::requested_bytes_for_capacity(text.capacity())
            .map_err(|e| allocation(runtime, e.into()))?
    };
    let receipt = admit(runtime, bytes, usize::from(bytes != 0))?;
    if let Some(receipt) = &receipt {
        let context = runtime.expect("selected managed context");
        let permit = receipt
            .output_permit(context)
            .map_err(|e| allocation(runtime, e))?;
        let text = text.clone().try_admit_exact(&permit);
        receipt
            .checkpoint(context)
            .map_err(|e| allocation(runtime, e))?;
        text.map_err(|e| allocation(runtime, e.source().into()))
    } else {
        Ok(text.clone())
    }
}
pub fn copy_str(text: &str, runtime: Option<&RuntimeContext>) -> Result<Utf8Text, RuntimeError> {
    let bytes = Utf8Text::requested_bytes_for_capacity(text.len())
        .map_err(|e| allocation(runtime, e.into()))?;
    let receipt = admit(runtime, bytes, usize::from(!text.is_empty()))?;
    if let Some(receipt) = &receipt {
        let context = runtime.expect("selected managed context");
        let permit = receipt
            .output_permit(context)
            .map_err(|e| allocation(runtime, e))?;
        let text =
            Utf8Text::try_copy_str_exact_with(text, &permit, |_| receipt.checkpoint(context));
        receipt
            .checkpoint(context)
            .map_err(|e| allocation(runtime, e))?;
        text.map_err(|e| copy_error(runtime, e))
    } else {
        Ok(text.into())
    }
}

/// Adopt a caller-owned argument's original bytes. Managed execution reserves
/// the actual capacity and shared header before constructing the text owner.
pub fn adopt_string(
    text: String,
    runtime: Option<&RuntimeContext>,
) -> Result<Utf8Text, RuntimeError> {
    let bytes = Utf8Text::requested_bytes_for_capacity(text.capacity())
        .map_err(|source| allocation(runtime, source.into()))?;
    let receipt = admit(runtime, bytes, usize::from(text.capacity() != 0))?;
    let Some(receipt) = receipt else {
        return Ok(Utf8Text::new(text));
    };
    let context = runtime.expect("managed argument text requires its context");
    let permit = receipt
        .output_permit(context)
        .map_err(|source| allocation(runtime, source))?;
    let text = Utf8Text::try_adopt_string_exact(text, &permit);
    receipt
        .checkpoint(context)
        .map_err(|source| allocation(runtime, source))?;
    text.map_err(|source| allocation(runtime, (*source.source()).into()))
}

pub fn copy_parts(
    parts: &[&str],
    runtime: Option<&RuntimeContext>,
) -> Result<Utf8Text, RuntimeError> {
    let owner =
        HostAllocationReceipt::capture_owner(runtime).map_err(|e| allocation(runtime, e))?;
    let mut length = 0usize;
    for part in parts {
        if let Some(owner) = &owner {
            owner
                .check(runtime.expect("managed text owner"))
                .map_err(|e| allocation(runtime, e))?;
        }
        length = length
            .checked_add(part.len())
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
    }
    let bytes = Utf8Text::requested_bytes_for_capacity(length)
        .map_err(|e| allocation(runtime, e.into()))?;
    let receipt = admit(runtime, bytes, usize::from(length != 0))?;
    if let Some(receipt) = &receipt {
        let context = runtime.expect("managed text owner");
        let permit = receipt
            .output_permit(context)
            .map_err(|e| allocation(runtime, e))?;
        let result =
            Utf8Text::try_copy_parts_exact_with(parts, &permit, |_| receipt.checkpoint(context));
        receipt
            .checkpoint(context)
            .map_err(|e| allocation(runtime, e))?;
        result.map_err(|e| copy_error(runtime, e))
    } else {
        Ok(Utf8Text::new(parts.iter().copied().collect()))
    }
}

// Rust Display's fixed scalar formats fit this control buffer, including a
// signed smallest f64 subnormal in non-exponent notation. Overflow is checked
// rather than silently truncating or falling back to an allocating String.
struct ScalarText {
    bytes: [u8; 512],
    len: usize,
}
impl std::fmt::Write for ScalarText {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let end = self
            .len
            .checked_add(text.len())
            .filter(|&n| n <= self.bytes.len())
            .ok_or(std::fmt::Error)?;
        self.bytes[self.len..end].copy_from_slice(text.as_bytes());
        self.len = end;
        Ok(())
    }
}
fn formatted(
    value: &impl std::fmt::Display,
    runtime: Option<&RuntimeContext>,
) -> Result<Utf8Text, RuntimeError> {
    let mut output = ScalarText {
        bytes: [0; 512],
        len: 0,
    };
    write!(&mut output, "{value}")
        .map_err(|_| invalid("scalar text conversion exceeds its bounded formatter"))?;
    let text =
        std::str::from_utf8(&output.bytes[..output.len]).expect("formatter writes UTF-8 strings");
    copy_str(text, runtime)
}
fn admit(
    runtime: Option<&RuntimeContext>,
    bytes: usize,
    claims: usize,
) -> Result<Option<HostAllocationReceipt>, RuntimeError> {
    let output = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|e| allocation(runtime, e))?;
    HostAllocationReceipt::admit(
        runtime,
        HostAllocationPlan::new(0, output).map_err(|e| allocation(runtime, e))?,
    )
    .map_err(|e| allocation(runtime, e))
}
fn copy_error(
    runtime: Option<&RuntimeContext>,
    source: runmat_value::ExactArrayInitializationError<HostAllocationError>,
) -> RuntimeError {
    match source {
        runmat_value::ExactArrayInitializationError::Allocation(e) => allocation(runtime, e.into()),
        runmat_value::ExactArrayInitializationError::Element(e) => allocation(runtime, e),
    }
}
fn utf16_copy_error(
    runtime: Option<&RuntimeContext>,
    source: runmat_value::ExactArrayInitializationError<Utf16TextCopyError<HostAllocationError>>,
) -> RuntimeError {
    match source {
        runmat_value::ExactArrayInitializationError::Allocation(error) => {
            allocation(runtime, error.into())
        }
        runmat_value::ExactArrayInitializationError::Element(Utf16TextCopyError::Checkpoint(
            error,
        )) => allocation(runtime, error),
        runmat_value::ExactArrayInitializationError::Element(Utf16TextCopyError::Decode(
            source,
        )) => crate::runtime_error::semantic_error_with_source(
            "ArgumentConversion",
            "argument text contains malformed UTF-16",
            source,
        ),
    }
}
fn invalid(message: impl Into<String>) -> RuntimeError {
    crate::runtime_error::semantic_error("ArgumentConversion", message)
}
fn allocation(runtime: Option<&RuntimeContext>, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:Call:Text",
        "argument text conversion failed",
        source,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_scalar_formatter_preserves_existing_conversion() {
        for value in [
            Value::Num(f64::from_bits(1)),
            Value::Num(-f64::from_bits(1)),
            Value::Num(f64::MAX),
            Value::Num(-0.0),
            Value::Num(f64::INFINITY),
            Value::Num(f64::NAN),
            Value::Bool(true),
            Value::Int(IntValue::I64(i64::MIN)),
            Value::Int(IntValue::U64(u64::MAX)),
        ] {
            let expected = String::try_from(&value).expect("existing scalar text conversion");
            assert_eq!(convert(&value, None).unwrap().as_str(), expected);
        }
    }

    #[test]
    fn original_text_is_retained_and_unicode_characters_are_encoded() {
        let mut original = String::with_capacity(43);
        original.push_str("member");
        let pointer = original.as_ptr();
        let adopted = adopt_string(original, None).unwrap();
        assert_eq!(adopted.as_str(), "member");
        assert_eq!(adopted.as_str().as_ptr(), pointer);
        let text = Utf8Text::new("a🙂é".to_owned());
        let pointer = text.as_str().as_ptr();
        let retained = convert(&Value::String(text.into()), None).unwrap();
        assert_eq!(retained.as_str().as_ptr(), pointer);
        let characters =
            runmat_value::CharArray::from_unicode_scalars(vec!['a', '🙂', 'é'], 1, 4).unwrap();
        assert_eq!(characters.shape(), &[1, 4]);
        assert_eq!(
            convert(&Value::CharArray(characters), None)
                .unwrap()
                .as_str(),
            "a🙂é"
        );
        let column = runmat_value::CharArray::from_unicode_scalars(vec!['a', 'b'], 2, 1).unwrap();
        assert!(convert(&Value::CharArray(column), None).is_err());
        assert_eq!(
            copy_parts(&["a🙂", "", "é"], None).unwrap().as_str(),
            "a🙂é"
        );
        let malformed = runmat_value::CharArray::from_code_units(
            vec![runmat_value::Utf16CodeUnit::from_raw(0xD83D)],
            1,
            1,
        )
        .unwrap();
        let error = convert(&Value::CharArray(malformed), None).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:ArgumentConversion"));
        assert!(std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<runmat_value::Utf16DecodeError>())
            .is_some());
    }
}
