//! Count and construct the same UTF-8 stream without conversion temporaries.
use std::fmt::{self, Write};

use runmat_value::{exact_capacity_vec, ExactHostAllocationPermit, StringElement, Utf8Text};

use super::{operand::Element, TextError};

struct ByteCount(usize);

impl Write for ByteCount {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.0 = self.0.checked_add(text.len()).ok_or(fmt::Error)?;
        Ok(())
    }
}

struct ExactText {
    text: String,
    length: usize,
}

impl Write for ExactText {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let end = self.text.len().checked_add(text.len()).ok_or(fmt::Error)?;
        if end > self.length {
            return Err(fmt::Error);
        }
        self.text.push_str(text);
        Ok(())
    }
}

fn write_pair(
    left: &Element<'_>,
    right: &Element<'_>,
    output: &mut impl Write,
) -> Result<(), TextError> {
    // Missing propagates; it is not literal text to append.
    if left.is_missing() || right.is_missing() {
        return Ok(());
    } else {
        left.write(output)?;
        right.write(output)?;
    }
    Ok(())
}

pub(super) fn byte_length(left: &Element<'_>, right: &Element<'_>) -> Result<usize, TextError> {
    let mut count = ByteCount(0);
    write_pair(left, right, &mut count)?;
    Ok(count.0)
}

pub(super) fn concatenate(
    left: &Element<'_>,
    right: &Element<'_>,
    permit: Option<&dyn ExactHostAllocationPermit>,
) -> Result<StringElement, TextError> {
    if left.is_missing() || right.is_missing() {
        return Ok(StringElement::Missing);
    }
    let length = byte_length(left, right)?;
    let bytes = exact_capacity_vec::<u8>(length).map_err(TextError::Allocation)?;
    let mut output = ExactText {
        text: String::from_utf8(bytes).expect("empty byte backing is UTF-8"),
        length,
    };
    write_pair(left, right, &mut output)?;
    if output.text.len() != length {
        return Err(TextError::Format(fmt::Error));
    }
    match permit {
        Some(permit) => Utf8Text::try_adopt_string_exact(output.text, permit)
            .map(StringElement::Text)
            .map_err(|error| TextError::Allocation(*error.source())),
        None => Ok(StringElement::text(output.text)),
    }
}
