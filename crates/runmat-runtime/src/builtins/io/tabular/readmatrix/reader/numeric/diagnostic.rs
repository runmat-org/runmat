//! One full-text numeric-token diagnostic for floating and integer replay.
//! Its checked byte length is an input to a future exact retained allocation;
//! ordinary `to_string` below does not by itself certify resource use.

use std::fmt;
use std::fmt::Write as _;

const PREFIX: &str = "readmatrix: unable to parse numeric value '";
const LINE: &str = "' on line ";
const COLUMN: &str = " column ";

pub(super) struct NumericTokenDiagnostic<'a> {
    token: &'a str,
    line: usize,
    column: usize,
}

#[derive(Debug, thiserror::Error)]
pub(super) enum NumericDiagnosticBuildError {
    #[error("numeric diagnostic length exceeds addressable storage")]
    LengthOverflow,
    #[error("numeric diagnostic allocation failed")]
    Allocation(#[source] std::collections::TryReserveError),
    #[error("numeric diagnostic formatting failed")]
    Formatting,
}

impl<'a> NumericTokenDiagnostic<'a> {
    pub(super) fn new(token: &'a str, line: usize, column: usize) -> Self {
        Self {
            token,
            line,
            column,
        }
    }

    pub(super) fn requested_text_bytes(&self) -> Option<usize> {
        PREFIX
            .len()
            .checked_add(self.token.len())?
            .checked_add(LINE.len())?
            .checked_add(decimal_digits(self.line))?
            .checked_add(COLUMN.len())?
            .checked_add(decimal_digits(self.column))
    }

    pub(super) fn try_render(&self) -> Result<String, NumericDiagnosticBuildError> {
        let requested = self
            .requested_text_bytes()
            .ok_or(NumericDiagnosticBuildError::LengthOverflow)?;
        let mut text = String::new();
        text.try_reserve_exact(requested)
            .map_err(NumericDiagnosticBuildError::Allocation)?;
        write!(&mut text, "{self}").map_err(|_| NumericDiagnosticBuildError::Formatting)?;
        debug_assert_eq!(text.len(), requested);
        Ok(text)
    }
}

impl fmt::Display for NumericTokenDiagnostic<'_> {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            output,
            "{PREFIX}{}{LINE}{}{COLUMN}{}",
            self.token, self.line, self.column
        )
    }
}

fn decimal_digits(mut number: usize) -> usize {
    let mut digits = 1;
    while number >= 10 {
        number /= 10;
        digits += 1;
    }
    digits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_text_length_and_spelling_cover_unicode_and_wide_coordinates() {
        for (token, line, column) in [
            ("?", 1, 2),
            ("not-a-number", 100, 20_000),
            ("α☃", usize::MAX, usize::MAX),
        ] {
            let diagnostic = NumericTokenDiagnostic::new(token, line, column);
            let actual = diagnostic.try_render().expect("bounded diagnostic");
            assert_eq!(diagnostic.requested_text_bytes(), Some(actual.len()));
            assert_eq!(
                actual,
                format!(
                    "readmatrix: unable to parse numeric value '{}' on line {} column {}",
                    token, line, column
                )
            );
        }
    }
}
