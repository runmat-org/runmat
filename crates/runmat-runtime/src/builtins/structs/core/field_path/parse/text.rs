//! Textual selectors share the numeric selector's one-based index contract.
use super::{error, IndexComponent, PathError, PathErrorKind};

pub(super) fn parse_text(
    text: &str,
    uses_textual_index: &mut bool,
) -> Result<IndexComponent, PathError> {
    *uses_textual_index = true;
    let text = text.trim();
    if text.eq_ignore_ascii_case("end") {
        return Ok(IndexComponent::End);
    }
    if let Ok(index) = text.parse::<usize>() {
        return (index > 0)
            .then_some(IndexComponent::Scalar(index))
            .ok_or_else(|| error(PathErrorKind::InvalidIndex, "index must be at least one"));
    }
    Err(error(
        PathErrorKind::InvalidIndex,
        format!("invalid textual index '{text}'"),
    ))
}
