use crate::{ExactHostAllocationPermit, ExactHostNumericAllocationError, Utf8Text};

pub(super) fn quote(
    name: &Utf8Text,
    source: Option<&Utf8Text>,
) -> Result<(usize, usize), ExactHostNumericAllocationError> {
    let mut bytes = 0usize;
    let mut claims = 0usize;
    // Published nonempty byte allocations cannot overlap. Count aliases of
    // one original text owner once, just as promotion attaches one claim.
    let source = source.filter(|source| source.as_str().as_ptr() != name.as_str().as_ptr());
    for text in std::iter::once(name).chain(source) {
        if let Some(unclaimed) = text.unclaimed_bytes()? {
            bytes = bytes
                .checked_add(unclaimed)
                .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
            claims += 1;
        }
    }
    Ok((bytes, claims))
}
pub(super) fn promote(
    name: &Utf8Text,
    source: Option<&Utf8Text>,
    permit: &dyn ExactHostAllocationPermit,
) -> Result<(), ExactHostNumericAllocationError> {
    for text in std::iter::once(name).chain(source) {
        text.clone()
            .try_admit_exact(permit)
            .map_err(|error| error.source())?;
    }
    Ok(())
}
