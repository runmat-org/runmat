use std::collections::HashSet;

pub(super) fn exact_field_set(
    current: &[String],
    requested: &[String],
) -> crate::BuiltinResult<()> {
    if requested.len() != current.len() {
        return Err(super::super::error::field_mismatch());
    }
    let current = current.iter().map(String::as_str).collect::<HashSet<_>>();
    let mut seen = HashSet::with_capacity(requested.len());
    for name in requested {
        if !current.contains(name.as_str()) {
            return Err(super::super::error::unknown_field(name));
        }
        if !seen.insert(name.as_str()) {
            return Err(super::super::error::duplicate_field(name));
        }
    }
    Ok(())
}
