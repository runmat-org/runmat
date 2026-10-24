//! Final documented FEA constructor cutover; no constructor algorithms live here.
mod domains;
mod entry;
mod interfaces;
mod quantities;
#[cfg(test)]
mod tests;

pub use self::domains::*;
pub use self::interfaces::*;
pub use self::quantities::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(quantities::ENTRIES.iter().copied());
    values.extend(domains::ENTRIES.iter().copied());
    values.extend(interfaces::ENTRIES.iter().copied());
}
