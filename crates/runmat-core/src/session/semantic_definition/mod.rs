//! Session definition identity, separate from closure capture instances.
mod fingerprint;
mod origin;
mod publication;
mod reuse;
pub(crate) use fingerprint::fingerprint;
pub(super) use publication::finalize_current_definitions;
pub(super) use reuse::unchanged_unit_remap;
#[cfg(test)]
mod tests;
