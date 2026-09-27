//! Session definition identity, separate from closure capture instances.
mod fingerprint;
mod reuse;
pub(crate) use fingerprint::fingerprint;
pub(super) use reuse::unchanged_unit_remap;
#[cfg(test)]
mod tests;
