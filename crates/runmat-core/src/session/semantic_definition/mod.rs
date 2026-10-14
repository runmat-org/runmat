//! Session definition identity, separate from closure capture instances.
mod fingerprint;
mod origin;
mod product_reuse;
mod publication;
mod reuse;
pub(crate) use fingerprint::fingerprint;
pub(super) use product_reuse::unchanged_product_remap;
pub(super) use publication::finalize_current_definitions;
#[cfg(test)]
pub(super) use reuse::unchanged_closed_product_remap;
#[cfg(test)]
mod tests;
