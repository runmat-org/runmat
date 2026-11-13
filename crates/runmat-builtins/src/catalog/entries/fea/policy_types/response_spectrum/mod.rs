//! Catalog identities of the `fea.ResponseSpectrum` typed policy class.

use crate::*;

mod new_method;
#[cfg(test)]
mod tests;

pub(super) const SPECTRUM_STUDY: &str = concat!(
    "Response-spectrum policies are assembled by `fea.ResponseSpectrum.new`. Register the",
    " spectrum with `fea.model(..., \"ResponseSpectra\", {spectrum})`; a modal ",
    "response-spectrum procedure step selects spectra by identifier through its ",
    "`fea.IndependentDomain.responseSpectra` domain and needs a modal-basis state from a ",
    "preceding modal step.",
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&new_method::FEA_RESPONSE_SPECTRUM_NEW_CATALOG_ENTRY];
