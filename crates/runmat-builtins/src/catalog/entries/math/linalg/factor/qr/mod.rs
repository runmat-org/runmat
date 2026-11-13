mod contract;
mod documentation;
mod inference;

pub use contract::*;

pub const QR_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::factor_entry(super::entry::FactorEntrySpec {
        name: "qr",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &QR_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(inference::infer),
        extensions: &QR_EXTENSIONS,
        integer_capabilities: &QR_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::Complete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&QR_CATALOG_ENTRY];
