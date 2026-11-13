mod contract;
mod documentation;
pub use contract::*;
pub const WRITETABLE_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::entry(super::entry::Definition {
        name: "writetable",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        descriptor: &WRITETABLE_DESCRIPTOR,
        documentation: documentation::DOCUMENTATION,
        inference: infer,
        extensions: &super::WRITETABLE_EXTENSIONS,
        integer_capabilities: &super::WRITETABLE_INTEGER_CAPABILITIES,
    });
fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::inference::infer(super::TabularWriteKind::Table, request, entry)
}
