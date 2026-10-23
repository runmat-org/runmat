mod contract;
mod documentation;
pub use contract::*;
pub const WRITETIMETABLE_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::entry(super::entry::Definition {
        name: "writetimetable",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        descriptor: &WRITETIMETABLE_DESCRIPTOR,
        documentation: documentation::DOCUMENTATION,
        inference: infer,
        extensions: &super::WRITETIMETABLE_EXTENSIONS,
        integer_capabilities: &super::WRITETIMETABLE_INTEGER_CAPABILITIES,
    });
fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::inference::infer(super::TabularWriteKind::Timetable, request, entry)
}
