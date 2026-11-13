mod descriptor;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub const FILE_DATASTORE_ENTRY: crate::BuiltinCatalogEntry = {
    let mut entry = super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fileDatastore",
        category: "io/tabular",
        documentation: documentation::DOCUMENTATION,
        descriptor: &FILE_DATASTORE_DESCRIPTOR,
        inference: inference::infer,
        extensions: &FILE_DATASTORE_EXTENSIONS,
        integer_capabilities: &FILE_DATASTORE_INTEGER_CAPABILITIES,
        integer_audit: None,
    });
    entry.contract.maturity = crate::BuiltinContractMaturity::Incomplete;
    entry
};
