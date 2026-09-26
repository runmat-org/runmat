mod documentation;
use crate::{BuiltinCatalogProvenance, BuiltinConstantCatalogEntry, BuiltinConstantKind};
pub(super) const CONSTANTS: &[BuiltinConstantCatalogEntry] = &[BuiltinConstantCatalogEntry::new(
    "pi",
    BuiltinConstantKind::RealDouble,
    BuiltinCatalogProvenance::new(file!(), module_path!()),
)
.with_documentation(documentation::DOCUMENTATION)];
