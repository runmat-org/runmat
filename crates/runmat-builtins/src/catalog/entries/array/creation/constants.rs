use crate::{BuiltinCatalogProvenance, BuiltinConstantCatalogEntry, BuiltinConstantKind};

pub(super) const CONSTANTS: &[BuiltinConstantCatalogEntry] = &[
    constant("inf", BuiltinConstantKind::RealDouble),
    constant("Inf", BuiltinConstantKind::RealDouble),
    constant("nan", BuiltinConstantKind::RealDouble),
    constant("NaN", BuiltinConstantKind::RealDouble),
    constant("true", BuiltinConstantKind::Logical),
    constant("false", BuiltinConstantKind::Logical),
];

const fn constant(name: &'static str, kind: BuiltinConstantKind) -> BuiltinConstantCatalogEntry {
    BuiltinConstantCatalogEntry::new(
        name,
        kind,
        BuiltinCatalogProvenance::new(file!(), module_path!()),
    )
}
