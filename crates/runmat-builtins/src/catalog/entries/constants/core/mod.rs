use crate::{BuiltinCatalogProvenance, BuiltinConstantCatalogEntry, BuiltinConstantKind};

pub(super) const CONSTANTS: &[BuiltinConstantCatalogEntry] = &[
    constant("pi", BuiltinConstantKind::RealDouble),
    constant("eps", BuiltinConstantKind::RealDouble),
    constant("sqrt2", BuiltinConstantKind::RealDouble),
    constant("i", BuiltinConstantKind::ComplexDouble),
    constant("j", BuiltinConstantKind::ComplexDouble),
];

const fn constant(name: &'static str, kind: BuiltinConstantKind) -> BuiltinConstantCatalogEntry {
    BuiltinConstantCatalogEntry::new(
        name,
        kind,
        BuiltinCatalogProvenance::new(file!(), module_path!()),
    )
}
