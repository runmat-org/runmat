/// Domain-owned alias declarations are composed here as their canonical
/// entries complete catalog migration. The alias itself remains a typed edge;
/// it never receives a copied catalog entry or runtime implementation.
pub(super) fn extend_aliases(_aliases: &mut Vec<&'static crate::BuiltinCatalogAlias>) {}
