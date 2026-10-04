//! Source topology checks inspect Rust items, not identifiers in imports or bodies.

use syn::parse::Parser;

pub(super) fn declares_catalog_entry(source: &str) -> bool {
    !catalog_entry_names(source).is_empty()
}

pub(super) fn catalog_entry_names(source: &str) -> Vec<syn::Ident> {
    let file = syn::parse_file(source).expect("catalog source must be valid Rust syntax");
    file.items
        .iter()
        .flat_map(|item| {
            if let syn::Item::Use(item) = item {
                return if matches!(item.vis, syn::Visibility::Public(_)) {
                    exported_entry_names(&item.tree)
                } else {
                    Vec::new()
                };
            }
            let declaration = match item {
                syn::Item::Const(item) => match &*item.ty {
                    syn::Type::Path(path)
                        if path
                            .path
                            .segments
                            .last()
                            .is_some_and(|segment| segment.ident == "BuiltinCatalogEntry") =>
                    {
                        Some(item.ident.clone())
                    }
                    _ => None,
                },
                syn::Item::Macro(item) if item.ident.is_none() => {
                    // Identity declaration macros take the named entry as their first
                    // argument. Macro definitions, use items and function bodies are
                    // different AST items and cannot supply declaration evidence.
                    let first_argument =
                        |input: syn::parse::ParseStream<'_>| -> syn::Result<syn::Ident> {
                            let identity: syn::Ident = input.parse()?;
                            input.parse::<syn::Token![,]>()?;
                            while !input.is_empty() {
                                input.step(|cursor| {
                                    cursor.token_tree().map(|(_, next)| ((), next)).ok_or_else(
                                        || cursor.error("expected macro argument token"),
                                    )
                                })?;
                            }
                            Ok(identity)
                        };
                    first_argument
                        .parse2(item.mac.tokens.clone())
                        .ok()
                        .filter(|identity| identity.to_string().ends_with("_CATALOG_ENTRY"))
                }
                _ => None,
            };
            declaration.into_iter().collect()
        })
        .collect()
}

fn exported_entry_names(tree: &syn::UseTree) -> Vec<syn::Ident> {
    match tree {
        syn::UseTree::Rename(rename)
            if rename.ident == "CATALOG_ENTRY"
                && rename.rename.to_string().ends_with("_CATALOG_ENTRY") =>
        {
            vec![rename.rename.clone()]
        }
        syn::UseTree::Path(path) => exported_entry_names(&path.tree),
        syn::UseTree::Group(group) => group.items.iter().flat_map(exported_entry_names).collect(),
        _ => Vec::new(),
    }
}

pub(super) fn registered_entry_names(source: &str) -> Vec<syn::Ident> {
    let file = syn::parse_file(source).expect("registration source must be valid Rust syntax");
    file.items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Const(item) if item.ident == "ENTRIES" => Some(&*item.expr),
            _ => None,
        })
        .flat_map(|expression| {
            let syn::Expr::Reference(reference) = expression else {
                return Vec::new();
            };
            let syn::Expr::Array(array) = &*reference.expr else {
                return Vec::new();
            };
            array
                .elems
                .iter()
                .filter_map(|element| {
                    let syn::Expr::Reference(reference) = element else {
                        return None;
                    };
                    let syn::Expr::Path(path) = &*reference.expr else {
                        return None;
                    };
                    path.path
                        .segments
                        .last()
                        .map(|segment| segment.ident.clone())
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn registration_resolves_qualified_entry_symbols() {
    let names = registered_entry_names("const ENTRIES: &[&BuiltinCatalogEntry] = &[&FIRST_CATALOG_ENTRY, &child::SECOND_CATALOG_ENTRY];");
    assert_eq!(
        names.iter().map(ToString::to_string).collect::<Vec<_>>(),
        ["FIRST_CATALOG_ENTRY", "SECOND_CATALOG_ENTRY"]
    );
}

#[test]
fn identifies_concrete_and_macro_contracts_without_counting_references() {
    assert!(declares_catalog_entry(
        "pub use CATALOG_ENTRY as INT16_CATALOG_ENTRY;"
    ));
    assert!(declares_catalog_entry(
        "pub use child::CATALOG_ENTRY as SAMPLE_CATALOG_ENTRY;"
    ));
    assert!(declares_catalog_entry(
        "pub const SAMPLE_CATALOG_ENTRY: BuiltinCatalogEntry = entry();"
    ));
    assert!(declares_catalog_entry(
        "entry!(SAMPLE_CATALOG_ENTRY, descriptor);"
    ));
    assert!(declares_catalog_entry(
        "pub const SAMPLE_CATALOG_ENTRY: crate::BuiltinCatalogEntry = entry();"
    ));
    for source in [
        "macro_rules! entry { () => { pub const $entry: BuiltinCatalogEntry = entry(); }; }",
        "use crate::{SAMPLE_CATALOG_ENTRY, OTHER_CATALOG_ENTRY};",
        "use CATALOG_ENTRY as SAMPLE_CATALOG_ENTRY;",
        "fn body() { entry!(SAMPLE_CATALOG_ENTRY, descriptor); }",
        "pub const ENTRIES: &[&BuiltinCatalogEntry] = &[&SAMPLE_CATALOG_ENTRY];",
        "pub const DOCUMENTATION: BuiltinDocumentation = documentation();",
    ] {
        assert!(!declares_catalog_entry(source), "{source}");
    }
}
