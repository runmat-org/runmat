use lsp_types::{HoverContents, Position};
use runmat_builtins::{
    builtin_catalog_aliases, builtin_catalog_entry_by_name, public_builtin_catalog_entry_by_name,
    BuiltinCatalogAudience,
};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, hover_at, signature_help_at, CompatMode,
};

const FIXTURE_NAMES: [&str; 2] = ["OverIdx.subsref", "OverIdx.subsasgn"];

#[test]
fn internal_fixture_lookup_stays_executable_but_public_projection_excludes_both_methods() {
    for name in FIXTURE_NAMES {
        let internal = builtin_catalog_entry_by_name(name).expect("internal fixture catalog entry");
        assert_eq!(internal.identity.name, name);
        assert_eq!(internal.audience, BuiltinCatalogAudience::Fixture);
        assert!(public_builtin_catalog_entry_by_name(name).is_none());
        assert!(
            !builtin_catalog_aliases()
                .iter()
                .any(|alias| alias.canonical.name.eq_ignore_ascii_case(name)),
            "public alias must not expose {name}"
        );
    }
}

#[test]
fn qualified_fixture_calls_have_no_public_signature_or_fixture_hover() {
    for (name, source) in [
        ("OverIdx.subsref", "out = OverIdx.subsref(obj, S);"),
        ("OverIdx.subsasgn", "out = OverIdx.subsasgn(obj, S, rhs);"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        let method = name.rsplit('.').next().expect("qualified method");
        let method_position = Position::new(0, source.find(method).expect("method token") as u32);
        assert!(
            signature_help_at(source, &analysis, &method_position).is_none(),
            "fixture signature became public: {name}"
        );

        // Hover currently resolves the token, not the qualified call name. It
        // may show the public unqualified subsref/subsasgn documentation, but
        // must never surface the OverIdx fixture descriptor or its prose.
        for position in [
            Position::new(0, source.find("OverIdx").expect("class token") as u32),
            method_position,
        ] {
            if let Some(hover) = hover_at(source, &analysis, &position) {
                let markdown = match hover.contents {
                    HoverContents::Markup(markup) => markup.value,
                    HoverContents::Scalar(marked) => format!("{marked:?}"),
                    HoverContents::Array(marked) => format!("{marked:?}"),
                };
                assert!(
                    !markdown.contains(name),
                    "fixture hover became public for {name}: {markdown}"
                );
                assert!(
                    !markdown.contains("Internal executable fixture"),
                    "fixture prose became public for {name}: {markdown}"
                );
            }
        }
    }
}

#[test]
fn completion_omits_exact_fixture_identities_without_hiding_public_indexing_builtins() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));
    for name in FIXTURE_NAMES {
        assert!(
            completions
                .iter()
                .all(|item| !item.label.eq_ignore_ascii_case(name)),
            "fixture completion became public: {name}"
        );
    }
    for name in ["subsref", "subsasgn"] {
        assert!(
            completions
                .iter()
                .any(|item| item.label.eq_ignore_ascii_case(name)),
            "public indexing completion disappeared: {name}"
        );
        let source = format!("{name}(obj, S);");
        let analysis = analyze_document_with_compat(&source, CompatMode::RunMat);
        assert!(
            signature_help_at(&source, &analysis, &Position::new(0, 0)).is_some(),
            "public indexing signature disappeared: {name}"
        );
    }
}
