use crate::{
    builtin_catalog_entry_by_name, BuiltinDocumentationAuthority, BuiltinExampleVerification,
};

#[test]
fn documentation_is_catalog_owned_and_executable() {
    for (name, example_count, faq_count) in [("abs", 7, 8), ("angle", 4, 7), ("sign", 7, 8)] {
        let entry = builtin_catalog_entry_by_name(name).expect("magnitude/phase/sign entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), example_count, "{name}");
        assert_eq!(documentation.faqs.len(), faq_count, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "GPU execution"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}
