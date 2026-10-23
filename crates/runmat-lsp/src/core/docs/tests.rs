use super::*;

#[test]
fn exported_sections_faqs_and_presentation_only_examples_survive_hover() {
    let document: builtins_json::BuiltinDocJson = serde_json::from_value(serde_json::json!({
        "title":"sample",
        "description":"A complete description.",
        "sections":[{"heading":"Options","paragraphs":["The first option.","The second option."]}],
        "faqs":[{"question":"Which policy?","answer":"The owned policy."}],
        "examples":[{"description":"An explanatory example.","output":null}],
    }))
    .unwrap();
    let hover = render_builtin_hover_from_json("sample", "sample()", "", document);
    for prose in [
        "**Options**",
        "The first option.",
        "The second option.",
        "Which policy?",
        "The owned policy.",
        "An explanatory example.",
    ] {
        assert!(hover.contains(prose), "missing {prose}");
    }
    assert!(
        !hover.contains("```matlab"),
        "prose-only examples are not fabricated programs"
    );
}

#[test]
fn canonical_aggregate_doc_is_available_without_a_physical_sidecar() {
    let entry = runmat_builtins::builtin_catalog_entry_by_name("colon").unwrap();
    let document = builtins_json::builtin_doc("COLON").expect("shared canonical export row");
    assert_eq!(
        document.summary.as_deref(),
        Some(entry.documentation.summary)
    );
    let hover = render_builtin_hover_from_json("colon", "colon()", "", document);
    for section in entry.documentation.sections {
        assert!(hover.contains(section.heading));
        for paragraph in section.paragraphs {
            assert!(hover.contains(&normalize_markdown((*paragraph).into())));
        }
    }
}
