use super::*;

#[test]
fn compact_hover_omits_reference_sections_and_preserves_one_complete_example() {
    let document: builtins_json::BuiltinDocJson = serde_json::from_value(serde_json::json!({
        "title":"sample",
        "description":"A complete description.",
        "sections":[{"heading":"Options","paragraphs":["The first option.","The second option."]}],
        "faqs":[{"question":"Which policy?","answer":"The owned policy."}],
        "examples":[{"description":"An explanatory example.","output":null},
            {"input":"x = sample(1);"},{"input":"x = sample(2);"}],
        "related":[{"label":"mean","url":"./mean"}],
        "links":[{"label":"mean","url":"./mean"},
            {"label":"Source","url":"https://github.com/example/source"}],
    }))
    .unwrap();
    let hover = render_builtin_hover_from_json("sample", "sample()", "", document, 0);
    for prose in [
        "**Options**",
        "The first option.",
        "The second option.",
        "Which policy?",
        "The owned policy.",
        "An explanatory example.",
    ] {
        assert!(
            !hover.contains(prose),
            "reference-only prose leaked: {prose}"
        );
    }
    assert!(
        !hover.contains("```matlab"),
        "all editor code fences use runmat"
    );
    assert!(hover.contains("A complete description."));
    assert!(hover.contains("```runmat\nx = sample(1);\n```"));
    assert!(!hover.contains("sample(2)"));
    assert!(!hover.contains("Source"));
    assert_eq!(hover.matches("[mean]").count(), 1);
    assert!(hover.contains("\n\nSee also:"));
    assert!(hover.contains("\n\nDocs:"));
}

#[test]
fn canonical_aggregate_doc_is_available_without_a_physical_sidecar() {
    let entry = runmat_builtins::builtin_catalog_entry_by_name("colon").unwrap();
    let document = builtins_json::builtin_doc("COLON").expect("shared canonical export row");
    assert_eq!(
        document.summary.as_deref(),
        Some(entry.documentation.summary)
    );
    let full: serde_json::Value =
        serde_json::from_str(builtins_json::builtin_json_str("colon").unwrap()).unwrap();
    assert_eq!(
        full["sections"].as_array().unwrap().len(),
        entry.documentation.sections.len()
    );
    assert_eq!(
        full["faqs"].as_array().unwrap().len(),
        entry.documentation.faqs.len()
    );
    let hover = render_builtin_hover_from_json("colon", "colon()", "", document, 0);
    assert!(hover.contains(entry.documentation.summary));
    assert!(!hover.contains("**Status:**"));
}
