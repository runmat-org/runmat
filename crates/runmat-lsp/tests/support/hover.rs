pub fn assert_catalog_hover(name: &str, help: &lsp_types::SignatureHelp, markdown: &str) {
    let labels = super::support::assert_catalog_signatures(name, help);
    let header = markdown
        .strip_prefix("```runmat\n")
        .and_then(|text| text.split_once("\n```"))
        .expect("leading canonical signature block")
        .0;
    let displayed = header.lines().collect::<Vec<_>>();
    let active = help.active_signature.unwrap() as usize;
    assert_eq!(displayed.first().copied(), Some(labels[active]));
    let mut seen = std::collections::BTreeSet::new();
    for label in &displayed {
        assert!(
            labels.contains(label),
            "noncanonical hover signature for {name}: {label}"
        );
        assert!(seen.insert(*label), "duplicate hover signature for {name}");
    }
    let remaining = labels.len() - displayed.len();
    if remaining > 0 {
        let suffix = if remaining == 1 { "" } else { "s" };
        assert!(markdown.contains(&format!("+{remaining} more overload{suffix}")));
    } else {
        assert!(!markdown.contains("more overload"));
    }
    assert!(markdown.contains(&format!(
        "Docs: https://runmat.com/docs/reference/builtins/{}",
        name.to_ascii_lowercase()
    )));
}
