use super::*;

#[test]
fn every_operation_and_argument_has_a_summary() {
    let mut seen = std::collections::BTreeSet::new();
    for op in operations() {
        assert!(
            seen.insert((format!("{:?}", op.group), op.word)),
            "{} twice",
            op.word
        );
        assert!(op.summary.ends_with('.'), "{}: {}", op.word, op.summary);
        for a in op.arguments {
            assert!(
                a.summary.ends_with('.'),
                "{}.{}: {}",
                op.word,
                a.name,
                a.summary
            );
            for c in a.choices {
                assert!(
                    c.summary.ends_with('.'),
                    "{}.{} = {}",
                    op.word,
                    a.name,
                    c.word
                );
            }
        }
    }
}

#[test]
fn regeneration_paths_find_their_quantity() {
    let kind = |k: &str, p: &str| parameter(k, p).map(|a| a.quantity);
    assert_eq!(kind("extrude", "distance"), Some(Quantity::Length));
    assert_eq!(kind("extrude", "second.distance"), Some(Quantity::Length));
    assert_eq!(kind("extrude", "from.distance"), Some(Quantity::Length));
    assert_eq!(kind("extrude", "draft.angle"), Some(Quantity::Angle));
    assert_eq!(kind("hole", "end.depth"), Some(Quantity::Length));
    assert_eq!(kind("hole", "form.angle"), Some(Quantity::Angle));
    assert_eq!(
        kind("hole", "form.counterbore.diameter"),
        Some(Quantity::Length)
    );
    assert_eq!(
        kind("fillet.variable", "variable[2].at"),
        Some(Quantity::Number)
    );
    assert_eq!(kind("fillet", "profile.rho"), Some(Quantity::Number));
    assert_eq!(kind("pattern", "rows[0].dx"), Some(Quantity::Length));
    assert_eq!(kind("pattern", "first_spacing"), Some(Quantity::Length));
    assert_eq!(kind("shell", "faces[1].thickness"), Some(Quantity::Length));
    assert_eq!(dimension("draft", "angle"), Some(Dimension::ANGLE));
}

#[test]
fn signatures_list_call_arguments_only() {
    let extrude = operation("extrude").unwrap();
    assert!(extrude
        .signature()
        .starts_with("extrude(profile, distance, through_all"));
    assert!(!extrude.signature().contains("thin.thickness"));
    let (arg, counterbore) = choice_call(operation("hole").unwrap(), "counterbore").unwrap();
    assert_eq!(arg.name, "form");
    assert_eq!(counterbore.fields.len(), 2);
}

#[test]
fn docs_blocks_rewrite_between_markers() {
    let page = "a\n<!-- arguments:table rib -->\nold\n<!-- /arguments -->\nb\n";
    let out = rewrite_docs(page);
    assert!(
        out.contains("| `thickness` | length (mm) | required |"),
        "{out}"
    );
    assert!(!out.contains("old"));
    assert_eq!(rewrite_docs(&out), out);
}
