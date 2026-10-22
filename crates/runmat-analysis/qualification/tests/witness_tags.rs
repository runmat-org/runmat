//! Every `// witness:` tag in the repository must name a real obligation, only required
//! envelopes, and sit on a real test function. A stale or broken tag fails here.
use runmat_analysis_qualification::{
    requirement::{Envelope, ExecutionEnvelope, Formulation, MeshOrder},
    requirements,
    witness::{parse_tag_text, repository_root, scan, validate_tags},
};

// witness: FEA-RELEASE-REGISTER-001/witness_tags_valid @ shared_infrastructure/mesh_independent/source_check;
//   boundary: Scans every crate for witness tags and checks obligation, required envelopes, boundary and test-function placement; does not execute the tests.
#[test]
fn all_witness_tags_are_valid_against_the_register() {
    let (tags, mut problems) = scan(&repository_root());
    let registry: Vec<_> = requirements().collect();
    problems.extend(validate_tags(&tags, &registry));
    assert!(
        problems.is_empty(),
        "{} witness problems:\n{}",
        problems.len(),
        problems
            .iter()
            .take(40)
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn tag_text_parses_alternatives_and_rejects_unknown_names() {
    let (requirement, obligation, envelopes, boundary, complete) = parse_tag_text(
        "FEA-X-001/obl @ thermal_steady|thermal_transient/tet4|tet10/numerical_kernel, \
         shared_infrastructure/mesh_independent/source_check; boundary: only a patch",
    )
    .unwrap();
    assert_eq!(
        (requirement.as_str(), obligation.as_str()),
        ("FEA-X-001", "obl")
    );
    assert_eq!(envelopes.len(), 5);
    assert!(envelopes.contains(&Envelope {
        formulation: Formulation::ThermalTransient,
        mesh: MeshOrder::Tet10,
        execution: ExecutionEnvelope::NumericalKernel,
    }));
    assert_eq!(boundary, "only a patch");
    assert!(!complete);
    assert!(
        parse_tag_text("A/b @ thermal_steady/tet4/durable; coverage: complete; boundary: x")
            .unwrap()
            .4
    );
    assert!(
        parse_tag_text("A/b @ thermal_steady/tet4/durable; coverage: most; boundary: x").is_err()
    );
    assert!(parse_tag_text("A/b @ thermal_steady/tet4/serial; boundary: x").is_err());
    assert!(parse_tag_text("A/b @ thermal_steady/tet4/durable;").is_err());
    assert!(parse_tag_text("A/b @ thermal_steady/tet4/durable; boundary:  ").is_err());
}

// witness: FEA-RELEASE-REGISTER-001/evidence_catalog_generated @ shared_infrastructure/mesh_independent/source_check;
//   boundary: Unit-level check of the verdict derivation (PASS only when every tagged test passed, FAIL on failure or missing test, OPEN without tags); the generated files come from the evidence_report command.
#[test]
fn verdicts_follow_test_results_and_never_exceed_tags() {
    use runmat_analysis_qualification::catalog::{cells, TestResult, Verdict};
    use std::collections::BTreeMap;
    let (tags, _) = scan(&repository_root());
    let registry: Vec<_> = requirements().collect();
    let all = cells(&registry, &tags, None);
    assert!(all
        .iter()
        .all(|c| c.verdict != Verdict::Pass && c.verdict != Verdict::Fail));
    let tagged = all.iter().filter(|c| c.verdict == Verdict::Tagged).count();
    assert!(tagged > 0);
    let mut results = BTreeMap::new();
    for tag in &tags {
        results.insert(tag.test.describe(), TestResult::Passed);
    }
    let passed = cells(&registry, &tags, Some(&results));
    assert_eq!(
        passed.iter().filter(|c| c.verdict == Verdict::Pass).count(),
        tagged
    );
    let broken = tags[0].test.describe();
    results.insert(broken.clone(), TestResult::Failed("x".into()));
    let failed = cells(&registry, &tags, Some(&results));
    assert!(failed
        .iter()
        .any(|c| c.verdict == Verdict::Fail
            && c.tests.iter().any(|t| t.locator.describe() == broken)));
    results.remove(&broken);
    assert!(cells(&registry, &tags, Some(&results))
        .iter()
        .any(|c| c.verdict == Verdict::Fail));
}

#[test]
fn path_attribute_modules_resolve_to_the_declared_module_name() {
    let (tags, _) = scan(&repository_root());
    let tag = tags
        .iter()
        .find(|t| {
            t.test
                .name
                .ends_with("acoustic_accuracy_admission_binds_retained_result_and_original_intent")
        })
        .expect("acoustic accuracy witness");
    assert_eq!(
        tag.test.name,
        "tests::acoustic_accuracy_tests::acoustic_accuracy_admission_binds_retained_result_and_original_intent"
    );
}
