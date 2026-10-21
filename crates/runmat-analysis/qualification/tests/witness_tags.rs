//! Every `// witness:` tag in the repository must name a real obligation, only required
//! envelopes, and sit on a real test function. A stale or broken tag fails here.
use runmat_analysis_qualification::{
    requirement::{Envelope, ExecutionEnvelope, Formulation, MeshOrder},
    requirements,
    witness::{parse_tag_text, repository_root, scan, validate_tags},
};

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
    let (requirement, obligation, envelopes, boundary) = parse_tag_text(
        "FEA-X-001/obl @ thermal_steady|thermal_transient/tet4|tet10/numerical_kernel, \
         shared_infrastructure/mesh_independent/source_check; boundary: only a patch",
    )
    .unwrap();
    assert_eq!((requirement.as_str(), obligation.as_str()), ("FEA-X-001", "obl"));
    assert_eq!(envelopes.len(), 5);
    assert!(envelopes.contains(&Envelope {
        formulation: Formulation::ThermalTransient,
        mesh: MeshOrder::Tet10,
        execution: ExecutionEnvelope::NumericalKernel,
    }));
    assert_eq!(boundary, "only a patch");
    assert!(parse_tag_text("A/b @ thermal_steady/tet4/serial; boundary: x").is_err());
    assert!(parse_tag_text("A/b @ thermal_steady/tet4/durable;").is_err());
    assert!(parse_tag_text("A/b @ thermal_steady/tet4/durable; boundary:  ").is_err());
}
