#![cfg(feature = "docs-export")]
use runmat_analysis_qualification::{export::write_requirements, requirements};

#[test]
fn deterministic_export_retains_exact_criteria_and_unwitnessed_scopes() {
    let mut first = Vec::new();
    let mut second = Vec::new();
    write_requirements(&mut first).unwrap();
    write_requirements(&mut second).unwrap();
    assert_eq!(first, second);
    let value: serde_json::Value = serde_json::from_slice(&first).unwrap();
    assert_eq!(value["posture"], "discovery_only");
    for (exported, original) in value["requirements"]
        .as_array()
        .unwrap()
        .iter()
        .zip(requirements())
    {
        assert_eq!(exported["id"], original.id);
        assert_eq!(exported["acceptance"], original.acceptance);
        assert_eq!(
            exported["obligations"].as_array().unwrap().len(),
            original.obligations.len()
        );
        for (obligation, original) in exported["obligations"]
            .as_array()
            .unwrap()
            .iter()
            .zip(original.obligations)
        {
            assert_eq!(obligation["criterion"], original.criterion);
            assert_eq!(
                obligation["required_envelopes"].as_array().unwrap().len(),
                original.required_envelopes.len()
            );
        }
    }
    let gold = value["requirements"]
        .as_array()
        .unwrap()
        .iter()
        .find(|requirement| requirement["id"] == "FEA-NONLINEAR-STATE-001")
        .unwrap();
    let producer = gold["obligations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|obligation| obligation["id"] == "producer_exit")
        .unwrap();
    assert!(producer["required_envelopes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|scope| scope["formulation"] == "small_strain_j2_penalty_coulomb_contact"));
    assert!(value.get("passed").is_none());
}
