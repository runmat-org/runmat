//! Pipeline checks that already exist as repository scripts, run as witnesses so a failing
//! script shows up in the evidence catalog instead of living outside it. Each test runs one
//! script from the repository root and requires a zero exit status.
use runmat_analysis_qualification::witness::repository_root;
use std::process::Command;

fn run(program: &str, script: &str) {
    let root = repository_root();
    let output = Command::new(program)
        .arg(script)
        .current_dir(&root)
        .output()
        .unwrap_or_else(|e| panic!("cannot run {program} {script}: {e}"));
    assert!(
        output.status.success(),
        "{script} failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .take(12)
            .collect::<Vec<_>>()
            .join("\n"),
        String::from_utf8_lossy(&output.stderr)
            .lines()
            .take(12)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

// witness: FEA-SHARED-OWNERSHIP-001/code_quality @ shared_infrastructure/canonical_mixed_topology/source_check;
//   coverage: partial;
//   boundary: scripts/development/check-architecture-boundaries.mjs passes: dependency direction, forbidden
//   imports and module-size debt ceilings; it does not review code quality beyond those rules.
#[test]
#[ignore = "pipeline check; evidence_report runs it as a witness and a failure is a finding, not a register defect"]
fn architecture_boundary_check_passes() {
    run(
        "node",
        "scripts/development/check-architecture-boundaries.mjs",
    );
}

// witness: FEA-SHARED-OWNERSHIP-001/bridge_dependency_boundary @
//   shared_infrastructure/canonical_mixed_topology/source_check; coverage: partial;
//   boundary: The same architecture script enforces the dependency edges between the analysis crates and the
//   shared platform; the full edge list in the obligation is not compared one by one.
#[test]
#[ignore = "pipeline check; evidence_report runs it as a witness and a failure is a finding, not a register defect"]
fn architecture_boundary_check_enforces_dependency_edges() {
    run(
        "node",
        "scripts/development/check-architecture-boundaries.mjs",
    );
}

// witness: FEA-AUTHORING-CUTOVER-001/surface_inventory @ shared_infrastructure/mesh_independent/source_check;
//   coverage: partial;
//   boundary: scripts/fea/check-language-surface-ledger.mjs passes: every accepted field and variant of the
//   authoring surface has a reviewed disposition in the ledger; semantic equivalence of old and new forms
//   is not checked by it.
#[test]
#[ignore = "pipeline check; evidence_report runs it as a witness and a failure is a finding, not a register defect"]
fn language_surface_ledger_has_no_unreviewed_field() {
    run("node", "scripts/fea/check-language-surface-ledger.mjs");
}

// witness: FEA-MESH-EXECUTION-001/performance_slo_and_governance @
//   shared_infrastructure/canonical_mixed_topology/source_check; coverage: partial;
//   boundary: The meshing conformance catalog validator passes (15 cases covering the 16 required feature
//   classes); the SLO report freshness and the other validators are separate witnesses.
#[test]
#[ignore = "pipeline check; evidence_report runs it as a witness and a failure is a finding, not a register defect"]
fn meshing_conformance_catalog_validates() {
    run("python3", "scripts/meshing/validate_conformance.py");
}

// witness: FEA-MESH-EXECUTION-001/performance_slo_and_governance @
//   shared_infrastructure/canonical_mixed_topology/source_check; coverage: partial;
//   boundary: The meshing corpus validator passes: the fixture inventory matches the registered corpus.
#[test]
#[ignore = "pipeline check; evidence_report runs it as a witness and a failure is a finding, not a register defect"]
fn meshing_corpus_catalog_validates() {
    run("python3", "scripts/meshing/validate_corpus.py");
}

// witness: FEA-MESH-EXECUTION-001/performance_slo_and_governance @
//   shared_infrastructure/canonical_mixed_topology/source_check; coverage: partial;
//   boundary: The meshing differential catalog validator passes (independent mesher cases dispositioned).
#[test]
#[ignore = "pipeline check; evidence_report runs it as a witness and a failure is a finding, not a register defect"]
fn meshing_differential_catalog_validates() {
    run("python3", "scripts/meshing/validate_differential.py");
}

// witness: FEA-MESH-EXECUTION-001/performance_slo_and_governance @ shared_infrastructure/canonical_mixed_topology/source_check;
//   coverage: partial;
//   boundary: The meshing reliability catalog validator passes: every reliability control (budgets, cancellation,
//   atomic writes, restart, worker loss, stale publication) anchors to an existing test.
#[test]
#[ignore = "pipeline check; evidence_report runs it as a witness and a failure is a finding, not a register defect"]
fn meshing_reliability_catalog_anchors_exist() {
    run("python3", "scripts/meshing/validate_reliability.py");
}

// witness: FEA-RELEASE-CI-001/preserved_existing_release_graph @ shared_infrastructure/canonical_mixed_topology/release_pipeline;
//   coverage: partial;
//   boundary: The meshing release-evidence validator passes: release promotion depends on meshing release
//   evidence; FEA release evidence and the rest of the graph are not checked.
#[test]
#[ignore = "pipeline check; evidence_report runs it as a witness and a failure is a finding, not a register defect"]
fn meshing_release_evidence_is_a_release_dependency() {
    run("python3", "scripts/meshing/validate_release_evidence.py");
}
