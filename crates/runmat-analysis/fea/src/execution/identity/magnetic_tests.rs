use super::*;

#[test]
fn magnetic_numerical_identity_binds_workload_mesh_and_distinct_physics() {
    let workload = AnalysisWorkloadIdentity::derive("test-workload", &"current").unwrap();
    let mesh = Digest::sha256(b"canonical mesh");
    let magnetic = NumericalPlanIdentity::from_magnetostatic_workload(workload, mesh).unwrap();
    assert_eq!(
        magnetic,
        NumericalPlanIdentity::from_magnetostatic_workload(workload, mesh).unwrap()
    );
    assert_ne!(
        magnetic,
        NumericalPlanIdentity::from_electrical_static_workload(workload, mesh).unwrap()
    );
    assert_ne!(
        magnetic,
        NumericalPlanIdentity::from_acoustic_workload(workload, mesh).unwrap()
    );
    assert_ne!(
        magnetic,
        NumericalPlanIdentity::from_magnetostatic_workload(
            workload,
            Digest::sha256(b"foreign mesh")
        )
        .unwrap()
    );
    assert_ne!(
        magnetic,
        NumericalPlanIdentity::from_magnetostatic_workload(
            AnalysisWorkloadIdentity::derive("test-workload", &"changed current").unwrap(),
            mesh
        )
        .unwrap()
    );
    assert_ne!(
        ExecutionPlanIdentity::new(magnetic, Digest::sha256(b"allocation one")).unwrap(),
        ExecutionPlanIdentity::new(magnetic, Digest::sha256(b"allocation two")).unwrap()
    );
}

#[test]
fn magnetic_dispatch_and_workload_have_explicit_wire_names() {
    use crate::execution::AnalysisNumericalObjectKind;
    assert_eq!(
        serde_json::to_value(AnalysisWorkKind::SolveMagnetostatic).unwrap(),
        serde_json::json!("solve_magnetostatic")
    );
    assert_eq!(
        serde_json::to_value(AnalysisNumericalObjectKind::MagnetostaticWorkload).unwrap(),
        serde_json::json!("magnetostatic_workload")
    );
    assert_ne!(
        AnalysisWorkKind::SolveMagnetostatic,
        AnalysisWorkKind::SolveElectricalStatic
    );
    assert_ne!(
        AnalysisNumericalObjectKind::MagnetostaticWorkload,
        AnalysisNumericalObjectKind::ElectricalStaticWorkload
    );
}
