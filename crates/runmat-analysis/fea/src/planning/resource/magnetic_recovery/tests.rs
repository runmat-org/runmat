use super::*;
use crate::execution::MagnetostaticWorkload;
use crate::physics::{electrical::tests::mesh, magnetic::test_support::authored_model};
use crate::results::field::canonical_field_partitions;
use runmat_analysis_core::FormulationRequirement;
use runmat_meshing_core::ElementOrder;

#[test]
fn magnetic_recovery_profiles_cover_actual_catalog_identity_and_storage_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = mesh(order, 2);
        let workload = MagnetostaticWorkload {
            model: authored_model(0.0),
            step_id: runmat_analysis_core::ProcedureStepId("potential".into()),
            formulation_requirement: FormulationRequirement::Experimental,
            relative_residual_target: 1e-10,
        };
        let result = workload.execute(&mesh).unwrap();
        let request = result.native_field_request(&workload, &mesh).unwrap();
        let fields = result
            .recover_native_fields(&workload, &mesh, &request)
            .unwrap();
        let (bounds, workspace) = magnetic_field_recovery_shape(&request).unwrap();
        let mut actual = vec![fields.manifest().unwrap().canonical_bytes().unwrap().len() as u64];
        for field in fields.fields() {
            actual.extend(
                canonical_field_partitions(field)
                    .unwrap()
                    .iter()
                    .map(|p| p.canonical_bytes().unwrap().len() as u64),
            );
        }
        assert_eq!(actual.len(), bounds.len());
        assert_eq!(
            magnetic_field_recovery_partition_count(&request).unwrap(),
            bounds.len()
        );
        assert!(actual
            .iter()
            .zip(&bounds)
            .all(|(actual, bound)| actual <= bound));
        assert!(workspace.0 > actual.iter().sum::<u64>());
        let bytes = bounds.iter().sum::<u64>();
        let output = PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(bytes),
            transport_closure_bytes_upper_bound: ByteCount(bytes + 1024),
            construction_resident_bytes_admission_budget: ByteCount(2 * bytes + 2048),
        };
        let plan = build_magnetic_field_recovery_resource_plan(
            &request,
            magnetic_recovery_source_validation_workspace(&workload, &mesh).unwrap(),
            PreparedPhaseResourceFacts {
                inputs: PreparedInputResourceFacts {
                    encoded_payload_bytes: ByteCount(8),
                    transport_closure_bytes: ByteCount(16),
                    decoded_resident_bytes_admission_budget: ByteCount(1024),
                },
                output: output.clone(),
                output_set: PreparedOutputSetResourceFacts {
                    encoded_bytes_upper_bound: ByteCount(1024),
                },
            },
        )
        .unwrap();
        assert_eq!(
            plan.input
                .demand
                .output_facts(crate::execution::AnalysisNumericalObjectKind::Field),
            Some(&output)
        );
        assert!(plan
            .input
            .demand
            .output_facts(crate::execution::AnalysisNumericalObjectKind::SolverResult)
            .is_none());
    }
}

#[test]
fn magnetic_recovery_shape_rejects_missing_or_relabelled_native_inventory() {
    let mesh = mesh(ElementOrder::Tet4, 1);
    let workload = MagnetostaticWorkload {
        model: authored_model(0.0),
        step_id: runmat_analysis_core::ProcedureStepId("potential".into()),
        formulation_requirement: FormulationRequirement::Experimental,
        relative_residual_target: 1e-10,
    };
    let result = workload.execute(&mesh).unwrap();
    let original = result.native_field_request(&workload, &mesh).unwrap();
    let mut plan = original.plan.clone();
    plan.requirements.pop();
    let request = crate::execution::FieldRecoveryRequest::new(
        original.producer,
        original.source,
        original.topology,
        plan,
    )
    .unwrap();
    assert!(magnetic_field_recovery_shape(&request).is_err());
    let mut plan = original.plan.clone();
    for requirement in &mut plan.requirements[1..] {
        requirement.cost.source_entity_count = 16;
        requirement.cost.output_entity_count = 16;
        requirement.cost.arithmetic_operation_count = 48;
    }
    let request = crate::execution::FieldRecoveryRequest::new(
        original.producer,
        original.source,
        original.topology,
        plan,
    )
    .unwrap();
    assert!(magnetic_field_recovery_shape(&request).is_err());
    let request = crate::execution::FieldRecoveryRequest::new(
        original.producer,
        original.source,
        crate::execution::FieldTopologyIdentity::VolumeMesh {
            mesh_digest: mesh.canonical_digest,
        },
        original.plan.clone(),
    )
    .unwrap();
    assert!(magnetic_field_recovery_shape(&request).is_err());
    let mut plan = original.plan.clone();
    plan.requirements.swap(1, 2);
    let request = crate::execution::FieldRecoveryRequest::new(
        original.producer,
        original.source,
        original.topology,
        plan,
    )
    .unwrap();
    assert!(magnetic_field_recovery_shape(&request).is_err());
}
