use super::*;
use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, PrescribedStructuralDofs, PrescribedValue,
    RegionTarget, StructuralConstraint, StructuralConstraintKind, StructuralTranslationDof,
    TranslationDofValue,
};

#[test]
fn scalar_modal_source_is_explicit_identity_bound_and_missing_inventory_rejects() {
    let (baseline, mesh, _, _) = modal_case(1, ComputeBackend::Cpu);
    let mut model = baseline.model().clone();
    model.constraints = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("scalar_support_{}", node.node_id),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    translations: [
                        StructuralTranslationDof::Ux,
                        StructuralTranslationDof::Uy,
                        StructuralTranslationDof::Uz,
                    ]
                    .into_iter()
                    .filter(|&dof| index == 0 || dof != StructuralTranslationDof::Ux)
                    .map(|dof| TranslationDofValue {
                        dof,
                        value: PrescribedValue::Constant(Displacement::new(0.)),
                    })
                    .collect(),
                    rotations: Vec::new(),
                }),
            })
        })
        .collect();
    let workload = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model,
        step_id: ProcedureStepId("modes".into()),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(1),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(workload.model(), Some(&mesh))
            .unwrap();
    let observations = resolve_structural_observation_plan(
        workload.procedure_step().unwrap(),
        workload.model(),
        Some(&mesh),
        &constraints,
        &structural_observable_field_definitions().unwrap(),
        ObservationLimits::default(),
    )
    .unwrap();
    let plan = build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &workload,
        None,
        &[],
        prepared(&mesh, &constraints, &workload),
    )
    .unwrap();
    let expected = 256
        + 512
        + 128 * mesh.topology.nodes.len() as u64
        + mesh
            .topology
            .volume_elements
            .iter()
            .map(|element| {
                let width = element.node_ids.len() as u64;
                32 + 8 * width + 8 * width * width
            })
            .sum::<u64>();
    let FeaResourceDemand::StructuralModalExtraction(demand) = &plan.input.demand else {
        panic!("modal demand")
    };
    let facts = super::super::structural_facts::derive_structural_resource_facts(
        workload.model(),
        Some(&mesh),
        &constraints,
    )
    .unwrap();
    let reduction = facts.constraint_matrix.affine_reduction().unwrap();
    let pattern = reduction
        .reduce_sparse_pattern(facts.symbolic.stiffness_upper_bound())
        .unwrap();
    let proof = crate::algebra::anchored_energy::SourceEnergyDemand::derive(
        crate::algebra::CoercivityWorkEnvelope::derive(&pattern).unwrap(),
        mesh.topology.nodes.len(),
        std::mem::size_of::<Option<usize>>() as u64 * mesh.topology.nodes.len() as u64,
        mesh.topology
            .volume_elements
            .iter()
            .map(|element| element.node_ids.len()),
    )
    .unwrap();
    assert_eq!(
        demand.original_energy_source_bytes.0,
        expected.max(proof.additional_resident_bytes())
    );
    assert!(demand.original_energy_source_bytes.0 >= expected);
    assert!(demand.operator.phase_resources.resident_bytes_upper_bound.0 >= expected);
    let mut changed = plan.input.clone();
    let FeaResourceDemand::StructuralModalExtraction(demand) = &mut changed.demand else {
        unreachable!()
    };
    demand.original_energy_source_bytes.0 = 0;
    let understated = super::super::build_resource_plan(changed).unwrap();
    assert_ne!(understated.identity, plan.identity);
    let mut wire = serde_json::to_value(&plan.input).unwrap();
    // The new source inventory is a required portable field, not a defaulted allowance.
    fn remove_source(value: &mut serde_json::Value) -> bool {
        match value {
            serde_json::Value::Object(fields) => {
                fields.remove("original_energy_source_bytes").is_some()
                    || fields.values_mut().any(remove_source)
            }
            serde_json::Value::Array(values) => values.iter_mut().any(remove_source),
            _ => false,
        }
    }
    assert!(remove_source(&mut wire));
    assert!(serde_json::from_value::<super::super::FeaResourcePlanningInput>(wire).is_err());
}
