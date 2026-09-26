use super::*;
use crate::{
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    execution::ThermalWorkload,
    fixtures::{fixture_model, FixtureId},
    planning::resource::ThermalPreassemblyResourceFacts,
};
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload() -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence =
        AnalysisSequence::from_steps("thermal", vec![ProcedureStep::thermal_transient("heat")]);
    ThermalWorkload {
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: crate::ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        procedure: crate::ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    }
}

#[test]
fn inventory_counts_all_actions_and_each_case_membership_without_copying_selection() {
    let mut workload = workload();
    let mut duplicate = workload.model.excitation_cases[0].clone();
    duplicate.case_id.0 = "second_case".repeat(512);
    workload.model.excitation_cases.push(duplicate);
    let step = &workload.model.sequence.steps[0];
    let facts = ThermalRealizationResourceFacts::derive(&workload.model, step).unwrap();
    let realized =
        realize_step_excitations(&workload.model, step, ExcitationSamplePoint::time(0.0)).unwrap();
    assert_eq!(facts.active_actions.0, realized.len() as u64);
    assert_eq!(
        facts.case_memberships.0,
        realized
            .iter()
            .map(|entry| entry.case_ids.len() as u64)
            .sum::<u64>()
    );
    assert!(facts.case_memberships.0 > facts.active_actions.0);
    let selected = active_actions(&workload.model, Some(step))
        .unwrap()
        .next()
        .unwrap();
    assert!(std::ptr::eq(
        selected.case_ids().next().unwrap(),
        &workload.model.excitation_cases[0].case_id
    ));
    let source = super::super::ThermalSourceEvidenceResourceFacts::derive(
        &workload.model,
        step,
        &workload.solve_schedule().unwrap(),
    )
    .unwrap();
    assert_eq!(source.records_per_sample.0, 0); // Structural load metadata still realizes.
    assert!(facts.active_actions.0 > 0);
    assert_eq!(
        facts.vector_bytes.0,
        facts.active_actions.0 * std::mem::size_of::<RealizedExcitation>() as u64
            + facts.case_memberships.0 * std::mem::size_of::<ExcitationCaseId>() as u64
    );
    workload.model.excitation_cases[1].activation =
        ExcitationCaseActivation::Steps(vec!["other".into()]);
    let reduced =
        ThermalRealizationResourceFacts::derive(&workload.model, &workload.model.sequence.steps[0])
            .unwrap();
    assert_eq!(reduced.active_actions, facts.active_actions);
    assert!(reduced.metadata_estimated_bytes.0 < facts.metadata_estimated_bytes.0);
    assert!(reduced.case_memberships.0 < facts.case_memberships.0);
    // Resolved target paths are cloned by realization and must affect metadata admission.
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut entity = mesh.topology.boundary_faces[0].provenance[0].clone();
    entity
        .assembly_path
        .extend((0..100).map(|index| format!("component_{index}_{}", "x".repeat(180))));
    let mut target = RegionTarget::boundary("resolved_face");
    target.bind_resolved_entities(vec![entity]).unwrap();
    let ExcitationAction::Force(action) = &mut workload.model.excitations[0] else {
        panic!("force fixture")
    };
    action.target = ExcitationTarget::Region(target);
    let nested =
        ThermalRealizationResourceFacts::derive(&workload.model, &workload.model.sequence.steps[0])
            .unwrap();
    assert_eq!(nested.vector_bytes, reduced.vector_bytes);
    assert!(nested.metadata_estimated_bytes.0 > reduced.metadata_estimated_bytes.0);
}

#[test]
fn realization_budget_is_composed_and_enforced_for_both_element_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut workload = workload();
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let facts = ThermalPreassemblyResourceFacts::derive(
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &workload.solve_schedule().unwrap(),
            &crate::admission::admit_nonstructural(
                runmat_analysis_core::PhysicsFamily::Thermal,
                workload.procedure.procedure(),
                Some(&mesh),
                runmat_analysis_core::SourceImplementation::CanonicalVolumeProjection,
                runmat_analysis_core::BoundaryTransferImplementation::CanonicalBoundaryProjection,
                workload.formulation_requirement,
            )
            .unwrap(),
        )
        .unwrap();
        let required = facts.reservation_bytes().unwrap().0;
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = required - facts.realization.reservation_bytes().unwrap().0;
        assert!(
            matches!(workload.execute(&mesh), Err(crate::FeaRunError::Resource(ResourcePlanningError::AccuracyCeiling { required: actual, .. })) if actual == required)
        );
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = required;
        workload.execute(&mesh).unwrap();
    }
}

#[test]
fn lookup_includes_unused_signals_and_realization_rejects_duplicate_ids() {
    let mut workload = workload();
    let base =
        ThermalRealizationResourceFacts::derive(&workload.model, &workload.model.sequence.steps[0])
            .unwrap();
    let mut unused = workload.model.signals[0].clone();
    let PhysicalSignal::Force(signal) = &mut unused else {
        panic!("force fixture")
    };
    signal.signal_id.0 = "unused_signal".into();
    workload.model.signals.push(unused);
    let step = &workload.model.sequence.steps[0];
    let grown = ThermalRealizationResourceFacts::derive(&workload.model, step).unwrap();
    assert_eq!(
        grown.signal_lookup_bytes.0,
        base.signal_lookup_bytes.0 + std::mem::size_of::<&PhysicalSignal>() as u64
    );
    assert_eq!(
        grown.metadata_estimated_bytes,
        base.metadata_estimated_bytes
    );
    realize_step_excitations(&workload.model, step, ExcitationSamplePoint::time(0.0)).unwrap();
    workload
        .model
        .signals
        .push(workload.model.signals[0].clone());
    assert!(realize_step_excitations(
        &workload.model,
        &workload.model.sequence.steps[0],
        ExcitationSamplePoint::time(0.0)
    )
    .is_err());
    let mut overflow = grown;
    overflow.vector_bytes = ByteCount(u64::MAX);
    assert!(overflow.reservation_bytes().is_err());
    assert!(scale(u64::MAX, 2).is_err());
}
