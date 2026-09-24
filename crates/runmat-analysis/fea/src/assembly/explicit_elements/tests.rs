use runmat_analysis_core::{StructuralElementKind, StructuralNode};

use super::{assemble_explicit_elements, StructuralDofLayout};
use crate::{
    assembly::dofs::StructuralNodeDofSet,
    fixtures::{fixture_model, FixtureId},
    planning::resource::{execute_scoped_chunks, LocalExecutionPlan, LocalThreadCount},
    progress::{replace_fea_progress_context, FeaCancellationToken, FeaProgressContext},
    properties::compile_property_map,
};

#[test]
fn scoped_beam_waves_preserve_serial_operators_and_recovery_order() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let structural = model.structural.as_mut().expect("beam fixture topology");
    let prototype = structural.elements[0].clone();
    structural.nodes = (1..=20)
        .map(|node_id| StructuralNode {
            node_id,
            coordinates_m: [f64::from(node_id - 1), 0.0, 0.0],
            solid_attachment: None,
        })
        .collect();
    structural.elements = (1..=19)
        .map(|node_id| {
            let mut element = prototype.clone();
            element.element_id = format!("beam_{node_id}");
            let StructuralElementKind::Beam(beam) = &mut element.kind else {
                panic!("beam fixture must contain a beam");
            };
            beam.node_ids = [node_id, node_id + 1];
            element
        })
        .collect();
    let properties = compile_property_map(&model, None).expect("beam property coverage");
    let layout = StructuralDofLayout::from_node_sets(vec![
        StructuralNodeDofSet::translational_rotational();
        20
    ]);
    let node_indices = (0..20).collect::<Vec<_>>();
    let structural = model.structural.as_ref().unwrap();
    let assemble =
        |plan| assemble_explicit_elements(structural, &properties, &layout, &node_indices, plan);
    let serial = assemble(LocalExecutionPlan::serial()).expect("serial beams");
    let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
    let parallel = assemble(plan).expect("scoped beams");
    let repeated = assemble(plan).expect("repeated scoped beams");
    assert_eq!(parallel.stiffness, serial.stiffness);
    assert_eq!(parallel.damping, serial.damping);
    assert_eq!(parallel.mass, serial.mass);
    assert_eq!(parallel.beam_recovery, serial.beam_recovery);
    assert_eq!(repeated.stiffness, parallel.stiffness);
    assert_eq!(repeated.beam_recovery, parallel.beam_recovery);
    assert_eq!(parallel.beam_count, 19);

    let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
        Ok::<_, ()>(assemble(plan))
    })
    .unwrap();
    assert!(matches!(
        &nested[0],
        Err(super::StructuralAssemblyError::LocalExecution(message))
            if message.contains("nested local numerical execution")
    ));

    let cancellation = FeaCancellationToken::new();
    cancellation.cancel();
    let _context = replace_fea_progress_context(FeaProgressContext {
        cancellation,
        ..FeaProgressContext::default()
    });
    assert!(matches!(
        assemble(plan),
        Err(super::StructuralAssemblyError::Cancelled)
    ));

    drop(_context);
    let mut invalid = model.clone();
    invalid.structural.as_mut().unwrap().elements[9].region_id = "unassigned".into();
    let error = assemble_explicit_elements(
        invalid.structural.as_ref().unwrap(),
        &properties,
        &layout,
        &node_indices,
        plan,
    )
    .err()
    .expect("worker failure must reach assembly");
    assert!(matches!(
        error,
        super::StructuralAssemblyError::UnassignedExplicitRegion { region_id, .. }
            if region_id == "unassigned"
    ));
}

#[test]
fn scoped_beam_scatter_preserves_interleaved_shell_operators() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let shell_model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
    model
        .material_assignments
        .extend(shell_model.material_assignments);
    let structural = model.structural.as_mut().expect("beam fixture topology");
    let shell = shell_model.structural.expect("shell fixture topology");
    structural.nodes.push(shell.nodes[2].clone());
    structural.elements.push(shell.elements[0].clone());
    structural.shell_sections = shell.shell_sections;
    let properties = compile_property_map(&model, None).expect("mixed explicit property coverage");
    let layout = StructuralDofLayout::from_node_sets(vec![
        StructuralNodeDofSet::translational_rotational();
        3
    ]);
    let node_indices = [0, 1, 2];
    let structural = model.structural.as_ref().unwrap();
    let serial = assemble_explicit_elements(
        structural,
        &properties,
        &layout,
        &node_indices,
        LocalExecutionPlan::serial(),
    )
    .unwrap();
    let parallel = assemble_explicit_elements(
        structural,
        &properties,
        &layout,
        &node_indices,
        LocalExecutionPlan::fixed(LocalThreadCount::new(2).unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(parallel.stiffness, serial.stiffness);
    assert_eq!(parallel.damping, serial.damping);
    assert_eq!(parallel.mass, serial.mass);
    assert_eq!(parallel.beam_recovery, serial.beam_recovery);
    assert_eq!(parallel.shell_recovery, serial.shell_recovery);
}
