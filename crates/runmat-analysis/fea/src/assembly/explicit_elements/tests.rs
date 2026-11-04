use runmat_analysis_core::{StructuralElementKind, StructuralNode};

use super::{
    assemble_explicit_elements, assemble_explicit_elements_with_owner, ExplicitCapacityPhase,
    StructuralDofLayout,
};
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
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([f64::from(node_id - 1), 0.0, 0.0]),
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
    // Element assembly composes with an enclosing serial chunk and stays canonical.
    let nested = nested[0].as_ref().expect("nested element assembly");
    assert_eq!(nested.stiffness, serial.stiffness);
    assert_eq!(nested.beam_recovery, serial.beam_recovery);

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
    let phases = std::cell::RefCell::new(Vec::new());
    let awaited = futures::executor::block_on(assemble_explicit_elements_with_owner(
        structural,
        &properties,
        &layout,
        &node_indices,
        LocalExecutionPlan::serial(),
        true,
        |phase, count| {
            phases.borrow_mut().push((phase, count));
            std::future::ready(Ok::<(), super::StructuralAssemblyError>(()))
        },
    ))
    .unwrap();
    assert_eq!(awaited.stiffness, serial.stiffness);
    assert_eq!(awaited.mass, serial.mass);
    assert_eq!(awaited.damping, serial.damping);
    assert_eq!(awaited.beam_recovery, serial.beam_recovery);
    assert_eq!(awaited.shell_recovery, serial.shell_recovery);
    assert!(phases
        .borrow()
        .iter()
        .any(|(phase, _)| matches!(phase, ExplicitCapacityPhase::Element { .. })));
    assert!(phases
        .borrow()
        .iter()
        .any(|(phase, _)| matches!(phase, ExplicitCapacityPhase::Scatter { .. })));
    let denied = futures::executor::block_on(assemble_explicit_elements_with_owner(
        structural,
        &properties,
        &layout,
        &node_indices,
        LocalExecutionPlan::serial(),
        true,
        |_, _| {
            std::future::ready(Err::<(), _>(
                super::StructuralAssemblyError::LocalExecution("admission denied".into()),
            ))
        },
    ));
    assert!(matches!(
        denied,
        Err(super::StructuralAssemblyError::LocalExecution(message)) if message == "admission denied"
    ));
}

#[test]
fn scoped_shell_waves_preserve_damped_operators_and_recovery_order() {
    let mut model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
    model.materials[0].damping = Some(runmat_analysis_core::MechanicalDampingModel::Rayleigh {
        mass_coefficient: runmat_analysis_core::RayleighMassCoefficient::new(0.1),
        stiffness_coefficient: runmat_analysis_core::RayleighStiffnessCoefficient::new(1.0e-5),
    });
    let structural = model.structural.as_mut().expect("shell fixture topology");
    let prototype_nodes = structural.nodes.clone();
    let prototype = structural.elements[0].clone();
    structural.nodes = (0..19)
        .flat_map(|element_index| {
            prototype_nodes
                .iter()
                .enumerate()
                .map(move |(local, node)| StructuralNode {
                    node_id: (3 * element_index + local + 1) as u32,
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new([
                        node.coordinates_m.canonical_value()[0] + 10.0 * element_index as f64,
                        node.coordinates_m.canonical_value()[1],
                        node.coordinates_m.canonical_value()[2],
                    ]),
                    solid_attachment: None,
                })
        })
        .collect();
    structural.elements = (0..19)
        .map(|element_index| {
            let mut element = prototype.clone();
            element.element_id = format!("shell_{element_index}");
            let StructuralElementKind::Shell(shell) = &mut element.kind else {
                panic!("shell fixture must contain a shell");
            };
            shell.node_ids = [
                (3 * element_index + 1) as u32,
                (3 * element_index + 2) as u32,
                (3 * element_index + 3) as u32,
            ];
            element
        })
        .collect();
    let properties = compile_property_map(&model, None).expect("shell property coverage");
    let layout = StructuralDofLayout::from_node_sets(vec![
        StructuralNodeDofSet::translational_rotational();
        57
    ]);
    let node_indices = (0..57).collect::<Vec<_>>();
    let structural = model.structural.as_ref().unwrap();
    let assemble =
        |plan| assemble_explicit_elements(structural, &properties, &layout, &node_indices, plan);
    let serial = assemble(LocalExecutionPlan::serial()).expect("serial shells");
    let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
    let parallel = assemble(plan).expect("scoped shells");
    let repeated = assemble(plan).expect("repeated scoped shells");
    assert_eq!(parallel.stiffness, serial.stiffness);
    assert_eq!(parallel.mass, serial.mass);
    assert_eq!(parallel.damping, serial.damping);
    assert_eq!(parallel.shell_recovery, serial.shell_recovery);
    assert_eq!(repeated.stiffness, parallel.stiffness);
    assert_eq!(repeated.damping, parallel.damping);
    assert_eq!(repeated.shell_recovery, parallel.shell_recovery);
    assert_eq!(parallel.shell_count, 19);
    for row in 0..layout.total_dof_count() {
        for column in 0..layout.total_dof_count() {
            let expected =
                0.1 * parallel.mass.get(row, column) + 1.0e-5 * parallel.stiffness.get(row, column);
            let actual = parallel.damping.get(row, column);
            assert!(
                (actual - expected).abs() <= 1.0e-8 * expected.abs().max(1.0),
                "shell Rayleigh damping differs at ({row}, {column}): {actual} vs {expected}"
            );
        }
    }

    let mut invalid = model.clone();
    invalid.structural.as_mut().unwrap().elements[9].region_id = "unassigned".into();
    assert!(matches!(
        assemble_explicit_elements(
            invalid.structural.as_ref().unwrap(),
            &properties,
            &layout,
            &node_indices,
            plan,
        ),
        Err(super::StructuralAssemblyError::UnassignedExplicitRegion { region_id, .. })
            if region_id == "unassigned"
    ));
}
