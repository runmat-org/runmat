use super::*;
use crate::physics::electrical::tests::{mesh, model};
use runmat_meshing_core::ElementOrder;

fn compiled(
    order: ElementOrder,
    equation: StaticElectricalEquation,
    invalid: bool,
) -> CompiledAnalysisStudy {
    let mut model = model(equation);
    if invalid {
        model.materials[0]
            .electrical
            .as_mut()
            .unwrap()
            .resistive_heating_coefficient = runmat_types::physical::InverseTemperature::new(0.1);
    }
    let mesh = mesh(order, 2);
    let properties = crate::properties::compile_property_map(&model, Some(&mesh)).unwrap();
    CompiledAnalysisStudy {
        module_path: "tests/electrical.fea".into(),
        study_id: "electrical".into(),
        study_inputs: vec![],
        assessments: Vec::new(),
        outputs: vec![],
        model,
        properties,
        solver_mesh_digest: mesh.canonical_digest,
    }
}

#[test]
fn physical_volume_owner_materials_are_required_for_each_equation_and_order() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for equation in [
            StaticElectricalEquation::Electrostatics,
            StaticElectricalEquation::SteadyConduction,
        ] {
            let study = compiled(order, equation, false);
            study.validate_against(&mesh(order, 2)).unwrap();
            validate_materials(&study, equation).unwrap();
            // A resistive-heating coefficient is ordinary material data (used by
            // electrothermal analyses), not an unsupported material.
            validate_materials(&compiled(order, equation, true), equation).unwrap();
        }
    }
}
