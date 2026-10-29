//! Source intent only: native reference tests do not borrow parent solver outputs.
use runmat_analysis_core::*;
use runmat_analysis_fea::execution::ElectricalStaticWorkload;
use runmat_meshing_core::{
    fixtures::{quadratic_slab_mesh, slab_mesh},
    ElementOrder, SolverMeshArtifact,
};

pub(crate) fn electrical_fixture(
    order: ElementOrder,
    equation: StaticElectricalEquation,
) -> (ElectricalStaticWorkload, SolverMeshArtifact) {
    let mesh = match order {
        ElementOrder::Tet4 => slab_mesh(2),
        ElementOrder::Tet10 => quadratic_slab_mesh(2),
    };
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
    );
    model.constraints.clear();
    model.boundaries.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.initial_conditions.clear();
    model.structural = None;
    model.thermal = None;
    model.thermo_mechanical = None;
    model.electro_thermal = None;
    model.cfd = None;
    model.electromagnetic = Some(ElectromagneticDomain::Static { equation });
    model.sequence = AnalysisSequence::from_steps(
        "conductor",
        vec![ProcedureStep::electromagnetic_static("potential")],
    );
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(300.0),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(3.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        conductivity_frequency_response: vec![],
    });
    for (face, volts) in [("slab_left", 0.0), ("slab_right", 2.0)] {
        model
            .constraints
            .push(AnalysisConstraint::Electrical(ElectricalConstraint {
                constraint_id: face.into(),
                kind: ElectricalConstraintKind::PrescribedPotential {
                    target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        face,
                    )),
                    value: PrescribedValue::Constant(ElectricPotential::new(volts)),
                },
            }));
    }
    let electrical = ElectricalStaticWorkload {
        model: model.clone(),
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-11,
        element_temperature_k: None,
        operating_time_s: None,
    };
    (electrical, mesh)
}
