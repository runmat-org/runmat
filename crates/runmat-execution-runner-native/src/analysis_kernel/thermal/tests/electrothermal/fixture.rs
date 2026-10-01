use runmat_analysis_core::*;
use runmat_analysis_fea::{
    execution::{ElectricalStaticWorkload, RetainedJouleHeatSource, ThermalWorkload},
    ComputeBackend, ThermalProcedureControl,
};
use runmat_meshing_core::{
    fixtures::{quadratic_slab_mesh, slab_mesh},
    ElementOrder, SolverMeshArtifact,
};

pub(crate) fn fixture(
    order: ElementOrder,
) -> (RetainedJouleHeatSource, ThermalWorkload, SolverMeshArtifact) {
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
    model.electromagnetic = Some(ElectromagneticDomain::Static {
        equation: StaticElectricalEquation::SteadyConduction,
    });
    model.sequence = AnalysisSequence::from_steps(
        "conductor",
        vec![ProcedureStep::electromagnetic_static("potential")],
    );
    model.material_assignments[0].target = PropertyRegionTarget::solid("slab");
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: 300.0,
        conductivity_s_per_m: 3.0,
        relative_permittivity: 1.0,
        relative_permeability: 1.0,
        resistive_heating_coefficient: 0.0,
        conductivity_frequency_response: vec![],
    });
    for (face, volts) in [("slab_left", 0.0), ("slab_right", 2.0)] {
        model
            .constraints
            .push(AnalysisConstraint::Electrical(ElectricalConstraint {
                constraint_id: face.into(),
                kind: ElectricalConstraintKind::PrescribedPotential {
                    target: RegionTarget::boundary(face),
                    value: PrescribedValue::Constant(ElectricPotential::new(volts)),
                },
            }));
    }
    let electrical = ElectricalStaticWorkload {
        model: model.clone(),
        step_id: ProcedureStepId("potential".into()),
        formulation_requirement: FormulationRequirement::Experimental,
        relative_residual_target: 1e-11,
    };
    let result = electrical.execute(&mesh).unwrap();
    let expected = result.source_result_identity().unwrap();
    model.constraints.clear();
    model.electromagnetic = None;
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(300.0),
    });
    model.materials[0].thermal = Some(MaterialThermalModel {
        reference_temperature_k: 300.0,
        modulus_temp_coeff_per_k: 0.0,
        conductivity_w_per_mk: 1.0,
        specific_heat_j_per_kgk: 1.0,
        expansion_coefficient_per_k: 0.0,
    });
    model.sequence =
        AnalysisSequence::from_steps("heat", vec![ProcedureStep::thermal_steady("heat")]);
    for face in ["slab_left", "slab_right"] {
        model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: face.into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(face),
                    value: PrescribedValue::Constant(Temperature::new(300.0)),
                },
            }));
    }
    let thermal = ThermalWorkload {
        model,
        step_id: ProcedureStepId("heat".into()),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        procedure: ThermalProcedureControl::Steady,
        residual_target: 1e-11,
        initial_fields: vec![],
    };
    let source =
        RetainedJouleHeatSource::new(electrical, result, expected, thermal, &mesh).unwrap();
    let bytes = source.canonical_bytes().unwrap();
    drop(source);
    let source = RetainedJouleHeatSource::from_canonical_bytes(&bytes).unwrap();
    let bound = source.bind(&mesh).unwrap();
    (source, bound, mesh)
}
