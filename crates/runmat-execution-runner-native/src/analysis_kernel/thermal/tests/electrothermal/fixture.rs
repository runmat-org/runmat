use runmat_analysis_core::*;
use runmat_analysis_fea::{
    execution::{RetainedJouleHeatSource, ThermalWorkload},
    ComputeBackend, ThermalProcedureControl,
};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

use super::electrical_fixture;

pub(crate) fn fixture(
    order: ElementOrder,
) -> (RetainedJouleHeatSource, ThermalWorkload, SolverMeshArtifact) {
    let (electrical, mesh) = electrical_fixture(order, StaticElectricalEquation::SteadyConduction);
    let mut model = electrical.model.clone();
    let result = electrical.execute(&mesh).unwrap();
    let expected = result.source_result_identity().unwrap();
    model.constraints.clear();
    model.electromagnetic = None;
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(300.0),
    });
    model.materials[0].thermal = Some(MaterialThermalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(300.0),
        modulus_temp_coeff_per_k: runmat_types::physical::InverseTemperature::new(0.0),
        conductivity_w_per_mk: runmat_types::physical::ThermalConductivity::new(1.0),
        specific_heat_j_per_kgk: runmat_types::physical::SpecificHeat::new(1.0),
        expansion_coefficient_per_k: runmat_types::physical::ThermalExpansion::new(0.0),
    });
    model.sequence =
        AnalysisSequence::from_steps("heat", vec![ProcedureStep::thermal_steady("heat")]);
    for face in ["slab_left", "slab_right"] {
        model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: face.into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(face),
                    ),
                    value: PrescribedValue::Constant(Temperature::new(300.0)),
                },
            }));
    }
    let thermal = ThermalWorkload {
        model,
        step_id: ProcedureStepId("heat".into()),
        backend: ComputeBackend::Cpu,
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
