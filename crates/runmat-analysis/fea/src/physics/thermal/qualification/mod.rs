//! Original independent continuum fixtures shared by core and native qualification.
//! These validation models do not certify arbitrary authored spatial/temporal accuracy.
use crate::{
    execution::ThermalWorkload,
    fixtures::{fixture_model, FixtureId},
    ComputeBackend, ThermalProcedureControl,
};
use runmat_analysis_core::*;
use runmat_meshing_core::{
    fixtures::{quadratic_slab_mesh, slab_mesh},
    SolverMeshArtifact,
};
pub mod cooling;
pub mod initial;
pub mod reference;

pub const TARGET: f64 = 1.0e-9;
pub const SIGMA: f64 = 5.670_374_419e-8;
pub const EMISSIVITY: f64 = 0.7;
pub const AMBIENT: f64 = 300.0;
pub const K: f64 = 2.0;

pub fn mesh(quadratic: bool) -> SolverMeshArtifact {
    if quadratic {
        quadratic_slab_mesh(1)
    } else {
        slab_mesh(2)
    }
}

pub fn workload() -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.boundaries.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.initial_conditions.clear();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.materials[0].mass_density_kg_per_m3 =
        (Some(1.0)).map(runmat_types::physical::MassDensity::new);
    let thermal = model.materials[0].thermal.as_mut().unwrap();
    thermal.conductivity_w_per_mk = runmat_types::physical::ThermalConductivity::new(K);
    thermal.specific_heat_j_per_kgk = runmat_types::physical::SpecificHeat::new(1.0);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(400.0),
    });
    model.sequence =
        AnalysisSequence::from_steps("radiation", vec![ProcedureStep::thermal_steady("heat")]);
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "gray-radiation".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: BoundaryOperator::ThermalRadiation {
            ambient_temperature: PrescribedValue::Constant(Temperature::new(AMBIENT)),
            emissivity: SurfaceEmissivity::new(EMISSIVITY),
        },
    });
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "insulated-sides".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
            "slab_insulated",
        )),
        operator: BoundaryOperator::ThermalHeatFlux {
            value: PrescribedValue::Constant(HeatFlux::new(0.0)),
        },
    });
    ThermalWorkload {
        model,
        initial_fields: Vec::new(),
        step_id: ProcedureStepId("heat".into()),
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::Steady,
        residual_target: TARGET,
    }
}

pub fn left_temperature(input: &mut ThermalWorkload, value: PrescribedValue<Temperature>) {
    input
        .model
        .constraints
        .push(AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "fixed-hot-left".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                value,
            },
        }));
}
