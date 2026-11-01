//! Original slab source shared by steady and temporal native witnesses.
use runmat_analysis_core::*;
use runmat_analysis_fea::{
    execution::FluidSteadyWorkload,
    fixtures::{fixture_model, FixtureId},
    physics::cfd::canonical::CanonicalFluidPolicy,
    topology::FiniteVolumeProjectionPolicy,
};
use runmat_meshing_core::{fixtures, SolverMeshArtifact};
pub fn workload(nonzero: bool, reference_pa: f64) -> (FluidSteadyWorkload, SolverMeshArtifact) {
    let mesh = fixtures::slab_mesh(1);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.structural = None;
    model.thermal = None;
    model.thermo_mechanical = None;
    model.electro_thermal = None;
    model.electromagnetic = None;
    model.constraints.clear();
    model.boundaries.clear();
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.initial_conditions.clear();
    model.interfaces.clear();
    model.connectors.clear();
    model.response_spectra.clear();
    model.random_excitations.clear();
    model.frame = CoordinateFrame::Global;
    model.materials[0].mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(1.0));
    model.materials[0].fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
        dynamic_viscosity: DynamicViscosity::new(1.0),
    });
    model.material_assignments[0].target =
        PropertyRegionTarget::cell(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.cfd = Some(CfdDomain {
        density: MassDensity::new(1.0),
        dynamic_viscosity: DynamicViscosity::new(1.0),
        turbulence_intensity: TurbulenceIntensity::new(0.0),
        rheology: None,
        porous_zones: Vec::new(),
    });
    model.sequence =
        AnalysisSequence::from_steps("native-fluid", vec![ProcedureStep::fluid_static("steady")]);
    let velocity = [
        Velocity::new(if nonzero { 1.0 } else { 0.0 }),
        Velocity::new(0.0),
        Velocity::new(0.0),
    ];
    model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "wall".into(),
            kind: if nonzero {
                FluidConstraintKind::MovingWall {
                    target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        "slab_insulated",
                    )),
                    velocity: PrescribedValue::Constant(velocity),
                }
            } else {
                FluidConstraintKind::NoSlipWall {
                    target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        "slab_insulated",
                    )),
                }
            },
        }));
    model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "inlet".into(),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                value: PrescribedValue::Constant(velocity),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "outlet".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: BoundaryOperator::FluidOutletPressure {
            value: PrescribedValue::Constant(Pressure::new(reference_pa)),
        },
    });
    for constraint in &mut model.constraints {
        let AnalysisConstraint::Fluid(fluid) = constraint else {
            unreachable!()
        };
        let target = match &mut fluid.kind {
            FluidConstraintKind::NoSlipWall { target }
            | FluidConstraintKind::MovingWall { target, .. }
            | FluidConstraintKind::PrescribedVelocity { target, .. } => target,
            _ => unreachable!(),
        };
        let entities = mesh
            .topology
            .boundary_faces
            .iter()
            .flat_map(|f| &f.provenance)
            .filter(|e| target.matches_persistent_entity(e))
            .cloned()
            .collect();
        target.bind_resolved_entities(entities).unwrap();
    }
    for boundary in &mut model.boundaries {
        let entities = mesh
            .topology
            .boundary_faces
            .iter()
            .flat_map(|f| &f.provenance)
            .filter(|e| boundary.target.matches_persistent_entity(e))
            .cloned()
            .collect();
        boundary.target.bind_resolved_entities(entities).unwrap();
    }
    let workload = FluidSteadyWorkload {
        model,
        step_id: ProcedureStepId("steady".into()),
        policy: CanonicalFluidPolicy {
            pressure_velocity_coupling: Default::default(),
            fixed_point_acceleration: None,
            maximum_outer_iterations: 200,
            maximum_nonorthogonal_iterations: 4,
            maximum_linear_iterations: 100,
            gmres_restart: 8,
            momentum_relaxation: 0.5,
            pressure_relaxation: 0.25,
            linear_relative_tolerance: 1e-10,
            physical_relative_tolerance: 1e-8,
            maximum_resident_bytes: Some(1_000_000_000),
            maximum_operator_work: Some(1_000_000_000_000),
            maximum_authority_items: 100,
        },
        geometry_policy: FiniteVolumeProjectionPolicy {
            max_cells: Some(100),
            max_face_incidences: Some(400),
            max_boundary_faces: Some(400),
            max_scratch_bytes: Some(10_000_000),
            max_traversal_operations: Some(1_000_000),
        },
    };
    (workload, mesh)
}
