//! Independent axial continuum, not an oscillator fitted to assembled K/M/C.
//! u_tt + alpha*u_t = c²*(u_xx + beta*u_xxt), u(0,t)=0, u_x(1,t)=0.
//! Unit area, rho=1, E=100, nu=0 and lateral translations held zero make this
//! an exact three-dimensional solid-elasticity solution. No eigensolve is used.
use crate::solve::transient::{StructuralTimeStepPolicy, TransientSolveOptions};
use crate::{
    fixtures::{
        fixture_model, replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind,
        FixtureId,
    },
    topology::test_meshes::{quadratic_slab_mesh, slab_mesh},
    ComputeBackend, FeaTransientRunResult,
};
use runmat_analysis_core::*;
use runmat_meshing_core::SolverMeshArtifact;

mod affine_forcing;
mod execution;
mod finer_envelope;
mod initial_equilibrium;
mod proof_workspace;
mod reference;
mod refinement;

const YOUNGS: f64 = 100.0;
const AMPLITUDE: f64 = 0.01;
const END_TIME: f64 = 0.2;
const ALPHA: f64 = 0.8;
const BETA: f64 = 0.002;

fn mesh(quadratic: bool, cells: usize) -> SolverMeshArtifact {
    if quadratic {
        quadratic_slab_mesh(cells)
    } else {
        slab_mesh(cells)
    }
}

fn model(mesh: &SolverMeshArtifact, damped: bool) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    let MaterialElasticityModel::IsotropicLinear(elasticity) =
        model.materials[0].elasticity.as_mut().unwrap()
    else {
        panic!("isotropic fixture")
    };
    elasticity.youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(YOUNGS);
    elasticity.poisson_ratio = runmat_types::physical::PoissonRatio::new(0.0);
    model.materials[0].mass_density_kg_per_m3 =
        (Some(1.0)).map(runmat_types::physical::MassDensity::new);
    model.materials[0].damping = damped.then_some(MechanicalDampingModel::Rayleigh {
        mass_coefficient: RayleighMassCoefficient::new(ALPHA),
        stiffness_coefficient: RayleighStiffnessCoefficient::new(BETA),
    });
    model.constraints = mesh
        .topology
        .nodes
        .iter()
        .map(|node| {
            let translations = [
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ]
            .into_iter()
            .filter(|dof| {
                node.coordinates_m.canonical_value()[0] == 0.0
                    || *dof != StructuralTranslationDof::Ux
            })
            .map(|dof| TranslationDofValue {
                dof,
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            })
            .collect();
            AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("rod_support_{}", node.node_id),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    translations,
                    rotations: Vec::new(),
                }),
            })
        })
        .collect();
    let mut step = ProcedureStep::structural_transient("rod_transient");
    step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: END_TIME,
        interval_count: 400,
    };
    model.sequence.steps = vec![step];
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "zero_traction".into(),
            region_id: "slab_right".into(),
            kind: FixtureExcitationKind::Traction([0.0; 3]),
        }],
    );
    // This original continuum fixture retains per-node typed inputs. Separate field-input
    // tests verify immutable nodal transport without replacing these accuracy gates.
    model.initial_conditions = mesh
        .topology
        .nodes
        .iter()
        .map(|node| {
            InitialCondition::Displacement(InitialConditionValue {
                initial_condition_id: format!("rod_initial_{}", node.node_id),
                target: InitialConditionTarget::Region(RegionTarget::point(format!(
                    "node:{}",
                    node.node_id
                ))),
                frame: CoordinateFrame::Global,
                value: InitialValue::UniformVector3(
                    [
                        reference::shape(node.coordinates_m.canonical_value()[0]) * AMPLITUDE,
                        0.0,
                        0.0,
                    ]
                    .map(Displacement::new),
                ),
                applies_before_step_id: "rod_transient".into(),
            })
        })
        .collect();
    model
}

fn run(mesh: &SolverMeshArtifact, damped: bool, intervals: usize) -> FeaTransientRunResult {
    crate::run_transient_with_options(
        &model(mesh, damped),
        ComputeBackend::Cpu,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed {
                interval_count: intervals,
            },
            residual_target: 1.0e-10,
            solver_mesh: Some(mesh.clone()),
            ..TransientSolveOptions::default()
        },
    )
    .unwrap()
}

mod field_inputs;
