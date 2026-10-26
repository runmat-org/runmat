//! Compare preassembly coercivity reservations with the assembled physical operator's proof.

use std::collections::BTreeSet;

use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, MaterialElasticityModel,
    PrescribedStructuralDofs, PrescribedValue, RegionTarget, StructuralConstraint,
    StructuralConstraintKind, StructuralTranslationDof, TranslationDofValue,
};
use runmat_meshing_core::{
    fixtures::{connected_quadratic_tetrahedron_solver_mesh, connected_tetrahedron_solver_mesh},
    CanonicalMeshingContract, SolverMeshArtifact,
};

use super::*;
use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    execution::StructuralLinearWorkload,
    post::fields::structural_static_observable_field_definitions,
    results::observation::ObservationExecutionPolicy,
    run_linear_static_with_prepared_constraint_plan, ComputeBackend, LinearStaticSolveOptions,
    StaticResourceAdmission,
};

fn affine_patch_model(mesh: &SolverMeshArtifact) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let MaterialElasticityModel::IsotropicLinear(elasticity) = model.materials[0]
        .elasticity
        .as_mut()
        .expect("fixture elasticity")
    else {
        panic!("expected isotropic linear elasticity")
    };
    elasticity.youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(100.0);
    elasticity.poisson_ratio = runmat_types::physical::PoissonRatio::new(0.25);
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints = mesh
        .topology
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|node_id| {
            let coordinates = mesh.topology.nodes[node_id as usize - 1]
                .coordinates_m
                .canonical_value();
            AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("affine_boundary_node_{node_id}"),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{node_id}")),
                    translations: [
                        StructuralTranslationDof::Ux,
                        StructuralTranslationDof::Uy,
                        StructuralTranslationDof::Uz,
                    ]
                    .into_iter()
                    .zip([0.1, 0.2, 0.3])
                    .enumerate()
                    .map(|(axis, (dof, gradient))| TranslationDofValue {
                        dof,
                        value: PrescribedValue::Constant(Displacement::new(
                            gradient * coordinates[axis],
                        )),
                    })
                    .collect(),
                    rotations: Vec::new(),
                }),
            })
        })
        .collect();
    model
}

#[test]
fn connected_tet4_and_tet10_exact_coercivity_proofs_fit_preassembly_envelopes() {
    for mesh in [
        connected_tetrahedron_solver_mesh(),
        connected_quadratic_tetrahedron_solver_mesh(),
    ] {
        mesh.validate_canonical().unwrap();
        let model = affine_patch_model(&mesh);
        let constraints =
            CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap();
        let fields = structural_static_observable_field_definitions().unwrap();
        let observations = resolve_structural_observation_plan(
            &model.sequence.steps[0],
            &model,
            Some(&mesh),
            &constraints,
            &fields,
            ObservationLimits::default(),
        )
        .unwrap();
        let options = LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            ..LinearStaticSolveOptions::default()
        };
        let workload = StructuralLinearWorkload {
            schema_version: 2,
            step_id: model.sequence.steps[0].step_id.clone(),
            model: model.clone(),
            backend: ComputeBackend::Cpu,
            preconditioner: options.preconditioner_kind,
            algebra_backend: options.algebra_backend_kind,
            observation_execution: ObservationExecutionPolicy::serial(),
            require_solver_mesh_for_solid: true,
            state_input: None,
        };
        let plan = build_structural_resource_plan(
            Some(&mesh),
            &constraints,
            &observations,
            &workload,
            &[],
            LocalExecutionPlan::serial(),
            None,
        )
        .unwrap();
        let SolveResourcePolicy::Iterative { admission, .. } =
            &plan.input.structural_operator().unwrap().solve
        else {
            panic!("physical static patch requires iterative solver admission")
        };
        let excitations = realize_step_excitations(
            &model,
            &model.sequence.steps[0],
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let summary = assemble_structural_system(
            &model,
            &excitations,
            Some(mesh.clone()),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        assert!(summary.affine_constraint_reduction.is_none());
        assert!(summary.constraint_matrix.is_none());
        let prepared = summary.operator.prepare_linear().unwrap();
        let proof = prepared.admissibility().unwrap().coercivity();
        let (dimension, factor_nonzeros, updates, resident_bytes) = proof.work_counts();
        assert!(admission.admits_certificate(proof));
        assert_eq!(dimension, admission.dimension.0);
        assert!(factor_nonzeros <= admission.factor_nonzero_upper_bound.0);
        assert!(updates <= admission.update_count_upper_bound.0);
        assert!(resident_bytes <= admission.resident_bytes_upper_bound.0);
        eprintln!(
            "connected {:?}: factor_nonzeros={factor_nonzeros}/{}, updates={updates}/{}, proof_resident_bytes={resident_bytes}/{}",
            mesh.resolved_request.element_order,
            admission.factor_nonzero_upper_bound.0,
            admission.update_count_upper_bound.0,
            admission.resident_bytes_upper_bound.0,
        );
        let run = run_linear_static_with_prepared_constraint_plan(
            &model,
            ComputeBackend::Cpu,
            options,
            &constraints,
            Some(StaticResourceAdmission {
                prepared_inputs: None,
                plan: &plan,
                observation_execution: ObservationExecutionPolicy::serial(),
            }),
        )
        .unwrap();
        assert!(run.structural_static_balance.is_some());
    }
}
