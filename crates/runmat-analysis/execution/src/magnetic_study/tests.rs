//! Metadata/phase guards. Native continuum and cold-root evidence live in runner tests.
use super::*;
use runmat_analysis_core::{
    AnalysisConstraint, ElectromagneticDomain, MagneticConstraint, MagneticConstraintKind,
    MagnetostaticFormulation, PropertyRegionTarget, RegionTarget,
};
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, StableDigest,
};

fn preparation(order: ElementOrder) -> MagneticStudyPreparation {
    let input = crate::electrical_study::tests::preparation();
    let mesh = canonical_tetrahedron_solver_mesh(order);
    let mut compiled = input.product.compiled;
    compiled.model.electromagnetic = Some(ElectromagneticDomain::Magnetostatic {
        formulation: MagnetostaticFormulation::MagneticVectorPotential,
    });
    compiled.model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("solid"));
    compiled.model.constraints.clear();
    for face in 0..4 {
        compiled
            .model
            .constraints
            .push(AnalysisConstraint::Magnetic(MagneticConstraint {
                constraint_id: format!("ground:{face}"),
                kind: MagneticConstraintKind::VectorPotentialGround {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(
                            &(format!("face:{face}")),
                        ),
                    ),
                },
            }));
    }
    compiled
        .model
        .constraints
        .push(AnalysisConstraint::Magnetic(MagneticConstraint {
            constraint_id: "gauge".into(),
            kind: MagneticConstraintKind::CoulombGauge {
                target: RegionTarget::volume(runmat_meshing_core::fixtures::fixture_solid("solid")),
            },
        }));
    compiled.properties =
        runmat_analysis_fea::properties::compile_property_map(&compiled.model, Some(&mesh))
            .unwrap();
    compiled.solver_mesh_digest = mesh.canonical_digest;
    let policy = MagneticStudyPlanningPolicy {};
    let workload = plan_magnetic_study(&compiled, policy).unwrap();
    MagneticStudyPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        policy,
        product: MagneticStudyProduct { compiled, workload },
        mesh,
        frozen_project_digest: input.frozen_project_digest,
        artifact_access: input.artifact_access,
        program_revision: input.program_revision,
        output_policy: input.output_policy,
        required_capabilities: input.required_capabilities,
    }
}

#[test]
fn magnetic_study_exact_authored_sources_and_phase_guards_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut study = MagneticStudyCoordinator::new(preparation(order)).unwrap();
        assert_eq!(study.phase(), MagneticStudyPhase::ReadyForSolve);
        assert!(study.prepare_field_recovery().is_err());
        assert!(study.prepare_observation_reduction().is_err());
        assert!(study.prepare_observation_assembly().is_err());
        assert!(study.prepare_accuracy().is_err());
        assert!(study.prepare_next_state().is_err());
        assert!(study.freeze_baseline().is_err());
        assert!(study.prepare_publication().is_err());
        let task = study.prepare_solve().unwrap();
        assert_eq!(
            task.host.workload.kind,
            runmat_analysis_fea::execution::AnalysisWorkKind::SolveMagnetostatic
        );
        assert!(study.prepare_solve().is_err());
        assert!(!study.is_complete());
        assert!(study.into_completed().is_err());
        let mut changed = preparation(order);
        changed.product.compiled.solver_mesh_digest = StableDigest::ZERO;
        assert!(MagneticStudyCoordinator::new(changed).is_err());
        let mut changed = preparation(order);
        changed.product.workload.relative_residual_target *= 0.5;
        assert!(MagneticStudyCoordinator::new(changed).is_err());
        let mut changed = preparation(order);
        changed.product.compiled.model.materials[0]
            .electrical
            .as_mut()
            .unwrap()
            .relative_permeability = runmat_types::physical::Quantity::new(
            changed.product.compiled.model.materials[0]
                .electrical
                .as_mut()
                .unwrap()
                .relative_permeability
                .canonical_value()
                * (2.0),
        );
        assert!(MagneticStudyCoordinator::new(changed).is_err());
    }
}

#[test]
fn magnetic_cold_completion_cannot_skip_solve_or_accept_empty_closure() {
    struct Empty;
    impl CacheImport for Empty {
        fn read_verified(
            &self,
            _: Digest,
        ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
            Ok(None)
        }
    }
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        assert!(import_magnetic_study_completion(&Empty, preparation(order), Vec::new()).is_err());
    }
}
