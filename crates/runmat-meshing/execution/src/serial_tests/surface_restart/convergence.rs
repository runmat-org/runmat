//! Protocol fixtures follow every requested restart before asserting terminal publication.
use super::*;

pub(super) struct SurfacePass {
    pub plan: crate::ExactSurfacePassPlan,
    pub curves: CompletedMeshingStage,
    pub partition: CompletedMeshingStage,
    pub joined: CompletedMeshingStage,
}

pub(super) fn converge(
    fixture: &mut Fixture,
    planner: &crate::ExactMeshingDagPlanner,
    topology: &runmat_geometry_core::ExactBRepTopology,
    mut current: SurfacePass,
) -> SurfacePass {
    for _ in 0..64 {
        let streams = current
            .joined
            .publication()
            .stage_objects()
            .decoded_streams()
            .unwrap();
        if streams
            .iter()
            .any(|stream| stream.media_type == MeshingChunkMediaType::SurfaceMesh)
        {
            return current;
        }
        let refinement = planner
            .curve_refinement(
                &current.plan,
                vec![root(current.partition.publication().root_output())],
                root(current.joined.publication().root_output()),
            )
            .unwrap();
        let curves = dag::execute_planned(
            fixture,
            &refinement,
            &crate::ExactCurveRefinementKernel::default(),
        );
        let plan = planner
            .next_surface_pass(
                &current.plan,
                topology,
                root(curves.publication().root_output()),
                32,
            )
            .unwrap();
        assert_eq!(
            plan.partitions().len(),
            1,
            "fixture owns one face partition"
        );
        let partition = dag::execute_planned(
            fixture,
            &plan.partitions()[0],
            &ExactSurfacePartitionKernel::default(),
        );
        let join = planner
            .surface_join(&plan, vec![root(partition.publication().root_output())])
            .unwrap();
        let joined = dag::execute_planned(fixture, &join, &crate::ExactSurfaceJoinKernel);
        current = SurfacePass {
            plan,
            curves,
            partition,
            joined,
        };
    }
    panic!("fixture surface did not converge within 64 restart passes")
}

pub(in crate::serial_tests) fn execute_from_fixture(
    fixture: Fixture,
) -> (
    Fixture,
    MeshingHostWorkload,
    ValueRef,
    CompletedMeshingStage,
    CompletedMeshingStage,
    CompletedMeshingStage,
) {
    let (mut fixture, host, exact_root, curves, partition, joined) =
        super::execute_surface_join_pipeline_from_fixture(
            &ExactSurfacePartitionKernel::default(),
            fixture,
        );
    let exact = import_exact_geometry_input(
        &fixture.store,
        host.geometry_document.clone().unwrap(),
        &exact_root,
        host.artifact_access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let planner =
        crate::ExactMeshingDagPlanner::from_exact_host(&host, exact_root.clone()).unwrap();
    let plan = planner
        .begin_surface_pass(
            &exact.geometry_objects().topology,
            root(curves.publication().root_output()),
            32,
        )
        .unwrap();
    let result = converge(
        &mut fixture,
        &planner,
        &exact.geometry_objects().topology,
        SurfacePass {
            plan,
            curves,
            partition,
            joined,
        },
    );
    let final_stage = planner
        .surface_join(
            &result.plan,
            vec![root(result.partition.publication().root_output())],
        )
        .unwrap();
    (
        fixture,
        final_stage.host().clone(),
        exact_root,
        result.curves,
        result.partition,
        result.joined,
    )
}
