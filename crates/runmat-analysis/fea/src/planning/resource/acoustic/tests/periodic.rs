use super::*;

#[test]
fn periodic_graph_workspace_is_admitted_without_raising_the_authored_ceiling() {
    for mesh in [slab_mesh(2), quadratic_slab_mesh(2)] {
        let base = workload();
        let base_plan = build_acoustic_resource_plan(&base, &mesh, facts(&base, &mesh)).unwrap();
        let mut periodic = base.clone();
        periodic.model.boundaries.clear();
        periodic
            .model
            .constraints
            .push(AnalysisConstraint::Acoustic(AcousticConstraint {
                constraint_id: "paired".into(),
                kind: AcousticConstraintKind::Periodic(PeriodicConstraint {
                    pair: PairedConstraint {
                        primary: RegionTarget::boundary(
                            runmat_meshing_core::fixtures::fixture_face("slab_left"),
                        ),
                        secondary: RegionTarget::boundary(
                            runmat_meshing_core::fixtures::fixture_face("slab_right"),
                        ),
                        pairing: PairingPolicy::Matching,
                    },
                    transform: PeriodicTransform::Translation(Translation::new([1.0, 0.0, 0.0])),
                }),
            }));
        let prepared = facts(&periodic, &mesh);
        let plan = build_acoustic_resource_plan(&periodic, &mesh, prepared.clone()).unwrap();
        let FeaResourceDemand::AcousticHarmonic(demand) = &plan.input.demand else {
            panic!("acoustic demand")
        };
        assert_eq!(demand.recovery_row_support_upper_bound.0, 2);
        assert!(plan.breakdown.workspace_bytes > base_plan.breakdown.workspace_bytes);
        assert_eq!(
            periodic.model.sequence.steps[0].accuracy.resources,
            base.model.sequence.steps[0].accuracy.resources
        );
        assert_ne!(plan.identity, base_plan.identity);
        let mut limited = periodic;
        limited.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(1);
        assert!(build_acoustic_resource_plan(&limited, &mesh, facts(&limited, &mesh)).is_err());
    }
}
