use super::*;
use runmat_analysis_fea::{
    execution::StructuralLinearWorkload,
    fixtures::{fixture_model, FixtureId},
    planning::resource::{LocalExecutionPlan, ResourcePlanningError, StaticSolveSelection},
    solve::{backend::kind::LinearAlgebraBackendKind, preconditioner::SpdPreconditionerKind},
    structural_linear_preparation_work_upper_bound, ComputeBackend,
};
use runmat_meshing_core::{ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest};
use runmat_simulation_usage::{LedgerError, OperationCount, UsageContext, UsageLedger};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
enum ObservedSolidMode {
    EmptyConnectors,
    SurfaceFoundation,
    BoundaryNormalSupport,
    WholeModelBodyForce,
}

fn observed_solid_fixture(
    order: ElementOrder,
    mode: ObservedSolidMode,
) -> (
    runmat_analysis_core::AnalysisModel,
    runmat_meshing_core::SolverMeshArtifact,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps[0].observations = vec![structural_displacement_observation()];
    if matches!(mode, ObservedSolidMode::SurfaceFoundation) {
        model
            .connectors
            .push(runmat_analysis_core::AnalysisConnector {
                connector_id: "tip_foundation".into(),
                kind: runmat_analysis_core::ConnectorKind::SurfaceNormalFoundation {
                    target: runmat_analysis_core::RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_face("tip"),
                    ),
                    modulus: runmat_analysis_core::FoundationModulus::new(1.0),
                },
            });
    }
    if matches!(mode, ObservedSolidMode::BoundaryNormalSupport) {
        use runmat_analysis_core::{
            AnalysisConstraint, CoordinateFrame, Displacement, PrescribedStructuralDofs,
            PrescribedValue, StructuralConstraintKind, StructuralTranslationDof,
            TranslationDofValue,
        };
        let [AnalysisConstraint::Structural(root)] = model.constraints.as_mut_slice() else {
            panic!("cantilever fixture must own exactly one structural root support");
        };
        let StructuralConstraintKind::Fixed { target } = &root.kind else {
            panic!("cantilever root support must originally be fixed");
        };
        let target = target.clone();
        root.frame = CoordinateFrame::BoundaryNormalTangent;
        root.kind = StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target,
            translations: [
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ]
            .into_iter()
            .map(|dof| TranslationDofValue {
                dof,
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            })
            .collect(),
            rotations: Vec::new(),
        });
    }
    if matches!(mode, ObservedSolidMode::WholeModelBodyForce) {
        use runmat_analysis_core::{
            BodyForceDensity, ExcitationAction, ExcitationDistribution, ExcitationTarget,
            PhysicalSignal, Signal, SignalRepresentation, SignalValue,
        };
        let ExcitationAction::Force(mut action) = model.excitations[0].clone() else {
            panic!("cantilever fixture must own a force excitation");
        };
        action.target = ExcitationTarget::WholeModel;
        action.distribution = ExcitationDistribution::IntensityPerVolume;
        let PhysicalSignal::Force(signal) = model.signals[0].clone() else {
            panic!("cantilever fixture must own a force signal");
        };
        model.signals[0] = PhysicalSignal::BodyForceDensity(Signal {
            signal_id: signal.signal_id.clone(),
            shape: signal.shape.clone(),
            frame: signal.frame.clone(),
            representation: SignalRepresentation::Constant(SignalValue::Vector3(
                [0.0, -1.0, 0.0].map(BodyForceDensity::new),
            )),
            precision: signal.precision,
            evidence: signal.evidence.clone(),
            provenance: signal.provenance.clone(),
        });
        model.excitations[0] = ExcitationAction::BodyForce(action);
    }
    let mut mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
    for (index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
        face.provenance.push(PersistentEntityId::new(
            PersistentEntityKind::Face,
            if index == 0 { "root" } else { "tip" }.into(),
            vec!["root".into()],
        ));
        face.provenance.sort();
    }
    let region = runmat_meshing_core::fixtures::fixture_entity_for_selection(
        model.material_assignments[0]
            .selected_region_id()
            .expect("cantilever fixture uses selected material region"),
    )
    .unwrap();
    mesh.topology.volume_elements[0].region_id = region.clone();
    mesh.topology.regions[0].region_id = region;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    (model, mesh)
}

#[test]
fn whole_structural_preparation_conserves_exact_composed_grant() {
    for (order, mode) in [
        (ElementOrder::Tet4, ObservedSolidMode::EmptyConnectors),
        (ElementOrder::Tet4, ObservedSolidMode::SurfaceFoundation),
        (ElementOrder::Tet4, ObservedSolidMode::BoundaryNormalSupport),
        (ElementOrder::Tet4, ObservedSolidMode::WholeModelBodyForce),
        (ElementOrder::Tet10, ObservedSolidMode::EmptyConnectors),
        (ElementOrder::Tet10, ObservedSolidMode::SurfaceFoundation),
        (
            ElementOrder::Tet10,
            ObservedSolidMode::BoundaryNormalSupport,
        ),
        (ElementOrder::Tet10, ObservedSolidMode::WholeModelBodyForce),
    ] {
        let (model, mesh) = observed_solid_fixture(order, mode);
        let quote = structural_linear_preparation_work_upper_bound(&model, Some(&mesh)).unwrap();
        let short = OperationCount::from_operations(quote.operations().checked_sub(1).unwrap());

        // The original fixed-support model supports the elastic Auto fallback. Its forced
        // tiny-ceiling refusal must not charge the shared projection twice.
        for (grant, machine_memory_bytes) in [(quote, None), (short, None), (quote, Some(1))]
            .into_iter()
            .filter(|(_, ceiling)| {
                matches!(mode, ObservedSolidMode::EmptyConnectors) || ceiling.is_none()
            })
        {
            let ledger = Arc::new(UsageLedger::new(quote, grant).unwrap());
            let result = UsageContext::new(Arc::clone(&ledger)).run_sync(|| {
                prepare_structural_linear_program(StructuralLinearProgramPreparation {
                    authored_study_digest: digest("exact-whole-preparation"),
                    frozen_project_digest: None,
                    program_revision: revision(),
                    artifact_access: access(),
                    workload: StructuralLinearWorkload {
                        schema_version: 2,
                        step_id: model.sequence.steps[0].step_id.clone(),
                        model: model.clone(),
                        backend: ComputeBackend::Cpu,
                        preconditioner: SpdPreconditionerKind::Auto,
                        algebra_backend: LinearAlgebraBackendKind::CpuReference,
                        observation_execution: runmat_analysis_fea::results::observation::ObservationExecutionPolicy::serial(),
                        require_solver_mesh_for_solid: true,
                        state_input: None,
                    },
                    mesh: Some(mesh.clone()),
                    state_input: None,
                    chunk_bytes: 4096,
                    inventory_limits: ObjectInventoryLimits::default(),
                    required_capabilities: BTreeSet::new(),
                    accelerator_requests: Vec::new(),
                    local_execution: LocalExecutionPlan::serial(),
                    solve_selection: StaticSolveSelection::WithinCeiling { machine_memory_bytes },
                })
            });
            let snapshot = ledger.snapshot().unwrap();
            assert_eq!(
                snapshot.reserved,
                OperationCount::ZERO,
                "{order:?}, mode={mode:?}"
            );
            if grant == short {
                let error = result.err().unwrap_or_else(|| {
                    panic!("one operation short must refuse: {order:?}, mode={mode:?}")
                });
                assert_eq!(
                    error.accounting_cause(),
                    Some(&LedgerError::MoreGrantRequired {
                        required: quote,
                        granted: short,
                    }),
                    "{order:?}, mode={mode:?}"
                );
                assert!(
                    snapshot.completed.assembly < quote,
                    "{order:?}, mode={mode:?}"
                );
            } else {
                assert_eq!(
                    snapshot.completed.assembly, quote,
                    "{order:?}, mode={mode:?}"
                );
                if machine_memory_bytes.is_some() {
                    assert!(matches!(
                        result,
                        Err(AnalysisBridgeError::Resource(
                            ResourcePlanningError::StaticSolveCeiling { maximum: 1, .. }
                        ))
                    ));
                } else {
                    result.unwrap_or_else(|error| {
                        panic!("exact composed grant must fund {order:?}, mode={mode:?}: {error:?}")
                    });
                }
            }
        }
    }
}

#[test]
fn owned_static_observations_acquire_capacity_before_consuming_their_quote() {
    use runmat_analysis_fea::{
        algebra::NumericalCapacityPhase, constraints::CanonicalStructuralConstraintPlan,
        run_linear_static_with_prepared_constraint_plan,
        run_linear_static_with_prepared_constraint_plan_tet4_owned, LinearStaticSolveOptions,
    };
    use std::sync::Mutex;

    for (order, mode) in [
        (ElementOrder::Tet4, ObservedSolidMode::EmptyConnectors),
        (ElementOrder::Tet4, ObservedSolidMode::SurfaceFoundation),
        (ElementOrder::Tet4, ObservedSolidMode::BoundaryNormalSupport),
        (ElementOrder::Tet4, ObservedSolidMode::WholeModelBodyForce),
        (ElementOrder::Tet10, ObservedSolidMode::EmptyConnectors),
        (ElementOrder::Tet10, ObservedSolidMode::SurfaceFoundation),
        (
            ElementOrder::Tet10,
            ObservedSolidMode::BoundaryNormalSupport,
        ),
        (ElementOrder::Tet10, ObservedSolidMode::WholeModelBodyForce),
    ] {
        let (model, mesh) = observed_solid_fixture(order, mode);
        let constraints =
            CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap();
        let quote = runmat_analysis_fea::results::observation::structural_observation_derivation_work_upper_bound(Some(&mesh))
            .unwrap().checked_mul(2).unwrap();
        let options = LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            ..Default::default()
        };
        let inline = run_linear_static_with_prepared_constraint_plan(
            &model,
            ComputeBackend::Cpu,
            options.clone(),
            &constraints,
            None,
        )
        .unwrap_or_else(|error| panic!("inline {order:?}, mode={mode:?}: {error:?}"));
        let ledger = Arc::new(
            UsageLedger::new(
                OperationCount::from_operations(1_000_000_000),
                OperationCount::ZERO,
            )
            .unwrap(),
        );
        let phases = Arc::new(Mutex::new(Vec::new()));
        let context = UsageContext::new(Arc::clone(&ledger));
        let owned = crate::serial::finish_inline(context.scope(
            run_linear_static_with_prepared_constraint_plan_tet4_owned(
                &model,
                ComputeBackend::Cpu,
                options,
                &constraints,
                None,
                usize::MAX,
                |phase, work| {
                    let ledger = Arc::clone(&ledger);
                    let phases = Arc::clone(&phases);
                    async move {
                        let before = ledger.snapshot().unwrap();
                        let required = before
                            .completed
                            .total()
                            .unwrap()
                            .checked_add(before.reserved)
                            .unwrap()
                            .checked_add(work)
                            .unwrap();
                        ledger.update_grant(required.max(before.granted)).unwrap();
                        phases.lock().unwrap().push((
                            phase,
                            work,
                            before.completed.total().unwrap(),
                        ));
                        Ok::<(), AnalysisBridgeError>(())
                    }
                },
            ),
        ))
        .unwrap_or_else(|error| {
            panic!(
                "owned {order:?}, mode={mode:?}: {error:?}; phases={:?}",
                phases.lock().unwrap()
            )
        });
        assert_eq!(owned.fields, inline.fields, "{order:?}, mode={mode:?}");
        assert_eq!(
            owned.observations, inline.observations,
            "{order:?}, mode={mode:?}"
        );
        let observation_phases = phases
            .lock()
            .unwrap()
            .iter()
            .filter(|(phase, _, _)| *phase == NumericalCapacityPhase::StaticObservations)
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            observation_phases.len(),
            1,
            "{order:?}, mode={mode:?}; phases={:?}",
            phases.lock().unwrap()
        );
        assert_eq!(observation_phases[0].1, quote, "{order:?}, mode={mode:?}");
        let completed = ledger.snapshot().unwrap();
        assert_eq!(completed.reserved, OperationCount::ZERO);
        assert_eq!(
            completed
                .completed
                .total()
                .unwrap()
                .operations()
                .checked_sub(observation_phases[0].2.operations())
                .unwrap(),
            quote.operations(),
        );
        let cause = LedgerError::MoreGrantRequired {
            required: quote,
            granted: OperationCount::ZERO,
        };
        assert!(matches!(
            runmat_analysis_fea::FeaRunError::from(
                runmat_analysis_fea::results::observation::ObservationError::Accounting(cause)
            ),
            runmat_analysis_fea::FeaRunError::Usage(actual) if actual == cause
        ));
    }
}
