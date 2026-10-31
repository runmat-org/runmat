//! Authored CAD/native governed lifecycle witness, not spatial/release qualification.
#[path = "fluid_authored/replay.rs"]
mod replay;
use runmat_execution::value::PublishedObjectSource;
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport, FilesystemObjectStore},
    object::ObjectInventoryLimits,
    ArtifactAccess, LogicalObject, ObjectNamespace,
};
use runmat_execution_runner_native::{
    execute_prepared_fea_study_locally, fluid_study_completion_from_source,
    prepare_checked_fea_study, CompletedFeaStudyResult, LocalFeaExecutionOptions,
    StudyRootPlanning,
};
use runmat_runtime::analysis::FeaRootSelector;

const SOURCE: &str = r#"
geometry part {
 file = "./box.brep"; units = m
 region bulk : cell { select result; expect exactly_one }
 region outlet : face { select persistent_id "FACE_2_ID"; expect exactly_one }
 region lid : face { select persistent_id "FACE_1_ID"; expect exactly_one }
 region wall3 : face { select persistent_id "FACE_3_ID"; expect exactly_one }
 region wall4 : face { select persistent_id "FACE_4_ID"; expect exactly_one }
 region wall5 : face { select persistent_id "FACE_5_ID"; expect exactly_one }
 region wall6 : face { select persistent_id "FACE_6_ID"; expect exactly_one }
}
mesh volume on geometry.part { tetrahedra { order = linear; target_size = 1.5 m; maximum_deviation = 0.02 m } }
material liquid { mass_density = 1 kg/m^3; fluid { law = incompressible_newtonian; dynamic_viscosity = 1 Pa*s } }
model flow {
 assign material.liquid to geometry.part.region.bulk
 constraint lid { fluid.BOUNDARY { target = geometry.part.region.lid; BOUNDARY_VALUE } }
 constraint wall3 { fluid.BOUNDARY { target = geometry.part.region.wall3; BOUNDARY_VALUE } }
 constraint wall4 { fluid.BOUNDARY { target = geometry.part.region.wall4; BOUNDARY_VALUE } }
 constraint wall5 { fluid.BOUNDARY { target = geometry.part.region.wall5; BOUNDARY_VALUE } }
 constraint wall6 { fluid.BOUNDARY { target = geometry.part.region.wall6; BOUNDARY_VALUE } }
 load outlet { fluid.outlet_pressure { target = geometry.part.region.outlet; value = 0 Pa } }
}
study steady {
 geometry = geometry.part; mesh = mesh.volume; model = model.flow
 analysis solve { fluid static {}; accuracy { algebraic_relative_tolerance = 1e-8 } }
 output velocity = analysis.solve.field.cell.fluid_velocity
 output pressure = analysis.solve.field.cell.fluid_pressure
 output flux = analysis.solve.field.face.volume_flow_rate
}
"#;

pub(super) async fn run() {
    let limits = ObjectInventoryLimits::default();
    let mut previous_receipt = None;
    for moving in [true, false] {
        eprintln!(
            "authored fluid lifecycle: rigid_translation={moving}, exact CAD/native execution"
        );
        let directory = tempfile::tempdir().unwrap();
        // Keep native worker/object-store artifacts inside this bounded test fixture,
        // never in the user's cache-directory default.
        std::env::set_var(
            "RUNMAT_EXECUTION_STATE_DIR",
            directory.path().join("native-state"),
        );
        std::env::set_var(
            "RUNMAT_FEA_ARTIFACT_ROOT",
            directory.path().join("runtime-artifacts"),
        );
        std::env::set_var(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            directory.path().join("study-artifacts"),
        );
        let path = directory.path().join("fluid.fea");
        // Untouched singleton CAD: aliases bind authentic imported topology.
        std::fs::write(
            directory.path().join("box.brep"),
            include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep"),
        )
        .unwrap();
        let mut source = SOURCE
            .replace(
                "BOUNDARY_VALUE",
                if moving {
                    "value = [0.001 m/s, 0 m/s, 0 m/s]"
                } else {
                    ""
                },
            )
            .replace("BOUNDARY", if moving { "velocity" } else { "no_slip_wall" });
        let imported = runmat_geometry_io::import_exact_cad(
            "box.brep",
            include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep"),
            runmat_geometry_io::GeometryFormat::Brep,
            &runmat_geometry_io::ExactCadImportOptions {
                source_units: Some(runmat_geometry_core::LengthUnit::Meter),
                ..Default::default()
            },
            &runmat_geometry_io::GeometryImportContext::new(),
        )
        .unwrap();
        for (index, face) in &imported.source_face_ids {
            source = source.replace(
                &format!("FACE_{}_ID", index + 1),
                &face.analysis_selection_id().unwrap(),
            );
        }
        std::fs::write(&path, source).unwrap();
        let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
        let checked = session.load_and_check_fea_source_path(&path).await.unwrap();
        assert_eq!(
            checked.checked().check_result().status,
            runmat_runtime::analysis::FeaCheckStatus::Valid,
            "{:?}",
            checked.checked().check_result().diagnostics
        );
        let options = LocalFeaExecutionOptions {
            root: Some(FeaRootSelector::Study {
                id: "steady".into(),
            }),
            workers: Some(1),
            ..Default::default()
        };
        let runmat_runtime::analysis::ResolvedFeaRoot::Study(resolved) =
            checked.resolve_root(options.root.as_ref()).unwrap()
        else {
            panic!("expected authored study")
        };
        let mut prepared = prepare_checked_fea_study(
            &checked,
            resolved,
            &[],
            "authored-fluid-policy-witness",
            &options,
        )
        .await
        .unwrap();
        let StudyRootPlanning::Fluid(planning) = &mut prepared.root.planning else {
            panic!("expected original authored fluid root")
        };
        planning.numerical.maximum_outer_iterations = 400;
        planning.numerical.linear_relative_tolerance = 1e-11;
        let case = execute_prepared_fea_study_locally(prepared, &options)
            .await
            .unwrap();
        let CompletedFeaStudyResult::Fluid(result) = case.result else {
            panic!("fluid domain substituted")
        };
        eprintln!("authored fluid lifecycle: original governed completion retained");
        assert_eq!(
            result.completion.workload().policy.maximum_outer_iterations,
            400
        );
        assert!(result.completion.result.original().outer_iterations <= 400);
        eprintln!(
            "authored fluid accepted iterations: outer={}, linear={}, original requested outer=400",
            result.completion.result.original().outer_iterations,
            result.completion.result.original().linear_iterations
        );
        assert!(
            result
                .completion
                .workload()
                .policy
                .linear_relative_tolerance
                <= 1e-11
        );
        assert_eq!(
            result
                .completion
                .workload()
                .policy
                .physical_relative_tolerance,
            1e-8
        );
        let original = result.completion.result.original();
        let worst_cell_momentum = original
            .residual
            .cell_momentum_relative
            .iter()
            .copied()
            .fold(0.0, f64::max);
        eprintln!("authored fluid original force balance: momentum={}, global={}, worst_cell={}, continuity={}, flux={}",
            original.residual.momentum_relative, original.residual.global_momentum_relative,
            worst_cell_momentum, original.residual.continuity_relative,
            original.residual.flux_constitutive_relative);
        assert!(original.residual.global_momentum_relative <= 1e-8);
        assert!(worst_cell_momentum <= 1e-8);
        assert_eq!(
            original.velocity_m_per_s.canonical_values().len(),
            result.mesh.topology.volume_elements.len()
        );
        assert!(!original
            .face_volume_flux_m3_per_s
            .canonical_values()
            .is_empty());
        assert_eq!(result.completion.fields.fields.len(), 3);
        assert_eq!(
            result.completion.fields.entity_counts.volume_elements,
            original.velocity_m_per_s.canonical_values().len()
        );
        assert_eq!(
            result.completion.fields.entity_counts.oriented_faces,
            original.face_volume_flux_m3_per_s.canonical_values().len()
        );
        let domains = [
            runmat_analysis_fea::results::field::RecoveredFieldEntityDomain::VolumeElements,
            runmat_analysis_fea::results::field::RecoveredFieldEntityDomain::VolumeElements,
            runmat_analysis_fea::results::field::RecoveredFieldEntityDomain::OrientedFaces,
        ];
        for (entry, domain) in result.completion.fields.fields.iter().zip(domains) {
            assert_eq!(entry.definition.entity_domain(), domain);
            let expected = if domain
                == runmat_analysis_fea::results::field::RecoveredFieldEntityDomain::OrientedFaces
            {
                original.face_volume_flux_m3_per_s.canonical_values().len()
            } else {
                original.velocity_m_per_s.canonical_values().len()
            };
            assert_eq!(
                entry
                    .partitions
                    .iter()
                    .map(|p| p.entity_count)
                    .sum::<usize>(),
                expected
            );
        }
        // The current authoring surface declares algebraic accuracy only. A stronger,
        // explicit mandatory spatial requirement must remain missing/Indeterminate;
        // real algebraic/conservation evidence cannot be relabelled as mesh accuracy.
        let mut spatial_plan = result.completion.workload().model.sequence.steps[0]
            .accuracy
            .clone();
        assert_eq!(
            result.completion.accuracy.evaluation().report().status,
            runmat_analysis_fea::results::accuracy::AccuracyStatus::Satisfied
        );
        assert!(spatial_plan
            .criteria
            .iter()
            .all(|criterion| criterion.tolerance
                == runmat_analysis_core::AccuracyTolerance::Relative(1e-8)));
        spatial_plan
            .criteria
            .push(runmat_analysis_core::AccuracyCriterion {
                criterion_id: "mandatory_spatial".into(),
                target: runmat_analysis_core::AccuracyTarget::NumericalResult,
                contribution: runmat_analysis_core::ErrorContributionKind::SpatialDiscretization,
                tolerance: runmat_analysis_core::AccuracyTolerance::Relative(0.01),
                evidence: runmat_analysis_core::RequiredAccuracyEvidence::ConvergenceStudy,
                mandatory: true,
            });
        let spatial = runmat_analysis_fea::results::accuracy::evaluate_accuracy(
            &spatial_plan,
            result.completion.accuracy.evidence(),
        )
        .unwrap();
        assert_eq!(
            spatial.report().status,
            runmat_analysis_fea::results::accuracy::AccuracyStatus::Indeterminate
        );
        assert_eq!(
            runmat_analysis_fea::results::accuracy::publication_gate(&spatial_plan, &spatial)
                .unwrap()
                .decision,
            runmat_analysis_fea::results::accuracy::PublicationDecision::Block
        );
        assert!(!result.completion.states.is_empty());
        assert!(result.completion.observation_catalog.is_some());
        for observation in &result.completion.observation_results {
            let (count, width, location) = match observation.quantity {
                runmat_analysis_core::ObservationQuantity::FluidVelocity => (
                    original.velocity_m_per_s.canonical_values().len(),
                    3,
                    runmat_analysis_core::FieldLocation::Cell,
                ),
                runmat_analysis_core::ObservationQuantity::FluidPressure => (
                    original.pressure_pa.canonical_values().len(),
                    1,
                    runmat_analysis_core::FieldLocation::Cell,
                ),
                runmat_analysis_core::ObservationQuantity::VolumeFlowRate => (
                    original.face_volume_flux_m3_per_s.canonical_values().len(),
                    1,
                    runmat_analysis_core::FieldLocation::Face,
                ),
                _ => panic!("original fluid public quantity substituted"),
            };
            assert_eq!(observation.location, location);
            assert_eq!(observation.entities.len(), count);
            assert_eq!(observation.field.shape[0], count);
            assert_eq!(
                observation.field.shape.iter().product::<usize>(),
                count * width
            );
        }
        for residual in [
            original.residual.momentum_relative,
            original.residual.continuity_relative,
            original.residual.flux_constitutive_relative,
        ] {
            assert!(
                residual
                    <= result
                        .completion
                        .workload()
                        .policy
                        .physical_relative_tolerance
            );
        }
        let max_velocity = original
            .velocity_m_per_s
            .canonical_values()
            .iter()
            .flatten()
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        if moving {
            let projection =
                runmat_analysis_fea::topology::AnalysisProjection::derive(&result.mesh).unwrap();
            let topology = runmat_analysis_fea::topology::FiniteVolumeTopology::derive(
                &projection,
                result.completion.workload().geometry_policy,
            )
            .unwrap();
            assert_eq!(
                topology.cells.len(),
                original.velocity_m_per_s.canonical_values().len()
            );
            assert_eq!(
                topology.faces.len(),
                original.face_volume_flux_m3_per_s.canonical_values().len()
            );
            let tolerance = result
                .completion
                .workload()
                .policy
                .physical_relative_tolerance;
            // Independent rigid-translation patch: U constant, p=0 and q=U dot S
            // at every original owner-oriented face, not only an exterior rendering.
            for u in original.velocity_m_per_s.canonical_values() {
                for (actual, expected) in u.iter().zip([0.001_f64, 0.0, 0.0]) {
                    assert!((*actual - expected).abs() <= tolerance * 0.001);
                }
            }
            assert!(original
                .pressure_pa
                .canonical_values()
                .iter()
                .all(|p| p.abs() <= tolerance * 0.001));
            let flux_scale = topology
                .faces
                .iter()
                .map(|f| 0.001 * f.area_m2.canonical_value())
                .fold(0.0, f64::max);
            let mut outlet_faces = 0;
            for (face, q) in topology
                .faces
                .iter()
                .zip(original.face_volume_flux_m3_per_s.canonical_values())
            {
                let expected = 0.001 * face.owner_area_vector_m2[0];
                assert!((*q - expected).abs() <= tolerance * flux_scale);
                if face.neighbor.is_none() && (face.centroid_m[0] - 1.0).abs() < 1e-12 {
                    assert!(face.owner_area_vector_m2[0] > 0.0);
                    assert!(*q > 0.0);
                    outlet_faces += 1;
                }
            }
            assert!(outlet_faces > 0);
            assert!(max_velocity > 0.0);
        } else {
            assert_eq!(max_velocity, 0.0);
            assert!(original
                .pressure_pa
                .canonical_values()
                .iter()
                .all(|p| *p == 0.0));
        }
        let expected = result.completion.clone();
        let mesh = result.mesh.clone();
        let meshing_evidence = result.meshing_evidence.clone();
        let root = expected.root.clone();
        let access = ArtifactAccess {
            authorization_scope: root.authorization_scope.clone(),
            encryption_context: root.encryption_context,
        };
        let retained_path = directory.path().join("retained");
        let mut retained =
            FilesystemObjectStore::open(&retained_path, limits.max_object_bytes).unwrap();
        for reference in &expected.catalog.objects {
            let object = LogicalObject::new(
                ObjectNamespace::ResultValue,
                "fluid-cold",
                "application/octet-stream",
                result.source.read_verified(reference).unwrap(),
            )
            .unwrap();
            assert_eq!(object.descriptor.digest, reference.logical_digest);
            retained.write_verified(&object).unwrap();
        }
        drop(retained);
        drop(result);
        drop(case.publication);
        drop(checked);
        drop(session);
        let cold = FilesystemObjectStore::open(&retained_path, limits.max_object_bytes).unwrap();
        assert!(runmat_analysis_execution::import_fluid_study_root(
            &cold,
            &root,
            runmat_execution::Digest::sha256(b"foreign-root"),
            &access,
            limits
        )
        .is_err());
        let missing = expected
            .catalog
            .objects
            .iter()
            .find(|o| o.logical_digest != root.logical_digest)
            .unwrap();
        assert!(runmat_analysis_execution::import_fluid_study_root(
            &MissingObject {
                source: &cold,
                missing: missing.logical_digest
            },
            &root,
            root.logical_digest,
            &access,
            limits
        )
        .is_err());
        let reopened = runmat_analysis_execution::import_fluid_study_root(
            &cold,
            &root,
            root.logical_digest,
            &access,
            limits,
        )
        .unwrap();
        // Exact retained codec regression: the transport builder excludes the
        // separately reserved canonical partition/materialization workspace.
        let recovery_plan = reopened.field_recovery_resource_plan();
        let runmat_analysis_fea::planning::resource::FeaResourceDemand::FluidFieldRecovery(
            recovery_demand,
        ) = &recovery_plan.input.demand
        else {
            panic!("original field phase changed")
        };
        let mut field_limits = limits;
        field_limits.max_total_bytes = field_limits
            .max_total_bytes
            .min(recovery_demand.output.transport_closure_bytes_upper_bound.0);
        let retained_fields = runmat_analysis_execution::import_analysis_objects(
            &cold,
            &reopened.fields.output.root,
            runmat_analysis_fea::execution::AnalysisNumericalObjectKind::Field,
            reopened.fields.output.logical_identity,
            reopened.fields.output.payload_digest,
            &access,
            field_limits,
        )
        .unwrap();
        let field_facts = runmat_analysis_execution::prepared_input_resource_facts(
            std::slice::from_ref(&retained_fields),
        )
        .unwrap();
        let codec_and_transport = field_facts
            .decoded_resident_bytes_admission_budget
            .0
            .checked_add(field_facts.transport_closure_bytes.0)
            .unwrap();
        let construction = recovery_demand
            .construction_workspace_bytes
            .0
            .checked_add(
                recovery_demand
                    .output
                    .construction_resident_bytes_admission_budget
                    .0,
            )
            .unwrap();
        assert!(
            codec_and_transport
                > recovery_demand
                    .output
                    .construction_resident_bytes_admission_budget
                    .0
        );
        assert!(codec_and_transport <= construction);
        let coexisting = recovery_demand
            .source_validation_workspace_bytes
            .0
            .checked_add(
                recovery_demand
                    .inputs
                    .decoded_resident_bytes_admission_budget
                    .0,
            )
            .and_then(|n| n.checked_add(recovery_demand.inputs.transport_closure_bytes.0))
            .and_then(|n| n.checked_add(codec_and_transport))
            .unwrap();
        assert!(coexisting <= recovery_plan.request.memory_bytes);
        eprintln!("authored fluid retained codec regression: decoded={}B, partition_materialization_workspace={}B, codec_and_transport={}B, transport_builder={}B, combined={}B, phase={}B",
            field_facts.decoded_resident_bytes_admission_budget.0, recovery_demand.construction_workspace_bytes.0,
            codec_and_transport, recovery_demand.output.construction_resident_bytes_admission_budget.0,
            construction, recovery_plan.request.memory_bytes);
        drop(retained_fields);
        for closure in reopened.result_closures().iter().filter(|closure| {
            matches!(
                closure.phase,
                runmat_analysis_execution::FluidStudyResultPhase::State { .. }
            )
        }) {
            eprintln!(
                "authored fluid lifecycle: cells={}, all_faces={}, admitted_state_heap={}B",
                expected
                    .result
                    .original()
                    .velocity_m_per_s
                    .canonical_values()
                    .len(),
                expected
                    .result
                    .original()
                    .face_volume_flux_m3_per_s
                    .canonical_values()
                    .len(),
                closure
                    .execution
                    .winning_accounting()
                    .resources
                    .allocation
                    .memory_bytes
            );
        }
        replay::check(&reopened, &cold);
        let (actual, source) = fluid_study_completion_from_source(reopened, cold).unwrap();
        assert_eq!(actual, expected);
        for reference in &actual.catalog.objects {
            source.read_verified(reference).unwrap();
        }
        let options = runmat_runtime::analysis::AnalysisCfdRunOptions {
            precision_mode: runmat_runtime::analysis::PrecisionMode::Fp64,
            numerical: actual.workload().policy,
            geometry: actual.workload().geometry_policy,
        };
        let publish =
            |candidate: &runmat_runtime::analysis::FluidStudyCompletion,
             options: &runmat_runtime::analysis::AnalysisCfdRunOptions| {
                runmat_runtime::analysis::publish_fluid_study_result(
                    candidate,
                    &source,
                    &mesh,
                    &meshing_evidence,
                    options,
                    candidate.compiled().definition().unwrap(),
                    runmat_runtime::operations::OperationContext::new(None, None),
                    runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!(
                        "cold-fluid:{}",
                        actual.root.logical_digest
                    ))
                    .unwrap(),
                )
            };
        publish(&actual, &options).unwrap();
        eprintln!("authored fluid lifecycle: cold original runtime publication verified");
        let trends = runmat_runtime::analysis::analysis_trends_op(
            runmat_runtime::analysis::AnalysisTrendsQuery { window_size: 2 },
            runmat_runtime::operations::OperationContext::new(None, None),
        )
        .unwrap();
        let fluid = trends
            .data
            .summaries
            .iter()
            .find(|summary| summary.identity.family() == runmat_analysis_core::PhysicsFamily::Fluid)
            .expect("authentic authored fluid publication contributes a trend");
        assert!(fluid.sample_count > 0);
        assert!(fluid.identity.realization().components.iter().all(|component|
            component.recovery == runmat_analysis_core::RecoveryImplementation::FluidOriginalCellVelocityPressureOwnerFaceFlux));
        let mut wrong = actual.clone();
        wrong.states.clear();
        assert!(publish(&wrong, &options).is_err());
        let mut wrong = actual.clone();
        wrong.fields.fields.pop();
        assert!(publish(&wrong, &options).is_err());
        let mut wrong_options = options.clone();
        wrong_options.numerical.physical_relative_tolerance *= 2.0;
        assert!(publish(&actual, &wrong_options).is_err());
        let mut wrong_options = options.clone();
        wrong_options.numerical.linear_relative_tolerance *= 2.0;
        assert!(publish(&actual, &wrong_options).is_err());
        let mut wrong_options = options.clone();
        wrong_options.precision_mode = runmat_runtime::analysis::PrecisionMode::Fp32;
        assert!(publish(&actual, &wrong_options).is_err());
        if let Some(receipt) = previous_receipt.take() {
            let mut wrong = actual.clone();
            wrong.numerical_receipt = receipt;
            assert!(publish(&wrong, &options).is_err());
        }
        previous_receipt = Some(actual.numerical_receipt.clone());
        // Live, replayed, accuracy and published validations of the catalog share one
        // (catalog, step, mesh) validation; a repeat is accepted, a foreign step is not.
        let catalog = actual
            .observation_catalog
            .as_ref()
            .expect("authored catalog");
        let step = &actual.workload().model.sequence.steps[0];
        let context = runmat_analysis_fea::execution::ObservationHistorySourceContext::VolumeMesh {
            mesh: &mesh,
        };
        catalog.validate_for(step, context).unwrap();
        catalog.validate_for(step, context).unwrap();
        let mut foreign = step.clone();
        foreign.step_id = runmat_analysis_core::ProcedureStepId("foreign".into());
        assert!(catalog.validate_for(&foreign, context).is_err());
    }
}

struct MissingObject<'a> {
    source: &'a FilesystemObjectStore,
    missing: runmat_execution::Digest,
}
impl CacheImport for MissingObject<'_> {
    fn read_verified(
        &self,
        digest: runmat_execution::Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        if digest == self.missing {
            Ok(None)
        } else {
            self.source.read_verified(digest)
        }
    }
}
