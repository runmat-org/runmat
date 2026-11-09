use super::*;
use runmat_analysis_core::*;
use runmat_analysis_fea::execution::{ThermalSolverResultManifest, ThermalWorkload};
use runmat_analysis_fea::results::field::RecoveredFieldSample;
use runmat_analysis_fea::results::observation::{ObservationFieldCatalog, ObservationLimits};
use runmat_analysis_fea::{ComputeBackend, ThermalProcedureControl};
use runmat_execution_artifact::{cache::CacheImport, ArtifactResult};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

struct RecordingStore {
    objects: BTreeMap<Digest, Vec<u8>>,
    reads: RefCell<Vec<Digest>>,
}
impl CacheImport for RecordingStore {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.reads.borrow_mut().push(digest);
        Ok(self.objects.get(&digest).cloned())
    }
}

fn spec(quantity: ObservationQuantity, location: FieldLocation) -> ObservationRequest {
    ObservationRequest::History(ObservationSpec {
        observation_id: ObservationId(format!("{quantity:?}")),
        quantity,
        value_shape: quantity.canonical_value_shape(),
        target: ObservationTarget::WholeModel,
        source_location: location,
        output_location: location,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::EveryAcceptedIncrement,
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        // This test checks partition access, not caller-authored cost ceilings.
        // Derive the admitted cost from the actual resolved history plan.
        cost: ObservationCostPolicy::Automatic,
    })
}

#[test]
fn thermal_observation_fanout_uses_published_sample_and_reads_only_selected_fields() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        // Separate single-quantity requests make unrelated-quantity non-reads observable.
        for (quantity, location) in [
            (ObservationQuantity::Temperature, FieldLocation::Node),
            (
                ObservationQuantity::TemperatureGradient,
                FieldLocation::Element,
            ),
            (ObservationQuantity::HeatFlux, FieldLocation::Element),
            (
                ObservationQuantity::ThermalEnergyStorageRateDensity,
                FieldLocation::Element,
            ),
        ] {
            let mut model = runmat_analysis_fea::fixtures::fixture_model(
                runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
            );
            model.thermal = Some(ThermalDomain {
                reference_temperature: Temperature::new(293.15),
            });
            let mut step = ProcedureStep::thermal_transient("heat");
            // The fixture has exactly three accepted samples; bound its retained history.
            step.accuracy.resources.maximum_samples = 3;
            step.observations = vec![spec(quantity, location)];
            model.sequence =
                AnalysisSequence::from_steps("thermal_observation", vec![step.clone()]);
            let workload = ThermalWorkload {
                initial_fields: Vec::new(),
                model,
                step_id: step.step_id.clone(),
                backend: ComputeBackend::Cpu,
                procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
                residual_target: 1e-4,
            };
            let mesh = canonical_tetrahedron_solver_mesh(order);
            let access = ArtifactAccess {
                authorization_scope: "thermal-observation".into(),
                encryption_context: digest("key"),
            };
            let limits = ObjectInventoryLimits::default();
            let task = prepare_thermal_task(ThermalTaskPreparation {
                maximum_resident_bytes: None,
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                field_source: None,
                authored_study_digest: digest("thermal-observation"),
                frozen_project_digest: None,
                program_revision: revision(),
                artifact_access: access.clone(),
                workload: workload.clone(),
                mesh: mesh.clone(),
                output_policy: AnalysisOutputPolicy::new(1024, limits).unwrap(),
                required_capabilities: BTreeSet::new(),
            })
            .unwrap();
            let result = workload.execute(&mesh).unwrap();
            let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
            let objects = prepare_thermal_solver_result_objects(
                &result,
                1024,
                ObjectNamespace::ResultValue,
                limits,
            )
            .unwrap();
            let output = AnalysisOutputRoot {
                kind: AnalysisNumericalObjectKind::SolverResult,
                logical_identity: objects.manifest.logical_identity,
                payload_digest: objects.manifest.payload_digest,
                root: objects
                    .root_reference(&access, ValueRefKind::ResultObject)
                    .unwrap(),
            };
            let store = RecordingStore {
                objects: objects
                    .objects
                    .iter()
                    .map(|object| (object.descriptor.digest, object.bytes.clone()))
                    .collect(),
                reads: RefCell::new(Vec::new()),
            };
            let prepare = |sample_ordinal| ObservationPreparation {
                source_host: task.host.clone(),
                source: ObservationPublishedSource::Thermal {
                    manifest: manifest.clone(),
                    output: output.clone(),
                    access: access.clone(),
                    sample_ordinal,
                },
                source_workload: ObservationWorkloadSource::Thermal(
                    task.inputs
                        .iter()
                        .find(|input| {
                            input.manifest.kind == AnalysisNumericalObjectKind::ThermalWorkload
                        })
                        .unwrap()
                        .clone(),
                ),
                mesh: Some(
                    task.inputs
                        .iter()
                        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::Mesh)
                        .unwrap()
                        .clone(),
                ),
                accelerator_requests: Vec::new(),
                sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap(),
                step: step.clone(),
                limits: ObservationLimits::default(),
                program_revision: revision(),
                chunk_bytes: 1024,
                inventory_limits: limits,
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            };
            let sample = RecoveredFieldSample::TimeSeconds(manifest.time_points_s[1]);
            let fanout = prepare_observation_fanout(&store, prepare(1), sample, None).unwrap();
            assert_eq!(fanout.partitions.len(), 1);
            let partition = &fanout.partitions[0];
            assert_eq!(partition.source_fields.sample, sample);
            assert_eq!(partition.source_fields.definition.quantity(), quantity);
            let catalog_input = partition
                .inputs
                .iter()
                .find(|input| {
                    input.manifest.kind == AnalysisNumericalObjectKind::ObservationFieldCatalog
                })
                .unwrap();
            let catalog =
                ObservationFieldCatalog::from_canonical_bytes(&catalog_input.payload()).unwrap();
            catalog
                .validate_thermal_selection(&fanout.source_request.selection, &workload, &mesh)
                .unwrap();
            assert!(partition
                .inputs
                .iter()
                .all(|input| input.manifest.kind != AnalysisNumericalObjectKind::SolverResult));
            let selected = &fanout.source_request.selection.fields()[0];
            let allowed = selected
                .partitions
                .iter()
                .map(|descriptor| descriptor.partition.digest())
                .collect::<BTreeSet<_>>();
            let reads = store.reads.borrow();
            for descriptor in &objects.manifest.partitions {
                if allowed.contains(&descriptor.logical_identity) {
                    continue;
                }
                let first = descriptor.first_chunk_ordinal as usize;
                for chunk in
                    &objects.manifest.chunks[first..first + descriptor.chunk_count as usize]
                {
                    assert!(
                        !reads.contains(&chunk.digest),
                        "unselected field, sample, or metadata payload read"
                    );
                }
            }
            drop(reads);
            assert!(prepare_observation_fanout(&store, prepare(0), sample, None).is_err());
        }
    }
}
