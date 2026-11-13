use super::*;
use runmat_analysis_core::{Temperature, ThermalDomain};
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, SemanticStatePublicationRequest, SemanticStatePublicationSource,
    ThermalSolverResultManifest,
};
use runmat_analysis_fea::{ComputeBackend, ThermalProcedureControl, ThermalSolveOptions};

#[test]
fn thermal_terminal_publication_reads_only_selected_validated_closure() {
    for order in [
        runmat_meshing_core::ElementOrder::Tet4,
        runmat_meshing_core::ElementOrder::Tet10,
    ] {
        let mut model = runmat_analysis_fea::fixtures::fixture_model(
            runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
        );
        model.thermal = Some(ThermalDomain {
            reference_temperature: Temperature::new(293.15),
        });
        model.sequence = AnalysisSequence::from_steps(
            "thermal_partial",
            vec![ProcedureStep::thermal_transient("heat")],
        );
        let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
        let result = runmat_analysis_fea::run_thermal_with_options(
            &model,
            ComputeBackend::Cpu,
            ThermalSolveOptions {
                procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 4 },
                solver_mesh: Some(mesh.clone()),
                ..ThermalSolveOptions::default()
            },
        )
        .unwrap();
        let expected = CanonicalSemanticState::from_thermal_result(&model, &mesh, &result).unwrap();
        let encoding = ThermalSolverResultManifest::from_result(&result)
            .unwrap()
            .encoding_profile()
            .unwrap();
        assert_eq!(encoding.contract_count, 1 + 4 * 4);
        for limits in [
            ObjectInventoryLimits {
                max_total_bytes: encoding.encoded_bytes_upper_bound - 1,
                ..ObjectInventoryLimits::default()
            },
            ObjectInventoryLimits {
                max_objects: encoding.contract_count as usize,
                ..ObjectInventoryLimits::default()
            },
        ] {
            let error = prepare_thermal_solver_result_objects(
                &result,
                1024,
                ObjectNamespace::ResultValue,
                limits,
            )
            .unwrap_err();
            assert!(error.to_string().contains("canonical payload projection"));
        }
        let request = SemanticStatePublicationRequest::new(
            SemanticStatePublicationSource::ThermalPrimaryField,
            result.producer,
            model.sequence.steps[0].state_outputs[0].clone(),
            result.topology_digest,
            result.source_result_identity().unwrap(),
        )
        .unwrap();
        for chunk_bytes in [1024, 8192] {
            let transport = ThermalSolverResultManifest::from_result(&result)
                .unwrap()
                .transport_profile(chunk_bytes)
                .unwrap();
            let admitted_limits = ObjectInventoryLimits {
                max_objects: transport.chunk_count_upper_bound as usize + 1,
                max_total_bytes: transport.total_bytes_upper_bound,
                max_object_bytes: transport
                    .root
                    .encoded_bytes_upper_bound
                    .max(transport.largest_chunk_bytes_upper_bound),
            };
            for limits in [
                ObjectInventoryLimits {
                    max_objects: admitted_limits.max_objects - 1,
                    ..admitted_limits
                },
                ObjectInventoryLimits {
                    max_total_bytes: admitted_limits.max_total_bytes - 1,
                    ..admitted_limits
                },
                ObjectInventoryLimits {
                    max_object_bytes: admitted_limits.max_object_bytes - 1,
                    ..admitted_limits
                },
            ] {
                let error = prepare_thermal_solver_result_objects(
                    &result,
                    chunk_bytes,
                    ObjectNamespace::ResultValue,
                    limits,
                )
                .unwrap_err();
                assert!(error.to_string().contains("canonical payload projection"));
            }
            let prepared = prepare_thermal_solver_result_objects(
                &result,
                chunk_bytes,
                ObjectNamespace::ResultValue,
                admitted_limits,
            )
            .unwrap();
            assert!(prepared.objects.len() as u64 <= transport.chunk_count_upper_bound + 1);
            assert!(
                prepared
                    .objects
                    .iter()
                    .map(|o| o.bytes.len() as u64)
                    .sum::<u64>()
                    <= transport.total_bytes_upper_bound
            );
            assert_eq!(prepared, collected_reference(&result, chunk_bytes));
            let workload = runmat_analysis_fea::execution::ThermalWorkload {
                initial_fields: Vec::new(),
                model: model.clone(),
                step_id: model.sequence.steps[0].step_id.clone(),
                backend: ComputeBackend::Cpu,
                procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 4 },
                residual_target: result.residual_target,
            };
            assert_eq!(
                thermal_receipt_from_prepared_result(&workload, &prepared).unwrap(),
                runmat_analysis_fea::execution::ThermalNumericalReceiptArtifact::new(
                    &workload, &result
                )
                .unwrap()
            );
            let mut wrong_workload = workload.clone();
            wrong_workload.residual_target *= 2.0;
            assert!(thermal_receipt_from_prepared_result(&wrong_workload, &prepared).is_err());
            assert_eq!(
                prepared.manifest.partitions.len() as u64,
                encoding.contract_count
            );
            let actual_payload_bytes = prepared
                .manifest
                .partitions
                .iter()
                .map(|partition| partition.encoded_length)
                .sum::<u64>();
            assert!(actual_payload_bytes <= encoding.encoded_bytes_upper_bound);
            for partition in &prepared.manifest.partitions {
                let bytes = prepared
                    .partition_payload(partition.logical_identity)
                    .unwrap();
                let actual = runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(&bytes).unwrap();
                assert!(actual <= encoding.largest_contract_materialization_bytes);
            }
            assert_eq!(
                decode_prepared_thermal_state(&request, &prepared).unwrap(),
                expected
            );
            let mut incomplete = prepared.clone();
            incomplete.manifest.partitions.pop();
            assert!(decode_prepared_thermal_state(&request, &incomplete).is_err());
            assert!(thermal_receipt_from_prepared_result(&workload, &incomplete).is_err());
            let mut header_only = prepared.clone();
            header_only
                .objects
                .truncate(1 + prepared.manifest.partitions[0].chunk_count as usize);
            // Numerical evidence needs the header, not field bytes. This is deliberately not
            // a proof that a complete field closure is present or valid for publication.
            assert_eq!(
                thermal_receipt_from_prepared_result(&workload, &header_only).unwrap(),
                thermal_receipt_from_prepared_result(&workload, &prepared).unwrap()
            );
            header_only.objects[1].bytes[0] ^= 1;
            assert!(thermal_receipt_from_prepared_result(&workload, &header_only).is_err());
            let manifest = ThermalSolverResultManifest::from_canonical_bytes(
                &prepared
                    .partition_payload(request.source_result.digest())
                    .unwrap(),
            )
            .unwrap();
            let selected = std::iter::once(manifest.identity().digest())
                .chain(
                    manifest
                        .terminal_temperature()
                        .unwrap()
                        .partitions
                        .iter()
                        .map(|p| p.partition.digest()),
                )
                .collect::<BTreeSet<_>>();
            let root = prepared
                .root_reference(&access(), ValueRefKind::ResultObject)
                .unwrap();
            let mut allowed = BTreeSet::from([prepared.root.digest]);
            let mut header_only = allowed.clone();
            let mut selected_bytes = root.encoded_length;
            let mut selected_objects = 1;
            for part in &prepared.manifest.partitions {
                if selected.contains(&part.logical_identity) {
                    selected_bytes += part.encoded_length;
                    selected_objects += part.chunk_count as usize;
                    let start = part.first_chunk_ordinal as usize;
                    for chunk in &prepared.manifest.chunks[start..start + part.chunk_count as usize]
                    {
                        allowed.insert(chunk.digest);
                        if part.logical_identity == manifest.identity().digest() {
                            header_only.insert(chunk.digest);
                        }
                    }
                }
            }
            let mut store = RecordingObjects {
                objects: prepared
                    .objects
                    .iter()
                    .filter(|o| allowed.contains(&o.descriptor.digest))
                    .map(|o| (o.descriptor.digest, o.bytes.clone()))
                    .collect(),
                reads: RefCell::new(Vec::new()),
            };
            assert!(store.objects.len() < prepared.objects.len());
            let import = |store: &RecordingObjects,
                          publication: &SemanticStatePublicationRequest,
                          limits| {
                import_thermal_state(
                    store,
                    ThermalStateImport {
                        root: &root,
                        publication,
                        expected_payload_digest: prepared.manifest.payload_digest,
                        access: &access(),
                        limits,
                    },
                )
            };
            assert_eq!(
                import(&store, &request, ObjectInventoryLimits::default()).unwrap(),
                expected
            );
            assert!(store.reads.borrow().iter().all(|id| allowed.contains(id)));
            assert!(store
                .reads
                .borrow()
                .iter()
                .any(|id| !header_only.contains(id)));

            // Valid request syntax with a wrong topology must fail before numerical reads.
            let mut wrong = request.clone();
            wrong.topology_digest = digest("wrong topology");
            store.reads.borrow_mut().clear();
            assert!(import(&store, &wrong, ObjectInventoryLimits::default()).is_err());
            assert!(store
                .reads
                .borrow()
                .iter()
                .all(|id| header_only.contains(id)));

            // Each selected partition can fit individually while the selected union cannot.
            store.reads.borrow_mut().clear();
            let error = import(
                &store,
                &request,
                ObjectInventoryLimits {
                    max_total_bytes: selected_bytes - 1,
                    ..ObjectInventoryLimits::default()
                },
            )
            .unwrap_err();
            assert!(error.to_string().contains("selected closure"), "{error}");
            assert!(store
                .reads
                .borrow()
                .iter()
                .all(|id| header_only.contains(id)));

            store.reads.borrow_mut().clear();
            let error = import(
                &store,
                &request,
                ObjectInventoryLimits {
                    max_objects: selected_objects - 1,
                    ..ObjectInventoryLimits::default()
                },
            )
            .unwrap_err();
            assert!(error.to_string().contains("selected closure"), "{error}");
            assert!(store
                .reads
                .borrow()
                .iter()
                .all(|id| header_only.contains(id)));

            let terminal = *allowed.difference(&header_only).next().unwrap();
            store.objects.get_mut(&terminal).unwrap()[0] ^= 1;
            assert!(import(&store, &request, ObjectInventoryLimits::default()).is_err());
        }
    }
}

// Retained-input reference exercises the bulk bridge boundary for byte-for-byte closure parity.
fn collected_reference(
    result: &runmat_analysis_fea::FeaThermalRunResult,
    chunk_bytes: usize,
) -> PreparedAnalysisObjects {
    use runmat_analysis_fea::results::field::{canonical_field_partitions, RecoveredField};
    let manifest = ThermalSolverResultManifest::from_result(result).unwrap();
    let mut payloads = vec![(
        manifest.identity().digest(),
        manifest.canonical_bytes().unwrap(),
    )];
    let histories = [
        &result.temperature_snapshots,
        &result.temperature_gradient_snapshots,
        &result.heat_flux_snapshots,
        &result.energy_storage_rate_snapshots,
    ];
    for (ordinal, entries) in manifest.fields.iter().enumerate() {
        for (entry, history) in entries.iter().zip(histories) {
            let field = RecoveredField::new(
                entry.definition.clone(),
                entry.sample,
                history[ordinal].clone(),
            )
            .unwrap();
            for partition in canonical_field_partitions(&field).unwrap() {
                payloads.push((
                    partition.identity.digest(),
                    partition.canonical_bytes().unwrap(),
                ));
            }
        }
    }
    let partitions = payloads
        .iter()
        .map(
            |(identity, bytes)| runmat_analysis_fea::execution::AnalysisPayloadPartition {
                logical_identity: *identity,
                payload: bytes,
            },
        )
        .collect::<Vec<_>>();
    prepare_partitioned_analysis_objects(
        AnalysisNumericalObjectKind::SolverResult,
        manifest.identity().digest(),
        &partitions,
        chunk_bytes,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap()
}
