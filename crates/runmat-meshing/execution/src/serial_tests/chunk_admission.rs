//! The real serial route must reject before publishing any understated payload.
use super::*;

struct ChunkKernel;
impl MeshingStageKernel for ChunkKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        let allocations = invocation.control.geometry_evaluation_control();
        let mut output = SurfaceKernel.execute(invocation)?;
        for (index, record) in output.streams[0].records.iter_mut().enumerate() {
            allocations
                .consume_allocation_bytes(70_000)
                .map_err(|error| {
                    crate::budget::failure(
                        runmat_meshing_core::MeshingStageKind::SurfaceMesh,
                        crate::geometry_failure::category(error.kind),
                        &error.reason,
                        None,
                    )
                })?;
            let replacement = vec![index as u8 + 1; 70_000];
            allocations.release_allocation_bytes(record.capacity() as u64);
            *record = replacement;
        }
        Ok(output)
    }
}

#[test]
fn serial_publication_requires_metadata_budget_beyond_its_exact_chunk_envelope() {
    let mut ceilings = crate::serial_tests::ceilings();
    let streams = vec![MeshingChunkStream {
        media_type: MeshingChunkMediaType::SurfacePartitions,
        schema_version: 2,
        records: vec![vec![1; 70_000], vec![2; 70_000], vec![3; 70_000]],
    }];
    let policy = MeshingChunkPolicy {
        maximum_chunk_bytes: 65_536,
        maximum_records_per_chunk: 3,
        maximum_total_encoded_bytes: 1_000_000,
    };
    let profile =
        runmat_meshing_core::profile_chunked_stage_payload(&streams, policy, &NeverCancelled)
            .unwrap();
    for cap in [
        profile.requested_storage_bytes - 1,
        profile.requested_storage_bytes,
        512 << 20,
    ] {
        let request = request();
        ceilings.maximum_memory_bytes = cap;
        let mut fixture = Fixture::with_request_and_ceilings(request, ceilings);
        let original_objects = fixture.store.objects.clone();
        let result = execute_serial_stage(
            &fixture.program,
            &mut fixture.store,
            &ChunkKernel,
            &NeverCancelled,
            &mut Progress::default(),
            policy,
            // Keep import below the payload envelope. Equal chunk capacity is
            // insufficient for the separately required publication metadata.
            ObjectInventoryLimits {
                max_object_bytes: 65_536,
                ..ObjectInventoryLimits::default()
            },
        );
        if cap <= profile.requested_storage_bytes {
            let MeshingSerialExecutionError::Stage(failure) = result.unwrap_err() else {
                panic!("chunk admission must retain its originating stage failure");
            };
            assert_eq!(
                failure.category,
                MeshingFailureCategory::MemoryBudgetExceeded
            );
            assert_eq!(fixture.store.objects, original_objects);
        } else {
            let completed = result.unwrap();
            assert_eq!(
                completed
                    .publication()
                    .stage_objects()
                    .decoded_streams()
                    .unwrap(),
                streams
            );
        }
    }
}
