//! Retained authority/reference metadata and original wire buffers; no payload
//! copies, serialized-size proxies or decoder scratch are counted as ownership.
use super::*;
impl PreparedMeshingResultPublication {
    pub(crate) fn retained_storage_bytes(
        &self,
        control: &dyn runmat_geometry_core::GeometryEvaluationControl,
    ) -> MeshingExecutionResult<u64> {
        let Self {
            stage_objects,
            root_output,
            result_objects,
        } = self;
        let ValuePayload::Object(root) = root_output else {
            return Err(MeshingExecutionError::Identity(
                "stage publication root is not an object reference",
            ));
        };
        let mut bytes = std::mem::size_of::<Self>() as u64;
        let mut add = |value| {
            control
                .checkpoint()
                .map_err(|error| MeshingExecutionError::Invalid(error.to_string()))?;
            bytes = bytes.checked_add(value).ok_or_else(|| {
                runmat_execution_artifact::ArtifactError::Limit(
                    "retained publication storage overflow".into(),
                )
            })?;
            Ok::<_, MeshingExecutionError>(())
        };
        add(stage_objects
            .retained_storage_bytes(control)?
            .checked_sub(std::mem::size_of::<PreparedMeshingStageObjects>() as u64)
            .ok_or(MeshingExecutionError::Identity(
                "stage metadata inventory lacks its inline value",
            ))?)?;
        add(std::mem::size_of::<ValueRef>() as u64)?;
        add(root.owned_buffer_bytes()?)?;
        add(crate::object_support::import_slots::<ValueRef>(
            result_objects.capacity(),
        )?)?;
        for reference in result_objects {
            add(reference.owned_buffer_bytes()?)?;
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Retained publication storage includes spare reference, object-payload, manifest and
    //   descriptor capacities without changing canonical content. Buffer capacities, encoded lengths
    //   and measured RSS remain separate quantities.
    #[test]
    fn retained_publication_counts_spare_capacities_without_changing_canonical_content() {
        let stage = crate::tests::fixture(vec![vec![5; 2048]], 1024);
        let limits = ObjectInventoryLimits::default();
        let access = ArtifactAccess {
            authorization_scope: "storage-owner".into(),
            encryption_context: runmat_execution::Digest::sha256(b"storage-owner"),
        };
        let mut publication = prepare_result_publication(stage, access, limits).unwrap();
        let control = crate::geometry_control::tests::bounded_control(64 << 20);
        let original_root = publication.stage_objects.root.digest;
        for field in 0..5 {
            let before = publication.retained_storage_bytes(&control).unwrap();
            let added = match field {
                0 => {
                    let ValuePayload::Object(root) = &mut publication.root_output else {
                        unreachable!()
                    };
                    let before = root.media_type.capacity();
                    root.media_type.reserve_exact(1024);
                    root.media_type.capacity() - before
                }
                1 => {
                    let before = publication.result_objects.capacity();
                    publication.result_objects.reserve_exact(100);
                    (publication.result_objects.capacity() - before)
                        * std::mem::size_of::<ValueRef>()
                }
                2 => {
                    let bytes = &mut publication.stage_objects.objects[1].bytes;
                    let before = bytes.capacity();
                    bytes.reserve_exact(1024);
                    bytes.capacity() - before
                }
                3 => {
                    let chunks = &mut publication.stage_objects.manifest.chunks;
                    let before = chunks.capacity();
                    chunks.reserve_exact(100);
                    (chunks.capacity() - before)
                        * std::mem::size_of::<runmat_meshing_core::MeshingChunkDescriptor>()
                }
                _ => {
                    let name = &mut publication.stage_objects.root.logical_name;
                    let before = name.capacity();
                    name.reserve_exact(1024);
                    name.capacity() - before
                }
            };
            assert!(added > 0);
            assert_eq!(
                publication.retained_storage_bytes(&control).unwrap(),
                before + added as u64
            );
            assert_eq!(publication.stage_objects.root.digest, original_root);
            publication.stage_objects.revalidate(limits).unwrap();
        }
    }
}
