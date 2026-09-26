//! Validate owned kernel closures in place before accepting them for publication.
use super::*;

impl PreparedAnalysisObjects {
    pub(crate) fn validate_result_output(
        &self,
        chunk_bytes: usize,
        limits: ObjectInventoryLimits,
    ) -> AnalysisBridgeResult<()> {
        self.manifest.validate()?;
        if chunk_bytes == 0 || self.objects.len() != self.manifest.chunks.len() + 1 {
            return Err(AnalysisBridgeError::Identity(
                "kernel output has an invalid object inventory",
            ));
        }
        validate_inventory(&self.objects, limits)?;
        let root = &self.objects[0];
        if self.root != root.descriptor
            || !conversion::matches_root_name(&self.root.logical_name, self.root.digest)
            || self.root.media_type != ANALYSIS_MANIFEST_MEDIA_TYPE
            || root.bytes != self.manifest.canonical_bytes()?
            || self
                .objects
                .iter()
                .any(|object| object.descriptor.namespace != ObjectNamespace::ResultValue)
        {
            return Err(AnalysisBridgeError::Identity(
                "kernel output root or namespace differs from its manifest",
            ));
        }
        for (object, chunk) in self.objects[1..].iter().zip(&self.manifest.chunks) {
            if object.descriptor.media_type != ANALYSIS_CHUNK_MEDIA_TYPE
                || !conversion::matches_chunk_name(
                    &object.descriptor.logical_name,
                    chunk.ordinal,
                    chunk.digest,
                )
                || object.descriptor.digest != chunk.digest
                || object.descriptor.encoded_length != chunk.encoded_length
                || object.bytes.len() > chunk_bytes
            {
                return Err(AnalysisBridgeError::Identity(
                    "kernel output chunk differs from its manifest or policy",
                ));
            }
        }
        for partition in &self.manifest.partitions {
            let start = partition.first_chunk_ordinal as usize + 1;
            let end = start + partition.chunk_count as usize;
            if self.objects[start..end - 1]
                .iter()
                .any(|object| object.bytes.len() != chunk_bytes)
            {
                return Err(AnalysisBridgeError::Identity(
                    "kernel output chunk boundaries differ from policy",
                ));
            }
        }
        validate_partition_payloads(&self.manifest, &self.objects)?;
        if Digest::sha256_parts(self.objects[1..].iter().map(|object| &object.bytes))
            != self.manifest.payload_digest
        {
            return Err(AnalysisBridgeError::Identity(
                "kernel output payload differs from its manifest",
            ));
        }
        Ok(())
    }
}
