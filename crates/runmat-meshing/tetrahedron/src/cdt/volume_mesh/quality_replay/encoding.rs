use super::super::representation::VolumeRepresentation;
use super::*;

/// Caller admits this flat dictionary/sort demand before preparing the borrowed view.
pub fn volume_quality_snapshot_preparation_storage_bytes(
    snapshot: &VolumeQualitySnapshot<'_>,
) -> Result<u64, DelaunayVolumeMeshError> {
    VolumeRepresentation::preparation_bytes(snapshot.topology, snapshot.provenance, None)
}
pub struct PreparedVolumeQualitySnapshot<'a> {
    binding: VolumeQualityBinding,
    input: &'a VolumeQualitySnapshot<'a>,
    volume: VolumeRepresentation<'a>,
}
pub fn prepare_volume_quality_snapshot<'a>(
    input: &'a VolumeQualitySnapshot<'a>,
    binding: VolumeQualityBinding,
    control: &dyn CanonicalCodecControl,
) -> Result<PreparedVolumeQualitySnapshot<'a>, DelaunayVolumeMeshError> {
    validate_binding(binding)?;
    Ok(PreparedVolumeQualitySnapshot {
        binding,
        input,
        volume: VolumeRepresentation::new(input.topology, input.provenance, None, control)?,
    })
}
impl Serialize for PreparedVolumeQualitySnapshot<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Input<'a, 'b> {
            volume: &'a VolumeRepresentation<'b>,
            metric: &'b MetricFieldRequest,
            fixed_nodes: &'b [StableDigest],
            options: DelaunayVolumeMeshOptions,
        }
        #[derive(Serialize)]
        struct Envelope<'a, 'b> {
            binding: VolumeQualityBinding,
            input: Input<'a, 'b>,
        }
        Envelope {
            binding: self.binding,
            input: Input {
                volume: &self.volume,
                metric: self.input.metric,
                fixed_nodes: self.input.fixed_nodes,
                options: self.input.options,
            },
        }
        .serialize(serializer)
    }
}
impl PreparedVolumeQualitySnapshot<'_> {
    pub fn profile(
        &self,
        limits: CanonicalLimits,
        control: &dyn CanonicalCodecControl,
    ) -> Result<(runmat_canonical_codec::CanonicalEncodingProfile, u64), DelaunayVolumeMeshError>
    {
        let observed = super::super::codec::control::Control::from_codec(control);
        let control: &dyn CanonicalCodecControl = &observed;
        let encoding =
            runmat_canonical_codec::profile_canonical_encoding_with_control(self, limits, control)
                .map_err(|failure| codec_error(failure, control))?;
        let materialization =
            runmat_canonical_codec::profile_serialized_with_control(self, limits, control)
                .map_err(|failure| codec_error(failure, control))?
                .materialization_resident_bytes_estimate;
        Ok((encoding, materialization))
    }
    pub fn encode(
        &self,
        limits: CanonicalLimits,
        control: &dyn CanonicalCodecControl,
    ) -> Result<Vec<u8>, DelaunayVolumeMeshError> {
        let observed = super::super::codec::control::Control::from_codec(control);
        let control: &dyn CanonicalCodecControl = &observed;
        runmat_canonical_codec::encode_contract_with_control(PREFIX, DOMAIN, self, limits, control)
            .map_err(|failure| codec_error(failure, control))
    }
}
