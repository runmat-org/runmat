//! Encode one accepted face, then release its rich producer storage. Wire records
//! have one owner until the ordinary governed chunk publication takes over.
use crate::codec_control::CodecControl;
use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};
use runmat_meshing_surface::{
    AcceptedExactFace, ExactFacePartitionCompletion, ExactFacePartitionError,
    ExactFacePartitionErrorKind, ExactSurfaceMeshError,
};

pub(super) struct Records<'a> {
    control: &'a dyn GeometryEvaluationControl,
    records: Vec<Vec<u8>>,
    retained: u64,
    root_storage: GeometryAllocationReservation<'a>,
    nodes: u64,
    triangles: u64,
}
pub(super) struct Output {
    pub records: Vec<Vec<u8>>,
    pub completed: u64,
    pub faces: u64,
    pub nodes: u64,
    pub triangles: u64,
    pub splits: u64,
    pub encoded_bytes: u64,
}
impl<'a> Records<'a> {
    pub fn new(
        control: &'a dyn GeometryEvaluationControl,
        count: u64,
    ) -> Result<Self, ExactFacePartitionError> {
        let capacity = usize::try_from(count)
            .ok()
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| invalid("face record inventory overflow"))?;
        let root_bytes = (capacity as u64)
            .checked_mul(std::mem::size_of::<Vec<u8>>() as u64)
            .ok_or_else(|| invalid("face record storage overflow"))?;
        let root_storage =
            GeometryAllocationReservation::new(control, root_bytes).map_err(control_error)?;
        let mut records = Vec::with_capacity(capacity);
        records.push(Vec::new());
        Ok(Self {
            control,
            records,
            retained: 0,
            root_storage,
            nodes: 0,
            triangles: 0,
        })
    }
    pub fn accept(
        &mut self,
        accepted: AcceptedExactFace<'_>,
    ) -> Result<(), ExactFacePartitionError> {
        if self.records.len() >= self.records.capacity() {
            return Err(invalid(
                "producer exceeds its admitted face record inventory",
            ));
        }
        let codec = CodecControl::new(self.control);
        let requested = accepted
            .encoding_preparation_storage_bytes(&codec)
            .map_err(|error| codec.take_failure().map_or(error, control_error))?;
        let _preparation =
            GeometryAllocationReservation::new(self.control, requested).map_err(control_error)?;
        let prepared = accepted
            .prepare_encoding(&codec)
            .map_err(|error| codec.take_failure().map_or(error, control_error))?;
        let profile = prepared
            .encoding_profile(&codec)
            .map_err(|error| codec_error(error, &codec))?;
        let mut storage = GeometryAllocationReservation::new(
            self.control,
            profile.resident_workspace_bytes_estimate,
        )
        .map_err(control_error)?;
        let bytes = prepared
            .encode_profiled(&profile, &codec)
            .map_err(|error| codec_error(error, &codec))?;
        if bytes.len() as u64 > profile.encoded_bytes_upper_bound
            || bytes.capacity() as u64 > profile.resident_workspace_bytes_estimate
        {
            return Err(invalid("face encoder exceeds admitted wire storage"));
        }
        self.nodes = self
            .nodes
            .checked_add(accepted.mesh().nodes.len() as u64)
            .ok_or_else(|| invalid("face node count overflow"))?;
        self.triangles = self
            .triangles
            .checked_add(accepted.mesh().triangles.len() as u64)
            .ok_or_else(|| invalid("face triangle count overflow"))?;
        let capacity = bytes.capacity() as u64;
        self.retained = self
            .retained
            .checked_add(capacity)
            .ok_or_else(|| invalid("face wire inventory overflow"))?;
        self.records.push(bytes);
        storage.retain_bytes(capacity);
        Ok(())
    }
    pub fn finish(
        mut self,
        completion: ExactFacePartitionCompletion,
    ) -> Result<Output, ExactFacePartitionError> {
        let split_count = completion.curve_splits.len() as u64;
        if split_count > 0 {
            self.records.truncate(1);
            self.control.release_allocation_bytes(self.retained);
            self.retained = 0;
            self.nodes = 0;
            self.triangles = 0;
        } else if completion.accepted_face_count != self.records.len() - 1 {
            return Err(invalid(
                "surface completion and encoded original face inventory differ",
            ));
        }
        let codec = CodecControl::new(self.control);
        let profile =
            runmat_meshing_surface::profile_exact_face_partition_completion(&completion, &codec)
                .map_err(|error| codec_error(error, &codec))?;
        let mut storage = GeometryAllocationReservation::new(
            self.control,
            profile.resident_workspace_bytes_estimate,
        )
        .map_err(control_error)?;
        let header =
            runmat_meshing_surface::encode_exact_face_partition_completion(&completion, &codec)
                .map_err(|error| codec_error(error, &codec))?;
        let capacity = header.capacity() as u64;
        if capacity > profile.resident_workspace_bytes_estimate {
            return Err(invalid(
                "surface header exceeds admitted encoding workspace",
            ));
        }
        self.retained = self
            .retained
            .checked_add(capacity)
            .ok_or_else(|| invalid("surface header storage overflow"))?;
        self.records[0] = header;
        storage.retain_bytes(capacity);
        let completed = completion
            .partition
            .entity_range
            .as_ref()
            .map_or(0, |range| range.entity_count);
        let encoded_bytes = self
            .records
            .iter()
            .try_fold(0u64, |sum, bytes| sum.checked_add(bytes.len() as u64))
            .ok_or_else(|| invalid("surface wire count overflow"))?;
        let output = Output {
            completed,
            faces: if split_count > 0 {
                0
            } else {
                completion.accepted_face_count as u64
            },
            nodes: self.nodes,
            triangles: self.triangles,
            splits: split_count,
            encoded_bytes,
            records: std::mem::take(&mut self.records),
        };
        self.retained = 0;
        self.root_storage
            .retain_bytes((output.records.capacity() * std::mem::size_of::<Vec<u8>>()) as u64);
        Ok(output)
    }
}
impl Drop for Records<'_> {
    fn drop(&mut self) {
        self.records.clear();
        self.control.release_allocation_bytes(self.retained);
    }
}
fn invalid(reason: impl Into<String>) -> ExactFacePartitionError {
    ExactFacePartitionError {
        kind: ExactFacePartitionErrorKind::Reuse,
        reason: reason.into(),
    }
}
fn control_error(error: runmat_geometry_core::GeometryEvaluationError) -> ExactFacePartitionError {
    ExactFacePartitionError {
        kind: ExactFacePartitionErrorKind::Coordinates(error.kind),
        reason: error.to_string(),
    }
}
fn codec_error(
    error: ExactSurfaceMeshError,
    control: &CodecControl<'_>,
) -> ExactFacePartitionError {
    match control.take_failure() {
        Some(error) => control_error(error),
        None => ExactFacePartitionError {
            kind: ExactFacePartitionErrorKind::Batch(error.kind),
            reason: error.to_string(),
        },
    }
}
