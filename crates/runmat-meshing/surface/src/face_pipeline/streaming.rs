//! Visit original accepted faces without retaining a rich batch. Persistence and
//! encoded output ownership belong to the caller; the numerical owner frees each
//! mesh before producing the next. No serialized flag can construct this capability.
use super::*;

pub struct ExactFacePartitionCompletion {
    pub partition: MeshingPartitionDescriptor,
    pub accepted_face_count: usize,
    pub curve_splits: Vec<runmat_meshing_curve::SharedCurveSegmentSplit>,
}

pub fn visit_exact_face_partition(
    partition: MeshingPartitionDescriptor,
    context: ExactFacePartitionContext<'_>,
    options: ExactFacePartitionOptions,
    visit: &mut dyn FnMut(AcceptedExactFace<'_>) -> Result<(), ExactFacePartitionError>,
) -> Result<ExactFacePartitionCompletion, ExactFacePartitionError> {
    visit_exact_face_partition_with_reuse(
        partition,
        context,
        options,
        &mut reuse::NoFaceReuse,
        visit,
    )
}

pub fn visit_exact_face_partition_with_reuse(
    partition: MeshingPartitionDescriptor,
    context: ExactFacePartitionContext<'_>,
    options: ExactFacePartitionOptions,
    reuse: &mut dyn ExactFaceMeshReuse,
    visit: &mut dyn FnMut(AcceptedExactFace<'_>) -> Result<(), ExactFacePartitionError>,
) -> Result<ExactFacePartitionCompletion, ExactFacePartitionError> {
    process_partition(partition, context, options, reuse, true, &mut |mesh| {
        visit(AcceptedExactFace::new(&mesh))
    })
}

pub(super) struct FaceRetention<'a> {
    control: &'a dyn GeometryEvaluationControl,
    bytes: u64,
}
impl<'a> FaceRetention<'a> {
    pub fn new(
        mesh: &ExactFaceMesh,
        control: &'a dyn GeometryEvaluationControl,
        release: bool,
    ) -> Result<Self, ExactFacePartitionError> {
        Ok(Self {
            control,
            bytes: if release {
                mesh.retained_storage_bytes(control).map_err(|error| {
                    failure(ExactFacePartitionErrorKind::Coordinates(error.kind), error)
                })?
            } else {
                0
            },
        })
    }
}
impl Drop for FaceRetention<'_> {
    fn drop(&mut self) {
        self.control.release_allocation_bytes(self.bytes);
    }
}
