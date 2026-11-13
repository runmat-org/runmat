//! Native representation bytes have exactly one inventory owner. Callers can
//! borrow them or consume the prepared closure and move the original buffer.
use super::*;
impl PreparedExactGeometryObjects {
    pub fn kernel_representation(&self) -> MeshingExecutionResult<Option<&[u8]>> {
        Ok(self
            .representation_index()?
            .map(|index| self.objects[index].bytes.as_slice()))
    }
    /// Ownership transfer only; this does not create a new validation proof.
    pub fn into_components(
        mut self,
    ) -> MeshingExecutionResult<runmat_geometry_core::AdmittedExactGeometry> {
        let representation = self
            .representation_index()?
            .map(|index| self.objects.swap_remove(index).bytes);
        Ok(runmat_geometry_core::AdmittedExactGeometry {
            manifest: self.manifest,
            topology: self.topology,
            evaluators: self.evaluators,
            kernel_representation: representation,
            healing_report: self.healing_report,
        })
    }
    fn representation_index(&self) -> MeshingExecutionResult<Option<usize>> {
        let mut objects =
            self.objects.iter().enumerate().filter(|(_, object)| {
                object.descriptor.media_type == KERNEL_REPRESENTATION_MEDIA_TYPE
            });
        match (&self.manifest.kernel_representation, objects.next()) {
            (None, None) => Ok(None),
            (Some(reference), Some((index, object))) => {
                if objects.next().is_some()
                    || object.descriptor.digest.bytes() != reference.digest.bytes()
                    || object.descriptor.encoded_length != reference.encoded_length
                    || object.bytes.len() as u64 != reference.encoded_length
                {
                    return Err(MeshingExecutionError::Identity(
                        "native geometry representation inventory differs from its manifest",
                    ));
                }
                Ok(Some(index))
            }
            _ => Err(MeshingExecutionError::Identity(
                "native geometry representation is missing or unreferenced",
            )),
        }
    }
}
