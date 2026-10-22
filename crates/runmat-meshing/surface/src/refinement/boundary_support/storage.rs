//! One output slot per exact boundary segment, allocated only on first use.
//! No geometric Vec growth can outrun the typed-capacity admission. A valid
//! trimmed surface has one incident kept triangle per boundary segment.
use super::*;
use runmat_geometry_core::{GeometryEvaluationError, PersistentEntityId};
pub(super) fn reserve_candidate<T>(
    output: &mut Vec<T>,
    maximum: usize,
    control: &dyn GeometryEvaluationControl,
    face: &PersistentEntityId,
) -> Result<(), ExactFaceRefinementError> {
    let invalid = |reason| {
        ExactFaceRefinementError::new(ExactFaceRefinementErrorKind::ResourceLimit, face, reason)
    };
    if output.len() >= maximum {
        return Err(invalid(
            "boundary support exceeds its admitted segment count",
        ));
    }
    if output.capacity() == 0 {
        let bytes = maximum
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| invalid("boundary support capacity overflow"))?;
        control
            .consume_allocation_bytes(bytes)
            .map_err(|failure: GeometryEvaluationError| {
                ExactFaceRefinementError::new(
                    ExactFaceRefinementErrorKind::Metric(
                        crate::ExactFaceMetricErrorKind::GeometryEvaluation(failure.kind),
                    ),
                    face,
                    failure.reason,
                )
            })?;
        output
            .try_reserve_exact(maximum)
            .map_err(|_| invalid("boundary support allocation failed"))?;
    }
    if output.capacity() < maximum {
        return Err(invalid("boundary support buffer has foreign capacity"));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{GeometryEvaluationErrorKind, PersistentEntityKind};
    struct Budget {
        maximum: u64,
        used: std::sync::atomic::AtomicU64,
    }
    impl GeometryEvaluationControl for Budget {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
            self.used
                .fetch_update(
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                    |used| used.checked_add(bytes).filter(|next| *next <= self.maximum),
                )
                .map_err(|_| {
                    GeometryEvaluationError::new(
                        GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                        "test capacity ceiling",
                    )
                })?;
            Ok(())
        }
        fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
    }
    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Actual vector capacity fits the admitted boundary-segment count for first and
    //   repeated insertions. A one-byte shortage rejects before allocation; exceeding the count cannot
    //   grow storage. This establishes typed scratch capacity admission, excluding allocator headers
    //   and whole-process RSS, not whole-model throughput or scientific acceptance.
    #[test]
    fn boundary_capacity_is_reserved_before_allocation_and_cannot_grow_past_the_boundary() {
        let (_, topology, _) = runmat_geometry_fixtures::exact_tetrahedron();
        let face = &topology
            .faces
            .iter()
            .find(|face| face.id.kind == PersistentEntityKind::Face)
            .unwrap()
            .id;
        for segments in [1, 3, 5, 17, 257] {
            let required = (segments * std::mem::size_of::<[u64; 3]>()) as u64;
            let denied = Budget {
                maximum: required - 1,
                used: std::sync::atomic::AtomicU64::new(0),
            };
            let mut output = Vec::<[u64; 3]>::new();
            let error = reserve_candidate(&mut output, segments, &denied, face).unwrap_err();
            assert_eq!(output.capacity(), 0);
            assert_eq!(
                error.kind,
                ExactFaceRefinementErrorKind::Metric(
                    crate::ExactFaceMetricErrorKind::GeometryEvaluation(
                        GeometryEvaluationErrorKind::AllocationBudgetExceeded
                    )
                )
            );
            let admitted = Budget {
                maximum: required,
                used: std::sync::atomic::AtomicU64::new(0),
            };
            for _ in 0..segments {
                reserve_candidate(&mut output, segments, &admitted, face).unwrap();
                output.push([0; 3]);
                assert!(output.capacity() <= segments);
                assert!(
                    output.capacity() as u64 * std::mem::size_of::<[u64; 3]>() as u64
                        <= admitted.used.load(std::sync::atomic::Ordering::Relaxed)
                );
            }
            let capacity = output.capacity();
            assert!(reserve_candidate(&mut output, segments, &admitted, face).is_err());
            assert_eq!(output.capacity(), capacity);
        }
    }
}
