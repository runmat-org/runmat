//! Display scaling respects the distinction between physical displacement and
//! arbitrarily normalized modal vectors. Neither changes the retained field.
/// A published mode shape, recognized from its declared state ordinal. Its amplitude is a
/// normalization, not a physical value.
pub(super) fn is_mode_shape(descriptor: &crate::analysis::AnalysisFieldDescriptor) -> bool {
    matches!(
        descriptor.layout.ordinal,
        runmat_analysis_core::FieldOrdinal::Mode(_)
    )
}

/// A nodal 3-vector displacement (static, stepped or modal) that can displace the shape.
pub(super) fn displaces(descriptor: &crate::analysis::AnalysisFieldDescriptor) -> bool {
    descriptor.layout.quantity == runmat_analysis_core::ObservationQuantity::Displacement
        && descriptor.layout.location == runmat_analysis_core::FieldLocation::Node
        && descriptor.layout.value_shape == runmat_analysis_core::ObservationValueShape::Vector3
        && descriptor.layout.representation == runmat_analysis_core::ComplexRepresentation::Real
}

pub(super) fn automatic_scale(diagonal: f64, amplitude: f64, normalized: bool) -> f64 {
    if !diagonal.is_finite() || amplitude <= 0.0 || !amplitude.is_finite() {
        return 0.0;
    }
    let scale = 0.12 * diagonal / amplitude;
    if normalized {
        scale
    } else {
        // Physical displacement remains visible at true scale when it already
        // exceeds the automatic display fraction. Modal normalization has no
        // physical metre-valued amplitude and must not use that lower bound.
        scale.max(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn modal_display_is_invariant_under_eigenvector_normalization() {
        for amplitude in [1e-100, 1e-6, 1.0, 1e6, 1e100] {
            let scale = automatic_scale(0.1, amplitude, true);
            assert!((scale * amplitude - 0.012).abs() < 1e-16);
        }
        assert_eq!(automatic_scale(0.1, 7.0, false), 1.0);
        assert!(automatic_scale(0.1, 1e-6, false) > 1.0);
        assert_eq!(automatic_scale(0.1, 0.0, true), 0.0);
    }
}

#[cfg(test)]
mod scene_tests {
    use super::super::{scene_from_topology, AnalysisFieldLocation, FieldData};
    use crate::analysis::{
        AnalysisRenderMesh, AnalysisRenderTopology, AnalysisRenderTopologySource,
    };

    // witness: FEA-SHARED-OBSERVATIONS-001/governed_observation_corpus @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: A modal scene's displayed positions are invariant under eigenvector rescaling while
    //   its original field values and scalar range remain intact. Synthetic scene only; no modal
    //   accuracy, mesh convergence, publication closure, distributed evaluation or whole-task resource
    //   claim.
    #[test]
    fn original_modal_vectors_keep_their_values_while_display_shape_is_normalization_invariant() {
        let topology = AnalysisRenderTopology {
            schema_version: "test".into(),
            source: AnalysisRenderTopologySource::AnalysisMesh,
            volume_element_count: 1,
            meshes: vec![AnalysisRenderMesh {
                mesh_id: "mesh".into(),
                vertices: vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.1]],
                triangles: vec![[0, 1, 2]],
                regions: vec![],
                vertex_volume_node_indices: vec![Some(0), Some(1), Some(2)],
                triangle_volume_element_indices: vec![Some(0)],
            }],
        };
        let capture = |amplitude| {
            let field = FieldData::inline(
                AnalysisFieldLocation::Node,
                3,
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, amplitude, 0.0, 0.0],
                "modal.mode_shape.1",
                "m",
            )
            .mode_shape();
            let (scene, scale) = scene_from_topology(
                "run",
                &topology,
                Some(&field),
                Some(&field),
                "exaggerated",
                1.0,
                true,
            )
            .unwrap();
            assert_eq!(field.range(6, 3).unwrap(), [amplitude, 0.0, 0.0]);
            assert!(scale < 1.0);
            assert_eq!(
                scene
                    .overlay
                    .as_ref()
                    .unwrap()
                    .scalar_legend
                    .as_ref()
                    .unwrap()
                    .maximum,
                amplitude
            );
            scene
        };
        let first = capture(7.0);
        let rescaled = capture(70.0);
        assert_eq!(first.chunks.len(), rescaled.chunks.len());
        for (first, rescaled) in first.chunks.iter().zip(&rescaled.chunks) {
            for (first, rescaled) in first.vertices.iter().zip(&rescaled.vertices) {
                for (first, rescaled) in first.position.iter().zip(&rescaled.position) {
                    assert!((first - rescaled).abs() < 1e-7);
                }
            }
        }
        assert!(first.bounds.max.x < 0.02);
        assert!(first.bounds.max.z < 0.11);
    }
}
