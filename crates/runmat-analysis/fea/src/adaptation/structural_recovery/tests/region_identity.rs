//! Recovery follows full canonical region occurrences, not display/source names.
use runmat_analysis_core::{AnalysisFieldLayout, ObservationQuantity, FieldLocation};
use super::*;

fn check_layout(integration_points: bool) {
    for quadratic in [false, true] {
        let mut mesh = if quadratic {
            crate::topology::test_meshes::quadratic_slab_mesh(1)
        } else {
            two_tetrahedron_artifact()
        };
        let count = mesh.topology.volume_elements.len();
        let points = if integration_points { 4 } else { 1 };
        let values = (0..count)
            .flat_map(|element| {
                let amplitude = (element + 1) as f64;
                let row = [
                    amplitude,
                    2.0 * amplitude,
                    -amplitude,
                    0.5 * amplitude,
                    0.0,
                    amplitude,
                ];
                (0..points).flat_map(move |_| row)
            })
            .collect();
        let stress = AnalysisField::host_f64(
            if integration_points {
                FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS
            } else {
                FEA_FIELD_STRUCTURAL_STRESS
            }, if integration_points { AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::IntegrationPoint) } else { AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::Element) },
            vec![count * points, 6],
            values,
        );
        let evaluate = |mesh: &SolverMeshArtifact| {
            estimate_structural_recovery_error(
                mesh,
                "elements",
                &stress,
                StructuralRecoveryEstimatorOptions::default(),
            )
            .unwrap()
        };
        // Equal canonical region values on adjacent elements must still blend.
        let blended = evaluate(&mesh);
        assert!(blended.total_error > 0.0);
        let original_region = mesh.topology.volume_elements[0].region_id.clone();
        mesh.topology.regions.clear();
        let topo = &mut *mesh.topology;
        for (index, element) in topo.volume_elements.iter_mut().enumerate() {
            let mut occurrence = original_region.clone();
            if index > 0 {
                occurrence.assembly_path = vec!["root".to_owned(), format!("part_{index}")];
            }
            element.region_id = occurrence.clone();
            element.provenance = vec![occurrence.clone()];
            topo.regions.push(MeshRegion {
                region_id: occurrence,
                element_ids: vec![element.element_id],
            });
        }
        mesh.seal_canonical_digest().unwrap();
        let separated = evaluate(&mesh);
        // Each region has one constant-stress cell, hence exact zero discrepancy.
        assert_eq!(
            separated.total_error, 0.0,
            "quadratic={quadratic}, points={integration_points}"
        );
        assert!(separated.marked_element_identities.is_empty());
        assert!((separated.reference_norm / blended.reference_norm - 1.0).abs() < 1.0e-12);
        assert!(separated.indicators.iter().all(|i| i.error == 0.0));
        let cold: SolverMeshArtifact =
            serde_json::from_value(serde_json::to_value(&mesh).unwrap()).unwrap();
        assert_eq!(separated, evaluate(&cold));
    }
}

#[test]
fn assembly_occurrences_remain_separate_for_element_stress() {
    check_layout(false);
}

#[test]
fn assembly_occurrences_remain_separate_for_point_stress() {
    check_layout(true);
}
