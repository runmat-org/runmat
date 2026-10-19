//! Quality report for the analysis-mesh artifact's tetrahedral elements.

use std::collections::BTreeMap;

use crate::contracts::AnalysisVolumeElement;

use super::{
    predicate::{tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_volume},
    AnalysisMeshQualityReport, ElementQuality,
};

pub fn quality_report(
    volume_elements: &[AnalysisVolumeElement],
    coordinates_by_node_id: &BTreeMap<u32, [f64; 3]>,
) -> AnalysisMeshQualityReport {
    let elements = volume_elements
        .iter()
        .filter_map(|element| {
            if element.node_ids.len() != 4 {
                return None;
            }
            let points = [
                *coordinates_by_node_id.get(&element.node_ids[0])?,
                *coordinates_by_node_id.get(&element.node_ids[1])?,
                *coordinates_by_node_id.get(&element.node_ids[2])?,
                *coordinates_by_node_id.get(&element.node_ids[3])?,
            ];
            Some(ElementQuality {
                element_id: element.element_id.clone(),
                scaled_jacobian: tetrahedron_scaled_jacobian(points),
                exact_scaled_jacobian: tetrahedron_scaled_jacobian(points),
                aspect_ratio: tetrahedron_edge_aspect_ratio(points),
                volume_m3: tetrahedron_volume(points),
            })
        })
        .collect::<Vec<_>>();
    let min_scaled_jacobian = elements
        .iter()
        .map(|element| element.scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let max_aspect_ratio = elements
        .iter()
        .map(|element| element.aspect_ratio)
        .fold(0.0_f64, f64::max);
    let mean_aspect_ratio = if elements.is_empty() {
        0.0
    } else {
        elements
            .iter()
            .map(|element| element.aspect_ratio)
            .sum::<f64>()
            / elements.len() as f64
    };
    AnalysisMeshQualityReport {
        min_scaled_jacobian: min_scaled_jacobian.min(1.0),
        min_exact_scaled_jacobian: min_scaled_jacobian.min(1.0),
        mean_aspect_ratio,
        max_aspect_ratio,
        inverted_element_count: elements
            .iter()
            .filter(|element| element.volume_m3 <= 0.0)
            .count(),
        mean_boundary_projection_error_m: 0.0,
        max_boundary_projection_error_m: 0.0,
        elements,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::VolumeElementKind;

    fn element() -> AnalysisVolumeElement {
        AnalysisVolumeElement {
            element_id: "tetrahedron_1".to_string(),
            kind: VolumeElementKind::Tetrahedron4,
            node_ids: vec![1, 2, 3, 4],
            material_region_id: "region_default".to_string(),
            provenance: Vec::new(),
        }
    }

    #[test]
    fn right_tetra_report_uses_the_canonical_predicates() {
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let coordinates = (1..=4).zip(points).collect();
        let report = quality_report(&[element()], &coordinates);
        let tetra = report.elements.first().expect("one tetrahedron");
        assert_eq!(tetra.volume_m3, 1.0 / 6.0);
        assert_eq!(tetra.aspect_ratio, 2.0_f64.sqrt());
        assert_eq!(tetra.scaled_jacobian, tetrahedron_scaled_jacobian(points));
        assert_eq!(
            tetra.exact_scaled_jacobian,
            report.min_exact_scaled_jacobian
        );
        assert_eq!(report.min_scaled_jacobian, tetra.scaled_jacobian);
        assert_eq!(report.mean_aspect_ratio, tetra.aspect_ratio);
        assert_eq!(report.max_aspect_ratio, tetra.aspect_ratio);
        assert_eq!(report.inverted_element_count, 0);
    }

    #[test]
    fn regular_tetra_report_preserves_single_element_aggregation() {
        let root_three = 3.0_f64.sqrt();
        let root_two_thirds = (2.0_f64 / 3.0).sqrt();
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, root_three / 2.0, 0.0],
            [0.5, root_three / 6.0, root_two_thirds],
        ];
        let coordinates = (1..=4).zip(points).collect();
        let report = quality_report(&[element()], &coordinates);
        assert!((report.elements[0].volume_m3 - 2.0_f64.sqrt() / 12.0).abs() < 1.0e-12);
        assert!((report.max_aspect_ratio - 1.0).abs() < 1.0e-12);
        assert!(report.min_scaled_jacobian > 0.0);
        assert_eq!(report.min_scaled_jacobian, report.min_exact_scaled_jacobian);
    }
}
