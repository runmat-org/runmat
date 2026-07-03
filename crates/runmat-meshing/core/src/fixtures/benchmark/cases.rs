use super::*;
use crate::{
    size::field::{MeshSizingField, SizingSample},
    validation::AnalysisMeshValidationOptions,
    MeshTargetSize, RefinementFocusLevel, VolumeMeshingOptions,
};
use runmat_geometry_core::GeometryAsset;

#[path = "cases/geometry.rs"]
mod geometry;
#[cfg(test)]
use geometry::annular_bore_block_surface;
use geometry::{
    box_geometry, box_surface, closed_surface_area_m2, closed_surface_volume_m3,
    faceted_cylinder_surface, geometry_from_surface, tapered_rectangular_prism_surface,
    through_hole_block_surface,
};

pub fn generic_mesh_benchmark_cases() -> Vec<MeshBenchmarkCase> {
    vec![
        solid_box_benchmark_case(
            "solid_cube",
            MeshBenchmarkTier::Solid3d,
            [1.0, 1.0, 1.0],
            1.0,
            6.0,
            1,
        ),
        solid_box_benchmark_case(
            "thin_slab",
            MeshBenchmarkTier::ThinFeature,
            [1.0, 1.0, 0.1],
            0.1,
            2.4,
            1,
        ),
        through_hole_block_benchmark_case(),
        faceted_cylinder_benchmark_case(),
        tapered_arm_benchmark_case(),
        disconnected_boxes_benchmark_case(),
        boundary_load_patch_benchmark_case(),
        adaptive_refinement_benchmark_case(),
    ]
}

fn solid_box_benchmark_case(
    benchmark_id: &str,
    tier: MeshBenchmarkTier,
    dimensions_m: [f64; 3],
    expected_volume_m3: f64,
    expected_boundary_area_m2: f64,
    max_volume_component_count: usize,
) -> MeshBenchmarkCase {
    let geometry = box_geometry(benchmark_id, dimensions_m, [0.0, 0.0, 0.0]);
    benchmark_case(
        benchmark_id,
        tier,
        geometry,
        expected_volume_m3,
        expected_boundary_area_m2,
        max_volume_component_count,
    )
}

fn disconnected_boxes_benchmark_case() -> MeshBenchmarkCase {
    let first = box_surface([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0);
    let second = box_surface([1.0, 1.0, 1.0], [1.6, 0.0, 0.0], 8);
    let mut vertices = first.0;
    vertices.extend(second.0);
    let mut triangles = first.1;
    triangles.extend(second.1);
    let geometry = geometry_from_surface(
        "disconnected_boxes",
        "generic_disconnected_boxes_surface",
        vertices,
        triangles,
    );
    benchmark_case(
        "disconnected_boxes",
        MeshBenchmarkTier::MultiBody,
        geometry,
        2.0,
        12.0,
        2,
    )
}

fn through_hole_block_benchmark_case() -> MeshBenchmarkCase {
    let outer = [1.0, 1.0, 1.0];
    let hole_min = [0.35, 0.35];
    let hole_max = [0.65, 0.65];
    let (vertices, triangles) = through_hole_block_surface(outer, hole_min, hole_max);
    let hole_width = hole_max[0] - hole_min[0];
    let hole_depth = hole_max[1] - hole_min[1];
    let expected_volume_m3 = outer[0] * outer[1] * outer[2] - hole_width * hole_depth * outer[2];
    let expected_boundary_area_m2 = 2.0 * (outer[0] * outer[1] - hole_width * hole_depth)
        + 2.0 * (outer[0] + outer[1]) * outer[2]
        + 2.0 * (hole_width + hole_depth) * outer[2];
    benchmark_case(
        "through_hole_block",
        MeshBenchmarkTier::HoleFeature,
        geometry_from_surface(
            "through_hole_block",
            "generic_through_hole_block_surface",
            vertices,
            triangles,
        ),
        expected_volume_m3,
        expected_boundary_area_m2,
        1,
    )
}

#[cfg(test)]
fn annular_bore_block_benchmark_case() -> MeshBenchmarkCase {
    let dimensions_m = [1.0, 1.0, 0.8];
    let bore_radius_m = 0.18;
    let segment_count = 12_usize;
    let (vertices, triangles) =
        annular_bore_block_surface(dimensions_m, bore_radius_m, segment_count);
    let expected_volume_m3 = closed_surface_volume_m3(&vertices, &triangles);
    let expected_boundary_area_m2 = closed_surface_area_m2(&vertices, &triangles);
    benchmark_case(
        "annular_bore_block",
        MeshBenchmarkTier::HoleFeature,
        geometry_from_surface(
            "annular_bore_block",
            "generic_annular_bore_block_surface",
            vertices,
            triangles,
        ),
        expected_volume_m3,
        expected_boundary_area_m2,
        1,
    )
}

fn faceted_cylinder_benchmark_case() -> MeshBenchmarkCase {
    let segment_count = 16_usize;
    let radius_m = 0.5_f64;
    let height_m = 1.0_f64;
    let (vertices, triangles) = faceted_cylinder_surface(segment_count, radius_m, height_m);
    let polygon_area = 0.5
        * segment_count as f64
        * radius_m.powi(2)
        * (std::f64::consts::TAU / segment_count as f64).sin();
    let polygon_perimeter =
        2.0 * segment_count as f64 * radius_m * (std::f64::consts::PI / segment_count as f64).sin();
    benchmark_case(
        "faceted_cylinder",
        MeshBenchmarkTier::CurvedSurface,
        geometry_from_surface(
            "faceted_cylinder",
            "generic_faceted_cylinder_surface",
            vertices,
            triangles,
        ),
        polygon_area * height_m,
        2.0 * polygon_area + polygon_perimeter * height_m,
        1,
    )
}

fn tapered_arm_benchmark_case() -> MeshBenchmarkCase {
    let (vertices, triangles) = tapered_rectangular_prism_surface([0.8, 0.5], [0.55, 0.35], 0.8);
    let expected_volume_m3 = closed_surface_volume_m3(&vertices, &triangles);
    let expected_boundary_area_m2 = closed_surface_area_m2(&vertices, &triangles);
    benchmark_case(
        "tapered_arm",
        MeshBenchmarkTier::Solid3d,
        geometry_from_surface(
            "tapered_arm",
            "generic_tapered_arm_surface",
            vertices,
            triangles,
        ),
        expected_volume_m3,
        expected_boundary_area_m2,
        1,
    )
}

fn adaptive_refinement_benchmark_case() -> MeshBenchmarkCase {
    let mut case = benchmark_case(
        "adaptive_refinement_marker",
        MeshBenchmarkTier::AdaptiveRefinement,
        box_geometry(
            "adaptive_refinement_marker",
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ),
        1.0,
        6.0,
        1,
    );
    case.options.target_size = MeshTargetSize::LengthM(2.0);
    case.options.refinement.focus.curvature = false;
    case.options.refinement.focus.small_features = false;
    case.options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    case.sizing = Some(MeshSizingField {
        samples: vec![SizingSample {
            position_m: [0.25, 0.25, 0.25],
            target_size_m: 0.50,
            reason: Some("structural.stress_gradient".to_string()),
        }],
        ..MeshSizingField::default()
    });
    case
}

fn boundary_load_patch_benchmark_case() -> MeshBenchmarkCase {
    let mut case = benchmark_case(
        "boundary_load_patch",
        MeshBenchmarkTier::SizingField,
        box_geometry("boundary_load_patch", [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
        1.0,
        6.0,
        1,
    );
    case.options.target_size = MeshTargetSize::LengthM(1.0);
    case.options.refinement.focus.curvature = false;
    case.options.refinement.focus.small_features = false;
    case.options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    case.validation.required_boundary_region_ids =
        vec!["benchmark_root".to_string(), "benchmark_tip".to_string()];
    case.validation.required_material_region_ids =
        vec!["benchmark_root".to_string(), "benchmark_tip".to_string()];
    case.sizing = Some(MeshSizingField {
        samples: vec![
            SizingSample {
                position_m: [1.0, 0.5, 0.5],
                target_size_m: 0.25,
                reason: Some("structural.load_regions".to_string()),
            },
            SizingSample {
                position_m: [0.5, 0.0, 0.5],
                target_size_m: 0.25,
                reason: Some("structural.constraint_regions".to_string()),
            },
        ],
        ..MeshSizingField::default()
    });
    case
}

fn benchmark_case(
    benchmark_id: &str,
    tier: MeshBenchmarkTier,
    geometry: GeometryAsset,
    expected_volume_m3: f64,
    expected_boundary_area_m2: f64,
    max_volume_component_count: usize,
) -> MeshBenchmarkCase {
    let characteristic_size = expected_volume_m3.cbrt() / 2.0;
    let options = VolumeMeshingOptions {
        target_size: MeshTargetSize::LengthM(characteristic_size.max(0.02)),
        max_elements: GENERIC_BENCHMARK_MAX_ELEMENTS,
        ..VolumeMeshingOptions::default()
    };
    MeshBenchmarkCase {
        benchmark_id: benchmark_id.to_string(),
        tier,
        geometry,
        options,
        sizing: None,
        validation: AnalysisMeshValidationOptions {
            max_volume_element_count: Some(GENERIC_BENCHMARK_MAX_ELEMENTS),
            expected_volume_m3: Some(expected_volume_m3),
            expected_boundary_area_m2: Some(expected_boundary_area_m2),
            max_volume_component_count: Some(max_volume_component_count),
            min_boundary_face_recovery_ratio: 1.0,
            min_boundary_edge_recovery_ratio: 1.0,
            require_no_fan_fallback: true,
            require_no_unrepaired_exact_quality: true,
            ..AnalysisMeshValidationOptions::default()
        },
    }
}
