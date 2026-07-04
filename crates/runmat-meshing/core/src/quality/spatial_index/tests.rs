use super::*;

#[test]
fn aabb_contains_and_intersects_points() {
    let bounds = Aabb3::from_points(&[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]).unwrap();

    assert!(bounds.contains_point([0.5, 1.0, 2.0]));
    assert!(!bounds.contains_point([2.0, 1.0, 2.0]));
    assert!(bounds.intersects(Aabb3::from_points(&[[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]]).unwrap()));
    assert!(!bounds.intersects(Aabb3::from_points(&[[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]).unwrap()));
}

#[test]
fn linear_spatial_index_queries_deterministically() {
    let mut index = LinearSpatialIndex::new();
    index.insert(
        Aabb3::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).unwrap(),
        "left",
    );
    index.insert(
        Aabb3::from_points(&[[2.0, 0.0, 0.0], [3.0, 1.0, 1.0]]).unwrap(),
        "right",
    );

    let hits = index
        .query_point([0.5, 0.5, 0.5])
        .map(|entry| entry.payload)
        .collect::<Vec<_>>();
    assert_eq!(hits, vec!["left"]);
    assert_eq!(
        index
            .nearest_by_center([2.8, 0.5, 0.5])
            .map(|entry| entry.payload),
        Some("right")
    );
}

#[test]
fn uniform_grid_spatial_index_filters_point_and_bounds_queries() {
    let entries = vec![
        SpatialEntry {
            bounds: Aabb3::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).unwrap(),
            payload: "left",
        },
        SpatialEntry {
            bounds: Aabb3::from_points(&[[2.0, 0.0, 0.0], [3.0, 1.0, 1.0]]).unwrap(),
            payload: "right",
        },
    ];
    let index = UniformGridSpatialIndex::from_entries(entries);

    let point_hits = index
        .query_point([0.5, 0.5, 0.5])
        .into_iter()
        .map(|entry| entry.payload)
        .collect::<Vec<_>>();
    assert_eq!(point_hits, vec!["left"]);
    let bounds_hits = index
        .query_bounds(Aabb3::from_points(&[[2.5, 0.5, 0.5], [4.0, 0.6, 0.6]]).unwrap())
        .into_iter()
        .map(|entry| entry.payload)
        .collect::<Vec<_>>();
    assert_eq!(bounds_hits, vec!["right"]);
}

#[test]
fn uniform_grid_spatial_index_traverses_ray_cells() {
    let entries = vec![
        SpatialEntry {
            bounds: Aabb3::from_points(&[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]).unwrap(),
            payload: "near",
        },
        SpatialEntry {
            bounds: Aabb3::from_points(&[[0.0, 1.0, 0.0], [0.25, 1.25, 0.25]]).unwrap(),
            payload: "off-axis",
        },
        SpatialEntry {
            bounds: Aabb3::from_points(&[[2.0, 0.0, 0.0], [2.25, 0.25, 0.25]]).unwrap(),
            payload: "far",
        },
    ];
    let index = UniformGridSpatialIndex::from_entries(entries);

    let hits = index
        .query_ray([-1.0, 0.1, 0.1], [1.0, 0.0, 0.0])
        .into_iter()
        .map(|entry| entry.payload)
        .collect::<Vec<_>>();

    assert_eq!(hits, vec!["near", "far"]);
}

#[test]
fn uniform_grid_spatial_index_rejects_parallel_ray_outside_bounds() {
    let index = UniformGridSpatialIndex::from_entries(vec![SpatialEntry {
        bounds: Aabb3::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).unwrap(),
        payload: "inside",
    }]);

    assert!(index
        .query_ray([-1.0, 2.0, 0.5], [1.0, 0.0, 0.0])
        .is_empty());
}
