use super::ProviderMatmulGeometry;

#[test]
fn nondefault_recipe_counts_matrix_tiles_and_diagonal_lanes_independently() {
    let geometry = ProviderMatmulGeometry {
        tile_size: 8,
        diagonal_workgroup_size: 64,
    };
    assert_eq!(geometry.dispatch_groups(129, 65, true), Some([9, 17, 2]));
    assert_eq!(geometry.dispatch_groups(129, 65, false), Some([9, 17, 0]));
    assert_eq!(geometry.dispatch_groups(0, 0, true), Some([0, 0, 0]));
    assert_eq!(geometry.dispatch_groups(u64::MAX, 1, false), None);
    assert_eq!(
        ProviderMatmulGeometry {
            tile_size: 0,
            ..geometry
        }
        .dispatch_groups(1, 1, false),
        None
    );
    assert_eq!(
        ProviderMatmulGeometry {
            diagonal_workgroup_size: 0,
            ..geometry
        }
        .dispatch_groups(1, 1, true),
        None
    );
    let boundary = 65_535 * 64;
    assert_eq!(
        geometry.dispatch_groups(boundary, boundary, true).unwrap()[2],
        65_535
    );
    assert_eq!(
        geometry
            .dispatch_groups(boundary + 1, boundary + 1, true)
            .unwrap()[2],
        65_536
    );
}
