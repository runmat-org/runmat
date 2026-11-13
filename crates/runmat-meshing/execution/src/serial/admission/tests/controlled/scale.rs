//! Analytic disconnected tetrahedra qualify validator storage and topology checks,
//! not native CAD meshing, thermal accuracy or whole-task RSS.
use super::*;
use runmat_geometry_core::{FacetedSolid, FacetedSolidModel, GeometryModel};
fn components(count: usize) -> (FacetedSolid, FacetedSolidModel) {
    let (document, seed) = runmat_geometry_fixtures::faceted_tetrahedron();
    let GeometryModel::FacetedSolid { mut model } = document.model else {
        unreachable!()
    };
    let mut solid = seed.clone();
    solid.vertices.clear();
    solid.triangles.clear();
    solid.shells.clear();
    for component in 0..count {
        let mut part = seed.clone();
        for vertex in &mut part.vertices {
            vertex.id.assembly_path.push(component.to_string());
            {
                let quantity_change = 2.0 * component as f64;
                let quantity_index = 0;
                let quantity_slot = &mut vertex.coordinates_m;
                let mut quantity_coordinates = quantity_slot.canonical_value();
                quantity_coordinates[quantity_index] += quantity_change;
                *quantity_slot = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(quantity_coordinates);
            };
        }
        for triangle in &mut part.triangles {
            triangle.id.assembly_path.push(component.to_string());
            triangle.shell_id.assembly_path.push(component.to_string());
            for index in &mut triangle.vertex_indices {
                *index += solid.vertices.len() as u32;
            }
        }
        for shell in &mut part.shells {
            shell.id.assembly_path.push(component.to_string());
            for index in &mut shell.triangle_indices {
                *index += solid.triangles.len() as u32;
            }
        }
        solid.vertices.extend(part.vertices);
        solid.triangles.extend(part.triangles);
        solid.shells.extend(part.shells);
    }
    model.vertex_count = solid.vertices.len() as u64;
    model.triangle_count = solid.triangles.len() as u64;
    model.shell_count = solid.shells.len() as u64;
    (solid, model)
}
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Analytic disconnected tetrahedra at 1/16/256 components have linear requested
//   validation scratch, exact/short admission and complete release. Vertex/triangle/shell
//   permutations preserve valid topology. This excludes native CAD meshing, accuracy of a numerical
//   study and physical memory measurements.
#[test]
fn flat_validator_scratch_scales_linearly_releases_and_preserves_unordered_inputs() {
    let mut unit_peak = 0;
    for count in [1, 16, 256] {
        let (mut solid, model) = components(count);
        let control = bounded_control(64 << 20);
        solid
            .validate_against_with_control(&model, &control)
            .unwrap();
        assert_eq!(control.usage().live_allocation_bytes, 0);
        let peak = control.usage().allocation_bytes;
        if count == 1 {
            unit_peak = peak;
        }
        assert_eq!(peak, unit_peak * count as u64);
        assert!(peak <= 512 * count as u64);
        let exact = bounded_control(peak);
        solid.validate_against_with_control(&model, &exact).unwrap();
        let short = bounded_control(peak - 1);
        assert!(solid.validate_against_with_control(&model, &short).is_err());
        assert_eq!(short.usage().live_allocation_bytes, 0);
        solid.vertices.reverse();
        for triangle in &mut solid.triangles {
            for index in &mut triangle.vertex_indices {
                *index = model.vertex_count as u32 - 1 - *index;
            }
        }
        solid.triangles.reverse();
        for shell in &mut solid.shells {
            for index in &mut shell.triangle_indices {
                *index = model.triangle_count as u32 - 1 - *index;
            }
            shell.triangle_indices.sort_unstable();
        }
        solid.shells.reverse();
        solid
            .validate_against_with_control(&model, &control)
            .unwrap();
        assert_eq!(control.usage().live_allocation_bytes, 0);
    }
}
#[test]
fn flat_boundary_validator_rejects_disconnected_shells_and_excess_edge_uses() {
    let (mut solid, mut model) = components(2);
    let id = solid.shells[0].id.clone();
    for triangle in &mut solid.triangles {
        triangle.shell_id = id.clone();
    }
    solid.shells.truncate(1);
    solid.shells[0].triangle_indices = (0..8).collect();
    model.shell_count = 1;
    assert!(solid.validate_against(&model).is_err());
    let (mut solid, mut model) = components(1);
    let mut repeated = solid.triangles.clone();
    for triangle in &mut repeated {
        triangle.id.source_topology_id.push_str("-extra");
    }
    solid.triangles.extend(repeated);
    solid.shells[0].triangle_indices = (0..8).collect();
    model.triangle_count = 8;
    assert!(solid.validate_against(&model).is_err());
}
