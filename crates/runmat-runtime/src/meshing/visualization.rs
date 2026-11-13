use std::collections::{BTreeSet, HashMap};

use runmat_meshing_core::SolverMeshArtifact;
use runmat_plot::plots::{Figure, MeshEdgeMode, MeshPlot};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SolverMeshFigureOptions {
    pub show_edges: bool,
    pub regions: Option<BTreeSet<String>>,
    pub section_plane: Option<[f64; 4]>,
}

pub fn solver_mesh_figure(
    mesh: &SolverMeshArtifact,
    title: impl Into<String>,
    options: SolverMeshFigureOptions,
) -> Result<Figure, String> {
    validate_selected_regions(mesh, options.regions.as_ref())?;
    let (vertices, triangles) = boundary_surface(mesh, &options)?;
    let mut plot = MeshPlot::new(vertices, triangles)?;
    plot.set_mesh_id(Some("solver_mesh_boundary".to_string()));
    plot.set_label(Some(format!(
        "{} tetrahedra, {} boundary faces",
        mesh.topology.volume_elements.len(),
        mesh.topology.boundary_faces.len()
    )));
    configure_edges(&mut plot, options.show_edges, 0.55);

    let mut figure = Figure::new()
        .with_title(title)
        .with_labels("X", "Y")
        .with_grid(true)
        .with_axis_equal(true);
    figure.z_label = Some("Z".to_string());
    figure.add_mesh_plot(plot);
    if let Some(plane) = options.section_plane {
        let (section_vertices, section_triangles) =
            section_cap(mesh, plane, options.regions.as_ref())?;
        if !section_triangles.is_empty() {
            let mut section = MeshPlot::new(section_vertices, section_triangles)?;
            section.set_mesh_id(Some("solver_mesh_section".to_string()));
            section.set_label(Some("Section plane".to_string()));
            section.set_face_color(glam::Vec4::new(0.94, 0.42, 0.16, 1.0));
            configure_edges(&mut section, options.show_edges, 0.7);
            figure.add_mesh_plot(section);
        }
    }
    Ok(figure)
}

fn boundary_surface(
    mesh: &SolverMeshArtifact,
    options: &SolverMeshFigureOptions,
) -> Result<(Vec<glam::Vec3>, Vec<[u32; 3]>), String> {
    let nodes = mesh
        .topology
        .nodes
        .iter()
        .map(|node| {
            Ok((
                node.node_id,
                glam::Vec3::new(
                    finite_f32(node.coordinates_m.canonical_value()[0])?,
                    finite_f32(node.coordinates_m.canonical_value()[1])?,
                    finite_f32(node.coordinates_m.canonical_value()[2])?,
                ),
            ))
        })
        .collect::<Result<HashMap<_, _>, String>>()?;
    let element_regions = mesh
        .topology
        .volume_elements
        .iter()
        .map(|element| {
            (
                element.element_id,
                element.region_id.source_topology_id.as_str(),
            )
        })
        .collect::<HashMap<_, _>>();
    let selected_faces = mesh.topology.boundary_faces.iter().filter(|face| {
        options.regions.as_ref().is_none_or(|regions| {
            face.adjacent_volume_element_ids.iter().any(|element_id| {
                element_regions
                    .get(element_id)
                    .is_some_and(|region| regions.contains(*region))
            })
        })
    });
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for face in selected_faces {
        if face.node_ids.len() < 3 {
            return Err("solver mesh boundary face has fewer than three nodes".to_string());
        }
        let points = face.node_ids[..3]
            .iter()
            .map(|node_id| {
                nodes
                    .get(node_id)
                    .copied()
                    .ok_or_else(|| "solver mesh boundary references a missing node".to_string())
            })
            .collect::<Result<Vec<_>, String>>()?;
        let polygon = match options.section_plane {
            Some(plane) => clip_triangle_to_plane(&points, plane),
            None => points,
        };
        if polygon.len() < 3 {
            continue;
        }
        let base = u32::try_from(vertices.len())
            .map_err(|_| "solver mesh visualization exceeds u32 vertex capacity".to_string())?;
        vertices.extend(polygon.iter().copied());
        for index in 1..polygon.len() - 1 {
            triangles.push([base, base + index as u32, base + index as u32 + 1]);
        }
    }
    Ok((vertices, triangles))
}

fn clip_triangle_to_plane(points: &[glam::Vec3], plane: [f64; 4]) -> Vec<glam::Vec3> {
    let mut clipped = Vec::with_capacity(4);
    for index in 0..points.len() {
        let start = points[index];
        let end = points[(index + 1) % points.len()];
        let start_distance = plane_distance(plane, start);
        let end_distance = plane_distance(plane, end);
        let start_inside = start_distance <= 0.0;
        let end_inside = end_distance <= 0.0;
        match (start_inside, end_inside) {
            (true, true) => clipped.push(end),
            (true, false) => {
                clipped.push(plane_intersection(start, end, start_distance, end_distance))
            }
            (false, true) => {
                clipped.push(plane_intersection(start, end, start_distance, end_distance));
                clipped.push(end);
            }
            (false, false) => {}
        }
    }
    clipped
}

fn plane_intersection(
    start: glam::Vec3,
    end: glam::Vec3,
    start_distance: f64,
    end_distance: f64,
) -> glam::Vec3 {
    let fraction = (start_distance / (start_distance - end_distance)) as f32;
    start + fraction * (end - start)
}

fn validate_selected_regions(
    mesh: &SolverMeshArtifact,
    selected: Option<&BTreeSet<String>>,
) -> Result<(), String> {
    let Some(selected) = selected else {
        return Ok(());
    };
    let available = mesh
        .topology
        .regions
        .iter()
        .map(|region| region.region_id.source_topology_id.as_str())
        .collect::<BTreeSet<_>>();
    if let Some(unknown) = selected
        .iter()
        .find(|region| !available.contains(region.as_str()))
    {
        return Err(format!("solver mesh does not contain region `{unknown}`"));
    }
    Ok(())
}

fn configure_edges(plot: &mut MeshPlot, show: bool, width: f32) {
    plot.set_edge_mode(if show {
        MeshEdgeMode::All
    } else {
        MeshEdgeMode::None
    });
    plot.set_edge_width(if show { width } else { 0.0 });
}

fn section_cap(
    mesh: &SolverMeshArtifact,
    plane: [f64; 4],
    regions: Option<&BTreeSet<String>>,
) -> Result<(Vec<glam::Vec3>, Vec<[u32; 3]>), String> {
    let normal = glam::DVec3::new(plane[0], plane[1], plane[2]);
    if !normal.is_finite() || normal.length_squared() <= f64::EPSILON || !plane[3].is_finite() {
        return Err(
            "section plane must contain a finite nonzero normal and finite offset".to_string(),
        );
    }
    let nodes = mesh
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m.canonical_value()))
        .collect::<HashMap<_, _>>();
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for element in &mesh.topology.volume_elements {
        if regions.is_some_and(|selected| !selected.contains(&element.region_id.source_topology_id))
        {
            continue;
        }
        let corners = element
            .node_ids
            .get(..4)
            .ok_or_else(|| "tetrahedron has fewer than four corner nodes".to_string())?;
        let points = corners
            .iter()
            .map(|id| {
                nodes
                    .get(id)
                    .copied()
                    .ok_or_else(|| "tetrahedron references a missing node".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let distances = points
            .iter()
            .map(|point| plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3])
            .collect::<Vec<_>>();
        let mut intersections = Vec::<[f64; 3]>::new();
        for (a, b) in EDGES {
            let da = distances[a];
            let db = distances[b];
            if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
                let t = da / (da - db);
                intersections.push(std::array::from_fn(|axis| {
                    points[a][axis] + t * (points[b][axis] - points[a][axis])
                }));
            } else if da.abs() <= 1.0e-12 {
                intersections.push(points[a]);
            } else if db.abs() <= 1.0e-12 {
                intersections.push(points[b]);
            }
        }
        intersections.sort_by(|left, right| {
            left.iter()
                .zip(right)
                .find_map(|(a, b)| a.partial_cmp(b).filter(|order| !order.is_eq()))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        intersections.dedup_by(|left, right| {
            left.iter()
                .zip(right.iter())
                .all(|(a, b)| (*a - *b).abs() <= 1.0e-10)
        });
        if intersections.len() < 3 {
            continue;
        }
        let center = intersections
            .iter()
            .fold([0.0; 3], |mut sum, point| {
                for axis in 0..3 {
                    sum[axis] += point[axis];
                }
                sum
            })
            .map(|value| value / intersections.len() as f64);
        let unit = normal.normalize();
        let reference = if unit.x.abs() < 0.9 {
            glam::DVec3::X
        } else {
            glam::DVec3::Y
        };
        let basis_u = unit.cross(reference).normalize();
        let basis_v = unit.cross(basis_u);
        intersections.sort_by(|left, right| {
            let angle = |point: &[f64; 3]| {
                let delta = glam::DVec3::from_array(*point) - glam::DVec3::from_array(center);
                delta.dot(basis_v).atan2(delta.dot(basis_u))
            };
            angle(left).total_cmp(&angle(right))
        });
        let base = u32::try_from(vertices.len())
            .map_err(|_| "section visualization exceeds u32 vertex capacity".to_string())?;
        vertices.extend(
            intersections
                .iter()
                .map(|point| {
                    Ok(glam::Vec3::new(
                        finite_f32(point[0])?,
                        finite_f32(point[1])?,
                        finite_f32(point[2])?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?,
        );
        for index in 1..intersections.len() - 1 {
            triangles.push([base, base + index as u32, base + index as u32 + 1]);
        }
    }
    Ok((vertices, triangles))
}

fn plane_distance(plane: [f64; 4], point: glam::Vec3) -> f64 {
    plane[0] * f64::from(point.x)
        + plane[1] * f64::from(point.y)
        + plane[2] * f64::from(point.z)
        + plane[3]
}

fn finite_f32(value: f64) -> Result<f32, String> {
    if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
        return Err("solver mesh contains a non-renderable coordinate".to_string());
    }
    Ok(value as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_clipping_keeps_boundary_triangles_inside_the_plane() {
        let points = [
            glam::Vec3::new(-1.0, 0.0, 0.0),
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(-1.0, 1.0, 0.0),
        ];

        let clipped = clip_triangle_to_plane(&points, [1.0, 0.0, 0.0, 0.0]);

        assert_eq!(clipped.len(), 4);
        assert!(clipped
            .iter()
            .all(|point| plane_distance([1.0, 0.0, 0.0, 0.0], *point) <= 1.0e-7));
    }
}
