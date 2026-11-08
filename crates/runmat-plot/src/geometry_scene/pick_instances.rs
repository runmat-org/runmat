//! Face picks on instanced chunks: one BVH per prototype in its own coordinates, tested per
//! placement with the ray moved into those coordinates (placements are rigid, so distances
//! carry over). The result names the placement's region, as an ordinary chunk's region would.
use super::pick::{ray_intersects_bounds, PickRay};
use super::*;

#[derive(Debug, Clone)]
pub(super) struct InstancedPick {
    local: GeometryScenePickIndex,
    chunk_id: String,
    mesh_id: Option<String>,
    placements: Vec<Placement>,
}

#[derive(Debug, Clone)]
struct Placement {
    forward: Mat4,
    inverse: Mat4,
    bounds: BoundingBox,
    region_id: String,
    label: Option<String>,
    tag: Option<String>,
}

/// The pick sets of a scene's visible instanced triangle chunks. Of a prototype drawn at
/// several levels of detail only the finest level (a band open above) is picked.
pub(super) fn build(scene: &GeometryScene) -> Vec<InstancedPick> {
    scene
        .chunks
        .iter()
        .filter(|c| {
            c.visible
                && c.is_instanced()
                && c.render_data.pipeline_type == PipelineType::Triangles
                && !c.is_coarse_level()
        })
        .map(|chunk| {
            let mut local = chunk.clone();
            local.instances = Vec::new();
            local.lod_pixels = None;
            local.bounds = bounds_from_vertices(&local.vertices);
            let local_bounds = local.bounds;
            let index = GeometryScenePickIndex::build(&GeometryScene::new("instanced", 0, vec![local]));
            let placements = chunk
                .instances
                .iter()
                .map(|instance| {
                    let forward = instance.raw().matrix();
                    let corners: [Vec3; 8] = std::array::from_fn(|k| {
                        forward.transform_point3(Vec3::new(
                            if k & 1 == 0 { local_bounds.min.x } else { local_bounds.max.x },
                            if k & 2 == 0 { local_bounds.min.y } else { local_bounds.max.y },
                            if k & 4 == 0 { local_bounds.min.z } else { local_bounds.max.z },
                        ))
                    });
                    Placement {
                        forward,
                        inverse: forward.inverse(),
                        bounds: BoundingBox::from_points(&corners),
                        region_id: instance.region_id.clone(),
                        label: instance.label.clone(),
                        tag: instance.tag.clone(),
                    }
                })
                .collect();
            InstancedPick {
                local: index,
                chunk_id: chunk.chunk_id.clone(),
                mesh_id: chunk.mesh_id.clone(),
                placements,
            }
        })
        .collect()
}

/// The owner nodes the sets' prototype triangles name.
pub(super) fn owner_ids(sets: &[InstancedPick]) -> impl Iterator<Item = &String> {
    sets.iter().flat_map(|set| set.local.owner_ids())
}

/// The view's hidden regions and owner nodes: a placement whose region is hidden, or a
/// prototype triangle whose region or owner node is hidden, is passed through.
pub(super) fn set_hidden(sets: &mut [InstancedPick], regions: &HashSet<String>, owners: &HashSet<String>) {
    for set in sets {
        set.local.set_hidden_sets(regions.clone(), owners.clone());
    }
}

/// The nearest placement the ray hits closer than `nearer_than`, as a pick result.
pub(super) fn pick(sets: &[InstancedPick], ray: &PickRay, nearer_than: f32) -> Option<GeometryScenePickResult> {
    let mut best: Option<(&InstancedPick, &Placement, usize, f32, PickRay)> = None;
    let mut limit = nearer_than;
    for set in sets {
        for placement in &set.placements {
            if set.local.region_hidden(&placement.region_id) {
                continue;
            }
            let Some(enter) = ray_intersects_bounds(ray, placement.bounds) else {
                continue;
            };
            if enter > limit {
                continue;
            }
            let local = PickRay {
                origin: placement.inverse.transform_point3(ray.origin),
                direction: placement.inverse.transform_vector3(ray.direction).normalize_or_zero(),
            };
            if let Some((triangle, distance)) = set.local.hit_ray(&local) {
                if distance < limit {
                    limit = distance;
                    best = Some((set, placement, triangle, distance, local));
                }
            }
        }
    }
    let (set, placement, triangle, distance, local) = best?;
    let local_position = local.origin + local.direction * distance;
    let mut result = set.local.result_for_triangle(triangle, local_position, distance, None)?;
    result.position = placement.forward.transform_point3(local_position).to_array();
    result.chunk_id = set.chunk_id.clone();
    result.mesh_id = set.mesh_id.clone();
    result.region_id = Some(placement.region_id.clone());
    result.region_label = placement.label.clone();
    result.region_tag = placement.tag.clone();
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pick_names_the_placement_it_hits() {
        // A unit square in the xy plane, placed at x = 0 and x = 10.
        let vertices = vec![
            vertex([0.0, 0.0, 0.0], [0.5; 4], [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], [0.5; 4], [0.0, 0.0, 1.0]),
            vertex([1.0, 1.0, 0.0], [0.5; 4], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.5; 4], [0.0, 0.0, 1.0]),
        ];
        let placed = |x: f32, id: &str| GeometrySceneInstance {
            transform: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, x, 0.0, 0.0],
            region_id: id.into(),
            label: Some(id.into()),
            tag: Some("occurrence".into()),
        };
        let chunk = GeometrySceneChunk::indexed_triangles("square", vertices, vec![0, 1, 2, 0, 2, 3], cad_default_material())
            .with_instances(vec![placed(0.0, "occ:0"), placed(10.0, "occ:1")], None);
        let scene = GeometryScene::new("s", 1, vec![chunk]);
        let index = GeometryScenePickIndex::build(&scene);
        let ray = PickRay { origin: Vec3::new(10.5, 0.5, 5.0), direction: Vec3::NEG_Z };
        let hit = pick(&index.instanced_sets(), &ray, f32::INFINITY).expect("hits the second square");
        assert_eq!(hit.region_id.as_deref(), Some("occ:1"));
        assert!((hit.position[0] - 10.5).abs() < 1e-5 && (hit.distance - 5.0).abs() < 1e-5, "{hit:?}");
        let miss = PickRay { origin: Vec3::new(5.0, 0.5, 5.0), direction: Vec3::NEG_Z };
        assert!(pick(&index.instanced_sets(), &miss, f32::INFINITY).is_none());
    }

    /// A hidden placement, or a hidden owner node of the prototype, lets the click through.
    #[test]
    fn a_pick_passes_through_hidden_placements_and_owners() {
        let vertices = vec![
            vertex([0.0, 0.0, 0.0], [0.5; 4], [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], [0.5; 4], [0.0, 0.0, 1.0]),
            vertex([1.0, 1.0, 0.0], [0.5; 4], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.5; 4], [0.0, 0.0, 1.0]),
        ];
        // Two squares stacked along z under the ray: occ:0 at z = 1 (nearer), occ:1 at z = 0.
        let placed = |z: f32, id: &str| GeometrySceneInstance {
            transform: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, z],
            region_id: id.into(),
            label: Some(id.into()),
            tag: Some("occurrence".into()),
        };
        let mut chunk = GeometrySceneChunk::indexed_triangles("square", vertices, vec![0, 1, 2, 0, 2, 3], cad_default_material())
            .with_instances(vec![placed(1.0, "occ:0"), placed(0.0, "occ:1")], None);
        chunk.owner_node_ids = vec!["part:square".into()];
        let scene = GeometryScene::new("s", 1, vec![chunk]);
        let mut index = GeometryScenePickIndex::build(&scene);
        let ray = PickRay { origin: Vec3::new(0.5, 0.5, 5.0), direction: Vec3::NEG_Z };
        assert_eq!(pick(index.instanced_sets(), &ray, f32::INFINITY).and_then(|h| h.region_id).as_deref(), Some("occ:0"));

        index.set_hidden(&GeometryScenePresentation { hidden_region_ids: vec!["occ:0".into()], ..Default::default() });
        assert_eq!(pick(index.instanced_sets(), &ray, f32::INFINITY).and_then(|h| h.region_id).as_deref(), Some("occ:1"));

        index.set_hidden(&GeometryScenePresentation { hidden_owner_node_ids: Some(vec!["part:square".into()]), ..Default::default() });
        assert!(pick(index.instanced_sets(), &ray, f32::INFINITY).is_none());

        index.set_hidden(&GeometryScenePresentation::default());
        assert_eq!(pick(index.instanced_sets(), &ray, f32::INFINITY).and_then(|h| h.region_id).as_deref(), Some("occ:0"));
    }
}
