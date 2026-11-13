//! Independently compare the canonical join to borrowed originals. Validation
//! indexes node references; it never reconstructs a second rich surface mesh.
use super::*;
use runmat_geometry_core::PersistentEntityId;

#[derive(Clone, Copy)]
pub(in crate::surface_mesh) struct Batch<'a> {
    pub schema_version: u16,
    pub partition: &'a runmat_meshing_core::MeshingPartitionDescriptor,
    pub faces: &'a [crate::ExactFaceMesh],
}

pub(in crate::surface_mesh) fn validate(
    result: &ExactSurfaceMesh,
    topology: &ExactBRepTopology,
    mut batches: Vec<Batch<'_>>,
    options: ExactSurfaceJoinOptions,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceMeshError> {
    checkpoint(control)?;
    validate_options(options)?;
    if batches.is_empty() || batches.len() > MAX_EXACT_FACE_PARTITIONS {
        return Err(invalid(
            "surface validation requires a bounded nonempty partition set",
        ));
    }
    batches.sort_by_key(|batch| batch.partition.partition_index);
    for (index, batch) in batches.iter().enumerate() {
        checkpoint(control)?;
        super::super::batch::validate_exact_face_mesh_batch_parts_with_cancellation(
            batch.schema_version,
            batch.partition,
            batch.faces,
            topology,
            control,
        )?;
        if batch.partition.partition_index != index as u32
            || batch.partition.partition_count != batches.len() as u32
        {
            return Err(invalid(
                "surface partitions do not form one complete canonical set",
            ));
        }
    }
    topology
        .validate_solid_shell_boundaries()
        .map_err(|error| invalid(error.to_string()))?;
    let faces = || batches.iter().flat_map(|batch| batch.faces);
    if result.nodes.len() as u64 > options.maximum_nodes
        || result.triangles.len() as u64 > options.maximum_triangles
        || result.boundary_segments.len() as u64 > options.maximum_boundary_segments
    {
        return Err(limit(
            "surface join exceeds its original entity hard limits",
        ));
    }
    if faces().count() != topology.faces.len()
        || !faces()
            .map(|face| &face.source_face_id)
            .eq(topology.faces.iter().map(|face| &face.id))
        || !result
            .face_ids
            .iter()
            .eq(topology.faces.iter().map(|face| &face.id))
        || result.schema_version != EXACT_SURFACE_MESH_SCHEMA_VERSION
    {
        return Err(invalid("surface schema or canonical face coverage differs"));
    }
    let mut chord = 0.0f64;
    let mut normal = 0.0f64;
    let mut triangles = result.triangles.iter();
    let mut boundaries = result.boundary_segments.iter();
    let count = faces()
        .try_fold(0usize, |sum, face| sum.checked_add(face.nodes.len()))
        .ok_or_else(|| limit("surface node reference inventory overflow"))?;
    let mut nodes = Vec::<(&PersistentEntityId, &ExactFaceMeshNode)>::with_capacity(count);
    // A compact borrowed identity inventory replaces the rich cloned expected mesh.
    let mut triangle_ids = Vec::with_capacity(result.triangles.len());
    for face in faces() {
        checkpoint(control)?;
        chord = chord.max(face.maximum_chordal_deviation_m.canonical_value());
        normal = normal.max(face.maximum_normal_deviation_rad);
        for node in &face.nodes {
            checkpoint(control)?;
            nodes.push((&face.source_face_id, node));
        }
        for triangle in &face.triangles {
            checkpoint(control)?;
            if triangles.next() != Some(triangle) {
                return Err(invalid("surface triangle differs from its original face"));
            }
            triangle_ids.push(triangle.triangle_id);
        }
        for boundary in &face.boundary_segments {
            checkpoint(control)?;
            if boundaries.next() != Some(boundary) {
                return Err(invalid("surface boundary differs from its original face"));
            }
        }
    }
    if triangles.next().is_some()
        || boundaries.next().is_some()
        || chord != result.maximum_chordal_deviation_m.canonical_value()
        || normal != result.maximum_normal_deviation_rad
    {
        return Err(invalid(
            "surface inventory or deviation bounds differ from original faces",
        ));
    }
    runmat_cpu_execution::sort_in_place_with_checkpoint(&mut triangle_ids, Ord::cmp, || {
        checkpoint(control)
    })?;
    for pair in triangle_ids.windows(2) {
        checkpoint(control)?;
        if pair[0] == pair[1] {
            return Err(invalid(
                "surface join contains a duplicate triangle identity",
            ));
        }
    }
    drop(triangle_ids);
    runmat_cpu_execution::sort_in_place_with_checkpoint(
        &mut nodes,
        |a, b| (a.1.node_id, a.0).cmp(&(b.1.node_id, b.0)),
        || checkpoint(control),
    )?;
    let tolerances = topology
        .edges
        .iter()
        .map(|edge| (&edge.id, edge.tolerance_m))
        .collect::<BTreeMap<_, _>>();
    let mut joined = result.nodes.iter();
    let mut start = 0;
    while start < nodes.len() {
        checkpoint(control)?;
        let first = nodes[start].1;
        let mut end = start + 1;
        while end < nodes.len() && nodes[end].1.node_id == first.node_id {
            checkpoint(control)?;
            end += 1;
        }
        let actual = joined
            .next()
            .ok_or_else(|| invalid("surface omits an original node"))?;
        if actual.node_id != first.node_id || actual.point_m != first.point_m {
            return Err(invalid(
                "surface node identity or original coordinate differs",
            ));
        }
        let mut uses = actual.uses.iter();
        let mut tolerance = options.coordinate_tolerance_m.canonical_value();
        for (_, node) in &nodes[start..end] {
            checkpoint(control)?;
            tolerance = node
                .uses
                .iter()
                .flat_map(|usage| &usage.exact_edge_parameters)
                .filter_map(|parameter| tolerances.get(&parameter.source_edge_id))
                .fold(tolerance, |value, local| value.max(local.canonical_value()));
            if distance(first.point_m, node.point_m) > tolerance {
                return Err(invalid(
                    "original shared node coordinates exceed the admitted tolerance",
                ));
            }
            for usage in &node.uses {
                checkpoint(control)?;
                if uses.next() != Some(usage) {
                    return Err(invalid("surface node use differs from its original face"));
                }
            }
        }
        if uses.next().is_some() {
            return Err(invalid("surface has extra node uses"));
        }
        if actual.uses.len() > 1 {
            let mut use_ids = BTreeSet::new();
            for usage in &actual.uses {
                checkpoint(control)?;
                let uv = usage
                    .uv
                    .map(|value| if value == 0.0 { 0 } else { value.to_bits() });
                if !use_ids.insert((&usage.source_face_id, usage.chart_id, uv)) {
                    return Err(invalid("surface join contains a duplicate node use"));
                }
            }
        }
        start = end;
    }
    if joined.next().is_some() {
        return Err(invalid("surface has extra joined nodes"));
    }
    validate_edge_conformity_with_control(topology, &result.boundary_segments, control)?;
    if result.shells != shell_evidence_with_control(topology, control)? {
        return Err(invalid("surface shell evidence differs from topology"));
    }
    Ok(())
}
