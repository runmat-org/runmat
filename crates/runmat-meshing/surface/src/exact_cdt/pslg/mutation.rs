use super::{ExactFacePslg, ExactFacePslgVertex, MAX_FACE_PSLG_ITEMS};

pub(crate) fn insert_pslg_vertices(
    pslg: &ExactFacePslg,
    additions: &[ExactFacePslgVertex],
) -> Result<(ExactFacePslg, Vec<u32>), &'static str> {
    if additions.is_empty()
        || pslg.vertices.len().saturating_add(additions.len()) > MAX_FACE_PSLG_ITEMS
        || pslg.vertices.len().saturating_add(additions.len()) > u32::MAX as usize
        || additions
            .iter()
            .flat_map(|vertex| vertex.uv)
            .any(|coordinate| !coordinate.is_finite())
    {
        return Err("face vertex additions are empty, invalid, or exceed the hard bound");
    }
    let old_count = pslg.vertices.len();
    let mut indexed = pslg
        .vertices
        .iter()
        .copied()
        .chain(additions.iter().copied())
        .enumerate()
        .collect::<Vec<_>>();
    indexed.sort_by(|(_, left), (_, right)| compare_vertex(*left, *right));
    if indexed
        .windows(2)
        .any(|pair| compare_vertex(pair[0].1, pair[1].1).is_eq())
    {
        return Err("face vertex addition duplicates an existing chart-local vertex");
    }
    let mut remap = vec![0u32; indexed.len()];
    let vertices = indexed
        .into_iter()
        .enumerate()
        .map(|(new_index, (old_index, vertex))| {
            remap[old_index] = new_index as u32;
            vertex
        })
        .collect();
    let mut updated = pslg.clone();
    updated.vertices = vertices;
    for segment in &mut updated.segments {
        segment.vertex_indices = segment.vertex_indices.map(|index| remap[index as usize]);
    }
    let inserted = (old_count..old_count + additions.len())
        .map(|index| remap[index])
        .collect();
    Ok((updated, inserted))
}

fn compare_vertex(left: ExactFacePslgVertex, right: ExactFacePslgVertex) -> std::cmp::Ordering {
    left.node_id
        .cmp(&right.node_id)
        .then_with(|| left.seam_image.cmp(&right.seam_image))
        .then_with(|| left.uv[0].total_cmp(&right.uv[0]))
        .then_with(|| left.uv[1].total_cmp(&right.uv[1]))
}
