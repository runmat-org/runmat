#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MissingFaceLink {
    #[cfg(test)]
    Edge,
    Node,
}

#[cfg(test)]
pub(super) fn missing_face_component_sizes(
    faces: &[[u32; 3]],
    link: MissingFaceLink,
) -> Vec<usize> {
    missing_face_components(faces, link)
        .into_iter()
        .map(|component| component.len())
        .collect()
}

pub(super) fn missing_face_components(
    faces: &[[u32; 3]],
    link: MissingFaceLink,
) -> Vec<Vec<usize>> {
    let mut visited = BTreeSet::<usize>::new();
    let mut components = Vec::<Vec<usize>>::new();
    for start in 0..faces.len() {
        if !visited.insert(start) {
            continue;
        }
        let mut component = Vec::<usize>::new();
        let mut pending = vec![start];
        while let Some(index) = pending.pop() {
            component.push(index);
            for neighbor in 0..faces.len() {
                if visited.contains(&neighbor)
                    || !missing_faces_connected(faces[index], faces[neighbor], link)
                {
                    continue;
                }
                visited.insert(neighbor);
                pending.push(neighbor);
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components.sort();
    components
}

#[cfg(test)]
pub(super) fn missing_face_component_common_node_ids(
    faces: &[[u32; 3]],
    component: &[usize],
) -> Vec<u32> {
    let Some(first) = component.first() else {
        return Vec::new();
    };
    let mut common = faces[*first].into_iter().collect::<BTreeSet<_>>();
    for index in component.iter().skip(1) {
        let face_nodes = faces[*index].into_iter().collect::<BTreeSet<_>>();
        common.retain(|node_id| face_nodes.contains(node_id));
    }
    common.into_iter().collect()
}

fn missing_faces_connected(left: [u32; 3], right: [u32; 3], link: MissingFaceLink) -> bool {
    if left == right {
        return true;
    }
    let shared_count = left
        .into_iter()
        .filter(|node_id| right.contains(node_id))
        .count();
    match link {
        #[cfg(test)]
        MissingFaceLink::Edge => shared_count >= 2,
        MissingFaceLink::Node => shared_count >= 1,
    }
}

#[cfg(test)]
pub(super) fn component_size_histogram(sizes: Vec<usize>) -> BTreeMap<usize, usize> {
    let mut histogram = BTreeMap::<usize, usize>::new();
    for size in sizes {
        *histogram.entry(size).or_default() += 1;
    }
    histogram
}
