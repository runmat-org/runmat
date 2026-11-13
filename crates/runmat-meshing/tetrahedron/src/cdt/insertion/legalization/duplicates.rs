//! Exact replacement membership through the admitted vertex-incidence inventory.
use super::*;

pub(super) fn replacement_duplicates_retained_tetrahedron(
    topology: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    removed: &[usize],
    created: &[[u32; 4]],
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    let mut created_keys = BTreeSet::new();
    for cell in created {
        work.checkpoint()?;
        let mut key = *cell;
        key.sort_unstable();
        if !created_keys.insert(key) {
            return Ok(true);
        }
        // Every matching cell belongs to all four vertex stars. Query the
        // smallest complete star; unrelated retained cells cannot match.
        let vertex = key
            .iter()
            .min_by_key(|node| topology.vertex_star_len(**node))
            .expect("a tetrahedron has four vertices");
        for index in topology.vertex_star(*vertex) {
            work.checkpoint()?;
            let index = index as usize;
            if removed.contains(&index) {
                continue;
            }
            let mut retained = topology
                .tetrahedron(index as u32)
                .expect("live retained cell")
                .vertex_indices;
            retained.sort_unstable();
            if retained == key {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests;
