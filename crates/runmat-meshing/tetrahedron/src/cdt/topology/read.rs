//! Identity-ordered local reads over the shared topology access contract.
//! Mutable slots are handles, not search ordinals. Normalize only the queried
//! cell/star; no mesh snapshot, point copy, or independent adjacency owner.
use crate::cdt::insertion::topology_access::{TetrahedralTopology, TetrahedronRef};
use crate::cdt::DelaunayVolumeNode;
use runmat_meshing_core::StableDigest;

pub(in crate::cdt) trait VolumeTopologyRead: TetrahedralTopology {
    fn node(&self, index: u32) -> DelaunayVolumeNode {
        self.nodes()[index as usize]
    }
    fn contains_cell(&self, index: u32) -> bool {
        self.tetrahedron(index).is_some()
    }
    fn cell(&self, index: u32) -> TetrahedronRef<'_> {
        let source = self
            .tetrahedron(index)
            .expect("local read names an active cell");
        let mut slots = [0, 1, 2, 3];
        slots.sort_unstable_by_key(|slot| self.node(source.vertex_indices[*slot]).identity);
        let odd = (0..4)
            .flat_map(|left| ((left + 1)..4).map(move |right| (left, right)))
            .filter(|(left, right)| slots[*left] > slots[*right])
            .count()
            % 2
            == 1;
        if odd {
            slots.swap(0, 1);
        }
        TetrahedronRef {
            vertex_indices: slots.map(|slot| source.vertex_indices[slot]),
            neighbors: slots.map(|slot| source.neighbors[slot]),
            region_id: source.region_id,
        }
    }
    fn cell_key(&self, index: u32) -> [StableDigest; 4] {
        let mut key = self
            .tetrahedron(index)
            .expect("live cell")
            .vertex_indices
            .map(|vertex| self.node(vertex).identity);
        key.sort_unstable();
        key
    }
    fn edge_star(&self, edge: [u32; 2]) -> Vec<u32> {
        let (first, second) = if self.vertex_star_len(edge[0]) > self.vertex_star_len(edge[1]) {
            (edge[1], edge[0])
        } else {
            (edge[0], edge[1])
        };
        let mut star = self
            .vertex_star(first)
            .into_iter()
            .filter(|index| {
                self.tetrahedron(*index)
                    .expect("live star cell")
                    .vertex_indices
                    .contains(&second)
            })
            .collect::<Vec<_>>();
        star.sort_unstable_by_key(|index| self.cell_key(*index));
        star
    }
    fn ordered_vertex_star(&self, vertex: u32) -> Vec<u32> {
        let mut star = self.vertex_star(vertex);
        star.sort_unstable_by_key(|index| self.cell_key(*index));
        star
    }
    fn vertex_is_boundary(&self, vertex: u32) -> bool {
        self.vertex_star(vertex).into_iter().any(|index| {
            let cell = self.cell(index);
            (0..4).any(|opposite| {
                cell.neighbors[opposite].is_none()
                    && cell
                        .vertex_indices
                        .iter()
                        .enumerate()
                        .any(|(slot, corner)| slot != opposite && *corner == vertex)
            })
        })
    }
    fn edge_is_boundary(&self, edge: [u32; 2]) -> bool {
        self.edge_star(edge).into_iter().any(|index| {
            let cell = self.cell(index);
            (0..4).any(|opposite| {
                cell.neighbors[opposite].is_none()
                    && cell
                        .vertex_indices
                        .iter()
                        .enumerate()
                        .filter(|(slot, _)| *slot != opposite)
                        .filter(|(_, vertex)| edge.contains(vertex))
                        .count()
                        == 2
            })
        })
    }
}
impl<T: TetrahedralTopology + ?Sized> VolumeTopologyRead for T {}
