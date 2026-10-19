use super::*;

#[test]
fn ordering_covers_disconnected_dense_star_and_path_graphs_without_duplicates() {
    for kind in 0..4 {
        let mut graph = vec![BTreeSet::new(); 96];
        for row in 0..96 {
            for column in 0..row {
                let edge = match kind {
                    0 => row / 12 == column / 12,
                    1 => column == 0,
                    2 => row == column + 1,
                    _ => false,
                };
                if edge {
                    graph[row].insert(column);
                    graph[column].insert(row);
                }
            }
        }
        let first = reverse_cuthill_mckee(&graph).unwrap();
        assert_eq!(first, reverse_cuthill_mckee(&graph).unwrap());
        let mut inventory = first;
        inventory.sort_unstable();
        assert_eq!(inventory, (0..96).collect::<Vec<_>>());
    }
}

#[test]
fn symbolic_inventory_matches_independent_dense_elimination() {
    const WIDTH: usize = 8;
    const N: usize = WIDTH * WIDTH;
    let mut dense = vec![vec![false; N]; N];
    for row in 0..N {
        dense[row][row] = true;
        for column in 0..row {
            let edge =
                row == column + WIDTH || (row == column + 1 && row / WIDTH == column / WIDTH);
            dense[row][column] = edge;
            dense[column][row] = edge;
        }
    }
    let mut offsets = vec![0];
    let mut columns = Vec::new();
    for row in &dense {
        columns.extend(
            row.iter()
                .enumerate()
                .filter_map(|(column, &edge)| edge.then_some(column as u64)),
        );
        offsets.push(columns.len() as u64);
    }
    let pattern = SparsePattern::try_new(N as u64, N as u64, offsets, columns).unwrap();
    let plan = CoercivitySymbolicPlan::derive(&pattern).unwrap();
    plan.validate_for(&pattern).unwrap();
    let mut live = [true; N];
    let mut factor_entries = N as u64;
    let mut updates = 0;
    for (position, &pivot) in plan.elimination_order().iter().enumerate() {
        live[pivot] = false;
        let neighbors: Vec<_> = (0..N).filter(|&v| live[v] && dense[pivot][v]).collect();
        let mut ordered_neighbors: Vec<_> =
            neighbors.iter().map(|&v| plan.inverse_order()[v]).collect();
        ordered_neighbors.sort_unstable();
        assert_eq!(ordered_neighbors, plan.pivot_neighbors()[position]);
        factor_entries += neighbors.len() as u64;
        updates += (neighbors.len() * (neighbors.len() + 1) / 2) as u64;
        for &left in &neighbors {
            for &right in &neighbors {
                dense[left][right] = true;
            }
        }
    }
    assert_eq!(plan.factor_nonzero_count(), factor_entries);
    assert_eq!(plan.update_count(), updates);
}
