//! Parent-forest validation and deterministic layout for `treelayout`.

use super::{graph_error, BuiltinResult};

pub(super) fn tree_coordinates(parents: &[usize]) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    validate_parent_forest(parents)?;
    let n = parents.len();
    let mut children = vec![Vec::<usize>::new(); n];
    let mut roots = Vec::new();
    for (node, parent) in parents.iter().copied().enumerate() {
        if parent == 0 {
            roots.push(node);
        } else {
            children[parent - 1].push(node);
        }
    }
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut leaf_cursor = 1.0;
    for root in roots {
        let mut stack = vec![(root, 0_usize, false)];
        while let Some((node, depth, expanded)) = stack.pop() {
            if expanded {
                if children[node].is_empty() {
                    x[node] = leaf_cursor;
                    leaf_cursor += 1.0;
                } else {
                    let first = children[node][0];
                    let last = *children[node].last().expect("nonempty children");
                    x[node] = (x[first] + x[last]) / 2.0;
                }
                continue;
            }
            y[node] = -(depth as f64);
            stack.push((node, depth, true));
            for &child in children[node].iter().rev() {
                stack.push((child, depth + 1, false));
            }
        }
    }
    Ok((x, y))
}

fn validate_parent_forest(parents: &[usize]) -> BuiltinResult<()> {
    let n = parents.len();
    let mut state = vec![0_u8; n];
    for start in 0..n {
        if state[start] == 2 {
            continue;
        }
        let mut chain = Vec::new();
        let mut node = start;
        loop {
            match state[node] {
                2 => break,
                1 => {
                    return Err(graph_error(
                        "treelayout",
                        "treelayout: parent vector contains a cycle",
                    ))
                }
                _ => {}
            }
            state[node] = 1;
            chain.push(node);
            let parent = parents[node];
            if parent == 0 {
                break;
            }
            if parent > n {
                return Err(graph_error(
                    "treelayout",
                    "treelayout: parent indices must be zero or valid one-based node ids",
                ));
            }
            node = parent - 1;
        }
        for node in chain {
            state[node] = 2;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_rejects_cycles_and_out_of_range_parents() {
        assert!(tree_coordinates(&[2, 1]).is_err());
        assert!(tree_coordinates(&[1]).is_err());
        assert!(tree_coordinates(&[0, 3]).is_err());
    }

    #[test]
    fn layout_preserves_depth_and_sibling_order() {
        let (x, y) = tree_coordinates(&[0, 1, 1, 2]).expect("forest");
        assert_eq!(x, vec![1.5, 1.0, 2.0, 1.0]);
        assert_eq!(y, vec![0.0, -1.0, -1.0, -2.0]);
    }
}
