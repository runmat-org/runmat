use super::*;

pub(super) fn fit_tree(spec: &FitSpec) -> BuiltinResult<TreeFit> {
    let mut nodes = Vec::new();
    let mut splits = 0usize;
    let rows = (0..classification_tree_matrix_dimensions(&spec.x).0).collect::<Vec<_>>();
    build_node(spec, &rows, &mut nodes, &mut splits, 0)?;
    Ok(TreeFit {
        nodes,
        num_splits: splits,
    })
}

fn build_node(
    spec: &FitSpec,
    rows: &[usize],
    nodes: &mut Vec<Node>,
    split_count: &mut usize,
    depth: usize,
) -> BuiltinResult<usize> {
    if depth > MAX_TREE_DEPTH {
        return Err(fitctree_invalid(
            "fitctree: tree depth exceeds RunMat safety limit",
        ));
    }
    let counts = class_counts(
        rows,
        &spec.class_indices,
        &spec.weights,
        spec.class_names.len(),
    );
    let total_weight = counts.iter().sum::<f64>();
    let probabilities = if total_weight > 0.0 {
        counts.iter().map(|value| value / total_weight).collect()
    } else {
        vec![0.0; counts.len()]
    };
    let class_index = probabilities
        .iter()
        .enumerate()
        .max_by(|left, right| {
            left.1
                .partial_cmp(right.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| right.0.cmp(&left.0))
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let node_index = nodes.len();
    nodes.push(Node {
        left: None,
        right: None,
        predictor: None,
        threshold: f64::NAN,
        class_index,
        probabilities,
    });
    let max_splits = spec.options.max_num_splits.unwrap_or(0);
    if *split_count >= max_splits
        || rows.len() < spec.options.min_parent_size
        || rows.len() < spec.options.min_leaf_size.saturating_mul(2)
        || counts.iter().filter(|value| **value > 0.0).count() <= 1
    {
        return Ok(node_index);
    }
    if let Some(best) = best_split(spec, rows, &counts)? {
        if best.gain <= EPS {
            return Ok(node_index);
        }
        *split_count += 1;
        let left = build_node(spec, &best.left_rows, nodes, split_count, depth + 1)?;
        let right = build_node(spec, &best.right_rows, nodes, split_count, depth + 1)?;
        nodes[node_index].left = Some(left);
        nodes[node_index].right = Some(right);
        nodes[node_index].predictor = Some(best.predictor);
        nodes[node_index].threshold = best.threshold;
    }
    Ok(node_index)
}

fn best_split(
    spec: &FitSpec,
    rows: &[usize],
    _parent_counts: &[f64],
) -> BuiltinResult<Option<SplitCandidate>> {
    let mut best = None;
    for col in 0..classification_tree_matrix_dimensions(&spec.x).1 {
        let mut entries = rows
            .iter()
            .filter(|row| !x_value(&spec.x, **row, col).is_nan())
            .map(|row| {
                (
                    x_value(&spec.x, *row, col),
                    spec.class_indices[*row],
                    spec.weights[*row],
                    *row,
                )
            })
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
        if entries.len() < 2 {
            continue;
        }
        let mut right_counts = vec![0.0; spec.class_names.len()];
        for (_, class_idx, weight, _) in &entries {
            right_counts[*class_idx] += *weight;
        }
        let parent_weight = right_counts.iter().sum::<f64>();
        let parent_impurity = impurity(&right_counts, spec.options.split_criterion);
        let mut left_counts = vec![0.0; right_counts.len()];
        let mut left_weight = 0.0;
        let mut right_weight = parent_weight;
        let mut left_len = 0usize;
        let mut right_len = entries.len();
        for idx in 0..entries.len() - 1 {
            let (_, class_idx, weight, _) = entries[idx];
            left_counts[class_idx] += weight;
            right_counts[class_idx] -= weight;
            left_weight += weight;
            right_weight -= weight;
            left_len += 1;
            right_len -= 1;
            let current = entries[idx].0;
            let next = entries[idx + 1].0;
            if (current - next).abs() <= EPS {
                continue;
            }
            if left_len < spec.options.min_leaf_size || right_len < spec.options.min_leaf_size {
                continue;
            }
            if left_weight <= 0.0 || right_weight <= 0.0 {
                continue;
            }
            let left_impurity = impurity(&left_counts, spec.options.split_criterion);
            let right_impurity = impurity(&right_counts, spec.options.split_criterion);
            let gain = parent_impurity
                - (left_weight / parent_weight) * left_impurity
                - (right_weight / parent_weight) * right_impurity;
            if gain
                > best
                    .as_ref()
                    .map(|candidate: &SplitCandidate| candidate.gain)
                    .unwrap_or(-1.0)
            {
                let threshold = current + (next - current) / 2.0;
                let mut left_rows = Vec::with_capacity(left_len);
                let mut right_rows = Vec::with_capacity(right_len);
                for (value, _, _, row) in &entries {
                    if *value <= threshold {
                        left_rows.push(*row);
                    } else {
                        right_rows.push(*row);
                    }
                }
                best = Some(SplitCandidate {
                    predictor: col,
                    threshold,
                    gain,
                    left_rows,
                    right_rows,
                });
            }
        }
    }
    Ok(best)
}

fn impurity(counts: &[f64], criterion: SplitCriterion) -> f64 {
    let total = counts.iter().sum::<f64>();
    if total <= 0.0 {
        return 0.0;
    }
    match criterion {
        SplitCriterion::Gdi => {
            1.0 - counts
                .iter()
                .map(|count| {
                    let p = count / total;
                    p * p
                })
                .sum::<f64>()
        }
        SplitCriterion::Deviance => counts
            .iter()
            .filter(|count| **count > 0.0)
            .map(|count| {
                let p = count / total;
                -p * p.ln()
            })
            .sum(),
    }
}

fn class_counts(
    rows: &[usize],
    classes: &[usize],
    weights: &[f64],
    class_count: usize,
) -> Vec<f64> {
    let mut counts = vec![0.0; class_count];
    for row in rows {
        counts[classes[*row]] += weights[*row];
    }
    counts
}
