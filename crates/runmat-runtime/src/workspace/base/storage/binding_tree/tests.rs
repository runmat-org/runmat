use runmat_value::Value;

use super::*;

fn binding(name: &str, value: f64) -> (WorkspaceName, BindingTreeValue) {
    (
        WorkspaceName::unbounded(name),
        BindingTreeValue::Local(
            RootedWorkspaceEntry::new(Value::Num(value)).expect("rooted test entry"),
        ),
    )
}

fn assert_avl(link: &Link) -> (usize, usize) {
    let Some(node) = link else {
        return (0, 0);
    };
    let node = node.get();
    let (left_height, left_len) = assert_avl(&node.left);
    let (right_height, right_len) = assert_avl(&node.right);
    assert!(left_height.abs_diff(right_height) <= 1);
    assert_eq!(node.height, left_height.max(right_height) + 1);
    assert_eq!(node.len, left_len + right_len + 1);
    (node.height, node.len)
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn path_copy_preserves_snapshots_and_sorted_lookup() {
    let mut current = BindingTree::default();
    for index in (0..128).rev() {
        let name = format!("key_{index:03}");
        let (name, binding) = binding(&name, f64::from(index));
        current = current.prepare_set(None, name, binding).unwrap();
        assert_avl(&current.root);
    }
    assert_eq!(current.len(), 128);
    let snapshot = current.clone();
    let (name, binding) = binding("key_064", 999.0);
    current = current.prepare_set(None, name, binding).unwrap();
    assert_eq!(
        snapshot.local("key_064").unwrap().value(),
        &Value::Num(64.0)
    );
    assert_eq!(
        current.local("key_064").unwrap().value(),
        &Value::Num(999.0)
    );

    for index in (0..128).step_by(2) {
        let name = format!("key_{index:03}");
        current = current.prepare_remove(None, &name).unwrap().unwrap();
        assert_avl(&current.root);
    }
    assert_eq!(current.len(), 64);
    assert_eq!(snapshot.len(), 128);
    assert_eq!(
        current
            .name_after(Some("key_065"))
            .map(|(name, _)| name.as_str()),
        Some("key_067")
    );
    let mut names = Vec::new();
    let mut previous = None;
    while let Some((name, _)) = current.name_after(previous) {
        names.push(name.as_str().to_owned());
        previous = Some(name.as_str());
    }
    assert!(names.windows(2).all(|pair| pair[0] < pair[1]));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn bulk_build_requires_strict_order_and_keeps_independent_paths() {
    let mut current = BindingTree::prepare_sorted_exact(
        None,
        (0..31).map(|index| {
            let name = format!("key_{index:03}");
            binding(&name, f64::from(index))
        }),
    )
    .unwrap();
    assert_eq!(assert_avl(&current.root).1, 31);
    let independent = {
        let (name, binding) = binding("new", 1.0);
        current.prepare_set(None, name, binding).unwrap()
    };
    let (name, current_binding) = binding("current", 2.0);
    current = current.prepare_set(None, name, current_binding).unwrap();
    assert!(independent.get("new").is_some());
    assert!(independent.get("current").is_none());
    assert!(current.get("new").is_none());
    assert!(current.get("current").is_some());

    let target = GlobalBindingTarget::Named(WorkspaceName::unbounded("session_target"));
    current = current
        .prepare_set(
            None,
            WorkspaceName::unbounded("global_without_value"),
            BindingTreeValue::Global(target.clone()),
        )
        .unwrap();
    assert_eq!(current.global_target("global_without_value"), Some(&target));
    assert!(current.local("global_without_value").is_none());
    assert_eq!(current.len(), 33);
    assert_eq!(
        current
            .name_after(Some("current"))
            .map(|(name, _)| name.as_str()),
        Some("global_without_value")
    );

    let invalid = [binding("b", 1.0), binding("a", 2.0)].into_iter();
    assert!(matches!(
        BindingTree::prepare_sorted_exact(None, invalid),
        Err(SortedBindingTreeError::NonAscendingNames)
    ));
}
