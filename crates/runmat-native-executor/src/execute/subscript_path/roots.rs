use runmat_runtime::object::indexing::{
    ObjectIndexComponent, ObjectIndexSelector, ObjectSubscriptPath,
};
use runmat_value::Value;

pub(super) fn subscript_roots(root: &Value, path: Option<&ObjectSubscriptPath>) -> Vec<Value> {
    let mut roots = vec![root.clone()];
    for step in path.into_iter().flat_map(ObjectSubscriptPath::steps) {
        if let ObjectIndexSelector::IndexValues { components } = step.selector() {
            roots.extend(components.iter().filter_map(|component| match component {
                ObjectIndexComponent::Value(value) => Some(value.clone()),
                ObjectIndexComponent::Colon => None,
            }));
        }
    }
    roots
}
