use runmat_runtime::object::indexing::{
    ObjectIndexComponent, ObjectIndexSelector, ObjectSubscriptPath,
};
use runmat_value::Value;

pub(super) fn subscript_values<'a>(
    root: &'a Value,
    path: Option<&'a ObjectSubscriptPath>,
) -> impl Iterator<Item = &'a Value> + Clone {
    std::iter::once(root).chain(
        path.into_iter()
            .flat_map(ObjectSubscriptPath::steps)
            .flat_map(|step| match step.selector() {
                ObjectIndexSelector::IndexValues { components } => components.as_slice(),
                _ => &[],
            })
            .filter_map(|component| match component {
                ObjectIndexComponent::Value(value) => Some(value),
                ObjectIndexComponent::Colon | ObjectIndexComponent::End => None,
            }),
    )
}

#[cfg(test)]
pub(super) fn subscript_roots(root: &Value, path: Option<&ObjectSubscriptPath>) -> Vec<Value> {
    let mut roots = vec![root.clone()];
    for step in path.into_iter().flat_map(ObjectSubscriptPath::steps) {
        if let ObjectIndexSelector::IndexValues { components } = step.selector() {
            roots.extend(components.iter().filter_map(|component| match component {
                ObjectIndexComponent::Value(value) => Some(value.clone()),
                ObjectIndexComponent::Colon | ObjectIndexComponent::End => None,
            }));
        }
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ScopedValueRoots;
    use runmat_runtime::object::indexing::ObjectSubscript;
    use runmat_value::HandleRef;

    #[test]
    fn pending_dotted_path_roots_receiver_and_argument_until_scope_drop() {
        if runmat_gc::test_support::run_isolated_gc_test(concat!(
            module_path!(),
            "::pending_dotted_path_roots_receiver_and_argument_until_scope_drop"
        )) {
            return;
        }
        runmat_gc::gc_test_context(|| {
            let receiver = runmat_gc::gc_allocate_rooted(Value::String("receiver".into()))
                .expect("allocate receiver");
            let argument = runmat_gc::gc_allocate_rooted(Value::String("argument".into()))
                .expect("allocate argument");
            let receiver_handle = receiver.handle();
            let argument_handle = argument.handle();
            let root = Value::HandleObject(HandleRef {
                class_name: "PendingReceiver".into(),
                target: receiver_handle,
                valid: true,
            });
            let argument_value = Value::HandleObject(HandleRef {
                class_name: "PendingArgument".into(),
                target: argument_handle,
                valid: true,
            });
            let path = ObjectSubscriptPath::new(
                ObjectSubscript::dotted_invoke("method", vec![argument_value]).to_vec(),
            )
            .expect("dotted path");
            let retained = ScopedValueRoots::register(
                subscript_roots(&root, Some(&path)),
                "native_pending_subscript_path_test",
            )
            .expect("retain path roots");
            drop(root);
            drop(path);
            drop(receiver);
            drop(argument);
            runmat_gc::gc_collect_major().expect("collect while path is retained");
            assert_eq!(
                runmat_gc::gc_clone_value(&receiver_handle).expect("receiver stays live"),
                Value::String("receiver".into())
            );
            assert_eq!(
                runmat_gc::gc_clone_value(&argument_handle).expect("argument stays live"),
                Value::String("argument".into())
            );
            drop(retained);
            runmat_gc::gc_collect_major().expect("collect after path scope drop");
            assert!(runmat_gc::gc_clone_value(&receiver_handle).is_err());
            assert!(runmat_gc::gc_clone_value(&argument_handle).is_err());
        });
    }
}
