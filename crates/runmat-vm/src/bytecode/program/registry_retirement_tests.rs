use super::{FunctionBytecode, FunctionRegistry};
use runmat_hir::{FunctionId, SourceId};
use std::collections::HashMap;

fn function(id: usize, name: &str, source: usize) -> FunctionBytecode {
    FunctionBytecode {
        function: FunctionId(id),
        display_name: name.into(),
        source_id: Some(SourceId(source)),
        ..Default::default()
    }
}

fn registry() -> FunctionRegistry {
    FunctionRegistry::new(HashMap::from([
        (FunctionId(1), function(1, "first", 5)),
        (FunctionId(2), function(2, "second", 5)),
        (FunctionId(3), function(3, "unrelated", 6)),
    ]))
}

#[test]
fn source_retirement_removes_all_target_aliases_and_preserves_rebound_names() {
    let mut registry = registry();
    registry.names.insert("first_alias".into(), FunctionId(1));
    registry.names.insert("second_alias".into(), FunctionId(2));
    registry.names.insert("rebound_alias".into(), FunctionId(1));
    registry.names.insert("rebound_alias".into(), FunctionId(3));
    registry.names.insert("first".into(), FunctionId(3));
    let removed = registry.remove_source(SourceId(5));
    assert_eq!(removed.len(), 2);
    for name in ["first_alias", "second_alias", "second"] {
        assert_eq!(registry.resolve_name(name), None);
    }
    for name in ["first", "rebound_alias", "unrelated"] {
        assert_eq!(registry.resolve_name(name), Some(FunctionId(3)));
    }
    assert_eq!(registry.functions.len(), 1);
    assert!(registry.functions_for_source(SourceId(5)).is_empty());
    assert_eq!(registry.functions_for_source(SourceId(6)), &[FunctionId(3)]);
}

#[test]
fn replacement_retires_old_aliases_without_removing_new_name() {
    let mut registry = registry();
    registry.names.insert("old_alias".into(), FunctionId(1));
    registry.names.insert("rebound_alias".into(), FunctionId(3));
    registry.insert_replacing_name(function(4, "first", 7));
    assert_eq!(registry.resolve_name("first"), Some(FunctionId(4)));
    assert_eq!(registry.resolve_name("old_alias"), None);
    assert_eq!(registry.resolve_name("rebound_alias"), Some(FunctionId(3)));
    assert!(registry.get(FunctionId(1)).is_none());
    assert_eq!(registry.functions_for_source(SourceId(5)), &[FunctionId(2)]);
    assert_eq!(registry.functions_for_source(SourceId(7)), &[FunctionId(4)]);
}

#[test]
fn same_identity_refresh_preserves_aliases_and_updates_source_membership() {
    let mut registry = registry();
    registry.names.insert("live_alias".into(), FunctionId(1));
    registry.insert_replacing_name(function(1, "first", 7));
    assert_eq!(registry.resolve_name("first"), Some(FunctionId(1)));
    assert_eq!(registry.resolve_name("live_alias"), Some(FunctionId(1)));
    assert_eq!(registry.functions_for_source(SourceId(5)), &[FunctionId(2)]);
    assert_eq!(registry.functions_for_source(SourceId(7)), &[FunctionId(1)]);
    registry.remove(FunctionId(1)).unwrap();
    assert_eq!(registry.resolve_name("first"), None);
    assert_eq!(registry.resolve_name("live_alias"), None);
    assert!(registry.functions_for_source(SourceId(7)).is_empty());
    assert!(registry.get(FunctionId(3)).is_some());
}
