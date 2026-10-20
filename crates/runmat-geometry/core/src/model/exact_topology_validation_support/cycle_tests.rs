use super::*;

#[test]
fn cyclic_references_preserve_traversal_with_one_canonical_rotation() {
    let a = id("a");
    let b = id("b");
    let c = id("c");
    let inventory = [a.clone(), b.clone(), c.clone()];
    let known = IdIndex::collect(inventory.iter(), ValidationWork(None)).unwrap();

    require_canonical_cycle_refs(
        "cycle",
        &[a.clone(), c.clone(), b.clone()],
        PersistentEntityKind::Coedge,
        &known,
        ValidationWork(None),
    )
    .unwrap();
    assert!(require_canonical_cycle_refs(
        "cycle",
        &[c, b.clone(), a.clone()],
        PersistentEntityKind::Coedge,
        &known,
        ValidationWork(None),
    )
    .is_err());
    assert!(require_canonical_cycle_refs(
        "cycle",
        &[a, b.clone(), b],
        PersistentEntityKind::Coedge,
        &known,
        ValidationWork(None),
    )
    .is_err());
}

fn id(name: &str) -> PersistentEntityId {
    PersistentEntityId::new(
        PersistentEntityKind::Coedge,
        name.into(),
        vec!["part".into()],
    )
}
