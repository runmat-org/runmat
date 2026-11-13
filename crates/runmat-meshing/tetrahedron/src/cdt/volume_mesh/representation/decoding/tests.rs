use super::*;
use runmat_meshing_core::NeverCancelled;

#[test]
fn dictionary_expansion_uses_the_charged_capacity_even_for_tiny_contexts() {
    let entity = PersistentEntityId::new(
        runmat_geometry_core::PersistentEntityKind::Region,
        "region".into(),
        vec!["assembly".into(), "body".into()],
    );
    for count in [0, 1, 2, 3, 5, 65] {
        let table = vec![entity.clone()];
        let indices = vec![0; count];
        let bytes = count as u64 * std::mem::size_of::<PersistentEntityId>() as u64;
        let mut budget = ExpansionBudget {
            remaining: bytes,
            allocations: None,
        };
        let expanded = budget.entities(&indices, &table, &NeverCancelled).unwrap();
        assert_eq!(expanded.capacity(), count);
        assert_eq!(expanded, vec![entity.clone(); count]);
        assert_eq!(budget.remaining, 0);
        if bytes > 0 {
            let mut short = ExpansionBudget {
                remaining: bytes - 1,
                allocations: None,
            };
            assert_eq!(
                short
                    .entities(&indices, &table, &NeverCancelled)
                    .unwrap_err()
                    .kind,
                DelaunayVolumeMeshErrorKind::ResourceLimit
            );
        }
    }
}

#[test]
fn failed_expansion_drops_original_values_without_consuming_later_items() {
    use std::cell::Cell;
    struct Observed<'a>(&'a Cell<usize>);
    impl Drop for Observed<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let dropped = Cell::new(0);
    let visited = Cell::new(0);
    let result = collect_exact((0..5).map(|index| {
        visited.set(visited.get() + 1);
        if index == 2 {
            Err(invalid("unknown original dictionary reference"))
        } else {
            Ok(Observed(&dropped))
        }
    }));
    assert!(result.is_err());
    assert_eq!(visited.get(), 3);
    assert_eq!(dropped.get(), 2);
}
