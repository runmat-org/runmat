use super::*;
use crate::cdt::test_fixtures::allocation::AllocationBudget;
use runmat_geometry_core::PersistentEntityKind;
use runmat_meshing_core::NeverCancelled;
use std::collections::BTreeSet;

#[test]
fn borrowed_parameter_images_preserve_ordered_reference_signed_zero_and_periodicity() {
    let edge = PersistentEntityId::new(
        PersistentEntityKind::Edge,
        "edge".into(),
        vec!["body".into()],
    );
    let face = PersistentEntityId::new(
        PersistentEntityKind::Face,
        "face".into(),
        vec!["body".into()],
    );
    let chart = StableDigest::from_bytes([7; 32]);
    let observations = [
        Row {
            node: 0,
            parameter: Parameter::Curve(&edge, -0.0),
        },
        Row {
            node: 0,
            parameter: Parameter::Curve(&edge, 0.0),
        },
        Row {
            node: 2,
            parameter: Parameter::Curve(&edge, std::f64::consts::TAU),
        },
        Row {
            node: 0,
            parameter: Parameter::Surface(&face, chart, [0.0, 0.5]),
        },
        Row {
            node: 0,
            parameter: Parameter::Surface(&face, chart, [-0.0, 0.5]),
        },
        // Negative parameters must precede positive ones in the canonical order.
        Row {
            node: 0,
            parameter: Parameter::Curve(&edge, -1.5),
        },
        Row {
            node: 0,
            parameter: Parameter::Curve(&edge, -0.25),
        },
        Row {
            node: 0,
            parameter: Parameter::Surface(&face, chart, [-2.0, -0.5]),
        },
    ];
    // `f64::total_cmp` order as an integer key.
    let total = |value: f64| {
        let bits = value.to_bits() as i64;
        bits ^ (((bits >> 63) as u64) >> 1) as i64
    };
    let key = |row: &Row<'_>| match row.parameter {
        Parameter::Curve(id, value) => (
            row.node,
            0,
            id.clone(),
            StableDigest::ZERO,
            [total(value), 0],
        ),
        Parameter::Surface(id, chart, uv) => (row.node, 1, id.clone(), chart, uv.map(total)),
    };
    for repeats in [1, 64, 4096] {
        // Repeated metadata observations; this is not a duplicate-cell mesh witness.
        let rows = observations
            .iter()
            .rev()
            .copied()
            .cycle()
            .take(observations.len() * repeats)
            .collect::<Vec<_>>();
        let expected = rows
            .iter()
            .map(key)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let storage = (rows.len() * std::mem::size_of::<Row<'_>>()) as u64;
        let budget = AllocationBudget::new(storage);
        let result = normalize_rows(
            rows,
            Some(&budget),
            DelaunaySolverTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        assert_eq!(result.iter().map(key).collect::<Vec<_>>(), expected);
        assert_eq!(budget.live_bytes(), 0);
        assert_eq!(budget.peak_bytes(), storage);
    }
    let denied = AllocationBudget::new(0);
    assert!(normalize_rows(
        observations.to_vec(),
        Some(&denied),
        DelaunaySolverTopologyOptions::default(),
        &NeverCancelled
    )
    .is_err());
    assert_eq!(denied.live_bytes(), 0);
}
