use super::*;
use runmat_analysis_core::{BodyForceDensity, FieldLocation, SignalPrecision};

fn shape() -> SignalShape {
    SignalShape::TopologyField {
        topology_id: Digest::from_bytes([7; 32]).to_string(),
        location: FieldLocation::Node,
        component_count: 3,
    }
}

fn tensor(rows: usize) -> SignalValue<BodyForceDensity> {
    SignalValue::Tensor {
        shape: vec![rows, 3],
        values: (0..rows * 3)
            .map(|i| BodyForceDensity::new(i as f64))
            .collect(),
    }
}

#[test]
fn spatial_real_and_complex_tables_preserve_exact_entity_rows() {
    let shape = shape();
    for ordinate in [
        SignalOrdinate::Real(tensor(2)),
        SignalOrdinate::Complex {
            real: tensor(2),
            imaginary: tensor(2),
        },
    ] {
        let precision = match &ordinate {
            SignalOrdinate::Real(_) => SignalPrecision::Real64,
            SignalOrdinate::Complex { .. } => SignalPrecision::Complex128,
        };
        let refs = vec![&ordinate, &ordinate];
        let count = table_entity_count(&shape, &refs).unwrap();
        assert_eq!(count, Some(2));
        let values = canonical_table_values(&shape, &refs, precision, count).unwrap();
        let recovered = ordinates::<BodyForceDensity>(&shape, 6, &values).unwrap();
        assert_eq!(recovered, vec![ordinate.clone(), ordinate]);
    }
}

#[test]
fn spatial_tables_reject_changed_rows_components_and_partial_ordinates() {
    let shape = shape();
    let first = SignalOrdinate::Real(tensor(2));
    let changed = SignalOrdinate::Real(tensor(3));
    assert!(canonical_table_values(
        &shape,
        &[&first, &changed],
        SignalPrecision::Real64,
        Some(2)
    )
    .is_err());
    let wrong = SignalOrdinate::Real(SignalValue::Tensor {
        shape: vec![3, 2],
        values: vec![BodyForceDensity::new(1.0); 6],
    });
    assert!(table_entity_count(&shape, &[&wrong]).is_err());
    assert!(ordinates::<BodyForceDensity>(
        &shape,
        6,
        &CanonicalSignalValues::Real { values: vec![0; 7] }
    )
    .is_err());
    assert!(ordinates::<BodyForceDensity>(
        &shape,
        0,
        &CanonicalSignalValues::Real { values: vec![] }
    )
    .is_err());
    assert!(table_entity_count::<BodyForceDensity>(&shape, &[]).is_err());
}
