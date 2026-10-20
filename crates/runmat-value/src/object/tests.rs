use super::*;
use crate::{
    ExactHostAllocationPermit, ExactHostAllocationReservation, ExactHostNumericAllocationError,
    Utf8Text, Value,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug)]
struct Permit {
    limit: usize,
    live: Arc<AtomicUsize>,
}
#[derive(Debug)]
struct Claim {
    bytes: usize,
    live: Arc<AtomicUsize>,
}
impl ExactHostAllocationReservation for Claim {}
impl Drop for Claim {
    fn drop(&mut self) {
        self.live.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}
impl ExactHostAllocationPermit for Permit {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        self.live
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes).filter(|total| *total <= self.limit)
            })
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        Ok(Arc::new(Claim {
            bytes,
            live: self.live.clone(),
        }))
    }
}

#[test]
fn one_generation_shares_children_and_retires_exact_claims_after_last_alias() {
    let permit = Permit {
        limit: usize::MAX,
        live: Arc::new(AtomicUsize::new(0)),
    };
    let class = PreparedObjectClassName::prepare("pkg.Result", &permit).unwrap();
    let mut builder =
        ExactObjectBuilder::try_new_with_dynamic_capacity(class, 1, 1, &permit).unwrap();
    let key = Utf8Text::try_copy_str_exact_with("value", &permit, |_| {
        Ok::<(), std::convert::Infallible>(())
    })
    .unwrap();
    let definition = DynamicPropertyDef::new("declaredName".into(), "pkg.Result")
        .try_admit_exact(&permit)
        .unwrap();
    builder
        .try_insert_dynamic_key_prepared(key.clone(), definition)
        .unwrap();
    builder.try_insert_prepared(key, Value::Num(7.0)).unwrap();
    let object = builder.finish().unwrap();
    let alias = object.clone();
    assert!(object.shares_generation_with(&alias));
    assert_eq!(alias.property("value"), Some(&Value::Num(7.0)));
    let removed = ExactObjectBuilder::prepare_copy_excluding_property(&alias, "value", &permit)
        .unwrap()
        .finish()
        .unwrap();
    assert_eq!(removed.property_count(), 0);
    assert_eq!(removed.dynamic_property_count(), 1);
    assert_eq!(removed.class_name(), alias.class_name());
    assert_eq!(alias.property("value"), Some(&Value::Num(7.0)));
    assert!(!removed.shares_generation_with(&alias));
    drop(removed);
    let removed_dynamic =
        ExactObjectBuilder::prepare_copy_excluding_dynamic_property(&alias, "value", &permit)
            .unwrap()
            .finish()
            .unwrap();
    assert_eq!(removed_dynamic.property_count(), 0);
    assert_eq!(removed_dynamic.dynamic_property_count(), 0);
    assert_eq!(
        alias.dynamic_property("value").unwrap().name.as_str(),
        "declaredName"
    );
    drop(removed_dynamic);
    assert!(object.clone().try_into_unbounded_builder().is_err());
    let mut replacement = ExactObjectBuilder::prepare_copy(&alias, 1, 1, &permit).unwrap();
    assert_eq!(
        replacement
            .try_replace_existing_property("value", Value::Num(8.0))
            .unwrap(),
        Value::Num(7.0)
    );
    let replacement = replacement.finish().unwrap();
    assert_eq!(replacement.property("value"), Some(&Value::Num(8.0)));
    assert_eq!(alias.property("value"), Some(&Value::Num(7.0)));
    drop(replacement);
    drop(object);
    assert!(permit.live.load(Ordering::SeqCst) > 0);
    drop(alias);
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);

    let mut ordinary = UnboundedObjectBuilder::new("pkg.Editable");
    ordinary.insert("first".into(), Value::Num(1.0));
    let mut ordinary = ordinary.finish().try_into_unbounded_builder().unwrap();
    ordinary.insert("second".into(), Value::Num(2.0));
    assert_eq!(ordinary.finish().property("second"), Some(&Value::Num(2.0)));
}

#[test]
fn exact_replacement_moves_children_and_preserves_source_on_preparation_failure() {
    let permit = Permit {
        limit: usize::MAX,
        live: Arc::new(AtomicUsize::new(0)),
    };
    let class = PreparedObjectClassName::prepare("pkg.Mutable", &permit).unwrap();
    let mut builder = ExactObjectBuilder::try_new(class, 1, &permit).unwrap();
    let first = Utf8Text::try_copy_str_exact_with("first", &permit, |_| {
        Ok::<(), std::convert::Infallible>(())
    })
    .unwrap();
    builder.try_insert_prepared(first, Value::Num(1.0)).unwrap();
    let object = builder.finish().unwrap();
    let mut input = ObjectEditInput::new(object);
    assert_eq!(
        ExactObjectBuilder::prepare_replacement(&mut input, 0, 0, &permit).unwrap_err(),
        ObjectEditPreparationError::CapacityInsufficient
    );
    assert_eq!(
        input.object().and_then(|object| object.property("first")),
        Some(&Value::Num(1.0))
    );

    let mut replacement =
        ExactObjectBuilder::prepare_replacement(&mut input, 2, 0, &permit).unwrap();
    assert!(input.object().is_none());
    let second = Utf8Text::try_copy_str_exact_with("second", &permit, |_| {
        Ok::<(), std::convert::Infallible>(())
    })
    .unwrap();
    replacement
        .try_insert_prepared(second, Value::Num(2.0))
        .unwrap();
    let replacement = replacement.finish().unwrap();
    assert_eq!(replacement.property("first"), Some(&Value::Num(1.0)));
    assert_eq!(replacement.property("second"), Some(&Value::Num(2.0)));
    drop(replacement);
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);
}

#[test]
fn containers_map_index_survives_growth_tombstones_and_swap_removal() {
    let mut storage = ContainersMapStorage::new(
        ContainersMapKeyType::UInt64,
        ContainersMapValueType::Double,
        true,
    );
    for key in 0_u64..96 {
        storage.insert_unique(ContainersMapEntry {
            normalized: ContainersMapKey::UInt(key),
            key_value: Value::Num(key as f64),
            value: Value::Num((key * 2) as f64),
        });
    }
    let removed = storage.remove_entry(&ContainersMapKey::UInt(0)).unwrap();
    assert_eq!(removed.normalized, ContainersMapKey::UInt(0));
    assert_eq!(
        storage
            .get(&ContainersMapKey::UInt(95))
            .map(|entry| &entry.value),
        Some(&Value::Num(190.0))
    );
    for key in (3_u64..96).step_by(3) {
        let removed = storage.remove_entry(&ContainersMapKey::UInt(key)).unwrap();
        assert_eq!(removed.normalized, ContainersMapKey::UInt(key));
    }
    for key in 0_u64..96 {
        let entry = storage.get(&ContainersMapKey::UInt(key));
        if key % 3 == 0 {
            assert!(entry.is_none());
        } else {
            assert_eq!(
                entry.map(|entry| &entry.value),
                Some(&Value::Num((key * 2) as f64))
            );
        }
    }
    for key in 96_u64..160 {
        storage.insert_unique(ContainersMapEntry {
            normalized: ContainersMapKey::UInt(key),
            key_value: Value::Num(key as f64),
            value: Value::Num((key * 2) as f64),
        });
    }
    assert_eq!(storage.len(), 128);
    for key in 1_u64..160 {
        if key < 96 && key % 3 == 0 {
            continue;
        }
        assert!(storage.contains(&ContainersMapKey::UInt(key)));
    }
    let replacement = ContainersMapEntry {
        normalized: ContainersMapKey::UInt(159),
        key_value: Value::Num(159.0),
        value: Value::Num(-1.0),
    };
    assert!(storage.replace_or_insert(replacement).is_some());
    assert_eq!(
        storage
            .get(&ContainersMapKey::UInt(159))
            .map(|entry| &entry.value),
        Some(&Value::Num(-1.0))
    );
}
