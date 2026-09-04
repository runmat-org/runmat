use std::collections::BTreeMap;

use super::KeyAtom;

pub(crate) type GroupKey = Vec<KeyAtom>;

/// The ordering contract selected by the builtin accepting the grouping value.
#[derive(Clone, Debug)]
pub(crate) enum KeyOrder {
    Sorted,
    FirstAppearance,
    Explicit(Vec<GroupKey>),
}

/// A deterministic index over nonmissing grouping observations.
#[derive(Clone, Debug)]
pub(crate) struct GroupIndex {
    pub(crate) ids: Vec<f64>,
    pub(crate) keys: Vec<GroupKey>,
    pub(crate) first_rows: Vec<Option<usize>>,
    pub(crate) row_groups: Vec<Vec<usize>>,
}

impl GroupIndex {
    pub(crate) fn build(rows: Vec<Option<GroupKey>>, order: KeyOrder) -> Result<Self, String> {
        match order {
            KeyOrder::Sorted => Self::sorted(rows),
            KeyOrder::FirstAppearance => Self::first_appearance(rows),
            KeyOrder::Explicit(keys) => Self::explicit(rows, keys),
        }
    }

    fn sorted(rows: Vec<Option<GroupKey>>) -> Result<Self, String> {
        let mut buckets = BTreeMap::<GroupKey, Vec<usize>>::new();
        for (row, key) in rows.iter().enumerate() {
            if let Some(key) = key {
                buckets.entry(key.clone()).or_default().push(row);
            }
        }
        let keys = buckets.keys().cloned().collect::<Vec<_>>();
        Self::from_keys_and_rows(rows, keys)
    }

    fn first_appearance(rows: Vec<Option<GroupKey>>) -> Result<Self, String> {
        let mut key_indices = BTreeMap::<GroupKey, usize>::new();
        let mut keys = Vec::new();
        for key in rows.iter().flatten() {
            if !key_indices.contains_key(key) {
                key_indices.insert(key.clone(), keys.len());
                keys.push(key.clone());
            }
        }
        Self::from_keys_and_rows(rows, keys)
    }

    fn explicit(rows: Vec<Option<GroupKey>>, keys: Vec<GroupKey>) -> Result<Self, String> {
        let mut seen = BTreeMap::new();
        for (index, key) in keys.iter().enumerate() {
            if seen.insert(key, index).is_some() {
                return Err("explicit group order contains a duplicate key".into());
            }
        }
        Self::from_keys_and_rows(rows, keys)
    }

    fn from_keys_and_rows(
        rows: Vec<Option<GroupKey>>,
        keys: Vec<GroupKey>,
    ) -> Result<Self, String> {
        let key_indices = keys
            .iter()
            .enumerate()
            .map(|(index, key)| (key, index))
            .collect::<BTreeMap<_, _>>();
        let mut ids = Vec::with_capacity(rows.len());
        let mut first_rows = vec![None; keys.len()];
        let mut row_groups = vec![Vec::new(); keys.len()];
        for (row, key) in rows.into_iter().enumerate() {
            let Some(key) = key else {
                ids.push(f64::NAN);
                continue;
            };
            let index = key_indices
                .get(&key)
                .copied()
                .ok_or_else(|| "group observation is absent from the explicit order".to_string())?;
            ids.push(index as f64 + 1.0);
            first_rows[index].get_or_insert(row);
            row_groups[index].push(row);
        }
        Ok(Self {
            ids,
            keys,
            first_rows,
            row_groups,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(value: &str) -> GroupKey {
        vec![KeyAtom::Text(value.into())]
    }

    #[test]
    fn sorted_and_first_appearance_are_distinct_policies() {
        let rows = vec![Some(text("b")), Some(text("a")), Some(text("b"))];
        let sorted = GroupIndex::build(rows.clone(), KeyOrder::Sorted).unwrap();
        let stable = GroupIndex::build(rows, KeyOrder::FirstAppearance).unwrap();
        assert_eq!(sorted.ids, vec![2.0, 1.0, 2.0]);
        assert_eq!(stable.ids, vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn explicit_order_retains_empty_levels_and_missing_rows() {
        let rows = vec![Some(text("b")), None, Some(text("b"))];
        let index = GroupIndex::build(
            rows,
            KeyOrder::Explicit(vec![text("a"), text("b"), text("c")]),
        )
        .unwrap();
        assert_eq!(index.ids[0], 2.0);
        assert!(index.ids[1].is_nan());
        assert_eq!(index.ids[2], 2.0);
        assert_eq!(index.first_rows, vec![None, Some(0), None]);
        assert_eq!(index.row_groups, vec![vec![], vec![0, 2], vec![]]);
    }
}
