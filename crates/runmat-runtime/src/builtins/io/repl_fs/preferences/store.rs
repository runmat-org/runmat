use std::cell::RefCell;
use std::collections::BTreeMap;

use runmat_value::{CharArray, StructValue, Value};

type PreferenceGroups = BTreeMap<String, BTreeMap<String, Value>>;

thread_local! {
    // Existing in-memory scope, pending an authoritative session/durable service.
    static PREFS: RefCell<PreferenceGroups> = const { RefCell::new(BTreeMap::new()) };
}

#[derive(Debug)]
pub(crate) struct MissingPreference {
    pub(crate) group: String,
    pub(crate) name: String,
}

pub(crate) fn all_groups() -> StructValue {
    PREFS.with(|prefs| {
        let mut out = StructValue::new();
        for (group, values) in prefs.borrow().iter() {
            out.insert(group.clone(), Value::Struct(as_struct(values)));
        }
        out
    })
}

pub(crate) fn group_values(group: &str) -> StructValue {
    PREFS.with(|prefs| {
        prefs
            .borrow()
            .get(group)
            .map_or_else(StructValue::new, as_struct)
    })
}

pub(crate) fn has_group(group: &str) -> bool {
    PREFS.with(|prefs| prefs.borrow().contains_key(group))
}

pub(crate) fn has_preference(group: &str, name: &str) -> bool {
    PREFS.with(|prefs| {
        prefs
            .borrow()
            .get(group)
            .is_some_and(|values| values.contains_key(name))
    })
}

pub(crate) fn read_or_create(
    group: &str,
    name: &str,
    default: Option<Value>,
) -> Result<Value, MissingPreference> {
    PREFS.with(|prefs| {
        let mut prefs = prefs.borrow_mut();
        if let Some(value) = prefs.get(group).and_then(|values| values.get(name)) {
            return Ok(value.clone());
        }
        let Some(default) = default else {
            return Err(MissingPreference {
                group: group.to_owned(),
                name: name.to_owned(),
            });
        };
        prefs
            .entry(group.to_owned())
            .or_default()
            .insert(name.to_owned(), default.clone());
        Ok(default)
    })
}

/// Validate every missing name before mutating the store. An absent later name
/// cannot leave earlier defaults installed after the call fails.
pub(crate) fn read_many(
    group: &str,
    names: &[String],
    defaults: Option<&[Value]>,
) -> Result<Vec<Value>, MissingPreference> {
    PREFS.with(|prefs| {
        let mut prefs = prefs.borrow_mut();
        for (index, name) in names.iter().enumerate() {
            if !prefs
                .get(group)
                .is_some_and(|values| values.contains_key(name))
                && defaults.and_then(|values| values.get(index)).is_none()
            {
                return Err(MissingPreference {
                    group: group.to_owned(),
                    name: name.clone(),
                });
            }
        }
        let mut out = Vec::with_capacity(names.len());
        for (index, name) in names.iter().enumerate() {
            if let Some(value) = prefs
                .get(group)
                .and_then(|values| values.get(name))
                .cloned()
            {
                out.push(value);
            } else if let Some(value) = defaults.and_then(|values| values.get(index)) {
                let value = value.clone();
                prefs
                    .entry(group.to_owned())
                    .or_default()
                    .insert(name.clone(), value.clone());
                out.push(value);
            }
        }
        Ok(out)
    })
}

pub(crate) fn set_many(group: String, entries: Vec<(String, Value)>) {
    if entries.is_empty() {
        return;
    }
    PREFS.with(|prefs| {
        let mut prefs = prefs.borrow_mut();
        let group = prefs.entry(group).or_default();
        for (name, value) in entries {
            group.insert(name, value);
        }
    });
}

pub(crate) fn session_pref_text(group: &str, preference: &str) -> Option<String> {
    PREFS.with(|prefs| {
        prefs
            .borrow()
            .get(group)
            .and_then(|values| values.get(preference))
            .and_then(pref_value_to_text)
    })
}

fn pref_value_to_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(array) if array.rows == 1 => {
            let mut text: String = array.data.iter().take(array.cols).collect();
            while text.ends_with(' ') {
                text.pop();
            }
            Some(text)
        }
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

fn as_struct(values: &BTreeMap<String, Value>) -> StructValue {
    let mut out = StructValue::new();
    for (name, value) in values {
        out.insert(name.clone(), value.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_later_name_does_not_install_an_earlier_default() {
        let group = "runmatPreferenceAtomicDefaults";
        let names = vec!["first".to_owned(), "missing".to_owned()];
        let error = read_many(group, &names, Some(&[Value::Num(1.0)]))
            .expect_err("later missing preference");
        assert_eq!(error.name, "missing");
        assert!(!has_group(group));

        // A later missing name in an otherwise populated group is also inert.
        set_many(group.to_owned(), vec![("first".into(), Value::Num(1.0))]);
        let error = read_many(group, &names, None).expect_err("later missing preference");
        assert_eq!(error.name, "missing");
        assert!(!has_preference(group, "missing"));
    }

    #[test]
    fn store_retains_wide_integer_payload_and_same_thread_host_text() {
        let group = "runmatPreferenceStoreWide";
        let exact = Value::Int(runmat_value::IntValue::U64(u64::MAX));
        set_many(group.into(), vec![("wide".into(), exact.clone())]);
        assert_eq!(read_or_create(group, "wide", None).expect("stored"), exact);
        set_many(
            group.into(),
            vec![("smtp".into(), Value::String("relay".into()))],
        );
        assert_eq!(session_pref_text(group, "smtp"), Some("relay".into()));
    }

    #[test]
    fn empty_name_batch_does_not_create_a_group() {
        let group = "runmatPreferenceEmptyBatch";
        set_many(group.into(), Vec::new());
        assert!(!has_group(group));
        assert_eq!(read_many(group, &[], None).expect("empty read"), Vec::new());
    }
}
