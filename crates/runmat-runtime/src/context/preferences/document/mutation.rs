//! Pure versioned-document mutation shared by native and browser hosts.

use super::PreferenceDocument;
use crate::context::{UserPathSetting, UserPathSnapshot};
use runmat_execution::value::ValuePayload;

pub(in crate::context::preferences) enum DocumentMutation {
    Set {
        group: String,
        entries: Vec<(String, ValuePayload)>,
    },
    ReadOrCreate {
        group: String,
        names: Vec<String>,
        defaults: Option<Vec<Option<ValuePayload>>>,
    },
    UserPath(UserPathSetting),
}

pub(in crate::context::preferences) struct DocumentPrepared {
    pub values: Vec<ValuePayload>,
    pub user_path: Option<UserPathSnapshot>,
    pub changed: bool,
}

#[derive(Debug, thiserror::Error)]
pub(in crate::context::preferences) enum DocumentMutationError {
    #[error("preference '{name}' does not exist in group '{group}'")]
    Missing { group: String, name: String },
    #[error("application user-path revision is exhausted")]
    RevisionExhausted,
}

pub(in crate::context::preferences) fn apply(
    mut document: PreferenceDocument,
    mutation: DocumentMutation,
) -> Result<(PreferenceDocument, DocumentPrepared), DocumentMutationError> {
    let mut values = Vec::new();
    let mut changed = false;
    let mut return_user_path = false;
    match mutation {
        DocumentMutation::Set { group, entries } => {
            if !entries.is_empty() {
                let target = document.groups.entry(group).or_default();
                for (name, payload) in entries {
                    target.insert(name, payload);
                }
                changed = true;
            }
        }
        DocumentMutation::ReadOrCreate {
            group,
            names,
            defaults,
        } => {
            for (index, name) in names.iter().enumerate() {
                let target = document.groups.entry(group.clone()).or_default();
                if !target.contains_key(name) {
                    let Some(default) = defaults
                        .as_ref()
                        .and_then(|values| values.get(index))
                        .and_then(Option::as_ref)
                    else {
                        return Err(DocumentMutationError::Missing {
                            group,
                            name: name.clone(),
                        });
                    };
                    target.insert(name.clone(), default.clone());
                    changed = true;
                }
                values.push(
                    target
                        .get(name)
                        .expect("existing or inserted preference")
                        .clone(),
                );
            }
        }
        DocumentMutation::UserPath(setting) => {
            document.user_path = setting;
            document.user_path_revision = document
                .user_path_revision
                .checked_add(1)
                .ok_or(DocumentMutationError::RevisionExhausted)?;
            changed = true;
            return_user_path = true;
        }
    }
    let prepared = DocumentPrepared {
        values,
        user_path: return_user_path.then(|| UserPathSnapshot {
            setting: document.user_path.clone(),
            revision: document.user_path_revision,
        }),
        changed,
    };
    Ok((document, prepared))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::value_codec::encode_durable_value;
    use runmat_value::Value;

    #[test]
    fn repeated_defaults_first_wins_and_user_path_is_a_distinct_revision() {
        let document = PreferenceDocument::empty();
        let first = encode_durable_value(&Value::Num(1.0)).unwrap();
        let later = encode_durable_value(&Value::Num(2.0)).unwrap();
        let (updated, result) = apply(
            document,
            DocumentMutation::ReadOrCreate {
                group: "app".into(),
                names: vec!["same".into(), "same".into()],
                defaults: Some(vec![Some(first.clone()), Some(later)]),
            },
        )
        .unwrap();
        assert!(result.changed);
        assert_eq!(result.values, vec![first.clone(), first]);
        let (document, setting) = apply(
            updated,
            DocumentMutation::UserPath(UserPathSetting::Custom("Rel/Exact".into())),
        )
        .unwrap();
        assert_eq!(setting.user_path.unwrap().revision, 1);
        assert_eq!(document.groups["app"].len(), 1);
        assert_eq!(
            document.user_path,
            UserPathSetting::Custom("Rel/Exact".into())
        );
    }

    #[test]
    fn empty_set_does_not_create_a_group_or_require_publication() {
        let (document, prepared) = apply(
            PreferenceDocument::empty(),
            DocumentMutation::Set {
                group: "app".into(),
                entries: Vec::new(),
            },
        )
        .unwrap();
        assert!(document.groups.is_empty());
        assert!(!prepared.changed);
        assert!(prepared.values.is_empty());
        assert!(prepared.user_path.is_none());
    }
}
