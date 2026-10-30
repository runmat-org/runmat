//! Versioned preference document, independent of execution ValueRef V2.

use super::PreferenceServiceError;
use crate::context::UserPathSetting;
use crate::execution::value_codec::{validate_durable_payload, ValueCodecError};
use runmat_execution::value::ValuePayload;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::collections::HashSet;

#[path = "document/mutation.rs"]
mod mutation;
#[cfg(not(target_arch = "wasm32"))]
pub(super) use mutation::DocumentPrepared;
pub(super) use mutation::{apply as apply_mutation, DocumentMutation, DocumentMutationError};

// V2 makes native NumericFactor DATA a required, typed part of a durable
// object rather than an ignorable JSON extension in an older reader.
pub(super) const PREFERENCE_DOCUMENT_VERSION: u32 = 2;
pub(super) type EncodedGroups = BTreeMap<String, BTreeMap<String, ValuePayload>>;

/// Indices whose first occurrence is absent from the authoritative document.
/// A repeated missing name uses its first default; later defaults are ignored.
pub(super) fn missing_default_indices(
    document: &PreferenceDocument,
    group: &str,
    names: &[String],
) -> Vec<usize> {
    let existing = document.groups.get(group);
    let mut seen = HashSet::with_capacity(names.len());
    names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| {
            (seen.insert(name.as_str())
                && !existing.is_some_and(|values| values.contains_key(name)))
            .then_some(index)
        })
        .collect()
}

#[derive(Debug, thiserror::Error)]
pub(super) enum PreferenceDocumentError {
    #[error("unsupported preference document version {0}")]
    Version(u32),
    #[error("invalid preference document: {0}")]
    Json(#[source] serde_json::Error),
    #[error("invalid durable preference value: {0}")]
    Value(#[source] ValueCodecError),
    #[error("invalid empty custom user path in application settings")]
    InvalidUserPath,
}

impl From<PreferenceDocumentError> for PreferenceServiceError {
    fn from(error: PreferenceDocumentError) -> Self {
        match error {
            PreferenceDocumentError::Version(version) => Self::UnsupportedDocumentVersion(version),
            PreferenceDocumentError::Json(source) => Self::Document(source),
            PreferenceDocumentError::Value(source) => Self::ValueEncoding(source),
            PreferenceDocumentError::InvalidUserPath => Self::InvalidStoredUserPath,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct PreferenceDocument {
    pub version: u32,
    pub groups: EncodedGroups,
    pub user_path: UserPathSetting,
    pub user_path_revision: u64,
    /// Application settings share this document. Retain fields owned by other
    /// schemas (including runtime settings), without interpreting or mirroring
    /// them as durable MATLAB preference values.
    #[serde(flatten)]
    application_fields: BTreeMap<String, serde_json::Value>,
}

/// Unlike Option<T>, this discriminator never treats explicit null as absence.
/// Typed preference fields must pass their own Deserialize implementation.
enum StoredField<T> {
    Absent,
    Present(T),
}

impl<T> Default for StoredField<T> {
    fn default() -> Self {
        Self::Absent
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for StoredField<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize(deserializer).map(Self::Present)
    }
}

#[derive(Deserialize)]
struct StoredApplicationDocument {
    #[serde(default)]
    version: StoredField<u32>,
    #[serde(default)]
    groups: StoredField<EncodedGroups>,
    #[serde(default)]
    user_path: StoredField<UserPathSetting>,
    #[serde(default)]
    user_path_revision: StoredField<u64>,
    #[serde(flatten)]
    application_fields: BTreeMap<String, serde_json::Value>,
}

impl PreferenceDocument {
    pub fn empty() -> Self {
        Self {
            version: PREFERENCE_DOCUMENT_VERSION,
            groups: BTreeMap::new(),
            user_path: UserPathSetting::Default,
            user_path_revision: 0,
            application_fields: BTreeMap::new(),
        }
    }

    /// Settings-only documents contain no preference fields and hydrate empty
    /// preferences. Once any durable preference field exists, its explicit
    /// current schema version remains mandatory. Old/future schemas are never
    /// silently reinterpreted as execution V2 values.
    pub fn decode(bytes: &[u8]) -> Result<Self, PreferenceDocumentError> {
        let stored: StoredApplicationDocument =
            serde_json::from_slice(bytes).map_err(PreferenceDocumentError::Json)?;
        let document = match stored.version {
            StoredField::Absent => {
                if !matches!(stored.groups, StoredField::Absent)
                    || !matches!(stored.user_path, StoredField::Absent)
                    || !matches!(stored.user_path_revision, StoredField::Absent)
                {
                    return Err(PreferenceDocumentError::Json(
                        <serde_json::Error as serde::de::Error>::custom(
                            "preference fields require an explicit current version",
                        ),
                    ));
                }
                Self {
                    application_fields: stored.application_fields,
                    ..Self::empty()
                }
            }
            StoredField::Present(version) => {
                if version != PREFERENCE_DOCUMENT_VERSION {
                    return Err(PreferenceDocumentError::Version(version));
                }
                let StoredField::Present(groups) = stored.groups else {
                    return Err(PreferenceDocumentError::Json(
                        <serde_json::Error as serde::de::Error>::missing_field("groups"),
                    ));
                };
                Self {
                    version,
                    groups,
                    user_path: match stored.user_path {
                        StoredField::Absent => UserPathSetting::Default,
                        StoredField::Present(setting) => setting,
                    },
                    user_path_revision: match stored.user_path_revision {
                        StoredField::Absent => 0,
                        StoredField::Present(revision) => revision,
                    },
                    application_fields: stored.application_fields,
                }
            }
        };
        if matches!(&document.user_path, UserPathSetting::Custom(text) if text.is_empty()) {
            return Err(PreferenceDocumentError::InvalidUserPath);
        }
        for values in document.groups.values() {
            for payload in values.values() {
                validate_durable_payload(payload).map_err(PreferenceDocumentError::Value)?;
            }
        }
        Ok(document)
    }

    pub fn encode(&self) -> Result<Vec<u8>, PreferenceDocumentError> {
        if self.version != PREFERENCE_DOCUMENT_VERSION {
            return Err(PreferenceDocumentError::Version(self.version));
        }
        serde_json::to_vec(self).map_err(PreferenceDocumentError::Json)
    }
}

impl Default for PreferenceDocument {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::value_codec::encode_durable_value;
    use runmat_value::{IntValue, ObjectInstance, Value};

    #[test]
    fn missing_default_selection_ignores_existing_and_repeated_names() {
        let mut document = PreferenceDocument::empty();
        document.groups.entry("app".into()).or_default().insert(
            "existing".into(),
            encode_durable_value(&Value::Num(3.0)).unwrap(),
        );
        let names = vec![
            "existing".into(),
            "new".into(),
            "new".into(),
            "other".into(),
        ];
        assert_eq!(
            missing_default_indices(&document, "app", &names),
            vec![1, 3]
        );
        assert_eq!(
            missing_default_indices(&document, "absent", &names),
            vec![0, 1, 3]
        );
        assert!(missing_default_indices(&document, "app", &[]).is_empty());
    }

    #[test]
    fn document_version_is_explicit_and_plain_objects_round_trip() {
        let mut object = ObjectInstance::new("PreferencePayload");
        object.replace_property_unbounded("revision".into(), Value::Int(IntValue::U64(u64::MAX)));
        let mut document = PreferenceDocument::empty();
        document.groups.entry("app".into()).or_default().insert(
            "object".into(),
            encode_durable_value(&Value::Object(object)).unwrap(),
        );
        let decoded = PreferenceDocument::decode(&document.encode().unwrap()).unwrap();
        assert_eq!(
            decoded.groups["app"]["object"],
            document.groups["app"]["object"]
        );
        let mut old = document;
        old.version = 0;
        let bytes = serde_json::to_vec(&old).unwrap();
        assert!(matches!(
            PreferenceDocument::decode(&bytes),
            Err(PreferenceDocumentError::Version(0))
        ));
    }

    #[test]
    fn version_two_defaults_only_the_distinct_user_setting_and_rejects_prior_schema() {
        let old = br#"{"version":1,"groups":{"app":{}}}"#;
        assert!(matches!(
            PreferenceDocument::decode(old),
            Err(PreferenceDocumentError::Version(1))
        ));
        let current = br#"{"version":2,"groups":{"app":{}}}"#;
        let decoded = PreferenceDocument::decode(current).unwrap();
        assert_eq!(decoded.user_path, UserPathSetting::Default);
        assert_eq!(decoded.user_path_revision, 0);
        assert!(decoded.groups.contains_key("app"));
        let mut updated = decoded;
        updated.user_path = UserPathSetting::Custom("relative/MiXeD".into());
        updated.user_path_revision = 7;
        let reopened = PreferenceDocument::decode(&updated.encode().unwrap()).unwrap();
        assert_eq!(reopened.user_path, updated.user_path);
        assert_eq!(reopened.user_path_revision, 7);
    }

    #[test]
    fn shared_application_settings_survive_preference_mutation_and_roundtrip() {
        let source = br#"{"runtime":{"jit":{"enabled":false,"future_knob":42}},"future_app":{"theme":"dark"}}"#;
        let document = PreferenceDocument::decode(source).unwrap();
        assert!(document.groups.is_empty());
        assert_eq!(document.user_path_revision, 0);
        let (updated, _) = apply_mutation(
            document,
            DocumentMutation::Set {
                group: "app".into(),
                entries: vec![(
                    "count".into(),
                    encode_durable_value(&Value::Num(7.0)).unwrap(),
                )],
            },
        )
        .unwrap();
        let (updated, _) = apply_mutation(
            updated,
            DocumentMutation::UserPath(UserPathSetting::Custom("project-independent/path".into())),
        )
        .unwrap();
        let bytes = updated.encode().unwrap();
        let saved: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let original: serde_json::Value = serde_json::from_slice(source).unwrap();
        assert_eq!(saved["runtime"], original["runtime"]);
        assert_eq!(saved["future_app"], original["future_app"]);
        let reopened = PreferenceDocument::decode(&bytes).unwrap();
        assert_eq!(reopened.groups["app"], updated.groups["app"]);
        assert_eq!(reopened.user_path, updated.user_path);
        assert_eq!(reopened.user_path_revision, 1);
    }

    #[test]
    fn settings_interop_does_not_weaken_durable_document_admission() {
        for invalid in [
            r#"{"groups":{}}"#,
            r#"{"user_path":"default"}"#,
            r#"{"user_path_revision":0}"#,
            r#"{"version":null,"groups":{}}"#,
            r#"{"version":2}"#,
            r#"{"version":2,"groups":null}"#,
            r#"{"version":2,"groups":{},"user_path":null}"#,
            r#"{"version":2,"groups":{},"user_path_revision":null}"#,
            r#"{"runtime":{},"groups":{},"version":1}"#,
            r#"{"runtime":{},"groups":{},"version":3}"#,
        ] {
            assert!(
                PreferenceDocument::decode(invalid.as_bytes()).is_err(),
                "accepted {invalid}"
            );
        }
        assert!(PreferenceDocument::decode(br#"{"runtime":{},"future_app":null}"#).is_ok());
        assert!(PreferenceDocument::decode(br#"{}"#).is_ok());
    }
}
