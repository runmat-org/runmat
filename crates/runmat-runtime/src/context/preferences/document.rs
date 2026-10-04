//! Versioned preference document, independent of execution ValueRef V2.

use super::PreferenceServiceError;
use crate::context::UserPathSetting;
use crate::execution::value_codec::{decode_durable_value, ValueCodecError};
use runmat_execution::value::ValuePayload;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[path = "document/mutation.rs"]
mod mutation;
pub(super) use mutation::{
    apply as apply_mutation, DocumentMutation, DocumentMutationError, DocumentPrepared,
};

pub(super) const PREFERENCE_DOCUMENT_VERSION: u32 = 1;
pub(super) type EncodedGroups = BTreeMap<String, BTreeMap<String, ValuePayload>>;

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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreferenceDocument {
    pub version: u32,
    pub groups: EncodedGroups,
    #[serde(default)]
    pub user_path: UserPathSetting,
    #[serde(default)]
    pub user_path_revision: u64,
}

impl PreferenceDocument {
    pub fn empty() -> Self {
        Self {
            version: PREFERENCE_DOCUMENT_VERSION,
            groups: BTreeMap::new(),
            user_path: UserPathSetting::Default,
            user_path_revision: 0,
        }
    }

    /// Unknown or old documents are rejected, never silently reset or
    /// reinterpreted as execution V2 values. Migration requires an explicit
    /// future versioned adapter owned by the host.
    pub fn decode(bytes: &[u8]) -> Result<Self, PreferenceDocumentError> {
        let document: Self =
            serde_json::from_slice(bytes).map_err(PreferenceDocumentError::Json)?;
        if document.version != PREFERENCE_DOCUMENT_VERSION {
            return Err(PreferenceDocumentError::Version(document.version));
        }
        if matches!(&document.user_path, UserPathSetting::Custom(text) if text.is_empty()) {
            return Err(PreferenceDocumentError::InvalidUserPath);
        }
        for values in document.groups.values() {
            for payload in values.values() {
                decode_durable_value(payload).map_err(PreferenceDocumentError::Value)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::value_codec::encode_durable_value;
    use runmat_value::{IntValue, ObjectInstance, Value};

    #[test]
    fn document_version_is_explicit_and_plain_objects_round_trip() {
        let mut object = ObjectInstance::new("PreferencePayload");
        object
            .properties
            .insert("revision".into(), Value::Int(IntValue::U64(u64::MAX)));
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
    fn prior_version_one_document_defaults_only_the_distinct_user_setting() {
        let old = br#"{"version":1,"groups":{"app":{}}}"#;
        let decoded = PreferenceDocument::decode(old).unwrap();
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
}
