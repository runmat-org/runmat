use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::identity::ValueId;
use crate::{ContractError, Digest};

use super::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};

const PUBLISHED_OBJECT_CATALOG_SCHEMA_VERSION: u16 = 1;
const MAX_PUBLISHED_OBJECTS: usize = 100_000;
const MAX_PUBLISHED_OBJECT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_PUBLISHED_TOTAL_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Portable inventory of a published result's content-addressed object closure.
///
/// Domain manifests name the semantic roots. This execution contract binds those roots to one
/// immutable result authority and carries the complete, bounded physical inventory. Storage and
/// retention remain responsibilities of an artifact repository implementation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishedObjectCatalog {
    pub schema_version: u16,
    pub roots: Vec<ValueRef>,
    pub objects: Vec<ValueRef>,
}

impl PublishedObjectCatalog {
    /// Multiple semantic roles may share the same immutable object. Identical
    /// references are retained once; conflicting descriptions of one ID fail.
    pub fn new_with_shared_objects(
        roots: Vec<ValueRef>,
        objects: Vec<ValueRef>,
    ) -> Result<Self, ContractError> {
        fn unique(mut references: Vec<ValueRef>) -> Result<Vec<ValueRef>, ContractError> {
            references.sort_by_key(|reference| reference.id);
            for pair in references.windows(2) {
                if pair[0].id == pair[1].id && pair[0] != pair[1] {
                    return Err(ContractError::invalid(
                        "published object catalog",
                        "shared object identity has conflicting references",
                    ));
                }
            }
            references.dedup();
            Ok(references)
        }
        Self::new(unique(roots)?, unique(objects)?)
    }

    pub fn new(
        mut roots: Vec<ValueRef>,
        mut objects: Vec<ValueRef>,
    ) -> Result<Self, ContractError> {
        roots.sort_by_key(|reference| reference.id);
        objects.sort_by_key(|reference| reference.id);
        let catalog = Self {
            schema_version: PUBLISHED_OBJECT_CATALOG_SCHEMA_VERSION,
            roots,
            objects,
        };
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != PUBLISHED_OBJECT_CATALOG_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                actual: self.schema_version,
                supported: PUBLISHED_OBJECT_CATALOG_SCHEMA_VERSION,
            });
        }
        if self.roots.is_empty()
            || self.objects.is_empty()
            || !strictly_ordered(&self.roots)
            || !strictly_ordered(&self.objects)
        {
            return Err(ContractError::invalid(
                "published object catalog",
                "catalog is empty, unordered, or contains duplicate identities",
            ));
        }
        if self.objects.len() > MAX_PUBLISHED_OBJECTS {
            return Err(ContractError::Limit {
                field: "published object count",
                limit: MAX_PUBLISHED_OBJECTS as u64,
            });
        }

        let first = self
            .objects
            .first()
            .expect("nonempty catalog checked above");
        validate_authority(first)?;
        let mut total_bytes = 0_u64;
        for reference in &self.objects {
            validate_reference(reference)?;
            if reference.authorization_scope != first.authorization_scope
                || reference.encryption_context != first.encryption_context
            {
                return Err(ContractError::invalid(
                    "published object catalog",
                    "catalog crosses artifact authorities",
                ));
            }
            if reference.encoded_length > MAX_PUBLISHED_OBJECT_BYTES {
                return Err(ContractError::Limit {
                    field: "published object bytes",
                    limit: MAX_PUBLISHED_OBJECT_BYTES,
                });
            }
            total_bytes = total_bytes
                .checked_add(reference.encoded_length)
                .ok_or_else(|| {
                    ContractError::invalid(
                        "published object catalog",
                        "total encoded byte count overflows",
                    )
                })?;
        }
        if total_bytes > MAX_PUBLISHED_TOTAL_BYTES {
            return Err(ContractError::Limit {
                field: "published object catalog bytes",
                limit: MAX_PUBLISHED_TOTAL_BYTES,
            });
        }

        let object_ids = self
            .objects
            .iter()
            .map(|reference| reference.id)
            .collect::<BTreeSet<_>>();
        for root in &self.roots {
            validate_reference(root)?;
            if root.authorization_scope != first.authorization_scope
                || root.encryption_context != first.encryption_context
                || !object_ids.contains(&root.id)
                || !self.objects.iter().any(|object| object == root)
            {
                return Err(ContractError::invalid(
                    "published object catalog root",
                    "root is absent or outside the catalog artifact authority",
                ));
            }
        }
        Ok(())
    }
}

/// Process-local port for reading a declared published object without coupling execution
/// consumers to a concrete artifact repository.
///
/// Implementations must fail unless the returned bytes have exactly the encoded length and
/// logical digest declared by `reference`. The receiving artifact store owns ingestion,
/// namespace selection, and retention.
pub trait PublishedObjectSource: Send + Sync {
    fn read_verified(&self, reference: &ValueRef) -> Result<Vec<u8>, String>;
}

pub fn artifact_object_value_id(
    logical_digest: Digest,
    encryption_context: Digest,
    authorization_scope: &str,
) -> ValueId {
    ValueId::derive(&[
        b"runmat-artifact-object/v1",
        logical_digest.bytes(),
        encryption_context.bytes(),
        authorization_scope.as_bytes(),
    ])
}

fn strictly_ordered(references: &[ValueRef]) -> bool {
    references.windows(2).all(|pair| pair[0].id < pair[1].id)
}

fn validate_authority(reference: &ValueRef) -> Result<(), ContractError> {
    if reference.authorization_scope.is_empty()
        || reference.authorization_scope.len() > 1024
        || reference.authorization_scope.chars().any(char::is_control)
        || reference
            .encryption_context
            .bytes()
            .iter()
            .all(|byte| *byte == 0)
    {
        return Err(ContractError::invalid(
            "published object authority",
            "authorization scope or encryption context is invalid",
        ));
    }
    Ok(())
}

fn validate_reference(reference: &ValueRef) -> Result<(), ContractError> {
    ValuePayload::Object(Box::new(reference.clone())).validate(ValueLimits::default())?;
    validate_authority(reference)?;
    if reference.kind != ValueRefKind::ResultObject
        || reference.resident_fence.is_some()
        || reference.encoded_length == 0
        || reference.id
            != artifact_object_value_id(
                reference.logical_digest,
                reference.encryption_context,
                &reference.authorization_scope,
            )
    {
        return Err(ContractError::invalid(
            "published object reference",
            "reference is nonportable or differs from its artifact authority",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::VALUE_PAYLOAD_SCHEMA_VERSION;

    fn reference(name: &str, scope: &str) -> ValueRef {
        let logical_digest = Digest::sha256(name);
        let encryption_context = Digest::sha256(scope);
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_VERSION,
            id: artifact_object_value_id(logical_digest, encryption_context, scope),
            logical_digest,
            encoded_length: name.len() as u64,
            media_type: "application/vnd.runmat.test".into(),
            value_schema: "runmat.test".into(),
            encryption_context,
            kind: ValueRefKind::ResultObject,
            authorization_scope: scope.into(),
            resident_fence: None,
        }
    }

    #[test]
    fn catalog_requires_one_authority_and_exact_roots() {
        let root = reference("root", "first");
        let child = reference("child", "first");
        let catalog =
            PublishedObjectCatalog::new(vec![root.clone()], vec![child.clone(), root.clone()])
                .unwrap();
        assert_eq!(catalog.roots, vec![root.clone()]);
        assert!(PublishedObjectCatalog::new(vec![root.clone()], vec![child]).is_err());
        assert!(PublishedObjectCatalog::new(
            vec![root.clone()],
            vec![root, reference("foreign", "second")],
        )
        .is_err());
    }

    #[test]
    fn shared_catalog_deduplicates_exact_references_and_rejects_aliases() {
        let object = reference("shared", "study:shared");
        let catalog = PublishedObjectCatalog::new_with_shared_objects(
            vec![object.clone(), object.clone()],
            vec![object.clone(), object.clone()],
        )
        .unwrap();
        assert_eq!(catalog.roots, vec![object.clone()]);
        assert_eq!(catalog.objects, vec![object.clone()]);
        let mut substituted = object.clone();
        substituted.encoded_length += 1;
        assert!(PublishedObjectCatalog::new_with_shared_objects(
            vec![object.clone()],
            vec![object, substituted],
        )
        .is_err());
    }

    #[test]
    fn catalog_rejects_a_forged_value_identity() {
        let mut forged = reference("root", "first");
        forged.id = ValueId::derive(&[b"forged"]);
        assert!(PublishedObjectCatalog::new(vec![forged.clone()], vec![forged]).is_err());
    }
}
