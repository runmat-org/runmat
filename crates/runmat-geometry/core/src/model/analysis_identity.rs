use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const PERSISTENT_SELECTION_PREFIX: &str = "persistent-entity:sha256:";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeometryContractError {
    pub field: String,
    pub reason: String,
}

impl GeometryContractError {
    pub fn invalid(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for GeometryContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for GeometryContractError {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryTolerancePolicy {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub source_tolerance_m: runmat_types::physical::Length,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub absolute_floor_m: runmat_types::physical::Length,
    pub model_relative_term: f64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub requested_deviation_m: runmat_types::physical::Length,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub maximum_healing_displacement_m: runmat_types::physical::Length,
}

impl GeometryTolerancePolicy {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        for (field, value) in [
            (
                "source_tolerance_m",
                self.source_tolerance_m.canonical_value(),
            ),
            ("absolute_floor_m", self.absolute_floor_m.canonical_value()),
            ("model_relative_term", self.model_relative_term),
            (
                "requested_deviation_m",
                self.requested_deviation_m.canonical_value(),
            ),
            (
                "maximum_healing_displacement_m",
                self.maximum_healing_displacement_m.canonical_value(),
            ),
        ] {
            validate_finite(field, value)?;
            if value < 0.0 {
                return Err(GeometryContractError::invalid(
                    field,
                    "must be non-negative",
                ));
            }
        }
        if self.requested_deviation_m.canonical_value() == 0.0 {
            return Err(GeometryContractError::invalid(
                "requested_deviation_m",
                "must be greater than zero",
            ));
        }
        Ok(())
    }

    pub fn equivalence_tolerance_m(
        &self,
        model_scale_m: f64,
    ) -> Result<f64, GeometryContractError> {
        self.validate()?;
        validate_finite("model_scale_m", model_scale_m)?;
        if model_scale_m < 0.0 {
            return Err(GeometryContractError::invalid(
                "model_scale_m",
                "must be non-negative",
            ));
        }
        Ok(self
            .source_tolerance_m
            .canonical_value()
            .max(self.absolute_floor_m.canonical_value())
            .max(self.model_relative_term * model_scale_m))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersistentEntityKind {
    Assembly,
    Instance,
    Body,
    Lump,
    Solid,
    Shell,
    Face,
    Wire,
    Coedge,
    Edge,
    Vertex,
    Region,
    Contact,
}

/// The value of one persistent exact-geometry identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersistentEntityIdData {
    pub kind: PersistentEntityKind,
    pub source_topology_id: String,
    #[serde(default)]
    pub assembly_path: Vec<String>,
}

/// A shared, interned handle to a persistent exact-geometry identity.
///
/// Meshes repeat a few thousand distinct identities on millions of records. Every handle with
/// equal contents created through `new` or deserialization shares one allocation, so a record
/// holds a pointer. Equality, ordering, hashing and the wire format are those of the value.
#[derive(Clone)]
pub struct PersistentEntityId(std::sync::Arc<PersistentEntityIdData>);

impl PersistentEntityId {
    pub fn new(
        kind: PersistentEntityKind,
        source_topology_id: String,
        assembly_path: Vec<String>,
    ) -> Self {
        Self::from_data(PersistentEntityIdData {
            kind,
            source_topology_id,
            assembly_path,
        })
    }

    pub fn from_data(data: PersistentEntityIdData) -> Self {
        Self(interner::intern(data))
    }

    pub fn data(&self) -> &PersistentEntityIdData {
        &self.0
    }

    /// Address of the shared allocation; equal for handles that share storage. Inventories use
    /// it to charge each distinct allocation once.
    pub fn shared_address(&self) -> usize {
        std::sync::Arc::as_ptr(&self.0) as usize
    }

    /// Heap owned by the one shared allocation behind this identity. Records holding a handle
    /// own only the pointer; charge this once per distinct identity.
    pub fn shared_heap_bytes(&self) -> u64 {
        let data = self.data();
        (std::mem::size_of::<PersistentEntityIdData>()
            + 2 * std::mem::size_of::<usize>()
            + data.source_topology_id.capacity()
            + data.assembly_path.capacity() * std::mem::size_of::<String>()
            + data
                .assembly_path
                .iter()
                .map(String::capacity)
                .sum::<usize>()) as u64
    }
}

impl std::ops::Deref for PersistentEntityId {
    type Target = PersistentEntityIdData;
    fn deref(&self) -> &PersistentEntityIdData {
        &self.0
    }
}

/// Value semantics: mutating a shared identity detaches this handle from the others.
impl std::ops::DerefMut for PersistentEntityId {
    fn deref_mut(&mut self) -> &mut PersistentEntityIdData {
        std::sync::Arc::make_mut(&mut self.0)
    }
}

impl std::fmt::Debug for PersistentEntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersistentEntityId")
            .field("kind", &self.kind)
            .field("source_topology_id", &self.source_topology_id)
            .field("assembly_path", &self.assembly_path)
            .finish()
    }
}

impl PartialEq for PersistentEntityId {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0
    }
}
impl Eq for PersistentEntityId {}
impl PartialOrd for PersistentEntityId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PersistentEntityId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if std::sync::Arc::ptr_eq(&self.0, &other.0) {
            std::cmp::Ordering::Equal
        } else {
            self.0.cmp(&other.0)
        }
    }
}
impl std::hash::Hash for PersistentEntityId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}
impl Serialize for PersistentEntityId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for PersistentEntityId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        PersistentEntityIdData::deserialize(deserializer).map(Self::from_data)
    }
}

/// Process-wide weak interner. Entries hold no ownership; dead entries are pruned as the table
/// grows, so identities of dropped geometry do not accumulate.
mod interner {
    use super::PersistentEntityIdData;
    use std::collections::HashMap;
    use std::hash::BuildHasher;
    use std::sync::{Arc, Mutex, OnceLock, Weak};

    const SHARDS: usize = 64;

    #[derive(Default)]
    struct Shard {
        entries: HashMap<u64, Vec<Weak<PersistentEntityIdData>>>,
        weak_count: usize,
        prune_at: usize,
    }

    struct Table {
        hasher: std::collections::hash_map::RandomState,
        shards: Vec<Mutex<Shard>>,
    }

    fn table() -> &'static Table {
        static TABLE: OnceLock<Table> = OnceLock::new();
        TABLE.get_or_init(|| Table {
            hasher: std::collections::hash_map::RandomState::new(),
            shards: (0..SHARDS).map(|_| Mutex::new(Shard::default())).collect(),
        })
    }

    pub(super) fn intern(data: PersistentEntityIdData) -> Arc<PersistentEntityIdData> {
        let table = table();
        let hash = table.hasher.hash_one(&data);
        let mut shard = table.shards[(hash as usize) % SHARDS]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let shard = &mut *shard;
        let bucket = shard.entries.entry(hash).or_default();
        let before = bucket.len();
        let mut found = None;
        bucket.retain(|weak| match weak.upgrade() {
            Some(live) => {
                if found.is_none() && *live == data {
                    found = Some(live);
                }
                true
            }
            None => false,
        });
        let removed = before - bucket.len();
        if let Some(live) = found {
            shard.weak_count -= removed;
            return live;
        }
        let shared = Arc::new(data);
        bucket.push(Arc::downgrade(&shared));
        shard.weak_count = shard.weak_count - removed + 1;
        if shard.weak_count > shard.prune_at {
            shard.entries.retain(|_, bucket| {
                bucket.retain(|weak| weak.strong_count() > 0);
                !bucket.is_empty()
            });
            shard.weak_count = shard.entries.values().map(Vec::len).sum();
            shard.prune_at = (shard.weak_count * 2).max(1024);
        }
        shared
    }
}

impl PersistentEntityId {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        validate_token(
            "persistent entity source topology id",
            &self.source_topology_id,
            512,
        )?;
        if self.assembly_path.len() > 256 {
            return Err(GeometryContractError::invalid(
                "persistent entity assembly path",
                "must contain at most 256 segments",
            ));
        }
        for segment in &self.assembly_path {
            validate_token("persistent entity assembly path segment", segment, 256)?;
        }
        Ok(())
    }

    /// Returns the canonical analysis selector for this exact geometry occurrence.
    ///
    /// Unlike `source_topology_id`, this identity includes the entity kind and assembly path, so
    /// repeated occurrences of the same source part cannot alias one another in solver boundary
    /// selection.
    pub fn analysis_selection_id(&self) -> Result<String, GeometryContractError> {
        self.validate()?;
        let mut digest = Sha256::new();
        digest.update(b"runmat.geometry.persistent-analysis-selection/v1\0");
        digest.update([persistent_kind_tag(self.kind)]);
        update_digest_token(&mut digest, &self.source_topology_id);
        digest.update((self.assembly_path.len() as u64).to_be_bytes());
        for segment in &self.assembly_path {
            update_digest_token(&mut digest, segment);
        }
        let digest = digest.finalize();
        let mut selection = String::with_capacity(PERSISTENT_SELECTION_PREFIX.len() + 64);
        selection.push_str(PERSISTENT_SELECTION_PREFIX);
        for byte in digest {
            use std::fmt::Write as _;
            write!(&mut selection, "{byte:02x}").expect("writing to a String cannot fail");
        }
        Ok(selection)
    }

    /// Validate the sole public string encoding of an exact entity selection.
    pub fn validate_analysis_selection_id(value: &str) -> Result<(), GeometryContractError> {
        let valid = value
            .strip_prefix(PERSISTENT_SELECTION_PREFIX)
            .is_some_and(|digest| {
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            });
        if valid {
            Ok(())
        } else {
            Err(GeometryContractError::invalid(
                "persistent selection",
                "requires the canonical occurrence-qualified ID returned by geometry inspection",
            ))
        }
    }

    /// Tests an analysis region selector against this exact entity.
    ///
    /// Only canonical, occurrence-qualified selectors are admitted. Source topology names and
    /// display ordinals are metadata, never alternate selection identities.
    pub fn matches_analysis_selection(&self, selection_id: &str) -> bool {
        self.analysis_selection_id().as_deref() == Ok(selection_id)
    }
}

fn update_digest_token(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_be_bytes());
    digest.update(value.as_bytes());
}

const fn persistent_kind_tag(kind: PersistentEntityKind) -> u8 {
    match kind {
        PersistentEntityKind::Assembly => 0,
        PersistentEntityKind::Instance => 1,
        PersistentEntityKind::Body => 2,
        PersistentEntityKind::Lump => 3,
        PersistentEntityKind::Solid => 4,
        PersistentEntityKind::Shell => 5,
        PersistentEntityKind::Face => 6,
        PersistentEntityKind::Wire => 7,
        PersistentEntityKind::Coedge => 8,
        PersistentEntityKind::Edge => 9,
        PersistentEntityKind::Vertex => 10,
        PersistentEntityKind::Region => 11,
        PersistentEntityKind::Contact => 12,
    }
}

pub(crate) fn validate_token(
    field: &str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), GeometryContractError> {
    if value.is_empty()
        || value.len() > maximum_bytes
        || !value.is_ascii()
        || value.chars().any(char::is_control)
        || value.trim() != value
    {
        return Err(GeometryContractError::invalid(
            field,
            format!("must be 1..={maximum_bytes} printable ASCII bytes without surrounding space"),
        ));
    }
    Ok(())
}

fn validate_finite(field: &str, value: f64) -> Result<(), GeometryContractError> {
    if !value.is_finite() {
        return Err(GeometryContractError::invalid(field, "must be finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerance_is_geometry_owned_and_scale_aware() {
        let policy = GeometryTolerancePolicy {
            source_tolerance_m: runmat_types::physical::Length::new(1.0e-8),
            absolute_floor_m: runmat_types::physical::Length::new(1.0e-10),
            model_relative_term: 1.0e-6,
            requested_deviation_m: runmat_types::physical::Length::new(1.0e-4),
            maximum_healing_displacement_m: runmat_types::physical::Length::new(1.0e-5),
        };
        assert_eq!(policy.equivalence_tolerance_m(2.0).unwrap(), 2.0e-6);
        assert!(policy.equivalence_tolerance_m(f64::NAN).is_err());
    }

    #[test]
    fn persistent_identity_rejects_ambiguous_tokens() {
        let valid = PersistentEntityId::new(
            PersistentEntityKind::Face,
            "step-label:42".into(),
            vec!["root".into(), "instance-7".into()],
        );
        valid.validate().unwrap();
        assert!({
            let __base = &(valid);
            PersistentEntityId::new(
                __base.kind,
                " step-label:42".into(),
                __base.assembly_path.clone(),
            )
        }
        .validate()
        .is_err());
    }

    #[test]
    fn analysis_selection_identity_qualifies_repeated_occurrences() {
        let first = PersistentEntityId::new(
            PersistentEntityKind::Face,
            "shared-part-face".into(),
            vec!["root".into(), "instance-a".into()],
        );
        let second = {
            let __base = &(first.clone());
            PersistentEntityId::new(
                __base.kind,
                __base.source_topology_id.clone(),
                vec!["root".into(), "instance-b".into()],
            )
        };

        let first_selection = first.analysis_selection_id().unwrap();
        let second_selection = second.analysis_selection_id().unwrap();
        assert_ne!(first_selection, second_selection);
        assert!(first.matches_analysis_selection(&first_selection));
        assert!(!second.matches_analysis_selection(&first_selection));
        assert!(!first.matches_analysis_selection("shared-part-face"));
        assert!(!second.matches_analysis_selection("shared-part-face"));
    }
}

impl From<runmat_types::physical::UnitError> for GeometryContractError {
    fn from(error: runmat_types::physical::UnitError) -> Self {
        Self::invalid("physical quantity", error.to_string())
    }
}

#[cfg(test)]
mod canonical_selection_contract_tests {
    use super::*;
    #[test]
    fn ordinal_aliases_and_unqualified_names_are_never_persistent_ids() {
        for value in [
            "face_000001",
            "solid_000001",
            "cad_label_0_1_1_1",
            "occt:face",
            "persistent-entity:sha256:ABC",
        ] {
            assert!(PersistentEntityId::validate_analysis_selection_id(value).is_err());
        }
        let id = PersistentEntityId::new(
            PersistentEntityKind::Face,
            "exact-face".into(),
            vec!["root".into()],
        );
        PersistentEntityId::validate_analysis_selection_id(&id.analysis_selection_id().unwrap())
            .unwrap();
        assert!(serde_json::from_str::<crate::GeometryRegionSelector>(
            r#"{"kind":"source_alias","value":"face_000001"}"#
        )
        .is_err());
    }

    #[test]
    fn equal_identities_share_one_allocation_and_keep_the_wire_format() {
        let first = PersistentEntityId::new(
            PersistentEntityKind::Edge,
            "interned-edge".into(),
            vec!["root".into()],
        );
        let second = PersistentEntityId::new(
            PersistentEntityKind::Edge,
            "interned-edge".into(),
            vec!["root".into()],
        );
        assert_eq!(first.shared_address(), second.shared_address());
        let json = serde_json::to_string(&first).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"edge","source_topology_id":"interned-edge","assembly_path":["root"]}"#
        );
        let decoded: PersistentEntityId = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.shared_address(), first.shared_address());
        let defaulted: PersistentEntityId =
            serde_json::from_str(r#"{"kind":"edge","source_topology_id":"interned-edge"}"#)
                .unwrap();
        assert_eq!(defaulted.assembly_path, Vec::<String>::new());

        let mut detached = first.clone();
        detached.source_topology_id = "detached-edge".into();
        assert_eq!(first.source_topology_id, "interned-edge");
        assert_ne!(detached, first);
        assert_ne!(detached.cmp(&first), std::cmp::Ordering::Equal);
    }

    #[test]
    fn dropped_identities_are_not_retained_by_the_interner() {
        let address = {
            let id = PersistentEntityId::new(
                PersistentEntityKind::Vertex,
                "transient-vertex".into(),
                Vec::new(),
            );
            let weak = std::sync::Arc::downgrade(&id.0);
            drop(id);
            assert!(weak.upgrade().is_none());
            1
        };
        assert_eq!(address, 1);
    }
}
