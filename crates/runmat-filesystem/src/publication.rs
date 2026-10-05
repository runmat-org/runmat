/// The durability established by a successful atomic publication.
///
/// Every successful publication is immediately visible as one complete byte sequence. Providers
/// report `Durable` only when they also guarantee that the published directory entry and content
/// survive a host crash. Remote and in-memory providers generally establish atomic visibility but
/// leave durable retention to their backing service.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationDurability {
    AtomicVisibility,
    Durable,
}

/// Result of publishing one complete byte sequence at a previously absent path.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(
    tag = "kind",
    content = "result",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum AtomicCreateResult {
    Published { durability: PublicationDurability },
    Existing(Vec<u8>),
}

/// Evidence from an atomic replacement. An error from the operation means the
/// replacement was not published; this value means it was made atomically visible.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtomicReplaceOutcome {
    pub durability: PublicationDurability,
    /// Original provider diagnostic when crash durability could not be confirmed.
    pub durability_diagnostic: Option<String>,
}
