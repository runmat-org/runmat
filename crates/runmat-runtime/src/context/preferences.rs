//! Application-owned preference storage, independent of project filesystems.

use super::{RuntimeContext, RuntimeServiceFuture};
use crate::RuntimeError;
use runmat_value::Value;
use std::collections::BTreeMap;

#[path = "preferences/memory.rs"]
mod memory;
pub use memory::IsolatedMemoryPreferenceService;
#[path = "preferences/decomposition.rs"]
mod decomposition;
#[path = "preferences/document.rs"]
mod document;
#[path = "preferences/input_roots.rs"]
mod input_roots;
#[cfg(not(target_arch = "wasm32"))]
#[path = "preferences/native.rs"]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::NativePreferenceService;
#[cfg(target_arch = "wasm32")]
#[path = "preferences/browser.rs"]
mod browser;
#[cfg(target_arch = "wasm32")]
pub use browser::BrowserPreferenceService;

pub type PreferenceValues = BTreeMap<String, Value>;
pub type PreferenceGroups = BTreeMap<String, PreferenceValues>;

async fn encode_selected_preference(
    value: &Value,
    budget: &mut crate::execution::CpuCooperationBudget,
    runtime: &RuntimeContext,
) -> Result<runmat_execution::value::ValuePayload, PreferenceServiceError> {
    decomposition::validate(value, runtime, budget).await?;
    crate::execution::value_codec::encode_durable_value(value)
        .map_err(PreferenceServiceError::ValueEncoding)
}

async fn decode_selected_preference(
    payload: &runmat_execution::value::ValuePayload,
    budget: &mut crate::execution::CpuCooperationBudget,
    runtime: &RuntimeContext,
) -> Result<Value, PreferenceServiceError> {
    use crate::execution::value_codec::{
        decode_durable_value_cooperatively_in_context, CooperativeValueCodecError,
    };
    let decoded = decode_durable_value_cooperatively_in_context(payload, budget, runtime)
        .await
        .map_err(|error| match error {
            CooperativeValueCodecError::Value(error) => {
                PreferenceServiceError::ValueEncoding(error)
            }
            CooperativeValueCodecError::Execution(error) => {
                PreferenceServiceError::Execution(error)
            }
        })?;
    decomposition::validate(decoded.value(), runtime, budget).await?;
    Ok(decoded.into_unrooted_value())
}

#[derive(Debug, thiserror::Error)]
pub enum PreferenceServiceError {
    #[error("preference '{name}' does not exist in group '{group}'")]
    Missing { group: String, name: String },
    #[error("invalid preference transaction: {0}")]
    InvalidTransaction(String),
    #[error("preference value encoding failed: {0}")]
    ValueEncoding(
        #[from]
        #[source]
        crate::execution::value_codec::ValueCodecError,
    ),
    #[error("decomposition preference binding failed: {0}")]
    Decomposition(#[source] RuntimeError),
    #[error("preference storage failed: {0}")]
    Storage(#[source] RuntimeError),
    #[error("preference execution failed: {0}")]
    Execution(#[source] crate::execution::ExecutionServiceError),
    #[error("preference commit outcome could not be determined: {0}")]
    CommitIndeterminate(#[source] RuntimeError),
    #[error("unsupported preference document version {0}")]
    UnsupportedDocumentVersion(u32),
    #[error("invalid preference document: {0}")]
    Document(#[source] serde_json::Error),
    #[error("application settings contain an invalid empty custom user path")]
    InvalidStoredUserPath,
    #[error("preference I/O failed: {0}")]
    Io(#[source] std::io::Error),
}

/// Installed by the application host, not created by a builtin or session.
/// Cloned sessions share this authority; changing the project filesystem does
/// not change it. Production implementations commit to application/user
/// storage, while deterministic hosts explicitly install isolated storage.
///
/// Each future checks its invocation's cancellation and session lease before
/// effects and final commit authorization. Preparation and staged writes are
/// abortable on cancellation or drop. Once an atomic host commit has been
/// authorized, a dropped caller cannot be promised rollback: the host-owned
/// active transaction must observe its definitive outcome, release its owner,
/// and never report cancellation as proof of rollback. A live caller receives
/// definitive success/failure or a typed indeterminate outcome if observation
/// is lost. Mutating operations are atomic across every supplied name: all
/// values are validated and encoded before authorization, and concurrent
/// transactions must not lose unrelated updates. Implementations report
/// encode, persistence and schema failures rather than resetting stored data.
pub trait RuntimePreferenceService {
    /// `None` means retained-value visibility is unavailable, not empty.
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        None
    }

    fn all_groups(
        &self,
        runtime: RuntimeContext,
    ) -> RuntimeServiceFuture<Result<PreferenceGroups, PreferenceServiceError>>;

    /// Distinguishes an absent group from an existing empty group.
    fn group_values(
        &self,
        runtime: RuntimeContext,
        group: String,
    ) -> RuntimeServiceFuture<Result<Option<PreferenceValues>, PreferenceServiceError>>;

    /// Preserves request order, including duplicate names and empty requests.
    fn contains_names(
        &self,
        runtime: RuntimeContext,
        group: String,
        names: Vec<String>,
    ) -> RuntimeServiceFuture<Result<Vec<bool>, PreferenceServiceError>>;

    /// Reads every name from one consistent transaction. Defaults, if present,
    /// have exactly the same cardinality as names and are inserted only for
    /// missing entries. A missing name without a default leaves storage intact.
    /// When a missing name is repeated, its first default is inserted and every
    /// occurrence reads that same value, even if later defaults differ.
    /// Output order matches names; an empty request must not create a group.
    fn read_or_create_many(
        &self,
        runtime: RuntimeContext,
        group: String,
        names: Vec<String>,
        defaults: Option<Vec<Value>>,
    ) -> RuntimeServiceFuture<Result<Vec<Value>, PreferenceServiceError>>;

    /// Commits all entries together; duplicate names follow supplied order
    /// (last assignment wins). Empty entries must not create a group.
    fn set_many(
        &self,
        runtime: RuntimeContext,
        group: String,
        entries: Vec<(String, Value)>,
    ) -> RuntimeServiceFuture<Result<(), PreferenceServiceError>>;
}
