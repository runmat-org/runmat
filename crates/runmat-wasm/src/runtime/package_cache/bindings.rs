use futures::future::LocalBoxFuture;
use js_sys::{Function, Reflect, Uint8Array};
use runmat_package::ContentDigest;
use runmat_package_cache::{
    BackendError, BackendSnapshot, CacheBackend, CacheState, CacheTransaction, CommitOutcome,
};
use serde::Serialize;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

#[derive(Clone)]
pub(crate) struct JsPackageCacheBackend {
    bindings: JsValue,
    snapshot: Function,
    initialize: Function,
    commit: Function,
    read_object_bytes: Function,
}

// Browser RunMat executes JavaScript bindings on one WASM agent. This mirrors the
// filesystem adapter's host contract and must not be moved across agents.
unsafe impl Send for JsPackageCacheBackend {}
unsafe impl Sync for JsPackageCacheBackend {}

impl JsPackageCacheBackend {
    pub(crate) fn new(bindings: &JsValue) -> Result<Self, JsValue> {
        if !bindings.is_object() {
            return Err(JsValue::from_str("packageCacheProvider must be an object"));
        }
        Ok(Self {
            bindings: bindings.clone(),
            snapshot: required_function(bindings, "snapshot")?,
            initialize: required_function(bindings, "initialize")?,
            commit: required_function(bindings, "commit")?,
            read_object_bytes: required_function(bindings, "readObjectBytes")?,
        })
    }

    async fn call_snapshot(&self) -> Result<BackendSnapshot, BackendError> {
        let value = call_promise(&self.snapshot, &self.bindings, &[]).await?;
        if value.is_null() || value.is_undefined() {
            let initial = BackendSnapshot {
                revision: 0,
                state: CacheState::default(),
            };
            let serialized = to_js(&initial)?;
            let initialized = call_promise(&self.initialize, &self.bindings, &[serialized]).await?;
            from_js(initialized)
        } else {
            from_js(value)
        }
    }
}

impl CacheBackend for JsPackageCacheBackend {
    fn snapshot(&self) -> LocalBoxFuture<'_, Result<BackendSnapshot, BackendError>> {
        Box::pin(async move {
            let snapshot = self.call_snapshot().await?;
            snapshot
                .state
                .validate()
                .map_err(|error| BackendError::IncompatibleSchema(error.to_string()))?;
            Ok(snapshot)
        })
    }

    fn commit(
        &self,
        transaction: CacheTransaction,
    ) -> LocalBoxFuture<'_, Result<CommitOutcome, BackendError>> {
        Box::pin(async move {
            let current = self.snapshot().await?;
            if current.revision != transaction.expected_revision {
                return Ok(CommitOutcome::Conflict {
                    actual_revision: current.revision,
                });
            }
            transaction
                .validate_transition(&current.state)
                .map_err(|error| BackendError::Failure(error.to_string()))?;
            let serialized = to_js(&transaction)?;
            let value = call_promise(&self.commit, &self.bindings, &[serialized]).await?;
            let outcome: CommitOutcome = from_js(value)?;
            if let CommitOutcome::Committed(commit) = &outcome {
                let expected = transaction
                    .expected_revision
                    .checked_add(1)
                    .ok_or_else(|| BackendError::Failure("cache revision overflow".to_string()))?;
                if commit.revision != expected {
                    return Err(BackendError::Failure(format!(
                        "browser backend committed revision {}, expected {expected}",
                        commit.revision
                    )));
                }
            }
            Ok(outcome)
        })
    }

    fn read_object_bytes(
        &self,
        digest: &ContentDigest,
    ) -> LocalBoxFuture<'_, Result<Option<Vec<u8>>, BackendError>> {
        let digest = digest.to_string();
        Box::pin(async move {
            let value = call_promise(
                &self.read_object_bytes,
                &self.bindings,
                &[JsValue::from_str(&digest)],
            )
            .await?;
            if value.is_null() || value.is_undefined() {
                return Ok(None);
            }
            Ok(Some(Uint8Array::new(&value).to_vec()))
        })
    }
}

fn required_function(bindings: &JsValue, name: &str) -> Result<Function, JsValue> {
    let value = Reflect::get(bindings, &JsValue::from_str(name))?;
    value
        .dyn_into::<Function>()
        .map_err(|_| JsValue::from_str(&format!("packageCacheProvider.{name} must be a function")))
}

async fn call_promise(
    function: &Function,
    receiver: &JsValue,
    arguments: &[JsValue],
) -> Result<JsValue, BackendError> {
    let array = js_sys::Array::new();
    for argument in arguments {
        array.push(argument);
    }
    let value = function.apply(receiver, &array).map_err(js_backend_error)?;
    let promise = js_sys::Promise::resolve(&value);
    JsFuture::from(promise).await.map_err(js_backend_error)
}

fn to_js<T: Serialize>(value: &T) -> Result<JsValue, BackendError> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    value
        .serialize(&serializer)
        .map_err(|error| BackendError::Failure(error.to_string()))
}

fn from_js<T: serde::de::DeserializeOwned>(value: JsValue) -> Result<T, BackendError> {
    serde_wasm_bindgen::from_value(value)
        .map_err(|error| BackendError::IncompatibleSchema(error.to_string()))
}

fn js_backend_error(value: JsValue) -> BackendError {
    let name = Reflect::get(&value, &JsValue::from_str("name"))
        .ok()
        .and_then(|value| value.as_string());
    let message = value
        .as_string()
        .or_else(|| {
            Reflect::get(&value, &JsValue::from_str("message"))
                .ok()
                .and_then(|value| value.as_string())
        })
        .unwrap_or_else(|| format!("{value:?}"));
    if name.as_deref() == Some("QuotaExceededError") {
        BackendError::QuotaExceeded {
            requested_bytes: 0,
            available_bytes: 0,
        }
    } else {
        BackendError::Failure(message)
    }
}
