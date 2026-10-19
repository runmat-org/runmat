// Rust-to-JS wire contracts and serialization helpers.
pub(crate) mod errors;
pub(crate) mod payloads;
pub(crate) mod value;

/// Preserve the canonical JSON contract at typed API boundaries: explicit optional
/// values are null, maps are objects, and serde field omissions stay omitted.
/// Serialize directly to JS without an intermediate string or JSON value tree.
pub(crate) fn json_contract_to_js<T: serde::Serialize + ?Sized>(
    value: &T,
) -> Result<wasm_bindgen::JsValue, serde_wasm_bindgen::Error> {
    value.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
}
