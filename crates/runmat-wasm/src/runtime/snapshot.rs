use wasm_bindgen::prelude::JsValue;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::runtime::config::InitOptions;
use crate::wire::errors::js_error;

pub(crate) async fn resolve_snapshot_bytes(opts: &InitOptions) -> Result<Option<Vec<u8>>, JsValue> {
    if let Some(bytes) = opts.snapshot_bytes.clone() {
        return Ok(Some(bytes));
    }

    #[cfg(target_arch = "wasm32")]
    {
        if let Some(stream_value) = opts.snapshot_stream.as_ref() {
            if !stream_value.is_null() && !stream_value.is_undefined() {
                let stream: web_sys::ReadableStream = stream_value
                    .clone()
                    .dyn_into()
                    .map_err(|_| js_error("snapshotStream must be a ReadableStream"))?;
                let bytes = read_stream(stream).await?;
                return Ok(Some(bytes));
            }
        }
    }

    if let Some(url) = &opts.snapshot_url {
        let bytes = fetch_snapshot_bytes(url).await?;
        return Ok(Some(bytes));
    }

    Ok(None)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_snapshot_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    use js_sys::Uint8Array;

    let window = web_sys::window().ok_or_else(|| js_error("window is unavailable"))?;
    let resp_value = JsFuture::from(window.fetch_with_str(url)).await?;
    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| js_error("Fetch response is not a Response"))?;

    if !resp.ok() {
        return Err(js_error(&format!(
            "Failed to fetch snapshot from {url}: status {}",
            resp.status()
        )));
    }

    if let Some(body) = resp.body() {
        return read_stream(body).await;
    }

    let buffer = JsFuture::from(resp.array_buffer()?).await?;
    let array = Uint8Array::new(&buffer);
    Ok(array.to_vec())
}

#[cfg(target_arch = "wasm32")]
async fn read_stream(stream: web_sys::ReadableStream) -> Result<Vec<u8>, JsValue> {
    use js_sys::{Reflect, Uint8Array};
    use web_sys::ReadableStreamDefaultReader;

    let reader_value = stream.get_reader();
    let reader: ReadableStreamDefaultReader = reader_value
        .dyn_into()
        .map_err(|_| js_error("Failed to cast reader"))?;

    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut total = 0usize;
    loop {
        let promise = reader.read();
        let result = JsFuture::from(promise).await?;
        let done = Reflect::get(&result, &JsValue::from_str("done"))?
            .as_bool()
            .unwrap_or(false);
        if done {
            break;
        }
        let value = Reflect::get(&result, &JsValue::from_str("value"))?;
        if value.is_undefined() || value.is_null() {
            continue;
        }
        let chunk = Uint8Array::new(&value);
        let mut buffer = vec![0u8; chunk.length() as usize];
        chunk.copy_to(&mut buffer[..]);
        total += buffer.len();
        chunks.push(buffer);
    }

    let mut merged = Vec::with_capacity(total);
    for chunk in chunks {
        merged.extend_from_slice(&chunk);
    }
    Ok(merged)
}

#[cfg(not(target_arch = "wasm32"))]
async fn fetch_snapshot_bytes(_url: &str) -> Result<Vec<u8>, JsValue> {
    Err(js_error(
        "Snapshot fetching is only supported on wasm32 targets",
    ))
}
