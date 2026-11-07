//! Display meshes as bytes beside the JSON.
//!
//! An assembly's display is mostly mesh arrays: positions, normals, triangle indices and edge
//! points. Written as JSON numbers they are several times their binary size, and the reader parses
//! every one. While [`collect`] runs, each such array of at least [`INLINE_MAX`] values is
//! appended to a byte buffer (little-endian `f32` or `u32`, so every offset is a multiple of 4) and
//! written in the JSON as a reference, `{"$f32": [offset, count]}` or `{"$u32": [offset, count]}`
//! (`offset` in bytes). [`collect`] returns the buffer beside the result. Outside it the arrays
//! serialize as JSON arrays, as before.
//!
//! The browser's CAD host returns the JSON and the bytes in one frame
//! (`runmat_cad_call_framed`); its worker sends the bytes to the page as a transferable buffer, and
//! the page reads each reference as a typed array over it.

use serde::ser::SerializeMap;
use serde::Serializer;
use std::cell::RefCell;

/// Arrays with fewer values than this stay inline: a reference costs about as much text.
pub const INLINE_MAX: usize = 16;

thread_local! {
    static SINK: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
}

/// Puts back the sink that was active before [`collect`], also when its closure panics.
struct Restore(Option<Vec<u8>>);

impl Drop for Restore {
    fn drop(&mut self) {
        let previous = self.0.take();
        SINK.with(|s| *s.borrow_mut() = previous);
    }
}

/// Runs `run` with mesh arrays written to a byte buffer; returns its result and the buffer.
pub fn collect<T>(run: impl FnOnce() -> T) -> (T, Vec<u8>) {
    let restore = Restore(SINK.with(|s| s.borrow_mut().replace(Vec::new())));
    let out = run();
    let bytes = SINK.with(|s| s.borrow_mut().take()).unwrap_or_default();
    drop(restore);
    (out, bytes)
}

/// Appends what `write` writes to the active buffer; their byte offset, or `None`
/// when no buffer is active.
fn append(write: impl FnOnce(&mut Vec<u8>)) -> Option<usize> {
    SINK.with(|s| {
        let mut sink = s.borrow_mut();
        let buffer = sink.as_mut()?;
        let offset = buffer.len();
        write(buffer);
        Some(offset)
    })
}

fn reference<S: Serializer>(serializer: S, tag: &str, offset: usize, count: usize) -> Result<S::Ok, S::Error> {
    let mut map = serializer.serialize_map(Some(1))?;
    map.serialize_entry(tag, &[offset as u64, count as u64])?;
    map.end()
}

/// `serialize_with` for a `Vec<f32>` mesh array.
#[allow(clippy::ptr_arg)]
pub fn f32s<S: Serializer>(values: &Vec<f32>, serializer: S) -> Result<S::Ok, S::Error> {
    if values.len() >= INLINE_MAX {
        let at = append(|b| {
            b.reserve(values.len() * 4);
            for v in values {
                b.extend_from_slice(&v.to_le_bytes());
            }
        });
        if let Some(offset) = at {
            return reference(serializer, "$f32", offset, values.len());
        }
    }
    serializer.collect_seq(values)
}

/// `serialize_with` for a `Vec<u32>` mesh array (triangle indices).
#[allow(clippy::ptr_arg)]
pub fn u32s<S: Serializer>(values: &Vec<u32>, serializer: S) -> Result<S::Ok, S::Error> {
    if values.len() >= INLINE_MAX {
        let at = append(|b| {
            b.reserve(values.len() * 4);
            for v in values {
                b.extend_from_slice(&v.to_le_bytes());
            }
        });
        if let Some(offset) = at {
            return reference(serializer, "$u32", offset, values.len());
        }
    }
    serializer.collect_seq(values)
}

#[cfg(test)]
mod tests {
    use super::collect;
    use crate::display::{BodyDisplay, DisplayLevel, FaceDisplay, PartDisplay};
    use serde_json::{json, Value};

    fn part() -> PartDisplay {
        let positions: Vec<f32> = (0..30).map(|i| i as f32 * 0.25).collect();
        let triangles: Vec<u32> = (0..18).collect();
        PartDisplay {
            bodies: vec![BodyDisplay {
                faces: vec![FaceDisplay { positions: positions.clone(), triangles: triangles.clone(), ..Default::default() }],
                ..Default::default()
            }],
            levels: vec![DisplayLevel { positions: vec![1.0, 2.0, 3.0], triangles: vec![0, 0, 0], normals: vec![] }],
            ..Default::default()
        }
    }

    /// Outside `collect` the arrays are JSON arrays; inside, the long ones are references to
    /// little-endian bytes that read back as the same values, and the short ones stay inline.
    #[test]
    fn long_mesh_arrays_travel_as_bytes_and_read_back_exactly() {
        let display = part();
        let plain = serde_json::to_value(&display).unwrap();
        assert_eq!(plain["bodies"][0]["faces"][0]["positions"].as_array().unwrap().len(), 30);

        let (framed, bytes) = collect(|| serde_json::to_value(&display).unwrap());
        let face = &framed["bodies"][0]["faces"][0];
        assert_eq!(face["positions"], json!({ "$f32": [0, 30] }));
        assert_eq!(face["triangles"], json!({ "$u32": [120, 18] }));
        assert_eq!(bytes.len(), 30 * 4 + 18 * 4);
        let read_f32 = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let read_u32 = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        assert!((0..30).all(|i| read_f32(i * 4) == display.bodies[0].faces[0].positions[i]));
        assert!((0..18).all(|i| read_u32(120 + i * 4) == i as u32));
        // A level of 3 values stays inline.
        assert!(framed["levels"][0]["positions"].is_array());
        assert_eq!(framed["levels"][0]["positions"], plain["levels"][0]["positions"]);

        // The sink is gone after `collect`.
        assert_eq!(serde_json::to_value(&display).unwrap(), plain);
        let _: Value = plain;
    }
}
