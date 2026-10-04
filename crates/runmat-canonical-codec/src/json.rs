//! Bounded JSON transport; canonical contract identity remains owned by the CBOR codec.
use crate::CanonicalCodecError;
use serde::Serialize;
use std::io::{self, Write};

/// Count the actual JSON representation without constructing a value tree or output buffer.
/// Custom serializers may allocate independently; callers retain semantic admission authority.
pub fn profile_json_bytes<T: Serialize + ?Sized>(
    value: &T,
    maximum_bytes: u64,
) -> Result<u64, CanonicalCodecError> {
    let mut writer = Counter {
        bytes: 0,
        maximum: maximum_bytes,
        exceeded: false,
    };
    serde_json::to_writer(&mut writer, value).map_err(|e| {
        if writer.exceeded {
            CanonicalCodecError::resource_limit("JSON byte profile", e.to_string())
        } else {
            CanonicalCodecError::invalid("JSON byte profile", e.to_string())
        }
    })?;
    Ok(writer.bytes)
}

/// Reserve the profiled size once, then reject any serializer that grows on its second pass.
/// This preserves serde_json's wire format without an intermediate serde_json::Value.
pub fn encode_json_bounded<T: Serialize + ?Sized>(
    value: &T,
    maximum_bytes: u64,
) -> Result<Vec<u8>, CanonicalCodecError> {
    let size = profile_json_bytes(value, maximum_bytes)?;
    let capacity = usize::try_from(size).map_err(|_| {
        CanonicalCodecError::resource_limit("JSON encoding", "size exceeds address space")
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|_| {
        CanonicalCodecError::resource_limit("JSON encoding", "output reservation failed")
    })?;
    let mut writer = Reserved {
        bytes,
        maximum: capacity,
    };
    serde_json::to_writer(&mut writer, value)
        .map_err(|e| CanonicalCodecError::invalid("JSON encoding", e.to_string()))?;
    if writer.bytes.len() != capacity {
        return Err(CanonicalCodecError::invalid(
            "JSON encoding",
            "serializer changed its profiled size",
        ));
    }
    Ok(writer.bytes)
}

struct Counter {
    bytes: u64,
    maximum: u64,
    exceeded: bool,
}
impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next = self
            .bytes
            .checked_add(bytes.len() as u64)
            .filter(|size| *size <= self.maximum)
            .ok_or_else(|| {
                self.exceeded = true;
                io::Error::other("JSON exceeds its byte budget")
            })?;
        self.bytes = next;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
struct Reserved {
    bytes: Vec<u8>,
    maximum: usize,
}
impl Write for Reserved {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes
            .len()
            .checked_add(bytes.len())
            .filter(|size| *size <= self.maximum)
            .ok_or_else(|| io::Error::other("JSON serializer exceeded its profiled size"))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn profiles_escaped_unicode_numeric_and_empty_payloads_against_original_wire() {
        let values = [
            serde_json::json!({"control": "\u{0}\n\t\\\"", "unicode": "λ水🙂",
            "numeric": [f64::MAX, f64::MIN_POSITIVE, -0.0], "integer": u64::MAX}),
            serde_json::json!([]),
            serde_json::json!({}),
            serde_json::Value::Null,
        ];
        for value in values {
            let original = serde_json::to_vec(&value).unwrap();
            let size = original.len() as u64;
            assert_eq!(profile_json_bytes(&value, size).unwrap(), size);
            assert_eq!(encode_json_bounded(&value, size).unwrap(), original);
            assert!(profile_json_bytes(&value, size - 1).is_err());
            assert!(encode_json_bounded(&value, size - 1).is_err());
        }
    }

    struct Changing {
        passes: AtomicUsize,
        growing: bool,
    }
    impl Serialize for Changing {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let pass = self.passes.fetch_add(1, Ordering::SeqCst);
            serializer.serialize_str(if (pass == 0) == self.growing {
                "a"
            } else {
                "a larger value"
            })
        }
    }
    #[test]
    fn serializer_changes_cannot_reallocate_or_return_a_false_admission() {
        for growing in [false, true] {
            let value = Changing {
                passes: AtomicUsize::new(0),
                growing,
            };
            assert!(encode_json_bounded(&value, 1024).is_err());
            assert_eq!(value.passes.load(Ordering::SeqCst), 2);
        }
    }
    #[test]
    fn oversized_projection_stops_before_allocating_and_second_serialization() {
        let value = Changing {
            passes: AtomicUsize::new(0),
            growing: false,
        };
        assert!(encode_json_bounded(&value, 1).is_err());
        assert_eq!(value.passes.load(Ordering::SeqCst), 1);
    }
}
