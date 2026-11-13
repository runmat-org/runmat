//! Structural admission for JSON decoding without constructing the value tree.
use crate::{
    materialization_resident_bytes_estimate, value::MaterializationProfile, CanonicalCodecError,
    CanonicalLimits,
};
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use std::fmt;

/// Shared canonical materialization estimate for an existing JSON transport.
/// Traversal retains no value tree or copied fields. The serde_json parser may
/// retain an escape-decoding scratch buffer; the shared estimate includes wire
/// buffers. This is requested residency, not measured heap/RSS or whole-task admission.
pub fn json_materialization_resident_bytes_estimate(
    bytes: &[u8],
    limits: CanonicalLimits,
    maximum_resident_bytes: u64,
) -> Result<u64, CanonicalCodecError> {
    let invalid = |reason| CanonicalCodecError::invalid("JSON materialization profile", reason);
    if bytes.len() > limits.maximum_encoded_bytes {
        return Err(CanonicalCodecError::resource_limit(
            "JSON materialization profile",
            "encoded byte limit exceeded",
        ));
    }
    let mut counter = Counter {
        profile: MaterializationProfile::default(),
        limits,
        encoded_bytes: bytes.len() as u64,
        maximum: maximum_resident_bytes,
        failure: None,
    };
    counter.check()?;
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    let result = Count {
        counter: &mut counter,
        depth: 0,
    }
    .deserialize(&mut decoder);
    if let Err(error) = result {
        return Err(counter
            .failure
            .take()
            .unwrap_or_else(|| invalid(error.to_string())));
    }
    decoder.end().map_err(|e| invalid(e.to_string()))?;
    counter.check()
}
struct Counter {
    profile: MaterializationProfile,
    limits: CanonicalLimits,
    encoded_bytes: u64,
    maximum: u64,
    failure: Option<CanonicalCodecError>,
}
impl Counter {
    fn check(&mut self) -> Result<u64, CanonicalCodecError> {
        let bytes = materialization_resident_bytes_estimate(self.profile, self.encoded_bytes)
            .inspect_err(|error| {
                self.failure = Some(error.clone());
            })?;
        if bytes > self.maximum {
            let error = CanonicalCodecError::resource_limit(
                "JSON materialization profile",
                "resident byte budget exceeded",
            );
            self.failure = Some(error.clone());
            return Err(error);
        }
        Ok(bytes)
    }
    fn limit<E: de::Error>(&mut self, reason: &'static str) -> E {
        let error = CanonicalCodecError::resource_limit("JSON materialization profile", reason);
        self.failure = Some(error.clone());
        E::custom(error)
    }
    fn text<E: de::Error>(&mut self, text: &str, key: bool) -> Result<(), E> {
        if text.len() > self.limits.maximum_string_bytes {
            return Err(self.limit("string byte limit exceeded"));
        }
        self.profile.string_bytes = self
            .profile
            .string_bytes
            .checked_add(text.len() as u64)
            .ok_or_else(|| self.limit("string byte count overflow"))?;
        if key {
            self.profile.map_entries = self
                .profile
                .map_entries
                .checked_add(1)
                .ok_or_else(|| self.limit("map entry count overflow"))?;
        }
        self.check().map_err(E::custom)?;
        Ok(())
    }
}
struct Count<'a> {
    counter: &'a mut Counter,
    depth: usize,
}
impl<'de> DeserializeSeed<'de> for Count<'_> {
    type Value = ();
    fn deserialize<D: de::Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        if self.depth > self.counter.limits.maximum_nesting_depth {
            return Err(self.counter.limit("nesting limit exceeded"));
        }
        self.counter.profile.value_nodes = self
            .counter
            .profile
            .value_nodes
            .checked_add(1)
            .ok_or_else(|| self.counter.limit("value count overflow"))?;
        self.counter.check().map_err(de::Error::custom)?;
        decoder.deserialize_any(self)
    }
}
impl<'de> Visitor<'de> for Count<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a bounded JSON value")
    }
    fn visit_unit<E: de::Error>(self) -> Result<(), E> {
        Ok(())
    }
    fn visit_bool<E: de::Error>(self, _: bool) -> Result<(), E> {
        Ok(())
    }
    fn visit_i64<E: de::Error>(self, _: i64) -> Result<(), E> {
        Ok(())
    }
    fn visit_u64<E: de::Error>(self, _: u64) -> Result<(), E> {
        Ok(())
    }
    fn visit_f64<E: de::Error>(self, value: f64) -> Result<(), E> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(E::custom("nonfinite JSON number"))
        }
    }
    fn visit_str<E: de::Error>(self, value: &str) -> Result<(), E> {
        self.counter.text(value, false)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<(), A::Error> {
        let mut items = 0usize;
        while sequence
            .next_element_seed(Count {
                counter: &mut *self.counter,
                depth: self.depth + 1,
            })?
            .is_some()
        {
            items = items
                .checked_add(1)
                .ok_or_else(|| self.counter.limit("array count overflow"))?;
            if items > self.counter.limits.maximum_collection_items {
                return Err(self.counter.limit("array item limit exceeded"));
            }
        }
        Ok(())
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let mut items = 0usize;
        while map.next_key_seed(Key(&mut *self.counter))?.is_some() {
            items = items
                .checked_add(1)
                .ok_or_else(|| self.counter.limit("map count overflow"))?;
            if items > self.counter.limits.maximum_collection_items {
                return Err(self.counter.limit("map item limit exceeded"));
            }
            map.next_value_seed(Count {
                counter: &mut *self.counter,
                depth: self.depth + 1,
            })?;
        }
        Ok(())
    }
}
struct Key<'a>(&'a mut Counter);
impl<'de> DeserializeSeed<'de> for Key<'_> {
    type Value = ();
    fn deserialize<D: de::Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        decoder.deserialize_str(self)
    }
}
impl Visitor<'_> for Key<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a bounded JSON key")
    }
    fn visit_str<E: de::Error>(self, value: &str) -> Result<(), E> {
        self.0.text(value, true)
    }
}

#[cfg(test)]
mod tests;
