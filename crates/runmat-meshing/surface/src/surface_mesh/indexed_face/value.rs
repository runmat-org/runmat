//! Four exact digest words use the shared canonical sequence grammar.
//! Foreign references remain foreign for independent semantic rejection.
use runmat_meshing_core::StableDigest;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DigestWords(pub StableDigest);
impl Serialize for DigestWords {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let words: [u64; 4] = std::array::from_fn(|index| {
            u64::from_be_bytes(
                self.0.bytes()[index * 8..index * 8 + 8]
                    .try_into()
                    .expect("fixed digest word"),
            )
        });
        words.serialize(serializer)
    }
}
fn read_digest<'de, A: de::SeqAccess<'de>>(mut sequence: A) -> Result<DigestWords, A::Error> {
    let mut bytes = [0; 32];
    for word in bytes.chunks_exact_mut(8) {
        let value: u64 = sequence
            .next_element()?
            .ok_or_else(|| de::Error::custom("digest requires four original words"))?;
        word.copy_from_slice(&value.to_be_bytes());
    }
    if sequence.next_element::<de::IgnoredAny>()?.is_some() {
        return Err(de::Error::custom("digest exceeds four original words"));
    }
    Ok(DigestWords(StableDigest::from_bytes(bytes)))
}
impl<'de> Deserialize<'de> for DigestWords {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> de::Visitor<'de> for Visitor {
            type Value = DigestWords;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("four original digest words")
            }
            fn visit_seq<A: de::SeqAccess<'de>>(
                self,
                sequence: A,
            ) -> Result<Self::Value, A::Error> {
                read_digest(sequence)
            }
        }
        deserializer.deserialize_tuple(4, Visitor)
    }
}
#[derive(Clone, Debug)]
pub(super) enum NodeReference {
    Index(u32),
    Foreign(Box<DigestWords>),
}
impl Serialize for NodeReference {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Index(index) => serializer.serialize_u32(*index),
            Self::Foreign(id) => id.serialize(serializer),
        }
    }
}
impl<'de> Deserialize<'de> for NodeReference {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> de::Visitor<'de> for Visitor {
            type Value = NodeReference;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a node index or four original foreign digest words")
            }
            fn visit_u64<E: de::Error>(self, index: u64) -> Result<Self::Value, E> {
                Ok(NodeReference::Index(
                    u32::try_from(index).map_err(E::custom)?,
                ))
            }
            fn visit_seq<A: de::SeqAccess<'de>>(
                self,
                sequence: A,
            ) -> Result<Self::Value, A::Error> {
                read_digest(sequence).map(|digest| NodeReference::Foreign(Box::new(digest)))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_references_do_not_retain_an_inline_foreign_digest_per_incidence() {
        assert!(std::mem::size_of::<NodeReference>() <= 2 * std::mem::size_of::<usize>());
    }
}
