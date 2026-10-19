use super::{invalid, key::Key, typed::Decoder, CanonicalCodecError};
use serde::de::{self, DeserializeSeed, Visitor};
pub(super) struct Sequence<'a, 'de, 'c> {
    pub(super) decoder: &'a mut Decoder<'de, 'c>,
    pub(super) remaining: u64,
}
impl<'de> de::SeqAccess<'de> for &mut Sequence<'_, 'de, '_> {
    type Error = CanonicalCodecError;
    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, Self::Error> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        seed.deserialize(&mut *self.decoder).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        usize::try_from(self.remaining).ok()
    }
}
pub(super) struct Map<'a, 'de, 'c> {
    pub(super) decoder: &'a mut Decoder<'de, 'c>,
    pub(super) remaining: u64,
    pub(super) pending: bool,
}
impl<'de> de::MapAccess<'de> for &mut Map<'_, 'de, '_> {
    type Error = CanonicalCodecError;
    fn next_key_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, Self::Error> {
        crate::checkpoint(self.decoder.control)?;
        if self.pending {
            return Err(invalid("map value must follow its key"));
        }
        if self.remaining == 0 {
            return Ok(None);
        }
        self.pending = true;
        seed.deserialize(Key(self.decoder.input.str().map_err(invalid)?))
            .map(Some)
    }
    fn next_value_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<T::Value, Self::Error> {
        if !self.pending {
            return Err(invalid("map value has no preceding key"));
        }
        self.pending = false;
        self.remaining -= 1;
        seed.deserialize(&mut *self.decoder)
    }
    fn size_hint(&self) -> Option<usize> {
        usize::try_from(self.remaining).ok()
    }
}
pub(super) struct Variant<'a, 'de, 'c> {
    pub(super) decoder: &'a mut Decoder<'de, 'c>,
    pub(super) name: &'de str,
    pub(super) payload: bool,
}
impl<'a, 'de, 'c> de::EnumAccess<'de> for Variant<'a, 'de, 'c> {
    type Error = CanonicalCodecError;
    type Variant = Self;
    fn variant_seed<T: DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> Result<(T::Value, Self), Self::Error> {
        let name = de::value::StringDeserializer::<Self::Error>::new(self.name.to_owned());
        Ok((seed.deserialize(name)?, self))
    }
}
impl<'de> de::VariantAccess<'de> for Variant<'_, 'de, '_> {
    type Error = CanonicalCodecError;
    fn unit_variant(self) -> Result<(), Self::Error> {
        if self.payload {
            serde::Deserialize::deserialize(&mut *self.decoder)
        } else {
            Ok(())
        }
    }
    fn newtype_variant_seed<T: DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> Result<T::Value, Self::Error> {
        if !self.payload {
            return Err(invalid("enum payload is missing"));
        }
        seed.deserialize(&mut *self.decoder)
    }
    fn tuple_variant<V: Visitor<'de>>(self, _: usize, visitor: V) -> Result<V::Value, Self::Error> {
        if !self.payload {
            return Err(invalid("enum tuple is missing"));
        }
        de::Deserializer::deserialize_seq(&mut *self.decoder, visitor)
    }
    fn struct_variant<V: Visitor<'de>>(
        self,
        _: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        if !self.payload {
            return Err(invalid("enum fields are missing"));
        }
        de::Deserializer::deserialize_map(&mut *self.decoder, visitor)
    }
}
