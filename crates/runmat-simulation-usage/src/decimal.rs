use serde::{de, Deserializer};
use std::{fmt, marker::PhantomData, str::FromStr};

pub(crate) fn deserialize<'de, D, T>(deserializer: D, digits: usize) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr,
    T::Err: fmt::Display,
{
    struct DecimalVisitor<T> {
        digits: usize,
        value: PhantomData<T>,
    }
    impl<T: FromStr> de::Visitor<'_> for DecimalVisitor<T>
    where
        T::Err: fmt::Display,
    {
        type Value = T;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a canonical unsigned decimal operation-count string")
        }

        fn visit_str<E: de::Error>(self, value: &str) -> Result<T, E> {
            if value.is_empty()
                || value.len() > self.digits
                || (value.len() > 1 && value.starts_with('0'))
                || !value.bytes().all(|byte| byte.is_ascii_digit())
            {
                return Err(E::custom("invalid decimal operation count"));
            }
            value.parse::<T>().map_err(E::custom)
        }
    }
    deserializer.deserialize_str(DecimalVisitor {
        digits,
        value: PhantomData,
    })
}
