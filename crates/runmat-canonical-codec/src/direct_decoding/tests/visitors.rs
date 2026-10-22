//! Deliberately permissive visitors expose differences hidden by ordinary derived
//! visitors: method type restrictions, numeric dispatch and discarded payloads.
use super::*;
use serde::de::{self, Visitor};
struct Flexible;
macro_rules! scalar {
    ($method:ident, $kind:ty) => {
        fn $method<E: de::Error>(self, value: $kind) -> Result<String, E> {
            Ok(format!("{}:{value}", stringify!($method)))
        }
    };
}
impl<'de> Visitor<'de> for Flexible {
    type Value = String;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("any original JSON value")
    }
    scalar!(visit_bool, bool);
    scalar!(visit_i64, i64);
    scalar!(visit_u64, u64);
    scalar!(visit_i128, i128);
    scalar!(visit_u128, u128);
    scalar!(visit_f64, f64);
    scalar!(visit_string, String);
    fn visit_str<E: de::Error>(self, value: &str) -> Result<String, E> {
        Ok(format!("borrowed:{value}"))
    }
    fn visit_unit<E: de::Error>(self) -> Result<String, E> {
        Ok("unit".into())
    }
    fn visit_none<E: de::Error>(self) -> Result<String, E> {
        Ok("none".into())
    }
    fn visit_some<D: de::Deserializer<'de>>(self, _: D) -> Result<String, D::Error> {
        Ok("discarded-some".into())
    }
    fn visit_newtype_struct<D: de::Deserializer<'de>>(self, _: D) -> Result<String, D::Error> {
        Ok("discarded-newtype".into())
    }
    fn visit_enum<D: de::EnumAccess<'de>>(self, _: D) -> Result<String, D::Error> {
        Ok("discarded-enum".into())
    }
    fn visit_seq<D: de::SeqAccess<'de>>(self, mut sequence: D) -> Result<String, D::Error> {
        let mut count = 0;
        while sequence.next_element::<de::IgnoredAny>()?.is_some() {
            count += 1;
        }
        Ok(format!("sequence:{count}"))
    }
    fn visit_map<D: de::MapAccess<'de>>(self, mut map: D) -> Result<String, D::Error> {
        let mut count = 0;
        while map
            .next_entry::<de::IgnoredAny, de::IgnoredAny>()?
            .is_some()
        {
            count += 1;
        }
        Ok(format!("map:{count}"))
    }
}
macro_rules! contract {
    ($name:ident,$method:ident $(,$argument:expr)*)=>{
        #[derive(Debug,PartialEq)]
        struct $name(String);
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: de::Deserializer<'de>>(decoder:D)->Result<Self,D::Error>{decoder.$method($($argument,)*Flexible).map(Self)}
        }
    };
}
contract!(Any, deserialize_any);
contract!(Number, deserialize_i128);
contract!(Float, deserialize_f64);
contract!(Boolean, deserialize_bool);
contract!(Text, deserialize_string);
contract!(Bytes, deserialize_byte_buf);
contract!(Unit, deserialize_unit);
contract!(Sequence, deserialize_seq);
contract!(Map, deserialize_map);
contract!(Record, deserialize_struct, "Record", &["a"]);
contract!(OptionValue, deserialize_option);
contract!(Newtype, deserialize_newtype_struct, "Newtype");
contract!(
    Enumeration,
    deserialize_enum,
    "Enumeration",
    &["Unit", "Payload"]
);
contract!(Ignored, deserialize_ignored_any);

fn compare<T: DeserializeOwned + PartialEq + std::fmt::Debug>(bytes: &[u8]) {
    let actual = decode_contract::<T>(PREFIX, DOMAIN, bytes, LIMITS);
    let expected = original::<T>(bytes, LIMITS);
    assert_eq!(
        actual.is_ok(),
        expected.is_ok(),
        "{} {actual:?} {expected:?}",
        std::any::type_name::<T>()
    );
    if let (Ok(actual), Ok(expected)) = (actual, expected) {
        assert_eq!(actual, expected, "{}", std::any::type_name::<T>());
    }
}
// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Permissive typed visitors compare method restrictions, numeric dispatch, owned
//   strings, containers and discarded option/newtype/enum payloads with the original owned JSON
//   representation. This finite protocol family is not whole-study cold replay or arbitrary
//   custom-Serde equivalence.
#[test]
fn method_dispatch_and_discarded_payloads_match_owned_value_semantics() {
    for value in [
        serde_json::json!(null),
        serde_json::json!(true),
        serde_json::json!(1),
        serde_json::json!(-1),
        serde_json::json!(1.25),
        serde_json::json!("Unit"),
        serde_json::json!([1, "a", []]),
        serde_json::json!({"Payload":[1,2,3]}),
        serde_json::json!({"a":1,"b":null}),
    ] {
        let bytes = encode_contract(PREFIX, DOMAIN, &value, LIMITS).unwrap();
        compare::<Any>(&bytes);
        compare::<Number>(&bytes);
        compare::<Float>(&bytes);
        compare::<Boolean>(&bytes);
        compare::<Text>(&bytes);
        compare::<Bytes>(&bytes);
        compare::<Unit>(&bytes);
        compare::<Sequence>(&bytes);
        compare::<Map>(&bytes);
        compare::<Record>(&bytes);
        compare::<OptionValue>(&bytes);
        compare::<Newtype>(&bytes);
        compare::<Enumeration>(&bytes);
        compare::<Ignored>(&bytes);
    }
}
