use runmat_types::NumericClass;

use super::ENTRIES;
use crate::{BuiltinInferenceRule, MathInferenceRule};

#[test]
fn every_integer_conversion_has_canonical_documentation_and_typed_inference() {
    let expected = [
        ("int8", NumericClass::Int8),
        ("int16", NumericClass::Int16),
        ("int32", NumericClass::Int32),
        ("int64", NumericClass::Int64),
        ("uint8", NumericClass::UInt8),
        ("uint16", NumericClass::UInt16),
        ("uint32", NumericClass::UInt32),
        ("uint64", NumericClass::UInt64),
    ];

    for (entry, (name, class)) in ENTRIES.iter().zip(expected) {
        assert_eq!(entry.identity.name, name);
        assert_eq!(entry.documentation.title, Some(name));
        assert_eq!(entry.documentation.examples.len(), 3);
        assert_eq!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Math(MathInferenceRule::NumericConversion(class))
        );
    }
}
