use runmat_types::{
    standard, DynamicReason, NumericClass, NumericDomain, NumericFact, ResidencyFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) struct NetworkInputFacts {
    pub data: Option<ValueFact>,
    pub dlarray: bool,
    pub unsupported_storage: bool,
    pub unsupported_resident_data: bool,
}

pub(super) fn project(input: Option<&ValueFact>) -> NetworkInputFacts {
    let (data, dlarray) = match input.map(|fact| &fact.kind) {
        Some(ValueKindFact::Object(object))
            if object
                .runtime_class
                .as_ref()
                .is_some_and(|class| class.is(standard::DL_ARRAY)) =>
        {
            let data = object
                .properties
                .get(crate::DLARRAY_DATA_PROPERTY)
                .cloned()
                .or_else(|| {
                    (!object.properties_complete)
                        .then(|| ValueFact::unknown(DynamicReason::RuntimeValue))
                });
            (data, true)
        }
        _ => (input.cloned(), false),
    };
    let unsupported_storage = data
        .as_ref()
        .is_some_and(|fact| fact.storage == StorageFact::Sparse);
    let unsupported_resident_data = data.as_ref().is_some_and(|fact| {
        matches!(&fact.residency, ResidencyFact::Device { .. })
            && match &fact.kind {
                ValueKindFact::Numeric(NumericFact { class, domain }) => {
                    *domain != NumericDomain::Real
                        || !matches!(class, NumericClass::Single | NumericClass::Double)
                }
                ValueKindFact::Unknown => false,
                _ => true,
            }
    });
    NetworkInputFacts {
        data,
        dlarray,
        unsupported_storage,
        unsupported_resident_data,
    }
}
