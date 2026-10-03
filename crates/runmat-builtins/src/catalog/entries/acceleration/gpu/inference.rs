use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, NumericFact, ResidencyFact,
    StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn numeric(class: NumericClass) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }))
}

pub(super) fn device(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut fields = BTreeMap::new();
    fields.insert("device_id".into(), numeric(NumericClass::UInt32));
    fields.insert("index".into(), numeric(NumericClass::UInt32));
    fields.insert("name".into(), ValueFact::scalar(ValueKindFact::String));
    fields.insert("vendor".into(), ValueFact::scalar(ValueKindFact::String));
    fields.insert("precision".into(), ValueFact::scalar(ValueKindFact::String));
    fields.insert(
        "supports_double".into(),
        ValueFact::scalar(ValueKindFact::Logical),
    );
    // `backend` and `memory_bytes` are conditional on provider-reported metadata.
    let mut output = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, false)));
    output.storage = StorageFact::Dense;
    output.residency = ResidencyFact::Host;
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM.GPU_DEVICE.TOO_MANY_INPUTS",
            "gpuDevice accepts at most one selector",
            1,
        ));
    }
    if let Some(selector) = request.arguments.first() {
        if selector
            .shape
            .element_count()
            .is_some_and(|count| count > 1)
        {
            diagnostics.push(argument_error(
                "RM.GPU_DEVICE.INVALID_INDEX",
                "gpuDevice requires a scalar selector",
                0,
            ));
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn info(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut output = ValueFact::scalar(ValueKindFact::String);
    output.storage = StorageFact::Dense;
    output.residency = ResidencyFact::Host;
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else {
        vec![argument_error(
            "RM.GPU_INFO.TOO_MANY_INPUTS",
            "gpuInfo accepts no inputs",
            0,
        )]
    };
    finish_fixed(entry, request, output, diagnostics)
}
