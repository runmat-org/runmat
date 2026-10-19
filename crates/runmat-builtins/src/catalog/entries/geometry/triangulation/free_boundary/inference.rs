use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, DynamicReason,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

const CLASS: runmat_types::StaticClassIdentity = super::super::delaunay_tri::DELAUNAY_TRI_CLASS;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FREEBOUNDARY-ARITY",
            "freeBoundary requires one DelaunayTri object",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let mut rows = None;
    match request.arguments.first().map(|value| &value.kind) {
        Some(ValueKindFact::Object(object)) => {
            if object
                .runtime_class
                .as_ref()
                .is_some_and(|class| class != &CLASS.owned())
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FREEBOUNDARY-RECEIVER",
                    "expected a DelaunayTri object",
                    0,
                ));
            }
            // With no triangles there are no boundary edges. Otherwise edge
            // count depends on topology values, not connectivity shape alone.
            if object.runtime_class.as_ref() == Some(&CLASS.owned()) {
                let connectivity = object
                    .properties
                    .get("ConnectivityList")
                    .or_else(|| object.properties.get("Triangulation"));
                if let Some(ValueFact {
                    shape: ShapeFact::Shaped { dims },
                    ..
                }) = connectivity
                {
                    if dims.len() == 2 && matches!(dims[0], DimensionFact::Known(0)) {
                        rows = Some(0);
                    }
                }
            }
        }
        Some(ValueKindFact::Unknown) | None => {}
        _ => diagnostics.push(argument_error(
            "RM-CATALOG-FREEBOUNDARY-RECEIVER",
            "expected a DelaunayTri object",
            0,
        )),
    }
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![rows, Some(2)]),
        StorageFact::Dense,
    );
    output.residency = ResidencyFact::Host;
    if request.arguments.is_empty() {
        output.certainty = runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
