use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, NumericClass,
    NumericDomain, NumericFact, ObjectFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

fn double_array(shape: ShapeFact) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    );
    fact.residency = ResidencyFact::Host;
    fact
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-DELAUNAYTRI-ARITY",
            "DelaunayTri accepts zero to three inputs",
            3,
        ));
    }
    for (index, input) in request.arguments.iter().enumerate() {
        match &input.kind {
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
            | ValueKindFact::Unknown => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-DELAUNAYTRI-CLASS",
                "coordinates and connectivity must be real numeric arrays",
                index,
            )),
        }
        if input.storage == StorageFact::Sparse {
            diagnostics.push(argument_error(
                "RM-CATALOG-DELAUNAYTRI-STORAGE",
                "triangulation inputs must be dense",
                index,
            ));
        }
    }
    let point_rows = if request.arguments.is_empty() {
        Some(0)
    } else if request.arguments.len() == 1 {
        match &request.arguments[0].shape {
            ShapeFact::Shaped { dims } if dims.len() == 1 => {
                if matches!(dims[0], DimensionFact::Known(columns) if columns != 2) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-DELAUNAYTRI-SHAPE",
                        "point matrix must have two columns",
                        0,
                    ));
                }
                Some(1)
            }
            ShapeFact::Shaped { dims }
                if dims.len() >= 2
                    && super::super::query_facts::possible_trailing_singletons(dims) =>
            {
                if matches!(dims[1], DimensionFact::Known(value) if value != 2) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-DELAUNAYTRI-SHAPE",
                        "point matrix must have two columns",
                        0,
                    ));
                }
                match &dims[0] {
                    DimensionFact::Known(rows) => Some(*rows),
                    _ => None,
                }
            }
            ShapeFact::Ranked { rank: 1.. } | ShapeFact::Unknown => None,
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-DELAUNAYTRI-SHAPE",
                    "point matrix must be N-by-2",
                    0,
                ));
                None
            }
        }
    } else {
        // Two-input dispatch distinguishes X,C from x,y using runtime value
        // geometry. Three-input forms require an empty constraint value.
        None
    };
    let triangle_rows = point_rows.filter(|rows| *rows < 3).map(|_| 0);
    let mut properties = BTreeMap::new();
    let points = double_array(ShapeFact::from(vec![point_rows, Some(2)]));
    let connectivity = double_array(ShapeFact::from(vec![triangle_rows, Some(3)]));
    properties.insert("X".into(), points.clone());
    properties.insert("Points".into(), points);
    properties.insert("Triangulation".into(), connectivity.clone());
    properties.insert("ConnectivityList".into(), connectivity);
    properties.insert(
        "Constraints".into(),
        double_array(ShapeFact::from(vec![Some(0), Some(2)])),
    );
    properties.insert(
        "Dimension".into(),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    );
    let mut output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::DELAUNAY_TRI_CLASS.owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    output.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
