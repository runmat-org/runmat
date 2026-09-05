use runmat_types::{
    standard, AliasFact, ContiguityFact, DynamicReason, LayoutFact, MutationFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn logical(shape: ShapeFact) -> ValueFact {
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    ValueFact {
        kind: ValueKindFact::Logical,
        shape,
        storage,
        layout: LayoutFact::ColumnMajor,
        contiguity: ContiguityFact::Contiguous,
        view: ViewFact::Materialized,
        residency: runmat_types::ResidencyFact::Host,
        alias: AliasFact::Unique,
        mutation: MutationFact::ValueSemantics,
        certainty: runmat_types::CertaintyFact::Proven,
        invalidation: Default::default(),
    }
}

pub(super) fn tabular(input: &ValueFact) -> Option<ValueFact> {
    let ValueKindFact::Object(object) = &input.kind else {
        return None;
    };
    let class = object.runtime_class.as_ref()?;
    if !standard::is_tabular(class) {
        return None;
    }
    let mut output = input.clone();
    let ValueKindFact::Object(object) = &mut output.kind else {
        unreachable!("tabular output cloned an object fact")
    };
    object.properties.clear();
    object.properties_complete = false;
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    output.residency = runmat_types::ResidencyFact::Host;
    Some(output)
}

pub(super) fn tabular_class(input: &ValueFact) -> Option<&runmat_types::ClassIdentity> {
    let ValueKindFact::Object(object) = &input.kind else {
        return None;
    };
    object
        .runtime_class
        .as_ref()
        .filter(|class| standard::is_tabular(class))
}

pub(super) fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}
