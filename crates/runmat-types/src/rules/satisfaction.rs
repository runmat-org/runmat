use crate::{
    AliasFact, CallableFact, CellFact, ClassReferenceFact, ContiguityFact, DimensionFact,
    DistributedFact, ExceptionFact, ExecutionFact, ForeignAffinityFact, ForeignFact,
    ForeignLifetimeFact, ForeignOwnershipFact, FutureStateFact, LayoutFact, MutationFact,
    ObjectFact, OutputListFact, ResidencyFact, ShapeFact, StorageFact, StructFact, ValueFact,
    ValueKindFact, ViewFact,
};

/// Whether one observed runtime fact satisfies a representation constraint.
///
/// Certainty and invalidation metadata are deliberately excluded: they explain
/// how a static fact was derived and when compiled code becomes stale, but are
/// not properties of the live value representation checked by a guard.
pub trait FactSatisfaction<Rhs = Self> {
    fn satisfies(&self, expected: &Rhs) -> bool;
}

impl FactSatisfaction for ValueFact {
    fn satisfies(&self, expected: &Self) -> bool {
        self.kind.satisfies(&expected.kind)
            && self.shape.satisfies(&expected.shape)
            && wildcard_eq(self.storage, expected.storage, StorageFact::Unknown)
            && wildcard_eq(self.layout, expected.layout, LayoutFact::Unknown)
            && wildcard_eq(
                self.contiguity,
                expected.contiguity,
                ContiguityFact::Unknown,
            )
            && wildcard_eq(self.view, expected.view, ViewFact::Unknown)
            && self.residency.satisfies(&expected.residency)
            && wildcard_eq(self.alias, expected.alias, AliasFact::Unknown)
            && wildcard_eq(self.mutation, expected.mutation, MutationFact::Unknown)
    }
}

impl FactSatisfaction for ValueKindFact {
    fn satisfies(&self, expected: &Self) -> bool {
        match (self, expected) {
            (_, Self::Unknown) => true,
            (Self::Cell(actual), Self::Cell(expected)) => actual.satisfies(expected),
            (Self::Struct(actual), Self::Struct(expected)) => actual.satisfies(expected),
            (Self::Object(actual), Self::Object(expected)) => actual.satisfies(expected),
            (Self::ClassReference(actual), Self::ClassReference(expected)) => {
                actual.satisfies(expected)
            }
            (Self::Callable(actual), Self::Callable(expected)) => actual.satisfies(expected),
            (Self::OutputList(actual), Self::OutputList(expected)) => actual.satisfies(expected),
            (Self::Exception(actual), Self::Exception(expected)) => actual.satisfies(expected),
            (Self::Execution(actual), Self::Execution(expected)) => actual.satisfies(expected),
            (Self::Distributed(actual), Self::Distributed(expected)) => actual.satisfies(expected),
            (Self::Foreign(actual), Self::Foreign(expected)) => actual.satisfies(expected),
            _ => self == expected,
        }
    }
}

impl FactSatisfaction for ShapeFact {
    fn satisfies(&self, expected: &Self) -> bool {
        match expected {
            Self::Unknown => true,
            Self::Scalar => self.element_count() == Some(1),
            Self::Ranked { rank } => self.rank() == Some(*rank),
            Self::Shaped { dims: expected } => {
                let Some(actual) = normalized_dimensions(self) else {
                    return false;
                };
                let mut expected = expected.clone();
                normalize_dimensions(&mut expected);
                actual.len() == expected.len()
                    && actual
                        .iter()
                        .zip(expected)
                        .all(|(actual, expected)| match expected {
                            DimensionFact::Unknown => true,
                            DimensionFact::Known(expected) => {
                                matches!(actual, DimensionFact::Known(actual) if *actual == expected)
                            }
                            DimensionFact::Symbolic(expected) => {
                                matches!(actual, DimensionFact::Symbolic(actual) if *actual == expected)
                            }
                        })
            }
        }
    }
}

impl FactSatisfaction for CellFact {
    fn satisfies(&self, expected: &Self) -> bool {
        self.element.satisfies(&expected.element)
            && sequence_satisfies(
                &self.elements,
                self.elements_complete,
                &expected.elements,
                expected.elements_complete,
            )
    }
}

impl FactSatisfaction for StructFact {
    fn satisfies(&self, expected: &Self) -> bool {
        fields_satisfy(
            &self.fields,
            self.fields_complete,
            &expected.fields,
            expected.fields_complete,
        )
    }
}

impl FactSatisfaction for ObjectFact {
    fn satisfies(&self, expected: &Self) -> bool {
        option_satisfies(&self.class, &expected.class)
            && option_satisfies(&self.runtime_class, &expected.runtime_class)
            && option_satisfies(&self.handle_semantics, &expected.handle_semantics)
            && fields_satisfy(
                &self.properties,
                self.properties_complete,
                &expected.properties,
                expected.properties_complete,
            )
    }
}

impl FactSatisfaction for ClassReferenceFact {
    fn satisfies(&self, expected: &Self) -> bool {
        option_satisfies(&self.class, &expected.class)
            && option_satisfies(&self.runtime_class, &expected.runtime_class)
    }
}

impl FactSatisfaction for CallableFact {
    fn satisfies(&self, expected: &Self) -> bool {
        option_satisfies(&self.identity, &expected.identity)
            && self.variadic_inputs == expected.variadic_inputs
            && self.variadic_outputs == expected.variadic_outputs
            && sequence_satisfies(
                &self.parameters,
                self.parameters_complete,
                &expected.parameters,
                expected.parameters_complete,
            )
            && sequence_satisfies(
                &self.outputs,
                self.outputs_complete,
                &expected.outputs,
                expected.outputs_complete,
            )
            && sequence_satisfies(
                &self.captures,
                self.captures_complete,
                &expected.captures,
                expected.captures_complete,
            )
    }
}

impl FactSatisfaction for OutputListFact {
    fn satisfies(&self, expected: &Self) -> bool {
        self.variadic == expected.variadic
            && sequence_satisfies(
                &self.outputs,
                !self.variadic,
                &expected.outputs,
                !expected.variadic,
            )
    }
}

impl FactSatisfaction for ExceptionFact {
    fn satisfies(&self, expected: &Self) -> bool {
        option_satisfies(&self.identifier, &expected.identifier)
    }
}

impl FactSatisfaction for ExecutionFact {
    fn satisfies(&self, expected: &Self) -> bool {
        match (self, expected) {
            (
                Self::Future {
                    output: actual_output,
                    state: actual_state,
                },
                Self::Future {
                    output: expected_output,
                    state: expected_state,
                },
            ) => {
                actual_output.satisfies(expected_output)
                    && (*expected_state == FutureStateFact::Unknown
                        || actual_state == expected_state)
            }
            (
                Self::Task {
                    output: actual_output,
                    spawn_safety: actual_safety,
                },
                Self::Task {
                    output: expected_output,
                    spawn_safety: expected_safety,
                },
            ) => actual_output.satisfies(expected_output) && actual_safety == expected_safety,
            (Self::Job { output: actual }, Self::Job { output: expected }) => {
                actual.satisfies(expected)
            }
            (Self::Pool, Self::Pool) => true,
            _ => false,
        }
    }
}

impl FactSatisfaction for DistributedFact {
    fn satisfies(&self, expected: &Self) -> bool {
        self.id == expected.id
            && self.owner == expected.owner
            && option_satisfies(&self.scheme, &expected.scheme)
            && self.value.satisfies(&expected.value)
            && self.materializable == expected.materializable
    }
}

impl FactSatisfaction for ForeignFact {
    fn satisfies(&self, expected: &Self) -> bool {
        self.family == expected.family
            && option_satisfies(&self.type_name, &expected.type_name)
            && option_satisfies(&self.type_version, &expected.type_version)
            && wildcard_eq(
                self.ownership,
                expected.ownership,
                ForeignOwnershipFact::Unknown,
            )
            && wildcard_eq(
                self.affinity,
                expected.affinity,
                ForeignAffinityFact::Unknown,
            )
            && wildcard_eq(
                self.lifetime,
                expected.lifetime,
                ForeignLifetimeFact::Unknown,
            )
    }
}

impl FactSatisfaction for ResidencyFact {
    fn satisfies(&self, expected: &Self) -> bool {
        match expected {
            Self::Unknown => true,
            Self::Device { provider: None } => matches!(self, Self::Device { .. }),
            Self::Remote { pool: None } => matches!(self, Self::Remote { .. }),
            _ => self == expected,
        }
    }
}

fn wildcard_eq<T: Copy + Eq>(actual: T, expected: T, wildcard: T) -> bool {
    expected == wildcard || actual == expected
}

fn option_satisfies<T: Eq>(actual: &Option<T>, expected: &Option<T>) -> bool {
    expected
        .as_ref()
        .is_none_or(|expected| actual.as_ref() == Some(expected))
}

fn sequence_satisfies(
    actual: &[ValueFact],
    actual_complete: bool,
    expected: &[ValueFact],
    expected_complete: bool,
) -> bool {
    (!expected_complete || (actual_complete && actual.len() == expected.len()))
        && actual.len() >= expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.satisfies(expected))
}

fn fields_satisfy(
    actual: &std::collections::BTreeMap<String, ValueFact>,
    actual_complete: bool,
    expected: &std::collections::BTreeMap<String, ValueFact>,
    expected_complete: bool,
) -> bool {
    (!expected_complete || (actual_complete && actual.len() == expected.len()))
        && expected.iter().all(|(name, expected)| {
            actual
                .get(name)
                .is_some_and(|actual| actual.satisfies(expected))
        })
}

fn normalized_dimensions(shape: &ShapeFact) -> Option<Vec<DimensionFact>> {
    let mut dimensions = match shape {
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => return None,
        ShapeFact::Scalar => vec![DimensionFact::Known(1), DimensionFact::Known(1)],
        ShapeFact::Shaped { dims } => dims.clone(),
    };
    normalize_dimensions(&mut dimensions);
    Some(dimensions)
}

fn normalize_dimensions(dimensions: &mut Vec<DimensionFact>) {
    while dimensions.len() > 2 && dimensions.last() == Some(&DimensionFact::Known(1)) {
        dimensions.pop();
    }
    dimensions.resize(2, DimensionFact::Known(1));
}

#[cfg(test)]
mod tests {
    use super::FactSatisfaction;
    use crate::{
        CellFact, DimensionFact, DynamicReason, NumericClass, NumericDomain, NumericFact,
        ResidencyFact, ShapeFact, ValueFact, ValueKindFact,
    };

    #[test]
    fn representation_satisfaction_treats_unknown_constraints_as_wildcards() {
        let actual = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        let expected = ValueFact::unknown(DynamicReason::RuntimeValue);
        assert!(actual.satisfies(&expected));
    }

    #[test]
    fn representation_satisfaction_checks_kind_shape_and_residency() {
        let actual = ValueFact::scalar(ValueKindFact::Logical);
        let mut expected = ValueFact::scalar(ValueKindFact::Logical);
        expected.shape = ShapeFact::Ranked { rank: 2 };
        assert!(actual.satisfies(&expected));

        expected.kind = ValueKindFact::String;
        assert!(!actual.satisfies(&expected));

        expected.kind = ValueKindFact::Logical;
        expected.residency = ResidencyFact::Device { provider: None };
        assert!(!actual.satisfies(&expected));
    }

    #[test]
    fn partial_shapes_and_nested_facts_accept_more_precise_runtime_values() {
        let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        let actual = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(numeric.clone()),
                elements: vec![numeric.clone(), numeric.clone()],
                elements_complete: true,
            }),
            ShapeFact::Shaped {
                dims: vec![DimensionFact::Known(1), DimensionFact::Known(2)],
            },
            crate::StorageFact::Dense,
        );
        let expected = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                elements: vec![numeric],
                elements_complete: false,
            }),
            ShapeFact::Shaped {
                dims: vec![DimensionFact::Known(1), DimensionFact::Unknown],
            },
            crate::StorageFact::Dense,
        );
        assert!(actual.satisfies(&expected));
    }
}
