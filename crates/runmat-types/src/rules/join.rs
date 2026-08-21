use crate::*;
use std::collections::{BTreeMap, BTreeSet};

pub trait FactJoin: Sized {
    fn join(&self, other: &Self) -> Self;
}

fn exact_or<T: Clone + PartialEq>(left: &T, right: &T, unknown: T) -> T {
    if left == right {
        left.clone()
    } else {
        unknown
    }
}

impl FactJoin for DimensionFact {
    fn join(&self, other: &Self) -> Self {
        exact_or(self, other, Self::Unknown)
    }
}

impl FactJoin for ShapeFact {
    fn join(&self, other: &Self) -> Self {
        if self == other {
            return self.clone();
        }
        match (self, other) {
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::Scalar, Self::Shaped { dims }) | (Self::Shaped { dims }, Self::Scalar)
                if dims
                    .iter()
                    .all(|dim| matches!(dim, DimensionFact::Known(1))) =>
            {
                Self::Scalar
            }
            (Self::Scalar, Self::Ranked { rank }) | (Self::Ranked { rank }, Self::Scalar)
                if *rank == 2 =>
            {
                Self::Ranked { rank: 2 }
            }
            (Self::Ranked { rank: left }, Self::Ranked { rank: right }) if left == right => {
                Self::Ranked { rank: *left }
            }
            (Self::Ranked { rank }, Self::Shaped { dims })
            | (Self::Shaped { dims }, Self::Ranked { rank })
                if *rank == dims.len() =>
            {
                Self::Ranked { rank: *rank }
            }
            (Self::Shaped { dims: left }, Self::Shaped { dims: right })
                if left.len() == right.len() =>
            {
                Self::Shaped {
                    dims: left.iter().zip(right).map(|(a, b)| a.join(b)).collect(),
                }
            }
            (left, right) if left.rank() == right.rank() => Self::Ranked {
                rank: left.rank().expect("equal known ranks"),
            },
            _ => Self::Unknown,
        }
    }
}

impl FactJoin for CertaintyFact {
    fn join(&self, other: &Self) -> Self {
        if self == other {
            self.clone()
        } else {
            Self::Dynamic(DynamicReason::ConflictingControlFlow)
        }
    }
}

impl FactJoin for InvalidationVector {
    fn join(&self, other: &Self) -> Self {
        Self(self.0.union(&other.0).cloned().collect::<BTreeSet<_>>())
    }
}

impl FactJoin for ValueKindFact {
    fn join(&self, other: &Self) -> Self {
        use ValueKindFact::*;
        if self == other {
            return self.clone();
        }
        match (self, other) {
            (Never, value) | (value, Never) => value.clone(),
            (Unknown, _) | (_, Unknown) => Unknown,
            (Numeric(left), Numeric(right)) if left == right => Numeric(*left),
            (Cell(left), Cell(right)) => {
                let elements_complete = left.elements_complete
                    && right.elements_complete
                    && left.elements.len() == right.elements.len();
                Cell(CellFact {
                    element: Box::new(left.element.join(&right.element)),
                    elements: if elements_complete {
                        join_ordered(&left.elements, &right.elements)
                    } else {
                        Vec::new()
                    },
                    elements_complete,
                })
            }
            (Struct(left), Struct(right)) => {
                let fields = left
                    .fields
                    .iter()
                    .filter_map(|(name, fact)| {
                        right
                            .fields
                            .get(name)
                            .map(|other| (name.clone(), fact.join(other)))
                    })
                    .collect::<BTreeMap<_, _>>();
                Struct(StructFact {
                    fields_complete: left.fields_complete
                        && right.fields_complete
                        && left.fields.keys().eq(right.fields.keys()),
                    fields,
                })
            }
            (Object(left), Object(right)) => Object(ObjectFact {
                class: exact_or(&left.class, &right.class, None),
                runtime_class: exact_or(&left.runtime_class, &right.runtime_class, None),
                properties: left
                    .properties
                    .iter()
                    .filter_map(|(name, fact)| {
                        right
                            .properties
                            .get(name)
                            .map(|other| (name.clone(), fact.join(other)))
                    })
                    .collect(),
                properties_complete: left.properties_complete
                    && right.properties_complete
                    && left.properties.keys().eq(right.properties.keys()),
                handle_semantics: exact_or(&left.handle_semantics, &right.handle_semantics, None),
            }),
            (ClassReference(left), ClassReference(right)) => ClassReference(ClassReferenceFact {
                class: exact_or(&left.class, &right.class, None),
                runtime_class: exact_or(&left.runtime_class, &right.runtime_class, None),
            }),
            (Callable(left), Callable(right)) => Callable(CallableFact {
                identity: exact_or(&left.identity, &right.identity, None),
                parameters: join_ordered(&left.parameters, &right.parameters),
                parameters_complete: left.parameters_complete
                    && right.parameters_complete
                    && left.parameters.len() == right.parameters.len(),
                outputs: join_ordered(&left.outputs, &right.outputs),
                outputs_complete: left.outputs_complete
                    && right.outputs_complete
                    && left.outputs.len() == right.outputs.len(),
                variadic_inputs: left.variadic_inputs
                    || right.variadic_inputs
                    || left.parameters.len() != right.parameters.len(),
                variadic_outputs: left.variadic_outputs
                    || right.variadic_outputs
                    || left.outputs.len() != right.outputs.len(),
                captures: join_ordered(&left.captures, &right.captures),
                captures_complete: left.captures_complete
                    && right.captures_complete
                    && left.captures.len() == right.captures.len(),
            }),
            (OutputList(left), OutputList(right)) => OutputList(OutputListFact {
                outputs: join_ordered(&left.outputs, &right.outputs),
                variadic: left.variadic
                    || right.variadic
                    || left.outputs.len() != right.outputs.len(),
            }),
            (Exception(left), Exception(right)) => Exception(ExceptionFact {
                identifier: exact_or(&left.identifier, &right.identifier, None),
            }),
            (Execution(left), Execution(right)) => join_execution(left, right),
            (Distributed(left), Distributed(right))
                if left.id == right.id && left.owner == right.owner =>
            {
                Distributed(DistributedFact {
                    id: left.id,
                    owner: left.owner,
                    scheme: exact_or(&left.scheme, &right.scheme, None),
                    value: Box::new(left.value.join(&right.value)),
                    materializable: left.materializable && right.materializable,
                })
            }
            (Foreign(left), Foreign(right)) if left.family == right.family => {
                Foreign(ForeignFact {
                    family: left.family.clone(),
                    type_name: exact_or(&left.type_name, &right.type_name, None),
                    type_version: exact_or(&left.type_version, &right.type_version, None),
                    ownership: exact_or(
                        &left.ownership,
                        &right.ownership,
                        ForeignOwnershipFact::Unknown,
                    ),
                    affinity: exact_or(
                        &left.affinity,
                        &right.affinity,
                        ForeignAffinityFact::Unknown,
                    ),
                    lifetime: exact_or(
                        &left.lifetime,
                        &right.lifetime,
                        ForeignLifetimeFact::Unknown,
                    ),
                })
            }
            _ => Unknown,
        }
    }
}

fn join_ordered(left: &[ValueFact], right: &[ValueFact]) -> Vec<ValueFact> {
    if left.len() == right.len() {
        left.iter().zip(right).map(|(a, b)| a.join(b)).collect()
    } else {
        Vec::new()
    }
}

fn join_execution(left: &ExecutionFact, right: &ExecutionFact) -> ValueKindFact {
    use ExecutionFact::*;
    let joined = match (left, right) {
        (
            Future {
                output: left,
                state: left_state,
            },
            Future {
                output: right,
                state: right_state,
            },
        ) => Future {
            output: Box::new(left.join(right)),
            state: exact_or(left_state, right_state, FutureStateFact::Unknown),
        },
        (
            Task {
                output: left,
                spawn_safety: left_safety,
            },
            Task {
                output: right,
                spawn_safety: right_safety,
            },
        ) => Task {
            output: Box::new(left.join(right)),
            spawn_safety: exact_or(
                left_safety,
                right_safety,
                SpawnSafetyFact::RequiresIsolation,
            ),
        },
        (Pool, Pool) => Pool,
        (Job { output: left }, Job { output: right }) => Job {
            output: Box::new(left.join(right)),
        },
        _ => return ValueKindFact::Unknown,
    };
    ValueKindFact::Execution(joined)
}

impl FactJoin for ValueFact {
    fn join(&self, other: &Self) -> Self {
        if matches!(self.kind, ValueKindFact::Never) {
            return other.clone();
        }
        if matches!(other.kind, ValueKindFact::Never) {
            return self.clone();
        }
        let kind = self.kind.join(&other.kind);
        let conflict = matches!(kind, ValueKindFact::Unknown) && self.kind != other.kind;
        Self {
            kind,
            shape: self.shape.join(&other.shape),
            storage: exact_or(&self.storage, &other.storage, StorageFact::Unknown),
            layout: exact_or(&self.layout, &other.layout, LayoutFact::Unknown),
            contiguity: exact_or(&self.contiguity, &other.contiguity, ContiguityFact::Unknown),
            view: exact_or(&self.view, &other.view, ViewFact::Unknown),
            residency: exact_or(&self.residency, &other.residency, ResidencyFact::Unknown),
            alias: exact_or(&self.alias, &other.alias, AliasFact::Unknown),
            mutation: exact_or(&self.mutation, &other.mutation, MutationFact::Unknown),
            certainty: if conflict {
                CertaintyFact::Dynamic(DynamicReason::ConflictingControlFlow)
            } else {
                self.certainty.join(&other.certainty)
            },
            invalidation: self.invalidation.join(&other.invalidation),
        }
    }
}
