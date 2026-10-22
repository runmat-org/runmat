use crate::execution::{RootedValue, RootedValueLease, RootedValueSequence};
use crate::object::indexing::{
    ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_gc::{Trace, Tracer};
use runmat_value::Value;

#[derive(Debug, Clone)]
pub enum SequenceEndpointSpec {
    Member(runmat_value::Utf8Text),
    CellContents(runmat_value::ExactArrayStorage<ObjectIndexComponent>),
    /// Ordinary assignment consumes one RHS value, not a destination-derived
    /// output count. Its selector sequence may still have variable width.
    Fixed {
        step: super::AssignmentStepSpec,
        mutation: runmat_types::PlaceMutationKind,
    },
}

impl SequenceEndpointSpec {
    pub(super) fn retained_values(&self) -> impl Iterator<Item = &Value> + Clone {
        let components: &[ObjectIndexComponent] = match self {
            Self::Member(_) => &[],
            Self::CellContents(components) => components,
            Self::Fixed { step, .. } => step.components(),
        };
        components.iter().filter_map(|component| match component {
            ObjectIndexComponent::Value(value) => Some(value),
            ObjectIndexComponent::Colon => None,
        })
    }
    pub(super) fn validate_fixed_mutation(&self) -> Result<(), RuntimeError> {
        use runmat_types::PlaceMutationKind;
        if let Self::Fixed { step, mutation } = self {
            let valid = matches!(
                (step, mutation),
                (
                    super::AssignmentStepSpec::Member(_),
                    PlaceMutationKind::MemberAssign
                ) | (
                    super::AssignmentStepSpec::Braces(_),
                    PlaceMutationKind::CellAssign
                ) | (
                    super::AssignmentStepSpec::Parentheses { .. },
                    PlaceMutationKind::IndexedAssign | PlaceMutationKind::Delete
                )
            );
            if !valid {
                return Err(semantic_error(
                    "InvalidAssignmentTarget",
                    "fixed destination and mutation kind disagree",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PreparedSequenceEndpoint {
    spec: SequenceEndpointSpec,
    cardinality: usize,
    fixed_plan: Option<crate::indexing::plan::IndexPlan>,
    fixed_cell_positions: Option<runmat_value::ExactArrayStorage<usize>>,
    _retained_roots: RootedValueLease,
    _source_roots: RootedValueLease,
}

impl PreparedSequenceEndpoint {
    pub fn visit_retained_values(&self, visitor: &mut impl FnMut(&Value)) {
        self.spec.visit_retained_values(visitor);
    }
    pub async fn prepare(base: &Value, spec: SequenceEndpointSpec) -> Result<Self, RuntimeError> {
        Self::prepare_with_context(None, base, spec).await
    }
    pub(super) async fn prepare_with_context(
        context: Option<&crate::context::RuntimeContext>,
        base: &Value,
        spec: SequenceEndpointSpec,
    ) -> Result<Self, RuntimeError> {
        Self::prepare_with_cardinality(context, base, spec, None).await
    }

    pub(super) async fn prepare_with_cardinality(
        context: Option<&crate::context::RuntimeContext>,
        base: &Value,
        spec: SequenceEndpointSpec,
        prepared: Option<
            crate::builtins::introspection::object_indexing::PreparedObjectAssignmentCardinality,
        >,
    ) -> Result<Self, RuntimeError> {
        spec.validate_fixed_mutation()?;
        if prepared
            .as_ref()
            .and_then(|prepared| prepared.method.as_ref())
            .is_some_and(|method| !crate::object::protocol::resolved_method_is_current(method))
        {
            return Err(semantic_error(
                "StaleObjectProtocolResolution",
                "object protocol binding changed after the destination was prepared",
            ));
        }
        let _inputs = match context {
            Some(runtime) => {
                RootedValueLease::for_values_iterator_in_context(spec.retained_values(), runtime)?
            }
            None => {
                RootedValueLease::for_traced(&spec).map_err(|error| error.into_runtime_error())?
            }
        };
        let _base = match context {
            Some(runtime) => {
                RootedValueLease::for_values_iterator_in_context(std::iter::once(base), runtime)?
            }
            None => RootedValueLease::for_values(std::iter::once(base))
                .map_err(|error| error.into_runtime_error())?,
        };
        let mut fixed_plan = None;
        let mut fixed_cell_positions = None;
        let cardinality = match &spec {
            SequenceEndpointSpec::Member(member) => {
                if let Some(prepared) = &prepared {
                    prepared.count
                } else {
                    assignment_cardinality_for_step(
                        context,
                        base,
                        ObjectSubscript::member_retained(member.clone()),
                    )
                    .await?
                }
            }
            SequenceEndpointSpec::CellContents(indices) => {
                if let Some(prepared) = &prepared {
                    prepared.count
                } else {
                    let step = ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
                        components: indices.clone(),
                    });
                    assignment_cardinality_for_step(context, base, step).await?
                }
            }
            SequenceEndpointSpec::Fixed { step, mutation } => {
                if *mutation == runmat_types::PlaceMutationKind::Delete
                    && !matches!(step, super::AssignmentStepSpec::Parentheses { .. })
                {
                    return Err(semantic_error(
                        "InvalidDeletionTarget",
                        "deletion requires parentheses indexing",
                    ));
                }
                if let super::AssignmentStepSpec::Parentheses { selectors } = step {
                    let shape = crate::indexing::value::shape_borrowed(base).ok_or_else(|| {
                        semantic_error("InvalidObjectSubscriptBase", "value has no indexed shape")
                    })?;
                    fixed_plan = Some(
                        crate::object::indexing::plan_for_components_with_context(
                            context,
                            shape.as_slice(),
                            selectors,
                            *mutation != runmat_types::PlaceMutationKind::Delete,
                        )
                        .await?,
                    );
                }
                if let super::AssignmentStepSpec::Braces(indices) = step {
                    let Value::Cell(cell) = base else {
                        return Err(semantic_error(
                            "CellIndexType",
                            "Brace assignment requires a cell array",
                        ));
                    };
                    fixed_cell_positions = Some(
                        crate::object::cell::resolve_cell_assignment_components_in_context(
                            context, cell, indices,
                        )?,
                    );
                }
                1
            }
        };
        let retained_roots = _inputs;
        Ok(Self {
            spec,
            cardinality,
            fixed_plan,
            fixed_cell_positions,
            _retained_roots: retained_roots,
            _source_roots: _base,
        })
    }

    pub const fn cardinality(&self) -> usize {
        self.cardinality
    }

    pub async fn assign(
        self,
        base: Value,
        values: Vec<Value>,
        caller_function_name: Option<&str>,
    ) -> Result<RootedValue, RuntimeError> {
        let values = runmat_value::ValueSequence::comma_separated(values)
            .map_err(crate::sequence::sequence_error_to_runtime)?;
        self.assign_with_context(None, base, values, caller_function_name)
            .await
    }

    pub async fn assign_in_context(
        self,
        context: &crate::context::RuntimeContext,
        base: Value,
        values: Vec<Value>,
        caller_function_name: Option<&str>,
    ) -> Result<RootedValue, RuntimeError> {
        let values = runmat_value::ValueSequence::comma_separated(values)
            .map_err(crate::sequence::sequence_error_to_runtime)?;
        self.assign_with_context(Some(context), base, values, caller_function_name)
            .await
    }

    pub(crate) async fn assign_sequence(
        self,
        base: Value,
        values: runmat_value::ValueSequence,
        caller: Option<&str>,
    ) -> Result<RootedValue, RuntimeError> {
        self.assign_with_context(None, base, values, caller).await
    }

    pub(crate) async fn assign_sequence_in_context(
        self,
        context: &crate::context::RuntimeContext,
        base: Value,
        values: runmat_value::ValueSequence,
        caller: Option<&str>,
    ) -> Result<RootedValue, RuntimeError> {
        self.assign_with_context(Some(context), base, values, caller)
            .await
    }

    async fn assign_with_context(
        self,
        context: Option<&crate::context::RuntimeContext>,
        base: Value,
        values: runmat_value::ValueSequence,
        caller_function_name: Option<&str>,
    ) -> Result<RootedValue, RuntimeError> {
        let Self {
            spec,
            cardinality,
            fixed_plan,
            fixed_cell_positions,
            _retained_roots,
            _source_roots,
        } = self;
        let _inputs = match context {
            Some(runtime) => RootedValueLease::for_values_iterator_in_context(
                std::iter::once(&base).chain(values.iter()),
                runtime,
            )?,
            None => RootedValueLease::for_values(std::iter::once(&base).chain(values.iter()))
                .map_err(|error| error.into_runtime_error())?,
        };
        if values.len() != cardinality {
            return Err(semantic_error(
                "RunMat:CommaSeparatedListAssignmentArity",
                format!(
                    "sequence assignment requires {} values, but received {}",
                    cardinality,
                    values.len()
                ),
            ));
        }
        match spec {
            SequenceEndpointSpec::Fixed { step, mutation } => {
                let rhs = match context {
                    Some(runtime) => RootedValueSequence::in_context(values, runtime)?,
                    None => RootedValueSequence::new(values)
                        .map_err(|error| error.into_runtime_error())?,
                }
                .try_into_single()?;
                let (rhs, _rhs_roots) = rhs.into_value_and_lease();
                match step {
                    super::AssignmentStepSpec::Parentheses { .. } => {
                        let plan = fixed_plan.ok_or_else(|| {
                            semantic_error(
                                "PreparedDestinationState",
                                "fixed assignment lost its validated plan",
                            )
                        })?;
                        let value = crate::indexing::value::mutate_with_plan(
                            context, base, &plan, rhs, mutation,
                        )
                        .await?;
                        RootedValue::in_optional_context(value, context)
                    }
                    super::AssignmentStepSpec::Member(member) => {
                        crate::object::resolve::store_member_text_with_context(
                            context,
                            base,
                            member,
                            rhs,
                            false,
                            caller_function_name,
                            runmat_gc::gc_record_write,
                        )
                        .await
                    }
                    super::AssignmentStepSpec::Braces(_) => {
                        // The existing scalar/cell assignment adapter owns
                        // replication; do not request extra RHS outputs.
                        let Value::Cell(cell) = base else {
                            return Err(semantic_error(
                                "CellIndexType",
                                "Brace assignment requires a cell array",
                            ));
                        };
                        let positions = fixed_cell_positions.ok_or_else(|| {
                            semantic_error(
                                "PreparedDestinationState",
                                "fixed cell assignment lost its validated positions",
                            )
                        })?;
                        let sequence = crate::call::arguments::adapt_legacy_builtin_result(rhs)?;
                        let value = match sequence.kind() {
                            runmat_value::ValueSequenceKind::Single => {
                                let value = sequence.as_slice().first().ok_or_else(|| {
                                    semantic_error(
                                        "CellAssignmentArityMismatch",
                                        "single cell assignment value is missing",
                                    )
                                })?;
                                crate::object::cell::assign_cell_value_scalar_in_context(
                                    context,
                                    cell,
                                    &positions,
                                    value,
                                    runmat_gc::gc_record_write,
                                )?
                            }
                            runmat_value::ValueSequenceKind::CommaSeparated => {
                                crate::object::cell::assign_cell_value_multi_in_context(
                                    context,
                                    cell,
                                    &positions,
                                    sequence.as_slice(),
                                    runmat_gc::gc_record_write,
                                )?
                            }
                        };
                        RootedValue::in_optional_context(value, context)
                    }
                }
            }
            SequenceEndpointSpec::Member(field) => match context {
                Some(context) => {
                    crate::object::resolve::store_member_sequence_values_in_context(
                        context,
                        base,
                        field,
                        values,
                        caller_function_name,
                    )
                    .await
                }
                None => {
                    crate::object::resolve::store_member_sequence_values(
                        base,
                        field,
                        values,
                        caller_function_name,
                    )
                    .await
                }
            },
            SequenceEndpointSpec::CellContents(indices) => {
                assign_cell_contents_with_context(context, base, indices, values).await
            }
        }
    }
}

impl SequenceEndpointSpec {
    pub(crate) fn visit_retained_values(&self, visitor: &mut impl FnMut(&Value)) {
        match self {
            Self::Member(_) => {}
            Self::Fixed { step, .. } => match step {
                super::AssignmentStepSpec::Member(_) => {}
                super::AssignmentStepSpec::Parentheses { selectors }
                | super::AssignmentStepSpec::Braces(selectors) => {
                    for component in selectors {
                        if let ObjectIndexComponent::Value(value) = component {
                            visitor(value);
                        }
                    }
                }
            },
            Self::CellContents(components) => {
                for component in components {
                    if let ObjectIndexComponent::Value(value) = component {
                        visitor(value);
                    }
                }
            }
        }
    }
}

async fn assignment_cardinality_for_step(
    context: Option<&crate::context::RuntimeContext>,
    base: &Value,
    step: ObjectSubscript,
) -> Result<usize, RuntimeError> {
    let path = ObjectSubscriptPath::single_in_context(context, step)?;
    let prepared = match context {
        Some(context) => crate::builtins::introspection::object_indexing::object_path_assignment_cardinality_in_context(
            context, base, &path,
        ).await?,
        None => crate::builtins::introspection::object_indexing::object_path_assignment_cardinality(
            base, &path,
        ).await?,
    };
    Ok(prepared.count)
}

impl Trace for SequenceEndpointSpec {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.visit_retained_values(&mut |value| value.trace(tracer));
    }
}

pub(crate) async fn assign_cell_contents_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    indices: runmat_value::ExactArrayStorage<ObjectIndexComponent>,
    values: runmat_value::ValueSequence,
) -> Result<RootedValue, RuntimeError> {
    let input_values =
        std::iter::once(&base)
            .chain(values.iter())
            .chain(indices.iter().filter_map(|component| match component {
                ObjectIndexComponent::Value(value) => Some(value),
                ObjectIndexComponent::Colon => None,
            }));
    let _inputs = match context {
        Some(context) => RootedValueLease::for_values_iterator_in_context(input_values, context)?,
        None => RootedValueLease::for_values(input_values)
            .map_err(|source| source.into_runtime_error())?,
    };
    match base {
        Value::Cell(cell) => {
            let positions = crate::object::cell::resolve_cell_assignment_components_in_context(
                context, &cell, &indices,
            )?;
            crate::object::cell::assign_cell_value_multi_in_context(
                context,
                cell,
                &positions,
                values.as_slice(),
                runmat_gc::gc_record_write,
            )
            .and_then(|value| RootedValue::in_optional_context(value, context))
        }
        base @ (Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)) => {
            let resolution = crate::object::dispatch::resolve_object_index_protocol(
                &base,
                crate::object::indexing::ObjectIndexOp::Subsasgn,
                None,
            )?;
            let Some(method) = resolution else {
                return Err(semantic_error(
                    "MissingObjectProtocol",
                    "brace assignment object does not define subsasgn",
                ));
            };
            let step = ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
                components: indices,
            });
            let path = match context {
                Some(context) => ObjectSubscriptPath::single_in_context(Some(context), step)?,
                None => ObjectSubscriptPath::single(step),
            };
            match context {
                Some(context) => {
                    crate::object::protocol::invoke_resolved_object_assignment_sequence_in_context(
                        context, &method, base, path, values,
                    )
                    .await
                }
                None => {
                    crate::object::protocol::invoke_resolved_object_assignment_sequence(
                        &method, base, path, values,
                    )
                    .await
                }
            }
        }
        _ => Err(semantic_error(
            "CellAssignmentOnNonCell",
            "brace sequence assignment requires a cell array or indexing object",
        )),
    }
}
