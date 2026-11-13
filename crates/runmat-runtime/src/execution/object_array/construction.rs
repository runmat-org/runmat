//! The move-only unpublished ObjectArray destination and its exact quote.
use super::*;
#[derive(Clone, Copy)]
pub(crate) enum Class<'a> {
    Owner(&'a SharedClassIdentity),
    Name(&'a str),
}
#[derive(Clone, Copy)]
enum Shape<'a> {
    Slice(&'a [usize]),
    Owner(&'a SharedShape),
}
enum Builder {
    Exact(ObjectArrayBuilder),
    Ordinary {
        class: SharedClassIdentity,
        data: Vec<Value>,
        shape: SharedShape,
        provenance: Vec<Rc<SelectedObjectClassFacts>>,
        unselected: bool,
    },
}
pub(crate) struct ObjectArrayConstruction<'a> {
    builder: Builder,
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a RuntimeContext>,
    count: usize,
    unselected: bool,
}

impl<'a> ObjectArrayConstruction<'a> {
    pub(crate) fn new<'p>(
        runtime: Option<&'a RuntimeContext>,
        class: Class<'_>,
        count: usize,
        shape: &[usize],
        provenance: impl Iterator<Item = &'p Rc<SelectedObjectClassFacts>> + Clone,
        unselected: bool,
    ) -> Result<Self, RuntimeError> {
        let capacity = if count == 0 {
            provenance.clone().count()
        } else {
            0
        };
        let mut result = Self::with_capacity(runtime, class, count, shape, capacity, unselected)?;
        for facts in provenance {
            result.push_provenance(facts.clone())?;
        }
        Ok(result)
    }
    pub(crate) fn new_retaining_shape<'p>(
        runtime: Option<&'a RuntimeContext>,
        class: Class<'_>,
        count: usize,
        shape: &SharedShape,
        provenance: impl Iterator<Item = &'p Rc<SelectedObjectClassFacts>> + Clone,
        unselected: bool,
    ) -> Result<Self, RuntimeError> {
        let capacity = if count == 0 {
            provenance.clone().count()
        } else {
            0
        };
        let mut result = Self::with_shape_capacity(
            runtime,
            class,
            count,
            Shape::Owner(shape),
            capacity,
            unselected,
        )?;
        for facts in provenance {
            result.push_provenance(facts.clone())?;
        }
        Ok(result)
    }
    pub(crate) fn with_capacity(
        runtime: Option<&'a RuntimeContext>,
        class: Class<'_>,
        count: usize,
        shape: &[usize],
        capacity: usize,
        unselected: bool,
    ) -> Result<Self, RuntimeError> {
        Self::with_shape_capacity(
            runtime,
            class,
            count,
            Shape::Slice(shape),
            capacity,
            unselected,
        )
    }
    pub(crate) fn with_capacity_retaining_shape(
        runtime: Option<&'a RuntimeContext>,
        class: Class<'_>,
        count: usize,
        shape: &SharedShape,
        capacity: usize,
        unselected: bool,
    ) -> Result<Self, RuntimeError> {
        Self::with_shape_capacity(
            runtime,
            class,
            count,
            Shape::Owner(shape),
            capacity,
            unselected,
        )
    }
    fn with_shape_capacity(
        runtime: Option<&'a RuntimeContext>,
        class: Class<'_>,
        count: usize,
        shape: Shape<'_>,
        capacity: usize,
        unselected: bool,
    ) -> Result<Self, RuntimeError> {
        let mut quote = match class {
            Class::Owner(class) => match shape {
                Shape::Slice(shape) => ObjectArrayBuilder::quote(class, count, shape, capacity),
                Shape::Owner(shape) => {
                    ObjectArrayBuilder::quote_retaining_shape(class, count, shape, capacity)
                }
            },
            Class::Name(_) => match shape {
                Shape::Slice(shape) => ObjectArrayBuilder::quote_storage(count, shape, capacity),
                Shape::Owner(shape) => {
                    ObjectArrayBuilder::quote_storage_retaining_shape(count, shape, capacity)
                }
            },
        }
        .map_err(|e| build_error(runtime, e))?;
        if let Class::Name(name) = class {
            quote.requested_bytes = quote
                .requested_bytes
                .checked_add(
                    SharedClassIdentity::requested_bytes_for_capacity(name.len())
                        .map_err(|e| allocation(runtime, e.into()))?,
                )
                .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
            quote.retained_claims = quote
                .retained_claims
                .checked_add(1)
                .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
        }
        let output = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(quote.requested_bytes)
                .map_err(|_| allocation(runtime, HostAllocationError::Overflow))?,
            quote.retained_claims,
        )
        .map_err(|e| allocation(runtime, e))?;
        let receipt = HostAllocationReceipt::admit(
            runtime,
            HostAllocationPlan::new(0, output).map_err(|e| allocation(runtime, e))?,
        )
        .map_err(|e| allocation(runtime, e))?;
        let builder = if let Some(receipt) = &receipt {
            let context = runtime.expect("managed receipt retains selected context");
            let permit = receipt
                .output_permit(context)
                .map_err(|e| allocation(runtime, e))?;
            let class = match class {
                Class::Owner(class) => class.clone(),
                Class::Name(name) => {
                    let prepared = PreparedObjectClassName::prepare(name, &permit);
                    receipt
                        .checkpoint(context)
                        .map_err(|e| allocation(runtime, e))?;
                    prepared
                        .map_err(|e| match e {
                            runmat_value::ObjectClassPreparationError::Allocation(source) => {
                                allocation(runtime, source.into())
                            }
                            source => crate::runtime_error::semantic_error_with_source(
                                "ObjectArrayConstruction",
                                "object class preparation failed",
                                source,
                            ),
                        })?
                        .into_identity()
                }
            };
            receipt
                .checkpoint(context)
                .map_err(|e| allocation(runtime, e))?;
            let builder = match shape {
                Shape::Slice(shape) => {
                    ObjectArrayBuilder::try_new(class, count, shape, capacity, &permit)
                }
                Shape::Owner(shape) => ObjectArrayBuilder::try_new_retaining_shape(
                    class,
                    count,
                    shape.clone(),
                    capacity,
                    &permit,
                ),
            };
            receipt
                .checkpoint(context)
                .map_err(|e| allocation(runtime, e))?;
            let mut builder = builder.map_err(|e| build_error(runtime, e))?;
            if unselected {
                builder = builder.with_unselected_provenance();
            }
            Builder::Exact(builder)
        } else {
            let class = match class {
                Class::Owner(class) => class.clone(),
                Class::Name(name) => name.into(),
            };
            Builder::Ordinary {
                class,
                data: Vec::with_capacity(count),
                shape: match shape {
                    Shape::Slice(shape) => SharedShape::new_unmanaged(shape.to_vec()),
                    Shape::Owner(shape) => shape.clone(),
                },
                provenance: Vec::with_capacity(capacity),
                unselected,
            }
        };
        let result = Self {
            builder,
            receipt,
            runtime,
            count,
            unselected,
        };
        result.checkpoint()?;
        Ok(result)
    }
    pub(crate) fn push_provenance(
        &mut self,
        facts: Rc<SelectedObjectClassFacts>,
    ) -> Result<(), RuntimeError> {
        self.checkpoint()?;
        match &mut self.builder {
            Builder::Exact(builder) => {
                builder
                    .try_push_selected_provenance(facts)
                    .map_err(|e| build_error(self.runtime, e))?;
            }
            Builder::Ordinary {
                class, provenance, ..
            } => {
                if facts.class() != class.as_identity() {
                    return Err(build_error(
                        self.runtime,
                        ObjectArrayBuildError::InvalidProvenance,
                    ));
                }
                if self.count == 0 {
                    provenance.push(facts);
                }
            }
        }
        Ok(())
    }
    pub(crate) fn mark_unselected(&mut self) {
        self.unselected = true;
    }
    pub(crate) fn checkpoint(&self) -> Result<(), RuntimeError> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("selected managed context"))
                .map_err(|e| allocation(self.runtime, e))?;
        }
        Ok(())
    }
    pub(crate) fn push(&mut self, value: Value) -> Result<(), RuntimeError> {
        self.checkpoint()?;
        match &mut self.builder {
            Builder::Exact(builder) => builder.try_push(value).map_err(|e| {
                let (kind, message, _value) = e.into_parts();
                crate::runtime_error::semantic_error_with_source(
                    "ObjectArrayConstruction",
                    message,
                    kind,
                )
            }),
            Builder::Ordinary { data, .. } => {
                if data.len() == self.count {
                    return Err(build_error(self.runtime, ObjectArrayBuildError::Capacity));
                }
                data.push(value);
                Ok(())
            }
        }
    }
    pub(crate) fn finish_value(self) -> Result<Value, RuntimeError> {
        self.checkpoint()?;
        let array = match self.builder {
            Builder::Exact(builder) => {
                let builder = if self.unselected {
                    builder.with_unselected_provenance()
                } else {
                    builder
                };
                builder.finish().map_err(|e| build_error(self.runtime, e))?
            }
            Builder::Ordinary {
                class,
                data,
                shape,
                provenance,
                unselected,
            } => {
                let mut array = ObjectArray::from_shared_storage_unbounded(
                    class,
                    runmat_value::OutputListStorage::shared_unbounded(data),
                    shape,
                )
                .map_err(|e| crate::runtime_error::semantic_error("ObjectArrayConstruction", e))?;
                array = array
                    .with_selected_provenance_unbounded(provenance)
                    .map_err(|e| {
                        crate::runtime_error::semantic_error("ObjectArrayConstruction", e)
                    })?;
                if unselected || self.unselected {
                    array = array.with_unselected_provenance();
                }
                array
            }
        };
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("selected managed context"))
                .map_err(|e| allocation(self.runtime, e))?;
        }
        Ok(Value::ObjectArray(array))
    }
    pub(crate) fn finish(self) -> Result<RootedValue, RuntimeError> {
        let runtime = self.runtime;
        let value = self.finish_value()?;
        match runtime {
            Some(runtime) => RootedValue::in_context(value, runtime),
            None => RootedValue::new(value).map_err(|e| e.into_runtime_error()),
        }
    }
}
