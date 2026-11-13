use crate::execution::RootedValueHandles;
use runmat_value::{
    ExactHostAllocationPermit, ExactHostNumericAllocationError, SharedValueGeneration, Value,
};

/// A workspace value whose borrowed publication alias or transferred owner
/// retains its independent exact header claim. Root handles live beside the
/// shared generation, never inside it: the same physical owner can therefore
/// also be traced as a GC activation child without introducing root cycles.
#[derive(Clone, Debug)]
pub(crate) struct WorkspaceRetainedValue {
    generation: SharedValueGeneration,
    _roots: Option<RootedValueHandles>,
}

impl WorkspaceRetainedValue {
    pub(in crate::workspace) fn from_shared_generation(
        generation: SharedValueGeneration,
        roots: Option<RootedValueHandles>,
    ) -> Self {
        Self {
            generation,
            _roots: roots,
        }
    }

    pub(in crate::workspace) fn requested_bytes() -> usize {
        SharedValueGeneration::requested_bytes()
    }

    pub(in crate::workspace) fn prepare(
        value: &Value,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Option<Self>, ExactHostNumericAllocationError> {
        if !is_allocation_free_workspace_alias(value) {
            return Ok(None);
        }
        SharedValueGeneration::prepare(value.clone(), permit).map(|generation| {
            Some(Self {
                generation,
                _roots: None,
            })
        })
    }

    /// Retain an already-owned Value after its operation has admitted this
    /// wrapper. The transferred root lease protects the moved Value for the
    /// payload's lifetime; this constructor does not certify a borrowed
    /// base-workspace alias.
    pub(in crate::workspace) fn prepare_owned(
        value: Value,
        roots: RootedValueHandles,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let generation = SharedValueGeneration::prepare(value, permit)?;
        Ok(Self {
            generation,
            _roots: Some(roots),
        })
    }

    pub(crate) fn value(&self) -> &Value {
        self.generation.value()
    }

    pub(crate) fn shared_generation(&self) -> SharedValueGeneration {
        self.generation.clone()
    }

    pub(crate) fn shared_roots(&self) -> Option<RootedValueHandles> {
        self._roots.clone()
    }

    #[cfg(test)]
    pub(crate) fn shares_owner_with(&self, other: &Self) -> bool {
        self.generation.ptr_eq(&other.generation)
    }
}

/// One immutable owner for an already-owned legacy Value or a value retained
/// by an exact publication claim. Cloning this lease never clones the Value.
/// Rooted payloads retain their GC lease alongside the Value.
#[derive(Clone, Debug)]
pub(crate) enum WorkspaceValueLease {
    Unbounded {
        generation: SharedValueGeneration,
        _roots: Option<RootedValueHandles>,
    },
    Admitted(WorkspaceRetainedValue),
}

/// Immutable, rooted ownership of one workspace value generation.
///
/// Cloning this handle never clones the language value. Executors use it to
/// seed workspace slots without constructing a second payload owner; a later
/// language assignment replaces the slot with an owned value.
#[derive(Clone, Debug)]
pub struct WorkspaceValueReadLease {
    value: WorkspaceValueLease,
    _roots: Option<RootedValueHandles>,
}

impl WorkspaceValueReadLease {
    pub(crate) fn new(value: WorkspaceValueLease) -> Self {
        Self {
            value,
            _roots: None,
        }
    }

    pub(crate) fn with_roots(
        value: WorkspaceValueLease,
        roots: Option<RootedValueHandles>,
    ) -> Self {
        Self {
            value,
            _roots: roots,
        }
    }

    pub fn value(&self) -> &Value {
        self.value.value()
    }

    pub(crate) fn shared_generation(&self) -> SharedValueGeneration {
        self.value.shared_generation()
    }

    pub(crate) fn shared_roots(&self) -> Option<RootedValueHandles> {
        self._roots.clone().or_else(|| self.value.shared_roots())
    }
}

impl WorkspaceValueLease {
    pub(in crate::workspace) fn unbounded(value: Value) -> Self {
        Self::Unbounded {
            generation: SharedValueGeneration::unbounded(value),
            _roots: None,
        }
    }

    pub(in crate::workspace) fn unbounded_rooted(value: Value, roots: RootedValueHandles) -> Self {
        Self::Unbounded {
            generation: SharedValueGeneration::unbounded(value),
            _roots: Some(roots),
        }
    }

    pub(in crate::workspace) fn from_retained(value: WorkspaceRetainedValue) -> Self {
        Self::Admitted(value)
    }

    /// Preserve the physical admission provenance of an existing generation.
    /// Sharing an unbounded source does not turn it into a finite claim merely
    /// because a new workspace entry retains the same owner.
    pub(in crate::workspace) fn from_shared_generation(
        generation: SharedValueGeneration,
        roots: Option<RootedValueHandles>,
    ) -> Self {
        if generation.has_retained_claim() {
            Self::Admitted(WorkspaceRetainedValue::from_shared_generation(
                generation, roots,
            ))
        } else {
            Self::Unbounded {
                generation,
                _roots: roots,
            }
        }
    }

    pub(in crate::workspace) fn value(&self) -> &Value {
        match self {
            Self::Unbounded { generation, .. } => generation.value(),
            Self::Admitted(value) => value.value(),
        }
    }

    pub(in crate::workspace) fn shared_generation(&self) -> SharedValueGeneration {
        match self {
            Self::Unbounded { generation, .. } => generation.clone(),
            Self::Admitted(value) => value.shared_generation(),
        }
    }

    pub(in crate::workspace) fn shared_roots(&self) -> Option<RootedValueHandles> {
        match self {
            Self::Unbounded { _roots, .. } => _roots.clone(),
            Self::Admitted(value) => value.shared_roots(),
        }
    }

    pub(in crate::workspace) fn is_admitted(&self) -> bool {
        matches!(self, Self::Admitted(value) if value.generation.has_retained_claim())
    }

    #[cfg(test)]
    pub(in crate::workspace) fn shares_owner_with(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Unbounded {
                    generation: left, ..
                },
                Self::Unbounded {
                    generation: right, ..
                },
            ) => left.ptr_eq(right),
            (Self::Admitted(left), Self::Admitted(right)) => left.shares_owner_with(right),
            _ => false,
        }
    }
}

/// These variants clone only fixed inline data or already-shared immutable
/// array owners. Other values require the general aggregate/GC admission path.
pub(crate) fn is_allocation_free_workspace_alias(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Complex(_, _) | Value::Bool(_) => true,
        Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::SparseTensor(_)
        | Value::GpuTensor(_) => true,
        Value::ComplexTensor(tensor) => matches!(
            tensor.complex_storage(),
            runmat_value::ComplexStorage::F32(_) | runmat_value::ComplexStorage::F64(_)
        ),
        Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::Object(_)
        | Value::ObjectArray(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::OutputList(_)
        | Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_) => false,
    }
}
