//! Immutable solver DATA, not a numerical certificate or execution capability.
//! Runtime validates every reused row under the current original invocation.
use super::LinearRowRole;
use crate::{ExactArrayStorage, HostNumericBuffer, SparseTensor, StructValue, Tensor, Value};
use runmat_gc_api::{Trace, Tracer};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WarmStartMemoryLimits {
    pub equalities: Option<usize>,
    pub inequalities: Option<usize>,
}

/// Actual completed constraint backing moves here once; there is no model,
/// objective, factor closure, Runtime, grant or historic-generation chain.
#[derive(Debug, PartialEq)]
pub struct LinearWorkingSetData {
    constraints: SparseTensor,
    rhs: HostNumericBuffer,
    roles: ExactArrayStorage<LinearRowRole>,
    active: ExactArrayStorage<bool>,
    equalities: usize,
}
impl LinearWorkingSetData {
    pub fn new(
        constraints: SparseTensor,
        rhs: HostNumericBuffer,
        roles: ExactArrayStorage<LinearRowRole>,
        active: ExactArrayStorage<bool>,
        equalities: usize,
    ) -> Result<Self, &'static str> {
        if constraints.rows != rhs.len()
            || roles.len() != rhs.len()
            || active.len() != rhs.len()
            || equalities > rhs.len()
        {
            return Err("warm-start constraint state has inconsistent row geometry");
        }
        if constraints
            .numeric_host_buffer()
            .is_none_or(|v| v.numeric_dtype() != crate::NumericDType::F64)
            || rhs.numeric_dtype() != crate::NumericDType::F64
        {
            return Err("warm-start constraint state must retain double storage");
        }
        Ok(Self {
            constraints,
            rhs,
            roles,
            active,
            equalities,
        })
    }
    pub fn constraints(&self) -> &SparseTensor {
        &self.constraints
    }
    pub fn rhs(&self) -> &HostNumericBuffer {
        &self.rhs
    }
    pub fn roles(&self) -> &[LinearRowRole] {
        &self.roles
    }
    pub fn active(&self) -> &[bool] {
        &self.active
    }
    pub fn equalities(&self) -> usize {
        self.equalities
    }
}

#[derive(Debug, PartialEq)]
pub struct ActiveSetWarmStartData {
    x: Value,
    options: Value,
    limits: WarmStartMemoryLimits,
    working: Option<LinearWorkingSetData>,
}
impl ActiveSetWarmStartData {
    pub fn new(
        x: Tensor,
        options: StructValue,
        limits: WarmStartMemoryLimits,
        working: Option<LinearWorkingSetData>,
    ) -> Result<Self, &'static str> {
        if x.shape() != [x.len(), 1] {
            return Err("warm-start X must be a column");
        }
        if x.numeric_dtype() != crate::NumericDType::F64 {
            return Err("warm-start X must retain double storage");
        }
        if limits.equalities == Some(0) || limits.inequalities == Some(0) {
            return Err("warm-start memory limits must be positive or unbounded");
        }
        if working
            .as_ref()
            .is_some_and(|w| w.constraints.cols != x.len())
        {
            return Err("warm-start point and constraint state dimensions disagree");
        }
        Ok(Self {
            x: Value::Tensor(x),
            options: Value::Struct(options),
            limits,
            working,
        })
    }
    pub fn x(&self) -> &Value {
        &self.x
    }
    pub fn options(&self) -> &Value {
        &self.options
    }
    pub fn limits(&self) -> WarmStartMemoryLimits {
        self.limits
    }
    pub fn working(&self) -> Option<&LinearWorkingSetData> {
        self.working.as_ref()
    }
    pub fn value_at(&self, index: usize) -> Option<&Value> {
        match index {
            0 => Some(&self.x),
            1 => Some(&self.options),
            _ => None,
        }
    }
}
impl Trace for ActiveSetWarmStartData {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.x.trace(tracer);
        self.options.trace(tracer);
    }
}
