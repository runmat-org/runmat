//! Shared concrete floating plot operands; rejected rich carriers are never
//! traversed to derive captures/properties or format a debugging value.
use crate::{BuiltinResult, RuntimeError};
use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueKindFact};
use runmat_value::Value;
#[derive(Clone, Copy)]
pub(super) struct Owner(runmat_builtins::BuiltinCatalogIdentity);
impl Owner {
    pub(super) const fn new(name: &'static str) -> Self {
        Self(runmat_builtins::BuiltinCatalogIdentity { name })
    }
    pub(super) fn real(self, value: &Value) -> BuiltinResult<()> {
        if matches!(
            value,
            Value::Num(_) | Value::Tensor(_) | Value::GpuTensor(_)
        ) && matches!(
            crate::value_fact::value_fact(value).kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single | NumericClass::Double,
                domain: NumericDomain::Real
            })
        ) {
            Ok(())
        } else {
            Err(self.invalid("expected real single/double data"))
        }
    }
    pub(super) fn scalar(self, value: &Value) -> BuiltinResult<f64> {
        self.real(value)?;
        match value {
            Value::Num(value) => Ok(*value),
            Value::Tensor(value) if value.len() == 1 => {
                Ok(crate::builtins::common::tensor::tensor_value_f64(value, 0))
            }
            _ => Err(self.invalid("optimization value fields must be floating scalars")),
        }
    }
    pub(super) fn invalid(self, detail: &str) -> RuntimeError {
        crate::build_runtime_error(format!("{}: {detail}", self.0.name))
            .with_builtin(self.0.name)
            .with_identifier(format!("RunMat:{}:InvalidArgument", self.0.name))
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    }
}
