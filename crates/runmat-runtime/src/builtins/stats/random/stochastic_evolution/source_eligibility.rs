//! Effect-free semantic classification of an already selected source loop.
//! This is not physical admission, a random draw, or provider authorization.

pub use crate::builtins::array::creation::randn::source_controls::ValueControlRef;
use runmat_accelerate_api::scalar_program::ScalarLiteral;
use runmat_accelerate_api::{GpuTensorStorage, NumericElementType};
use runmat_types::{NumericClass, OperatorKind, RealFloatingRecipe, StorageFact};
use runmat_value::{NumericDType, NumericScalar, Value};

mod preparation;
pub use preparation::{prepare_stochastic_source, StochasticPreparedSource};

/// The VM supplies an indexed borrow, not an allocated `Vec<&Value>` or an
/// already-cloned argument list. Runtime owns the exact parser/plan quote and
/// must retain its preparation permit through classification and execution.
pub struct StochasticSourceControls<'a> {
    pub state: &'a Value,
    pub steps: ValueControlRef<'a>,
    pub drift: ValueControlRef<'a>,
    pub scale: ValueControlRef<'a>,
    pub randn_arg_count: usize,
    pub randn_arg: &'a dyn Fn(usize) -> Option<ValueControlRef<'a>>,
}

/// Publication phases for the exact source loop, in program order. The VM
/// adapter must use its ordinary binding-store path for each event, so a
/// terminal error leaves the already-completed prefix visible. In particular
/// an arithmetic error after `Noise` must not roll back `t` or `Z`.
#[derive(Debug)]
pub enum StochasticCheckpoint {
    IterationBinding(f64),
    Noise(Value),
    State(Value),
}

/// The whole-call coordinator may decline only before its first observable
/// checkpoint, RNG draw, upload, or provider submission. After that boundary
/// every error is terminal and retains the published source prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StochasticSourceExecution {
    Declined,
    Completed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StochasticTypedPattern {
    /// Both source operands have this exact geometry. A broadcasting route
    /// needs its own proof rather than silently changing the source program.
    shape: Vec<usize>,
    state_class: NumericClass,
    state_storage: StorageFact,
    noise_class: NumericClass,
    noise_storage: StorageFact,
    drift: ScalarLiteral,
    scale: ScalarLiteral,
    scale_noise: RealFloatingRecipe,
    drift_scaled: RealFloatingRecipe,
    first_state_factor: RealFloatingRecipe,
    continuing_state_factor: RealFloatingRecipe,
    steps: u32,
    last_counter: f64,
}

impl StochasticTypedPattern {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub const fn state_class(&self) -> NumericClass {
        self.state_class
    }

    pub const fn state_storage(&self) -> StorageFact {
        self.state_storage
    }

    pub const fn noise_class(&self) -> NumericClass {
        self.noise_class
    }

    pub const fn noise_storage(&self) -> StorageFact {
        self.noise_storage
    }

    pub const fn drift(&self) -> ScalarLiteral {
        self.drift
    }

    pub const fn scale(&self) -> ScalarLiteral {
        self.scale
    }

    pub const fn scale_noise(&self) -> RealFloatingRecipe {
        self.scale_noise
    }

    pub const fn drift_scaled(&self) -> RealFloatingRecipe {
        self.drift_scaled
    }

    pub const fn first_state_factor(&self) -> RealFloatingRecipe {
        self.first_state_factor
    }

    pub const fn continuing_state_factor(&self) -> RealFloatingRecipe {
        self.continuing_state_factor
    }

    pub const fn steps(&self) -> u32 {
        self.steps
    }

    pub const fn last_counter(&self) -> f64 {
        self.last_counter
    }

    pub const fn exponential_class(&self) -> NumericClass {
        self.drift_scaled.result_class()
    }

    pub const fn result_class(&self) -> NumericClass {
        self.first_state_factor.result_class()
    }
}

fn state_description(value: &Value) -> Option<(&[usize], NumericClass, StorageFact)> {
    const SCALAR_SHAPE: [usize; 2] = [1, 1];
    match value {
        Value::Num(_) => Some((&SCALAR_SHAPE, NumericClass::Double, StorageFact::Scalar)),
        Value::Tensor(tensor) => {
            let class = match tensor.numeric_dtype() {
                NumericDType::F32 => NumericClass::Single,
                NumericDType::F64 => NumericClass::Double,
                _ => return None,
            };
            Some((tensor.shape(), class, StorageFact::Dense))
        }
        Value::GpuTensor(handle)
            if handle.physical().storage() == GpuTensorStorage::Real
                && !handle.view().is_logical() =>
        {
            let class = match handle.physical().element_type() {
                NumericElementType::F32 => NumericClass::Single,
                NumericElementType::F64 => NumericClass::Double,
                _ => return None,
            };
            // This is a type/geometry description, not an operational lease or
            // permission to download. The selected executor must revalidate
            // the live handle and exact provider owner before its first effect.
            Some((handle.shape(), class, StorageFact::Opaque))
        }
        // Logical, complex, integer, sparse, and non-tensor states retain the source
        // For until their arithmetic/ownership route is independently proven.
        _ => None,
    }
}

fn scalar_literal(value: &Value) -> Option<ScalarLiteral> {
    match value {
        Value::Num(value) => Some(ScalarLiteral::F64Bits(value.to_bits())),
        Value::Tensor(tensor) if tensor.len() == 1 => match tensor.numeric_value_at(0)? {
            NumericScalar::F32(value) => Some(ScalarLiteral::F32Bits(value.to_bits())),
            NumericScalar::F64(value) => Some(ScalarLiteral::F64Bits(value.to_bits())),
            _ => None,
        },
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct RecipeFacts {
    scale_noise: RealFloatingRecipe,
    drift_scaled: RealFloatingRecipe,
    first_state_factor: RealFloatingRecipe,
    continuing_state_factor: RealFloatingRecipe,
}

fn recipe_facts(
    state_class: NumericClass,
    noise_class: NumericClass,
    drift: ScalarLiteral,
    scale: ScalarLiteral,
) -> Option<RecipeFacts> {
    let scale_noise = RealFloatingRecipe::for_classes(
        OperatorKind::ElementwiseMultiply,
        scale.class(),
        noise_class,
    )?;
    let drift_scaled = RealFloatingRecipe::for_classes(
        OperatorKind::Add,
        drift.class(),
        scale_noise.result_class(),
    )?;
    let first_state_factor = RealFloatingRecipe::for_classes(
        OperatorKind::ElementwiseMultiply,
        state_class,
        drift_scaled.result_class(),
    )?;
    let continuing_state_factor = RealFloatingRecipe::for_classes(
        OperatorKind::ElementwiseMultiply,
        first_state_factor.result_class(),
        drift_scaled.result_class(),
    )?;
    (continuing_state_factor.result_class() == first_state_factor.result_class()).then_some(
        RecipeFacts {
            scale_noise,
            drift_scaled,
            first_state_factor,
            continuing_state_factor,
        },
    )
}

/// Cheap guard used before evaluating source-body operands. An unentered or
/// unsupported range must follow the original For path without reading any
/// randn controls or the prior state.
pub fn positive_stochastic_source_steps(value: &Value) -> Option<u32> {
    match value {
        Value::Num(value) if value.is_finite() && *value > 0.0 && value.fract() == 0.0 => {
            if *value <= f64::from(u32::MAX) {
                Some(*value as u32)
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use runmat_value::Tensor;
    use std::rc::Rc;

    #[test]
    fn resident_description_distinguishes_logical_views_from_numeric_storage() {
        use runmat_accelerate_api::{test_support::TestAllocationOwner, ProviderPrecision};

        let owner = TestAllocationOwner::new();
        for (element, precision, class) in [
            (
                NumericElementType::F32,
                ProviderPrecision::F32,
                NumericClass::Single,
            ),
            (
                NumericElementType::F64,
                ProviderPrecision::F64,
                NumericClass::Double,
            ),
        ] {
            let numeric = owner
                .allocate_zeroed(vec![2, 1], element, GpuTensorStorage::Real)
                .unwrap();
            let logical = owner
                .allocate_logical_zeroed(vec![2, 1], precision)
                .unwrap();
            assert!(matches!(
                state_description(&Value::GpuTensor(numeric.handle)),
                Some((shape, actual_class, StorageFact::Opaque))
                    if shape == [2, 1] && actual_class == class
            ));
            assert!(state_description(&Value::GpuTensor(logical.handle)).is_none());
        }
    }

    #[test]
    fn default_noise_does_not_inherit_single_state_class() {
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let state = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let drift = Value::Num(0.1);
        let scale = Value::Num(0.2);
        let steps = Value::Num(3.0);
        let noise_dimensions = [Value::Num(2.0), Value::Num(1.0)];
        let prepared = prepare_stochastic_source(
            &context,
            StochasticSourceControls {
                state: &state,
                steps: ValueControlRef::Borrowed(&steps),
                drift: ValueControlRef::Borrowed(&drift),
                scale: ValueControlRef::Borrowed(&scale),
                randn_arg_count: noise_dimensions.len(),
                randn_arg: &|index| noise_dimensions.get(index).map(ValueControlRef::Borrowed),
            },
        )
        .expect("admitted source preparation")
        .expect("ordinary mixed-class source should be described");
        let plan = prepared.pattern();
        assert_eq!(plan.state_class(), NumericClass::Single);
        assert_eq!(plan.state_storage(), StorageFact::Dense);
        assert_eq!(plan.noise_class(), NumericClass::Double);
        assert_eq!(plan.noise_storage(), StorageFact::Dense);
        assert_eq!(plan.scale_noise().computation_class(), NumericClass::Double);
        assert_eq!(plan.scale_noise().result_class(), NumericClass::Double);
        assert_eq!(plan.drift_scaled().result_class(), NumericClass::Double);
        assert_eq!(
            plan.first_state_factor().computation_class(),
            NumericClass::Double
        );
        assert_eq!(
            plan.first_state_factor().result_class(),
            NumericClass::Single
        );
        assert_eq!(
            plan.continuing_state_factor().result_class(),
            NumericClass::Single
        );
        assert_eq!(plan.steps(), 3);
        assert_eq!(plan.last_counter(), 3.0);

        let zero_steps = Value::Num(0.0);
        let zero = prepare_stochastic_source(
            &context,
            StochasticSourceControls {
                state: &state,
                steps: ValueControlRef::Borrowed(&zero_steps),
                drift: ValueControlRef::Borrowed(&drift),
                scale: ValueControlRef::Borrowed(&scale),
                randn_arg_count: noise_dimensions.len(),
                randn_arg: &|index| noise_dimensions.get(index).map(ValueControlRef::Borrowed),
            },
        )
        .expect("zero-step preparation");
        assert!(zero.is_none());

        let scalar_state = Value::Num(2.0);
        let scalar_steps = Value::Num(1.0);
        let scalar = prepare_stochastic_source(
            &context,
            StochasticSourceControls {
                state: &scalar_state,
                steps: ValueControlRef::Borrowed(&scalar_steps),
                drift: ValueControlRef::Borrowed(&drift),
                scale: ValueControlRef::Borrowed(&scale),
                randn_arg_count: 0,
                randn_arg: &|_| None,
            },
        )
        .expect("scalar source preparation")
        .expect("ordinary scalar source should be described");
        let scalar = scalar.pattern();
        assert_eq!(scalar.shape(), &[1, 1]);
        assert_eq!(scalar.state_storage(), StorageFact::Scalar);
        assert_eq!(scalar.noise_storage(), StorageFact::Scalar);
        assert_eq!(scalar.result_class(), NumericClass::Double);
    }
}
