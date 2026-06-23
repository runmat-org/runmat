use runmat_builtins::{IntValue, Value};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static ABI_GC_ROOTS: RefCell<HashMap<usize, runmat_gc::ExplicitRoot>> =
        RefCell::new(HashMap::new());
}

/// Stable value tag used by the Turbine host ABI.
///
/// The f64-only JIT path remains an optimization. Cross-runtime calls that can
/// carry cells, tensors, strings, objects, or output lists use this tagged value
/// representation so semantic identity is preserved without falling back to
/// legacy name dispatch.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurbineValueTag {
    Empty = 0,
    Num = 1,
    Bool = 2,
    Int = 3,
    GcHandle = 4,
}

/// C-compatible Turbine value slot.
///
/// `payload` is interpreted by `tag`:
///
/// - `Num`: raw `f64::to_bits()`
/// - `Bool`: `0` or `1`
/// - `Int`: signed `i64` bits
/// - `GcHandle`: opaque identity token for a rooted GC-managed `Value`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurbineValue {
    pub tag: TurbineValueTag,
    pub reserved: u32,
    pub payload: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurbineArgSpec {
    pub is_expand: u32,
    pub num_indices: u32,
    pub expand_all: u32,
    pub reserved: u32,
}

impl TurbineArgSpec {
    pub const fn new(is_expand: bool, num_indices: usize, expand_all: bool) -> Self {
        Self {
            is_expand: is_expand as u32,
            num_indices: num_indices as u32,
            expand_all: expand_all as u32,
            reserved: 0,
        }
    }
}

impl TurbineValue {
    pub const fn empty() -> Self {
        Self {
            tag: TurbineValueTag::Empty,
            reserved: 0,
            payload: 0,
        }
    }

    pub fn from_runtime_value(value: Value) -> crate::Result<Self> {
        match value {
            Value::Num(value) => Ok(Self {
                tag: TurbineValueTag::Num,
                reserved: 0,
                payload: value.to_bits(),
            }),
            Value::Bool(value) => Ok(Self {
                tag: TurbineValueTag::Bool,
                reserved: 0,
                payload: u64::from(value),
            }),
            Value::Int(value) => Ok(Self {
                tag: TurbineValueTag::Int,
                reserved: 0,
                payload: value.to_i64() as u64,
            }),
            value => {
                let root = runmat_gc::gc_allocate_rooted(value)
                    .map_err(|err| crate::execution_error(err.to_string()))?;
                let payload = runmat_gc::gc_handle_addr(&root.handle());
                ABI_GC_ROOTS.with(|roots| {
                    roots.borrow_mut().insert(payload, root);
                });
                Ok(Self {
                    tag: TurbineValueTag::GcHandle,
                    reserved: 0,
                    payload: payload as u64,
                })
            }
        }
    }

    pub fn to_runtime_value(self) -> crate::Result<Value> {
        match self.tag {
            TurbineValueTag::Empty => Ok(Value::Num(0.0)),
            TurbineValueTag::Num => Ok(Value::Num(f64::from_bits(self.payload))),
            TurbineValueTag::Bool => Ok(Value::Bool(self.payload != 0)),
            TurbineValueTag::Int => Ok(Value::Int(IntValue::I64(self.payload as i64))),
            TurbineValueTag::GcHandle => {
                if self.payload == 0 {
                    return Err(crate::execution_error("null Turbine GC value handle"));
                }
                let addr = self.payload as usize;
                let rooted =
                    ABI_GC_ROOTS.with(|roots| roots.borrow().get(&addr).map(|root| root.handle()));
                let ptr = if let Some(handle) = rooted {
                    handle
                } else {
                    // SAFETY: this fallback only exists for ABI tokens that
                    // predate the local root table or are supplied by legacy JIT
                    // entrypoints. The GC validates that the address currently
                    // names a live GC Value before any clone occurs below.
                    unsafe { runmat_gc::gc_handle_from_addr(addr) }
                        .map_err(|err| crate::execution_error(err.to_string()))?
                };
                runmat_gc::gc_clone_value(&ptr)
                    .map_err(|err| crate::execution_error(err.to_string()))
            }
        }
    }
}

#[cfg(test)]
fn clear_abi_gc_roots_for_test() {
    ABI_GC_ROOTS.with(|roots| roots.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::{TurbineValue, TurbineValueTag};
    use runmat_builtins::{IntValue, Value};

    #[test]
    fn scalar_values_round_trip_through_turbine_value() {
        let num = TurbineValue::from_runtime_value(Value::Num(42.5)).unwrap();
        assert_eq!(num.tag, TurbineValueTag::Num);
        assert_eq!(num.to_runtime_value().unwrap(), Value::Num(42.5));

        let bool_value = TurbineValue::from_runtime_value(Value::Bool(true)).unwrap();
        assert_eq!(bool_value.tag, TurbineValueTag::Bool);
        assert_eq!(bool_value.to_runtime_value().unwrap(), Value::Bool(true));

        let int_value = TurbineValue::from_runtime_value(Value::Int(IntValue::I64(-7))).unwrap();
        assert_eq!(int_value.tag, TurbineValueTag::Int);
        assert_eq!(
            int_value.to_runtime_value().unwrap(),
            Value::Int(IntValue::I64(-7))
        );
    }

    #[test]
    fn non_scalar_values_round_trip_through_gc_handle() {
        runmat_gc::gc_test_context(|| {
            super::clear_abi_gc_roots_for_test();
            let value = Value::String("semantic-handle".to_string());
            let turbine_value = TurbineValue::from_runtime_value(value.clone()).unwrap();

            assert_eq!(turbine_value.tag, TurbineValueTag::GcHandle);
            assert_ne!(turbine_value.payload, 0);
            assert_eq!(turbine_value.to_runtime_value().unwrap(), value);
            super::clear_abi_gc_roots_for_test();
        });
    }

    #[test]
    fn gc_handle_payload_is_rooted_until_abi_roots_are_cleared() {
        runmat_gc::gc_test_context(|| {
            super::clear_abi_gc_roots_for_test();

            let value = Value::String("rooted-token".to_string());
            let turbine_value = TurbineValue::from_runtime_value(value.clone()).unwrap();

            runmat_gc::gc_collect_minor().unwrap();
            assert_eq!(turbine_value.to_runtime_value().unwrap(), value);

            super::clear_abi_gc_roots_for_test();
            runmat_gc::gc_collect_minor().unwrap();
            assert!(turbine_value.to_runtime_value().is_err());
        });
    }
}
