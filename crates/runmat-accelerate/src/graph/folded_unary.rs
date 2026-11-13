//! Exact selected unary cast fold, independent of a consumer's use of the value.

use super::*;
use runmat_types::{NumericClass, ScalarOperation};

impl AccelGraph {
    /// A compiler-marked unary fold is immutable only when the selected
    /// canonical cast reproduces its stored scalar bits from the source.
    /// Binding/liveness admission remains with the consuming execution owner.
    pub fn exact_folded_unary_scalar(&self, id: ValueId) -> Option<&BuiltinValue> {
        let info = self.value(id)?;
        let ValueOrigin::NodeOutput { node, output: 0 } = info.origin else {
            return None;
        };
        let producer = self.node(node)?;
        if producer.id != node
            || producer.outputs.as_slice() != [id]
            || producer.span.start > producer.span.end
            || producer.category != AccelOpCategory::Elementwise
            || producer.tags.as_slice()
                != [AccelGraphTag::Elementwise, AccelGraphTag::UnaryScalarFold]
            || info.shape != ShapeInfo::Tensor(vec![Some(1), Some(1)])
            || info.ty
                != (Type::Tensor {
                    shape: Some(vec![Some(1), Some(1)]),
                })
        {
            return None;
        }
        let AccelNodeLabel::Builtin {
            name,
            origin: FusionCallOrigin::Named { binding },
        } = &producer.label
        else {
            return None;
        };
        let entry = runmat_builtins::builtin_catalog_entry_by_name(name)?;
        if entry.identity != binding.builtin {
            return None;
        }
        let declaration =
            runmat_runtime::builtins::common::spec::canonical_fusion_declaration(*binding)?;
        let ScalarOperation::Cast(class) = declaration.semantics?.scalar_operation? else {
            return None;
        };
        let [source] = producer.inputs.as_slice() else {
            return None;
        };
        let source = match self.source_literal_constant(*source)? {
            BuiltinValue::Num(value) => *value,
            BuiltinValue::Bool(value) => f64::from(u8::from(*value)),
            _ => return None,
        };
        let represented = info.constant.as_ref()?;
        let BuiltinValue::Tensor(result) = represented else {
            return None;
        };
        if result.shape() != [1, 1] {
            return None;
        }
        let exact = match class {
            NumericClass::Single => result.as_f32_slice().is_some_and(|values| {
                values.len() == 1 && values[0].to_bits() == (source as f32).to_bits()
            }),
            NumericClass::Double => result
                .as_f64_slice()
                .is_some_and(|values| values.len() == 1 && values[0].to_bits() == source.to_bits()),
            _ => false,
        };
        exact.then_some(represented)
    }
}
