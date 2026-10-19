//! The selected reduction source and its physical shader requirements travel
//! together from the one WGSL producer through provider admission.

use runmat_accelerate_api::{ProviderPrecision, ProviderShaderRequirements};

pub(crate) struct GeneratedReductionShader {
    source: String,
    requirements: ProviderShaderRequirements,
}

impl GeneratedReductionShader {
    pub(super) fn new(source: String, precision: ProviderPrecision) -> Self {
        let requirements = match precision {
            ProviderPrecision::F32 => ProviderShaderRequirements::Basic,
            // This producer emits IEEE classification and canonical NaN helpers
            // for every F64 reduction, independently of the selected operation.
            ProviderPrecision::F64 => ProviderShaderRequirements::F64IeeeWords,
        };
        Self {
            source,
            requirements,
        }
    }

    pub(crate) fn source(&self) -> &str {
        &self.source
    }

    pub(crate) fn requirements(&self) -> ProviderShaderRequirements {
        self.requirements
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn legacy_reduction_templates_validate_both_precisions_axes_and_flavors() {
        use crate::fusion::{FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp};
        use crate::graph::{InstrSpan, PrimitiveOp, ShapeInfo};
        use runmat_accelerate_api::ReductionFlavor;
        use std::collections::HashMap;
        for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
            for axis in [0, 1] {
                for flavor in [ReductionFlavor::Sum, ReductionFlavor::Mean] {
                    let plan = FusionGroupPlan {
                        index: 0,
                        group: FusionGroup {
                            id: 0,
                            kind: FusionKind::Reduction,
                            nodes: vec![],
                            shape: ShapeInfo::Tensor(vec![Some(2), Some(3)]),
                            span: InstrSpan { start: 0, end: 2 },
                            pattern: None,
                            stack_layout: None,
                        },
                        operations: vec![
                            FusionOp::Primitive {
                                op: PrimitiveOp::ElemMul,
                                inputs: vec![0, 0],
                                output: Some(1),
                            },
                            FusionOp::Builtin {
                                name: if matches!(flavor, ReductionFlavor::Mean) {
                                    "mean"
                                } else {
                                    "sum"
                                }
                                .into(),
                                origin: Default::default(),
                                inputs: vec![1, 2],
                                output: Some(3),
                            },
                        ],
                        inputs: vec![0],
                        stack_pattern: vec![0],
                        constants: HashMap::new(),
                        const_values: HashMap::from([(
                            2,
                            runmat_value::Value::Num(f64::from(axis + 1)),
                        )]),
                        materialized_stores: vec![],
                        output: Some(3),
                        kernel: FusionKernelSpec {
                            kind: FusionKind::Reduction,
                            supported: true,
                        },
                        reduction_data: Some(1),
                        reduction_dim: Some(2),
                        reduction_flavor: Some(flavor),
                        reduction_axes: None,
                        pattern: None,
                    };
                    let generated = plan
                        .prepare_reduction_shader(precision, flavor, None)
                        .unwrap();
                    let scalar = if precision == ProviderPrecision::F64 {
                        "f64"
                    } else {
                        "f32"
                    };
                    let zero = if precision == ProviderPrecision::F64 {
                        "f64(0.0)"
                    } else {
                        "0.0"
                    };
                    assert!(generated
                        .source()
                        .contains(&format!("var acc: {scalar} = {zero};")));
                    assert_eq!(
                        generated.source().contains("1.0 /"),
                        matches!(flavor, ReductionFlavor::Mean)
                    );
                    assert!(generated.source().contains(if axis == 0 {
                        "let col = wid.x;"
                    } else {
                        "let row = wid.x;"
                    }));
                    #[cfg(feature = "wgpu")]
                    crate::wgsl::validation_tests::validate_generated(
                        "legacy-reduction-complete-template",
                        generated.source(),
                        64,
                        8,
                        if precision == ProviderPrecision::F64 {
                            crate::wgsl::validation_tests::RequiredCaps::Float64Int64
                        } else {
                            crate::wgsl::validation_tests::RequiredCaps::None
                        },
                    );
                }
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn selected_reduction_source_retains_its_typed_physical_requirement() {
        let single = GeneratedReductionShader::new("single".into(), ProviderPrecision::F32);
        let double = GeneratedReductionShader::new("double".into(), ProviderPrecision::F64);
        assert_eq!(single.source(), "single");
        assert_eq!(single.requirements(), ProviderShaderRequirements::Basic);
        assert_eq!(double.source(), "double");
        assert_eq!(
            double.requirements(),
            ProviderShaderRequirements::F64IeeeWords
        );
    }
}
