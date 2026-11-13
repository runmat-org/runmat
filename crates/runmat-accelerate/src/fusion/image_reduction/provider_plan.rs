//! Direct provider-hook projection of canonical image mean and power sources.
//! Logical calls are not dispatch, fallback, peak-memory, or numerical proof.

use super::{ImageMeanAxis, ImageMeanForm, ImageMeanOutput, ImageMeanRoute, ImageReductionRoutes};
use crate::graph::{AccelGraph, AccelNodeLabel, NodeId, PrimitiveOp, ValueId};
use runmat_accelerate_api::{
    scalar_program::ScalarLiteral, ImageNormalizeDescriptor, PowerScalarControl,
    ProviderImageMeanCall, ProviderImageMeanClass, ProviderImageMeanHook,
    ProviderImagePowerProgram, ProviderImageStageWorkload,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageMeanProviderHook {
    ReduceNd([usize; 2]),
    ReduceDim(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageMeanProviderPlan {
    pub calls: Vec<ImageMeanProviderCall>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageMeanProviderCall {
    pub hook: ImageMeanProviderHook,
    pub output_class: ImageMeanOutput,
    pub shape: [usize; 3],
}

/// The two actual `.^` source nodes. A matcher simplification such as
/// `gamma == 1` cannot remove the final power operation from this transcript.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageSourcePower {
    pub node: NodeId,
    pub base: ValueId,
    pub exponent_value: ValueId,
    pub represented_exponent: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageMeanAndPowerSource {
    pub mu: ImageMeanProviderPlan,
    pub variance: ImageMeanProviderPlan,
    pub square: ImageSourcePower,
    pub final_power: ImageSourcePower,
}

impl ImageMeanAndPowerSource {
    /// Project the fixed hook transcript once. The same typed value is used
    /// by the held session and by future provider-owned physical admission;
    /// it does not replace selected source/binding or numerical proof.
    pub fn staged_workload(
        &self,
        desc: &ImageNormalizeDescriptor,
    ) -> Option<ProviderImageStageWorkload> {
        self.staged_workload_with_power(desc, None, None)
    }

    pub fn staged_workload_selected(
        &self,
        desc: &ImageNormalizeDescriptor,
        powers: ProviderImagePowerProgram,
        epsilon: ScalarLiteral,
    ) -> Option<ProviderImageStageWorkload> {
        self.staged_workload_with_power(desc, Some(powers), Some(epsilon))
    }

    fn staged_workload_with_power(
        &self,
        desc: &ImageNormalizeDescriptor,
        selected_powers: Option<ProviderImagePowerProgram>,
        selected_epsilon: Option<ScalarLiteral>,
    ) -> Option<ProviderImageStageWorkload> {
        if selected_powers.is_some() != selected_epsilon.is_some() {
            return None;
        }
        fn shape(shape: [usize; 3]) -> Option<[u64; 3]> {
            Some([
                u64::try_from(shape[0]).ok()?,
                u64::try_from(shape[1]).ok()?,
                u64::try_from(shape[2]).ok()?,
            ])
        }
        fn mean(calls: &[ImageMeanProviderCall]) -> Option<Vec<ProviderImageMeanCall>> {
            calls
                .iter()
                .map(|call| {
                    Some(ProviderImageMeanCall {
                        hook: match call.hook {
                            ImageMeanProviderHook::ReduceNd(axes) => {
                                ProviderImageMeanHook::ReduceNd([
                                    u8::try_from(axes[0]).ok()?,
                                    u8::try_from(axes[1]).ok()?,
                                ])
                            }
                            ImageMeanProviderHook::ReduceDim(axis) => {
                                ProviderImageMeanHook::ReduceDim(u8::try_from(axis).ok()?)
                            }
                        },
                        output_class: match call.output_class {
                            ImageMeanOutput::Default => ProviderImageMeanClass::Default,
                            ImageMeanOutput::Native => ProviderImageMeanClass::Native,
                        },
                        shape: shape(call.shape)?,
                    })
                })
                .collect()
        }
        let exact_f32 = |value: f64| {
            (value.is_finite() && f64::from(value as f32) == value)
                .then_some((value as f32).to_bits())
        };
        if let Some(control) = selected_epsilon {
            let exact = match control {
                ScalarLiteral::F32Bits(bits) => f64::from(f32::from_bits(bits)),
                ScalarLiteral::F64Bits(bits) => f64::from_bits(bits),
            };
            if exact.to_bits() != desc.epsilon.to_bits() {
                return None;
            }
        }
        let program = ProviderImageStageWorkload {
            input_shape: shape([desc.batch, desc.height, desc.width])?,
            mu: mean(&self.mu.calls)?,
            variance: mean(&self.variance.calls)?,
            // Zero is an intentionally unused legacy slot for a typed
            // selected program; no source class is inferred from it.
            square_exponent_f32_bits: if selected_powers.is_some() {
                0
            } else {
                exact_f32(self.square.represented_exponent)?
            },
            final_exponent_f32_bits: if selected_powers.is_some() {
                0
            } else {
                exact_f32(self.final_power.represented_exponent)?
            },
            selected_powers,
            epsilon: match selected_epsilon {
                Some(control) => control,
                None => ScalarLiteral::F32Bits(exact_f32(desc.epsilon)?),
            },
            output_program: None,
            moment_program: None,
        };
        (program.validate()
            && desc.gamma.unwrap_or(1.0).to_bits()
                == self.final_power.represented_exponent.to_bits())
        .then_some(program)
    }

    /// Source-exact reduction/power operations only; centered subtraction,
    /// epsilon, sqrt, final division, casts, domain, and provider effects
    /// require their own staged admission before this can execute.
    pub fn from_graph(
        graph: &AccelGraph,
        routes: &ImageReductionRoutes,
        input_shape: [usize; 3],
    ) -> Option<Self> {
        Self::from_selected_graph(graph, routes, input_shape, None)
    }

    /// A live final-power control can be represented by the selected source
    /// only after its exact request value has been bound. The value ID must
    /// still be the final producer's operand; this is not permission to use a
    /// compiler constant fact in place of a published binding.
    pub(crate) fn from_selected_graph(
        graph: &AccelGraph,
        routes: &ImageReductionRoutes,
        input_shape: [usize; 3],
        bound_final_exponent: Option<(ValueId, PowerScalarControl)>,
    ) -> Option<Self> {
        if !routes.matches_source_graph(graph) {
            return None;
        }
        let producers = routes.stage_producers()?;
        Some(Self {
            mu: routes.mu.direct_provider_plan(input_shape)?,
            variance: routes.variance.direct_provider_plan(input_shape)?,
            square: source_power(graph, routes.variance.input(), None)?,
            final_power: source_power(graph, producers.output, bound_final_exponent)?,
        })
    }
}

fn source_power(
    graph: &AccelGraph,
    output: ValueId,
    bound_exponent: Option<(ValueId, PowerScalarControl)>,
) -> Option<ImageSourcePower> {
    let (node, producer) = super::node_from_value(graph, output)?;
    if !matches!(
        producer.label,
        AccelNodeLabel::Primitive(PrimitiveOp::ElemPow)
    ) || producer.inputs.len() != 2
        || producer.outputs.as_slice() != [output]
    {
        return None;
    }
    let exponent_value = producer.inputs[1];
    let represented_exponent = match bound_exponent {
        Some((id, actual)) => (id == exponent_value).then_some(actual.value())?,
        None => super::super::collect_scalar_constant(graph, exponent_value)?.value,
    };
    Some(ImageSourcePower {
        node,
        base: producer.inputs[0],
        exponent_value,
        represented_exponent,
    })
}

impl ImageMeanRoute {
    pub fn input(&self) -> ValueId {
        match &self.form {
            ImageMeanForm::VecDims { call, .. } => call.input,
            ImageMeanForm::Nested { inner, .. } => inner.input,
        }
    }

    /// The first direct hooks ordinary resident mean attempts for this exact
    /// form. `MeanAxes::Vec` sorts zero-based dimensions before `reduce_mean_nd`
    /// while separate nested `MeanAxes::Dim` calls preserve their order and
    /// per-call output-class boundaries.
    pub fn direct_provider_plan(&self, input_shape: [usize; 3]) -> Option<ImageMeanProviderPlan> {
        let [batch, _, _] = input_shape;
        match &self.form {
            ImageMeanForm::VecDims { call, axes } => {
                if !matches!(
                    *axes,
                    [ImageMeanAxis::Height, ImageMeanAxis::Width]
                        | [ImageMeanAxis::Width, ImageMeanAxis::Height]
                ) {
                    return None;
                }
                Some(ImageMeanProviderPlan {
                    calls: vec![ImageMeanProviderCall {
                        hook: ImageMeanProviderHook::ReduceNd([1, 2]),
                        output_class: call.output_class,
                        shape: [batch, 1, 1],
                    }],
                })
            }
            ImageMeanForm::Nested {
                inner,
                inner_axis,
                outer,
                outer_axis,
            } => {
                if inner_axis == outer_axis {
                    return None;
                }
                let mut after_inner = input_shape;
                after_inner[inner_axis.zero_based()] = 1;
                Some(ImageMeanProviderPlan {
                    calls: vec![
                        ImageMeanProviderCall {
                            hook: ImageMeanProviderHook::ReduceDim(inner_axis.zero_based()),
                            output_class: inner.output_class,
                            shape: after_inner,
                        },
                        ImageMeanProviderCall {
                            hook: ImageMeanProviderHook::ReduceDim(outer_axis.zero_based()),
                            output_class: outer.output_class,
                            shape: [batch, 1, 1],
                        },
                    ],
                })
            }
        }
    }
}
