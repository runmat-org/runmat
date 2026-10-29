//! Exact mathematical equivalence for the compatibility matrix reduction ABI.
//! A selected name/flavor alone cannot authorize provider-precision arithmetic.
use super::*;
use runmat_types::{
    FusionReductionContract, FusionReductionCount, FusionReductionEmpty, FusionReductionTies,
    NumericClass, ReductionNaN,
};

pub(super) struct LegacyEquivalent<'a> {
    _program: &'a super::super::super::ResolvedReductionProgram<'a>,
    flavor: ReductionFlavor,
}
impl LegacyEquivalent<'_> {
    pub(super) fn flavor(&self) -> ReductionFlavor {
        self.flavor
    }
}

pub(super) fn admit<'a>(
    request: &FusionExecutionRequest<'_>,
    program: &'a super::super::super::ResolvedReductionProgram<'a>,
    flavor: ReductionFlavor,
    precision: ProviderPrecision,
    reduce_len: usize,
    slices: usize,
    output_shape: &[usize],
) -> Option<LegacyEquivalent<'a>> {
    let native = native_class(precision);
    if program.execution_precision() != precision
        || program.data_classes().iter().any(|class| *class != native)
        || !equivalent(
            program.contract(),
            flavor,
            precision,
            (reduce_len, slices),
            output_shape,
            request.plan.legacy_reduction_axis()?,
        )
    {
        return None;
    }
    program.require_current_controls(request).ok()?;
    Some(LegacyEquivalent {
        _program: program,
        flavor,
    })
}

fn native_class(precision: ProviderPrecision) -> NumericClass {
    match precision {
        ProviderPrecision::F32 => NumericClass::Single,
        ProviderPrecision::F64 => NumericClass::Double,
    }
}

fn equivalent(
    contract: &FusionReductionContract,
    flavor: ReductionFlavor,
    precision: ProviderPrecision,
    geometry: (usize, usize),
    output_shape: &[usize],
    axis: usize,
) -> bool {
    let native = native_class(precision);
    let policy_matches = match (contract.operation, flavor) {
        (FusionReductionOperation::Sum, ReductionFlavor::Sum) => {
            contract.count == FusionReductionCount::None
                && contract.empty == FusionReductionEmpty::PositiveZero
        }
        (FusionReductionOperation::Mean, ReductionFlavor::Mean) => {
            contract.count == FusionReductionCount::SliceLength
                && contract.empty == FusionReductionEmpty::NaN
                // The old shader scales by reciprocal and represents its
                // count through f32. Only an exact power-of-two count proves
                // this is the canonical division, without count narrowing.
                && geometry.0.is_power_of_two()
                && u32::try_from(geometry.0).is_ok()
        }
        _ => false,
    };
    if !policy_matches
        || contract.identity
        || contract.nan_mode != ReductionNaN::Include
        || contract.ties != FusionReductionTies::NotApplicable
        || contract.input_class != native
        || contract.accumulator_class != native
        || contract.output_class != native
        || contract.output_shape != output_shape
        || contract.input_shape.is_empty()
        || contract.input_shape.len() > 2
        || geometry.0 == 0
        || geometry.1 == 0
        || contract.axes.as_slice() != [axis]
    {
        return false;
    }
    let rows = contract.input_shape[0];
    let columns = *contract.input_shape.get(1).unwrap_or(&1);
    match axis {
        0 => (rows, columns) == geometry,
        1 => (columns, rows) == geometry,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legacy_equivalence_uses_public_owner_contract_not_flavor() {
        for (name, flavor) in [
            ("sum", ReductionFlavor::Sum),
            ("mean", ReductionFlavor::Mean),
        ] {
            let entry = runmat_builtins::builtin_catalog_entry_by_name(name).unwrap();
            let [binding] = entry.bindings else {
                panic!("canonical fixture binding");
            };
            let declaration = runmat_runtime::builtins::common::spec::canonical_fusion_declaration(
                entry.binding_identity(binding),
            )
            .unwrap()
            .reduction
            .unwrap();
            let controls = [Value::Num(1.0)];
            let prepared = futures::executor::block_on((declaration.prepare)(
                runmat_runtime::builtins::common::spec::FusionReductionControls {
                    values: &controls,
                    requested_outputs: 1,
                },
            ))
            .unwrap()
            .unwrap();
            let fact = runmat_runtime::value_fact::value_fact(&Value::Tensor(
                runmat_value::Tensor::new(vec![1.25, 2.5], vec![2, 1]).unwrap(),
            ));
            let contract = prepared.resolve(&fact).unwrap().unwrap();
            assert!(equivalent(
                &contract,
                flavor,
                ProviderPrecision::F64,
                (2, 1),
                &[1, 1],
                0
            ));
            assert!(!equivalent(
                &contract,
                flavor,
                ProviderPrecision::F32,
                (2, 1),
                &[1, 1],
                0
            ));
            if name == "mean" {
                let mut nonpower = contract.clone();
                nonpower.input_shape[0] = 3;
                assert!(!equivalent(
                    &nonpower,
                    flavor,
                    ProviderPrecision::F64,
                    (3, 1),
                    &[1, 1],
                    0,
                ));
            }
            for change in 0..6 {
                let mut changed = contract.clone();
                match change {
                    0 => changed.accumulator_class = NumericClass::Single,
                    1 => changed.output_class = NumericClass::Single,
                    2 => changed.axes = vec![1],
                    3 => changed.nan_mode = ReductionNaN::Omit,
                    4 => changed.identity = true,
                    _ => changed.output_shape = vec![1],
                }
                assert!(!equivalent(
                    &changed,
                    flavor,
                    ProviderPrecision::F64,
                    (2, 1),
                    &[1, 1],
                    0
                ));
            }
        }
    }
}
