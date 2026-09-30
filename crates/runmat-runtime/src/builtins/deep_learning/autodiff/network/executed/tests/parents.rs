//! Public derivative boundaries for prepared precision and canonical parents.
use super::*;

#[derive(Clone, Copy, Debug)]
enum ParentRole {
    Learnable(ParameterRole),
    State(StateRole),
}

fn replace_parent(net: &Value, role: ParentRole, replacement: Value) -> Value {
    let table = match role {
        ParentRole::Learnable(_) => block_on(model::get_learnables_builtin(net.clone())),
        ParentRole::State(_) => block_on(model::get_state_builtin(net.clone())),
    };
    let table = table.unwrap();
    let Value::Object(object) = &table else {
        panic!("canonical table")
    };
    let mut variables = crate::builtins::table::table_variables(object).unwrap();
    let Value::Object(network) = net else {
        panic!("network")
    };
    let index = match role {
        ParentRole::Learnable(role) => model::parameters::view(network, "test")
            .unwrap()
            .iter()
            .position(|row| row.key.role == role)
            .unwrap(),
        ParentRole::State(role) => model::state::view(network, "test")
            .unwrap()
            .iter()
            .position(|row| row.key.role == role)
            .unwrap(),
    };
    let Some(Value::Cell(column)) = variables.fields.get_mut("Value") else {
        panic!("Value column")
    };
    column.data[index] = replacement;
    let updated = crate::builtins::table::table_replace_variables_like(object, variables).unwrap();
    match role {
        ParentRole::Learnable(_) => block_on(model::set_learnables_builtin(net.clone(), updated)),
        ParentRole::State(_) => block_on(model::set_state_builtin(net.clone(), updated)),
    }
    .unwrap()
}

fn objective(net: &Value, data: &[f64]) -> f64 {
    prediction_and_states(net, input(data.to_vec(), NumericDType::F32))
        .iter()
        .flat_map(|value| tensor(value).materialize_f64())
        .map(|value| value * value)
        .sum()
}

#[test]
fn prepared_single_reverse_restores_each_double_learnable_and_initial_state_parent() {
    let data = vec![-0.7, 0.3, 0.5, -0.4, 1.2, 0.8];
    for kind in [K::Lstm, K::Bilstm] {
        let net = network(kind, NumericDType::F64);
        let parents = {
            let tape = Rc::new(RefCell::new(Tape::new()));
            let _activation = TapeActivation::enter(tape);
            let traced =
                crate::builtins::deep_learning::autodiff::tape::annotate_tree(net.clone()).unwrap();
            let Value::Object(network) = &traced else {
                panic!("network")
            };
            let mut targets: Vec<_> = model::parameters::view(network, "test")
                .unwrap()
                .into_iter()
                .map(|row| {
                    (
                        ParentRole::Learnable(row.key.role),
                        row.value.unwrap().clone(),
                    )
                })
                .collect();
            targets.extend(
                model::state::view(network, "test")
                    .unwrap()
                    .into_iter()
                    .map(|row| (ParentRole::State(row.key.role), row.value.clone())),
            );
            let outputs = prediction_and_states(&traced, input(data.clone(), NumericDType::F32));
            assert_eq!(tensor(&outputs[0]).numeric_dtype(), NumericDType::F32);
            let mut losses = outputs.into_iter().map(|output| {
                dlarray_sum_builtin(
                    dlarray_times_builtin(output.clone(), output).unwrap(),
                    vec![Value::String("all".into())],
                )
                .unwrap()
            });
            let loss = dlarray_plus_builtin(
                dlarray_plus_builtin(losses.next().unwrap(), losses.next().unwrap()).unwrap(),
                losses.next().unwrap(),
            )
            .unwrap();
            targets
                .into_iter()
                .map(|(role, target)| {
                    let gradient =
                        block_on(dlgradient_builtin(loss.clone(), vec![target.clone()])).unwrap();
                    assert_eq!(tensor(&gradient).numeric_dtype(), NumericDType::F64);
                    assert_eq!(tensor(&gradient).shape, tensor(&target).shape);
                    (role, tensor(&target).clone(), tensor(&gradient).clone())
                })
                .collect::<Vec<_>>()
        };
        for (role, original, gradient) in parents {
            let original_values = original.as_f64_slice().unwrap();
            let step = 0.001;
            for index in 0..original.len() {
                let evaluate = |delta| {
                    let mut values = original_values.to_vec();
                    values[index] += delta;
                    let changed = replace_parent(
                        &net,
                        role,
                        Value::Tensor(Tensor::new(values, original.shape.clone()).unwrap()),
                    );
                    objective(&changed, &data)
                };
                let expected = (evaluate(step) - evaluate(-step)) / (2.0 * step);
                let actual = gradient.as_f64_slice().unwrap()[index];
                assert!(
                    (actual - expected).abs() < 0.003,
                    "{kind:?} {role:?} element {index}: {actual} vs {expected}"
                );
            }
        }
    }
}
