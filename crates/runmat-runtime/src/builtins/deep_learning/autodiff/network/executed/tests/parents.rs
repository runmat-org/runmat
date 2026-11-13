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
    let Some(Value::Cell(column)) =
        crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Value")
    else {
        panic!("Value column")
    };
    crate::builtins::deep_learning::tests::fixture_array_mut(column)[index] = replacement;
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

fn squared_joint_loss(outputs: impl IntoIterator<Item = Value>) -> Value {
    outputs
        .into_iter()
        .map(|output| {
            dlarray_sum_builtin(
                dlarray_times_builtin(output.clone(), output).unwrap(),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                    "all".into(),
                )]),
            )
            .unwrap()
        })
        .reduce(|left, right| dlarray_plus_builtin(left, right).unwrap())
        .expect("at least one network output")
}

fn record_for(output: &Value, tape: &Rc<RefCell<Tape>>) -> Rc<NetworkTapeRecord> {
    let node = dlarray_node_id(output).expect("traced network output");
    let graph = tape.borrow();
    let NodeKind::Network { execution, .. } = &graph.nodes[node].kind else {
        panic!("network tape record expected")
    };
    execution.clone()
}

#[test]
fn two_recurrent_traces_share_one_upstream_parent_without_losing_either_adjoint() {
    let data = vec![-0.7, 0.3, 0.5, -0.4, 1.2, 0.8];
    let first = network(K::Lstm, NumericDType::F64);
    let second = network(K::Bilstm, NumericDType::F64);
    let analytic = {
        let tape = Rc::new(RefCell::new(Tape::new()));
        let _activation = TapeActivation::enter(tape.clone());
        let x = annotate_dlarray_value(input(data.clone(), NumericDType::F64)).unwrap();
        let first_outputs = prediction_and_states(&first, x.clone());
        let second_outputs = prediction_and_states(&second, x.clone());
        let first_record = record_for(&first_outputs[0], &tape);
        let second_record = record_for(&second_outputs[0], &tape);
        assert!(!Rc::ptr_eq(&first_record, &second_record));
        let loss = squared_joint_loss(first_outputs.into_iter().chain(second_outputs));
        let gradient = block_on(dlgradient_direct(
            loss.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![x.clone()]),
        ))
        .unwrap();
        assert_eq!(first_record.reverse_calls.get(), 1);
        assert_eq!(second_record.reverse_calls.get(), 1);
        let repeated = block_on(dlgradient_direct(
            loss,
            crate::builtins::common::test_support::builtin_arguments(vec![x]),
        ))
        .unwrap();
        assert_eq!(first_record.reverse_calls.get(), 2);
        assert_eq!(second_record.reverse_calls.get(), 2);
        assert_eq!(tensor(&gradient), tensor(&repeated));
        tensor(&gradient).materialize_f64()
    };
    let objective = |values: Vec<f64>| {
        [first.clone(), second.clone()]
            .into_iter()
            .flat_map(|net| prediction_and_states(&net, input(values.clone(), NumericDType::F64)))
            .flat_map(|output| tensor(&output).materialize_f64())
            .map(|value| value * value)
            .sum::<f64>()
    };
    for index in 0..data.len() {
        let mut upper = data.clone();
        upper[index] += 0.00001;
        let mut lower = data.clone();
        lower[index] -= 0.00001;
        let expected = (objective(upper) - objective(lower)) / 0.00002;
        assert!(
            (analytic[index] - expected).abs() < 0.00002,
            "shared input {index}: {} vs {expected}",
            analytic[index]
        );
    }
}

#[test]
fn aliased_input_and_initial_hidden_state_accumulate_to_one_parent() {
    let base = network(K::Lstm, NumericDType::F64);
    let sequence = |values: Vec<f64>| {
        object(
            "dlarray",
            vec![
                ("Data", super::values(values, vec![2, 1], NumericDType::F64)),
                ("Format", Value::String("CT".into())),
                ("Labels", Value::String("CT".into())),
            ],
        )
        .expect("fixture object")
    };
    let data = vec![0.3, -0.4];
    let analytic = {
        let tape = Rc::new(RefCell::new(Tape::new()));
        let _activation = TapeActivation::enter(tape.clone());
        let x = annotate_dlarray_value(sequence(data.clone())).unwrap();
        let net = replace_parent(&base, ParentRole::State(StateRole::HiddenState), x.clone());
        let outputs = prediction_and_states(&net, x.clone());
        let record = record_for(&outputs[0], &tape);
        assert_eq!(
            record.input.as_ref().unwrap().node,
            record
                .states
                .get(&(1, StateRole::HiddenState))
                .unwrap()
                .node,
            "input and HiddenState must retain the same tape parent"
        );
        let loss = squared_joint_loss(outputs);
        let gradient = block_on(dlgradient_direct(
            loss,
            crate::builtins::common::test_support::builtin_arguments(vec![x]),
        ))
        .unwrap();
        assert_eq!(record.reverse_calls.get(), 1);
        tensor(&gradient).materialize_f64()
    };
    let objective = |values: Vec<f64>| {
        let x = sequence(values);
        let net = replace_parent(&base, ParentRole::State(StateRole::HiddenState), x.clone());
        prediction_and_states(&net, x)
            .iter()
            .flat_map(|output| tensor(output).materialize_f64())
            .map(|value| value * value)
            .sum::<f64>()
    };
    for index in 0..data.len() {
        let mut upper = data.clone();
        upper[index] += 0.00001;
        let mut lower = data.clone();
        lower[index] -= 0.00001;
        let expected = (objective(upper) - objective(lower)) / 0.00002;
        assert!(
            (analytic[index] - expected).abs() < 0.00002,
            "aliased input/State {index}: {} vs {expected}",
            analytic[index]
        );
    }
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
                crate::builtins::deep_learning::autodiff::tape::annotate_tree(net.value().clone())
                    .unwrap();
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
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                        "all".into(),
                    )]),
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
                    let gradient = block_on(dlgradient_direct(
                        loss.clone(),
                        crate::builtins::common::test_support::builtin_arguments(vec![
                            target.clone()
                        ]),
                    ))
                    .unwrap();
                    assert_eq!(tensor(&gradient).numeric_dtype(), NumericDType::F64);
                    assert_eq!(tensor(&gradient).shape(), tensor(&target).shape());
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
                        Value::Tensor(Tensor::new(values, original.shape().to_vec()).unwrap()),
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
