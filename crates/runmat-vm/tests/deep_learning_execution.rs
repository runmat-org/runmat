#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_nested_gradient_targets_preserve_native_leaf_values_and_requested_prefix() {
    execute_source(
        r#"
        x = dlarray(single([1 2 3]), 'CB');
        [loss, gradients] = dlfeval(@nestedGradients, x);
        assert(loss.Data == single(14));
        assert(iscell(gradients) && isequal(size(gradients), [1 2]));
        inner = gradients{1};
        first = inner{1};
        second = gradients{2};
        assert(strcmp(class(first.Data), 'single'));
        assert(strcmp(class(second.Data), 'single'));
        assert(strcmp(first.Format, 'CB'));
        assert(isequal(first.Data, single([2 4 6])));
        assert(isequal(second.Data, single([2 4 6])));
        function [loss, gradients] = nestedGradients(x)
            loss = sum(x .* x, 'all');
            targets = {{x}, x};
            gradients = dlgradient(loss, targets, uint8(9));
        end
        "#,
    )
    .expect(
        "compiled nested target traversal preserves class/format and ignores unrequested targets",
    );
}

#[test]
fn compiled_temporal_pooling_preserves_axes_precision_and_state_output() {
    execute_source(
        r#"
        net = dlnetwork({sequenceInputLayer(2); globalAveragePooling1dLayer(); regressionLayer()});
        data = single(reshape(1:24, 2, 3, 4));
        x = dlarray(data, 'CBT');
        [y, state] = forward(net, x);
        assert(strcmp(class(y.Data), 'single'));
        assert(strcmp(y.Format, 'CB'));
        assert(isequal(y.Data, single([10 12 14; 11 13 15])));
        assert(strcmp(class(state), 'table'));
        assert(isempty(state.Value));
        [raw, nextState] = predict(net, data);
        assert(strcmp(class(raw), 'single'));
        assert(isequal(raw, single([10 11; 12 13; 14 15])));
        assert(strcmp(class(nextState), 'table'));
        assert(isempty(nextState.Value));
        "#,
    )
    .expect("compiled public forward/predict use their distinct semantic axis presentations");
}

#[test]
fn compiled_prediction_reverse_inverts_rank_changing_output_presentation() {
    execute_source(
        r#"
        net = dlnetwork({sequenceInputLayer(2); globalAveragePooling1dLayer()});
        x = dlarray(single(reshape(1:24, 2, 3, 4)));
        [loss, gradient] = dlfeval(@weightedPrediction, net, x);
        assert(strcmp(class(loss.Data), 'single'), 'weighted loss must retain single');
        assert(strcmp(class(gradient.Data), 'single'), 'prediction gradient must retain single');
        assert(loss.Data == single(275), 'weighted prediction loss must equal 275');
        expected = repmat(single([1 2 3; 4 5 6]) / single(4), 1, 1, 4);
        assert(isequal(size(gradient.Data), [2 3 4]), 'prediction gradient must restore CBT geometry');
        assert(isequal(gradient.Data, expected), 'prediction gradient must invert the presentation permutation');
        function [loss, gradient] = weightedPrediction(net, x)
            prediction = predict(net, x);
            assert(strcmp(prediction.Format, ''), 'unformatted prediction must remain unformatted');
            weights = dlarray(single([1 4; 2 5; 3 6]));
            loss = sum(prediction .* weights, 'all');
            gradient = dlgradient(loss, x);
        end
        "#,
    )
    .expect("compiled public callback retains the source CBT adjoint behind a BC prediction");
}
