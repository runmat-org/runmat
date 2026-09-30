#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_network_learnables_writeback_updates_execution_without_a_second_authority() {
    execute_source(
        r#"
        input = featureInputLayer(2, 'Name', 'input');
        dense = fullyConnectedLayer(1, 'Name', 'dense');
        net = dlnetwork({input, dense});
        original = net;
        originalPrediction = predict(original, [7 11]);
        learnables = net.Learnables;
        assert(strcmp(class(learnables.Layer), 'string'));
        assert(strcmp(class(learnables.Parameter), 'string'));
        learnables.Value{1} = dlarray([2 3]);
        learnables.Value{2} = dlarray(5);
        net.Learnables = learnables;
        actual = predict(net, [7 11]);
        assert(isequal(actual, 52));
        assert(isequal(predict(original, [7 11]), originalPrediction));
        net.Learnables.Value{2} = dlarray(9);
        assert(isequal(predict(net, [7 11]), 56));
        assert(isequal(predict(original, [7 11]), originalPrediction));
        "#,
    )
    .expect("compiled dependent table property writeback updates only the owned network");
}

#[test]
fn compiled_dlfeval_traces_public_named_callback_and_preserves_single_gradient() {
    execute_source(
        r#"
        X = dlarray(single([1 2 3]));
        [loss, gradients] = dlfeval(@squareLoss, X);
        assert(strcmp(class(loss.Data), 'single'));
        assert(strcmp(class(gradients.Data), 'single'));
        assert(isequal(loss.Data, single(14)));
        assert(isequal(gradients.Data, single([2 4 6])));
        function [loss, gradients] = squareLoss(X)
            squared = X .* X;
            loss = sum(squared, 'all');
            gradients = dlgradient(loss, X);
        end
        "#,
    )
    .expect("compiled public dlfeval callback and repeated Single gradient");
}

#[test]
fn compiled_dlfeval_forwards_callback_requested_outputs() {
    execute_source(
        r#"
        [left, right] = dlfeval(@deal, uint16(7), single(2));
        assert(strcmp(class(left), 'uint16') && left == uint16(7));
        assert(strcmp(class(right), 'single') && right == single(2));
        "#,
    )
    .expect("compiled callback forwards output count and typed values");
}

#[test]
fn compiled_nested_dlfeval_restores_outer_tape() {
    execute_source(
        r#"
        X = dlarray(single([1 2]));
        [loss, outerGradient, innerGradient] = dlfeval(@outerLoss, X);
        assert(isequal(loss.Data, single(5)));
        assert(isequal(outerGradient.Data, single([2 4])));
        assert(isequal(innerGradient.Data, single(6)));
        function [loss, outerGradient, innerGradient] = outerLoss(X)
            Y = dlarray(single(3));
            [innerValue, innerGradient] = dlfeval(@innerLoss, Y);
            assert(isequal(innerValue.Data, single(9)));
            loss = sum(X .* X, 'all');
            outerGradient = dlgradient(loss, X);
        end
        function [loss, gradient] = innerLoss(Y)
            loss = sum(Y .* Y, 'all');
            gradient = dlgradient(loss, Y);
        end
        "#,
    )
    .expect("nested compiled callbacks restore separate autodiff tapes");
}

#[test]
fn compiled_layer_constructors_preserve_sizes_options_and_defaults() {
    execute_source(
        r#"
        fc = fullyConnectedLayer(uint16(7), 'Name', 'dense', 'Description', 'projection');
        assert(fc.OutputSize == 7 && strcmp(fc.Name, 'dense'));
        assert(strcmp(fc.Description, 'projection'));
        feature = featureInputLayer(4, 'Name', 'features');
        assert(feature.InputSize == 4 && strcmp(feature.Name, 'features'));
        sequence = sequenceInputLayer([4 2], 'Name', 'sequence');
        assert(isequal(sequence.InputSize, [4 2]));
        relu = reluLayer('Name', 'relu');
        elu = eluLayer(0.2, 'Name', 'elu');
        softmax = softmaxLayer('Name', 'probability');
        classification = classificationLayer('Name', 'labels');
        regression = regressionLayer('Name', 'response');
        assert(strcmp(relu.Name, 'relu') && strcmp(relu.Type, 'ReLU'));
        assert(elu.Alpha == 0.2 && strcmp(elu.Name, 'elu'));
        assert(strcmp(softmax.Name, 'probability'));
        assert(strcmp(classification.Classes, 'auto'));
        assert(strcmp(regression.Name, 'response'));

        norm = layerNormalizationLayer('Name', 'norm');
        pool = globalAveragePooling1dLayer('Name', 'pool');
        lstm = lstmLayer(uint8(8), 'Name', 'memory');
        bilstm = bilstmLayer(6, 'OutputMode', 'last', 'Name', 'bidirectional');
        conv = convolution1dLayer(uint8(3), uint16(16), 'Padding', 'causal');
        assert(norm.Epsilon == 1e-5 && strcmp(norm.Name, 'norm'));
        assert(strcmp(pool.Name, 'pool'));
        assert(lstm.NumHiddenUnits == 8 && strcmp(lstm.OutputMode, 'sequence'));
        assert(bilstm.NumHiddenUnits == 6 && strcmp(bilstm.OutputMode, 'last'));
        assert(conv.FilterSize == 3 && conv.NumFilters == 16);
        assert(conv.Stride == 1 && strcmp(conv.Padding, 'causal'));
        "#,
    )
    .expect("compiled layer constructors preserve structural metadata and options");
}

#[test]
fn compiled_layer_graph_and_analysis_are_metadata_not_training() {
    execute_source(
        r#"
        input = featureInputLayer(3, 'Name', 'input');
        hidden = fullyConnectedLayer(2, 'Name', 'hidden');
        output = regressionLayer('Name', 'output');
        graph = layerGraph({input, hidden, output});
        assert(numel(graph.Layers) == 3);
        assert(numel(graph.LayerNames) == 3);
        report = analyzeNetwork(graph);
        assert(report.NumLayers == 3);
        assert(isequal(report.LayerNames, graph.LayerNames));
        assert(strcmp(report.Diagnostics, 'RunMat metadata analysis completed without launching UI'));
        empty = layerGraph();
        assert(isempty(empty.Layers));
        empty_report = analyzeNetwork(empty);
        assert(empty_report.NumLayers == 0);
        "#,
    )
    .expect("compiled layer graph and analysis return structural metadata");
}

#[test]
fn compiled_combvec_preserves_column_cartesian_order() {
    execute_source(
        r#"
        combinations = combvec([1 3; 2 4], [10 20]);
        assert(isequal(size(combinations), [3 4]));
        assert(isequal(combinations, [1 3 1 3; 2 4 2 4; 10 10 20 20]));
        "#,
    )
    .expect("compiled combvec cartesian column order");
}

#[test]
fn compiled_padsequences_preserves_class_shape_and_logical_mask() {
    for class_name in [
        "double", "single", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
        "uint64",
    ] {
        let program = format!(
            "sequences = {{{class_name}([1 2]), {class_name}(3)}}; [data, mask] = padsequences(sequences, 2); assert(isa(data, '{class_name}')); assert(isequal(size(data), [1 2 2])); assert(data(1,1,1) == 1 && data(1,2,1) == 2 && data(1,1,2) == 3 && data(1,2,2) == 0); assert(islogical(mask) && ~mask(1,2,2));"
        );
        execute_source(&program).unwrap_or_else(|error| {
            panic!("compiled padding of {class_name} sequences failed: {error}")
        });
    }
    execute_source(
        r#"
        sequences = {uint16([1 2]), uint16(3)};
        [padded, mask] = padsequences(sequences, 2);
        assert(isa(padded, 'uint16') && isequal(size(padded), [1 2 2]));
        assert(isequal(reshape(padded, 1, 4), uint16([1 2 3 0])));
        assert(islogical(mask) && isequal(reshape(mask, 1, 4), logical([1 1 1 0])));

        wide = uint64(9007199254740992) + uint64(1);
        [exact, wide_mask] = padsequences({[wide intmax('uint64')], uint64(7)}, 2);
        assert(isa(exact, 'uint64') && exact(1,1,1) == wide);
        assert(exact(1,2,1) == intmax('uint64') && exact(1,2,2) == uint64(0));
        assert(islogical(wide_mask) && ~wide_mask(1,2,2));
        "#,
    )
    .expect("compiled sequence padding retains integer storage and two-output mask");
}

#[test]
fn compiled_padsequences_applies_direction_length_and_cell_output_options() {
    execute_source(
        r#"
        sequences = {[1 2 3], 4};
        [cells, masks] = padsequences(sequences, 2, 'Length', 2, 'PaddingDirection', 'left', 'UniformOutput', false, 'PaddingValue', -1);
        assert(iscell(cells) && iscell(masks));
        assert(isequal(cells{1}, [2 3]));
        assert(isequal(cells{2}, [-1 4]));
        assert(isequal(masks{1}, logical([1 1])));
        assert(isequal(masks{2}, logical([0 1])));
        "#,
    )
    .expect("compiled sequence options preserve truncation, padding direction and cell masks");
}

#[test]
fn compiled_deep_learning_extension_and_exact_metadata_boundaries() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let error = execute_source("layer = featureInputLayer([2 3]);")
        .expect_err("vector feature size is a RunMat extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FeatureInputVectorExtension")
    );
    let error = execute_source("choices = combvec(uint8([1 2]), [3 4]);")
        .expect_err("integer combvec data is a RunMat extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:CombvecIntegerInputExtension")
    );
    drop(_matlab);

    execute_source(
        "wide = uint64(9223372036854775808); layer = fullyConnectedLayer(wide); assert(layer.OutputSize == 9223372036854775808);",
    )
    .expect("exactly representable wide integer metadata");
    let error = execute_source(
        "wide = uint64(9007199254740992) + uint64(1); layer = fullyConnectedLayer(wide);",
    )
    .expect_err("inexact wide integer cannot become double-valued metadata");
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
}

#[test]
fn compiled_dlarray_wraps_typed_host_data_and_projects_shape() {
    execute_source(
        r#"
        X = dlarray(single([1 2; 3 4]), 'CB');
        assert(strcmp(class(X), 'dlarray'));
        assert(isequal(size(X), [2 2]) && numel(X) == 4);
        assert(isa(X.Data, 'single') && isequal(X.Data, single([1 2; 3 4])));
        assert(strcmp(X.Format, 'CB') && strcmp(X.Labels, 'CB'));
        flags = dlarray(logical([1 0 1]));
        assert(isa(flags.Data, 'logical') && isequal(size(flags), [1 3]));
        assert(strcmp(flags.Format, '') && strcmp(flags.Labels, ''));
        "#,
    )
    .expect("compiled dlarray preserves typed payloads, labels and array geometry");
}

#[test]
fn compiled_training_options_preserve_defaults_and_exact_integer_controls() {
    execute_source(
        r#"
        sgdm = trainingOptions('sgdm');
        adam = trainingOptions('adam', 'MaxEpochs', uint16(5), 'Verbose', false);
        rmsprop = trainingOptions('rmsprop');
        assert(strcmp(class(sgdm), 'nnet.cnn.TrainingOptionsSGDM'));
        assert(strcmp(class(adam), 'nnet.cnn.TrainingOptionsADAM'));
        assert(strcmp(class(rmsprop), 'nnet.cnn.TrainingOptionsRMSProp'));
        assert(strcmp(sgdm.SolverName, 'sgdm'));
        assert(strcmp(adam.SolverName, 'adam'));
        assert(strcmp(rmsprop.SolverName, 'rmsprop'));
        assert(sgdm.MaxEpochs == 30);
        assert(sgdm.MiniBatchSize == 128 && sgdm.InitialLearnRate == 0.01);
        assert(adam.InitialLearnRate == 0.001 && isa(adam.MaxEpochs, 'uint16'));
        assert(adam.MaxEpochs == uint16(5) && adam.Verbose == false);
        assert(rmsprop.InitialLearnRate == 0.001);
        wide = uint64(9007199254740992) + uint64(1);
        metadata = trainingOptions('adam', 'MaxEpochs', wide);
        assert(isa(metadata.MaxEpochs, 'uint64') && metadata.MaxEpochs == wide);
        "#,
    )
    .expect("compiled training options retain class, defaults and exact control metadata");
}

#[test]
fn compiled_deep_learning_object_constructors_check_arity_even_when_discarded() {
    execute_source("dlarray(1); trainingOptions('adam');")
        .expect("discarded objects still execute valid constructors");
    for (source, identifier) in [
        ("[a,b] = dlarray(1);", "RunMat:deepLearning:InvalidInput"),
        (
            "[a,b] = trainingOptions('adam');",
            "RunMat:deepLearning:InvalidInput",
        ),
        (
            "trainingOptions('unavailable');",
            "RunMat:deepLearning:InvalidInput",
        ),
        (
            "trainingOptions('adam', 'MaxEpochs');",
            "RunMat:deepLearning:InvalidInput",
        ),
    ] {
        let error = execute_source(source).expect_err("invalid constructor invocation");
        assert_eq!(error.identifier(), Some(identifier), "{source}: {error}");
    }
    execute_source("trainingOptions();").expect_err("solver argument is required");
}
