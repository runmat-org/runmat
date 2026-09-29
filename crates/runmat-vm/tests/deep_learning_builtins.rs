#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

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
