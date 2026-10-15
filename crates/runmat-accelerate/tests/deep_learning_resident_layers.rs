#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]

//! Public dlarray/forward/dlgradient witnesses on a required native adapter.
//! Run each ignored precision filter in its own process with the matching
//! RUNMAT_WGPU_FORCE_PRECISION; the provider is a process-local singleton.

use futures::executor::block_on;
use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
use runmat_accelerate_api::{AccelProvider, ProviderPrecision};
use runmat_hir::LoweringContext;

struct VmThreadState;

impl VmThreadState {
    fn enter() -> Self {
        runmat_vm::reset_thread_state_for_tests();
        Self
    }
}

impl Drop for VmThreadState {
    fn drop(&mut self) {
        runmat_vm::reset_thread_state_for_tests();
    }
}

fn run_source(
    source: &str,
    provider: &'static dyn AccelProvider,
) -> Result<(), Box<runmat_runtime::RuntimeError>> {
    let source: std::sync::Arc<str> = std::sync::Arc::from(source);
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            let _thread = VmThreadState::enter();
            let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
            let ast = runmat_parser::parse(&source).expect("parse public DL program");
            let hir = runmat_hir::lower(&ast, &LoweringContext::empty())
                .expect("lower public DL program");
            let mir = runmat_mir::lowering::lower_assembly(&hir.assembly)
                .expect("lower public DL assembly");
            let entrypoint = hir.assembly.entrypoints[0].id;
            let bytecode = runmat_vm::compile(&hir.assembly, &mir, entrypoint)
                .expect("compile public DL program");
            let ordinal = bytecode
                .source_id
                .and_then(runmat_vm::BytecodeSourceRef::program)
                .expect("public DL bytecode retains its compiled source ordinal");
            let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_program_sources([runmat_runtime::source_context::ProgramSourceRecord {
                ordinal,
                name: std::sync::Arc::from("<resident-layer>"),
                fullpath_name: None,
                text: source,
            }])
            .expect("bind the exact public DL source to its VM session");
            block_on(runtime.scope(runmat_vm::interpret(&bytecode)))
                .map(|_| ())
                .map_err(Box::new)
        })
        .expect("start public VM stack")
        .join()
        .expect("public VM thread panicked")
}

fn execute_source(source: &str, provider: &'static dyn AccelProvider) {
    run_source(source, provider).expect("run public DL program");
}

fn required_provider(precision: ProviderPrecision) -> &'static dyn AccelProvider {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU adapter required for resident layer qualification");
    assert_eq!(
        provider.precision(),
        precision,
        "set RUNMAT_WGPU_FORCE_PRECISION"
    );
    provider
}

fn run_layer_witnesses(precision: ProviderPrecision) {
    let provider = required_provider(precision);
    let class = match precision {
        ProviderPrecision::F32 => "single",
        ProviderPrecision::F64 => "double",
    };
    let tolerance = match precision {
        ProviderPrecision::F32 => "5e-4",
        ProviderPrecision::F64 => "1e-9",
    };
    let normalization = r#"
        net = dlnetwork({featureInputLayer(3); layerNormalizationLayer(); fullyConnectedLayer(1)});
        learnables = net.Learnables;
        assert(strcmp(learnables.Parameter(1), 'Scale'));
        assert(strcmp(learnables.Parameter(2), 'Offset'));
        assert(strcmp(learnables.Parameter(3), 'Weights'));
        assert(strcmp(learnables.Parameter(4), 'Bias'));
        learnables.Value{1} = dlarray($CLASS$([1; 1; 1]));
        learnables.Value{2} = dlarray($CLASS$([0; 0; 0]));
        learnables.Value{3} = dlarray($CLASS$([1 0 0]));
        learnables.Value{4} = dlarray($CLASS$(0));
        net.Learnables = learnables;
        x = dlarray(gpuArray($CLASS$([1; 2; 4])), 'CB');
        [loss, gradient, parameterGradients] = dlfeval(@objective, net, x);
        assert(isgpuarray(loss.Data));
        assert(isgpuarray(gradient.Data));
        assert(strcmp(gradient.Format, 'CB'));
        assert(isequal(size(parameterGradients.Value), [4 1]));
        $NUMERICAL$
        function [loss, gradient, parameterGradients] = objective(net, x)
            loss = forward(net, x);
            [gradient, parameterGradients] = dlgradient(loss, x, net);
        end
    "#;
    let pooling = r#"
        net = dlnetwork({sequenceInputLayer(1); globalAveragePooling1dLayer()});
        x = dlarray(gpuArray($CLASS$(reshape(1:4, 1, 1, 4))), 'CBT');
        [loss, gradient] = dlfeval(@objective, net, x);
        assert(isgpuarray(loss.Data));
        assert(isgpuarray(gradient.Data));
        assert(strcmp(gradient.Format, 'CBT'));
        $NUMERICAL$
        function [loss, gradient] = objective(net, x)
            loss = forward(net, x);
            gradient = dlgradient(loss, x);
        end
    "#;
    let cases = [
        (
            normalization,
            r#"
                score = gather(loss.Data);
                assert(strcmp(class(score), '$CLASS$'));
                assert(isequal(size(score), [1 1]));
                assert(abs(score - $CLASS$(-1.0690415314502972)) < $TOL$);
                values = gather(gradient.Data);
                assert(strcmp(class(values), '$CLASS$'));
                expected = $CLASS$([0.22908229170096814; -0.3436200013686196; 0.11453770966765141]);
                assert(isequal(size(values), [3 1]));
                assert(max(abs(values - expected)) < $TOL$);
                scale = parameterGradients.Value{1};
                offset = parameterGradients.Value{2};
                weights = parameterGradients.Value{3};
                bias = parameterGradients.Value{4};
                assert(isgpuarray(scale.Data) && isgpuarray(offset.Data));
                assert(isgpuarray(weights.Data) && isgpuarray(bias.Data));
                dScale = gather(scale.Data);
                dOffset = gather(offset.Data);
                dWeights = gather(weights.Data);
                dBias = gather(bias.Data);
                assert(strcmp(class(dScale), '$CLASS$') && strcmp(class(dOffset), '$CLASS$'));
                assert(strcmp(class(dWeights), '$CLASS$') && strcmp(class(dBias), '$CLASS$'));
                assert(isequal(size(dScale), [3 1]) && isequal(size(dOffset), [3 1]));
                assert(isequal(size(dWeights), [1 3]) && isequal(size(dBias), [1 1]));
                assert(max(abs(dScale - $CLASS$([-1.0690415314502972; 0; 0]))) < $TOL$);
                assert(isequal(dOffset, $CLASS$([1; 0; 0])));
                assert(max(abs(dWeights - $CLASS$([-1.0690415314502972 -0.2672603828625743 1.3363019143128716]))) < $TOL$);
                assert(isequal(dBias, $CLASS$(1)));
            "#,
        ),
        (
            pooling,
            "score = gather(loss.Data); assert(strcmp(class(score), '$CLASS$')); assert(isequal(size(score), [1 1])); assert(isequal(score, $CLASS$(2.5))); values = gather(gradient.Data); assert(strcmp(class(values), '$CLASS$')); expected = $CLASS$(ones(1, 1, 4) * 0.25); assert(isequal(size(values), [1 1 4])); assert(isequal(values, expected));",
        ),
    ];
    for (template, numerical) in cases {
        let resident_only = template
            .replace("$CLASS$", class)
            .replace("$NUMERICAL$", "");
        provider.reset_telemetry();
        execute_source(&resident_only, provider);
        assert_eq!(
            provider.telemetry_snapshot().download_bytes,
            0,
            "resident public forward/dlgradient must not gather before the oracle"
        );
        let numerical = numerical
            .replace("$CLASS$", class)
            .replace("$TOL$", tolerance);
        execute_source(
            &template
                .replace("$CLASS$", class)
                .replace("$NUMERICAL$", &numerical),
            provider,
        );
    }

    // The adjoint must use the Scale captured by the executed forward pass.
    // Changing the network's canonical Learnables table afterward is not a
    // request to replay forward or to read its replacement Scale in reverse.
    let snapshot = r#"
        net = dlnetwork({featureInputLayer(3); layerNormalizationLayer(); fullyConnectedLayer(1)});
        learnables = net.Learnables;
        learnables.Value{1} = dlarray($CLASS$([2; 1; 1]));
        learnables.Value{2} = dlarray($CLASS$([0; 0; 0]));
        learnables.Value{3} = dlarray($CLASS$([1 0 0]));
        learnables.Value{4} = dlarray($CLASS$(0));
        net.Learnables = learnables;
        x = dlarray(gpuArray($CLASS$([1; 2; 4])), 'CB');
        [loss, gradient] = dlfeval(@objective, net, x);
        assert(isgpuarray(loss.Data));
        assert(isgpuarray(gradient.Data));
        $NUMERICAL$
        function [loss, gradient] = objective(net, x)
            loss = forward(net, x);
            replacement = net.Learnables;
            replacement.Value{1} = dlarray($CLASS$([0; 0; 0]));
            replacement.Value{2} = dlarray($CLASS$([100; 100; 100]));
            net.Learnables = replacement;
            gradient = dlgradient(loss, x);
        end
    "#;
    provider.reset_telemetry();
    execute_source(
        &snapshot
            .replace("$CLASS$", class)
            .replace("$NUMERICAL$", ""),
        provider,
    );
    assert_eq!(
        provider.telemetry_snapshot().download_bytes,
        0,
        "the saved Scale adjoint must remain resident after network mutation"
    );
    let numerical = "score = gather(loss.Data); assert(strcmp(class(score), '$CLASS$')); assert(abs(score - $CLASS$(-2.1380830629005945)) < $TOL$); values = gather(gradient.Data); assert(strcmp(class(values), '$CLASS$')); expected = $CLASS$([0.45816458340193628; -0.6872400027372392; 0.22907541933530282]); assert(isequal(size(values), [3 1])); assert(max(abs(values - expected)) < $TOL$);";
    execute_source(
        &snapshot.replace("$CLASS$", class).replace(
            "$NUMERICAL$",
            &numerical
                .replace("$CLASS$", class)
                .replace("$TOL$", tolerance),
        ),
        provider,
    );
}

fn run_convolution_witness(precision: ProviderPrecision) {
    let provider = required_provider(precision);
    let class = match precision {
        ProviderPrecision::F32 => "single",
        ProviderPrecision::F64 => "double",
    };
    let source = r#"
        net = dlnetwork({sequenceInputLayer(1); convolution1dLayer(2, 1, 'Padding', 0)});
        learnables = net.Learnables;
        assert(strcmp(learnables.Parameter(1), 'Weights'));
        assert(strcmp(learnables.Parameter(2), 'Bias'));
        learnables.Value{1} = dlarray(reshape($CLASS$([2 3]), 2, 1, 1));
        learnables.Value{2} = dlarray($CLASS$(5));
        net.Learnables = learnables;
        configured = net.Learnables;
        assert(configured.Value{1}.Data(1) == $CLASS$(2));
        assert(configured.Value{1}.Data(2) == $CLASS$(3));
        assert(configured.Value{2}.Data(1) == $CLASS$(5));
        x = dlarray(gpuArray(reshape($CLASS$([1 4]), 1, 1, 2)), 'CBT');
        [loss, gradient, parameterGradients] = dlfeval(@objective, net, x);
        assert(isgpuarray(loss.Data));
        assert(isgpuarray(gradient.Data));
        assert(strcmp(gradient.Format, 'CBT'));
        assert(isequal(size(parameterGradients.Value), [2 1]));
        $NUMERICAL$
        function [loss, gradient, parameterGradients] = objective(net, x)
            loss = forward(net, x);
            [gradient, parameterGradients] = dlgradient(loss, x, net);
        end
    "#;
    provider.reset_telemetry();
    execute_source(
        &source.replace("$CLASS$", class).replace("$NUMERICAL$", ""),
        provider,
    );
    assert_eq!(
        provider.telemetry_snapshot().download_bytes,
        0,
        "learned convolution forward and adjoint must remain resident"
    );
    let numerical = r#"
        score = gather(loss.Data);
        assert(strcmp(class(score), '$CLASS$'));
        assert(numel(score) == 1);
        assert(size(score, 1) == 1 && size(score, 2) == 1 && size(score, 3) == 1);
        assert(abs(score(1) - $CLASS$(19)) < $CLASS$(1e-4));
        values = gather(gradient.Data);
        assert(strcmp(class(values), '$CLASS$'));
        assert(isequal(size(values), [1 1 2]));
        assert(isequal(values, reshape($CLASS$([2 3]), 1, 1, 2)));
        weights = parameterGradients.Value{1};
        bias = parameterGradients.Value{2};
        assert(isgpuarray(weights.Data) && isgpuarray(bias.Data));
        dWeights = gather(weights.Data);
        dBias = gather(bias.Data);
        assert(strcmp(class(dWeights), '$CLASS$') && strcmp(class(dBias), '$CLASS$'));
        assert(numel(dWeights) == 2 && size(dWeights, 1) == 2 && size(dWeights, 2) == 1 && size(dWeights, 3) == 1);
        assert(numel(dBias) == 1 && size(dBias, 1) == 1 && size(dBias, 2) == 1);
        assert(isequal(dWeights, reshape($CLASS$([1 4]), 2, 1, 1)));
        assert(isequal(dBias, $CLASS$(1)));
    "#;
    execute_source(
        &source
            .replace("$CLASS$", class)
            .replace("$NUMERICAL$", &numerical.replace("$CLASS$", class)),
        provider,
    );
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F32 precision"]
fn resident_normalization_and_pooling_public_gradient_f32() {
    run_layer_witnesses(ProviderPrecision::F32);
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F64 precision"]
fn resident_normalization_and_pooling_public_gradient_f64() {
    run_layer_witnesses(ProviderPrecision::F64);
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F32 precision"]
fn resident_convolution_public_gradient_f32() {
    run_convolution_witness(ProviderPrecision::F32);
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F64 precision"]
fn resident_convolution_public_gradient_f64() {
    run_convolution_witness(ProviderPrecision::F64);
}

fn mixed_precision_parameter_gradient_witness(
    provider: &'static dyn AccelProvider,
    formatted_cb: bool,
) {
    let source = r#"
        net = dlnetwork({featureInputLayer(2); fullyConnectedLayer(1)});
        learnables = net.Learnables;
        learnables.Value{1} = dlarray([2 3]);
        learnables.Value{2} = dlarray(5);
        net.Learnables = learnables;
        x = $INPUT$;
        [loss, inputGradient, parameterGradients] = dlfeval(@objective, net, x);
        assert(isgpuarray(loss.Data) && isgpuarray(inputGradient.Data));
        assert(isequal(size(parameterGradients.Value), [2 1]));
        assert(isgpuarray(parameterGradients.Value{1}.Data));
        assert(isgpuarray(parameterGradients.Value{2}.Data));
        $NUMERICAL$
        function [loss, inputGradient, parameterGradients] = objective(net, x)
            loss = forward(net, x);
            [inputGradient, parameterGradients] = dlgradient(loss, x, net);
        end
    "#;
    let source = if formatted_cb {
        source.replace("$INPUT$", "dlarray(gpuArray(single([1; 4])), 'CB')")
    } else {
        source.replace("$INPUT$", "dlarray(gpuArray(single([1 4])))")
    };
    provider.reset_telemetry();
    execute_source(&source.replace("$NUMERICAL$", ""), provider);
    assert_eq!(
        provider.telemetry_snapshot().download_bytes,
        0,
        "mixed-class resident gradient must not gather before presentation"
    );
    let numerical = r#"
        score = gather(loss.Data);
        dx = gather(inputGradient.Data);
        dw = gather(parameterGradients.Value{1}.Data);
        db = gather(parameterGradients.Value{2}.Data);
        assert(strcmp(class(score), 'single') && abs(score - single(19)) < single(1e-5));
        assert(strcmp(class(dx), 'single') && isequal(size(dx), size($DX$)) && isequal(dx, $DX$));
        assert(strcmp(class(dw), 'double') && isequal(dw, [1 4]));
        assert(strcmp(class(db), 'double') && isequal(db, 1));
    "#;
    let numerical = numerical.replace(
        "$DX$",
        if formatted_cb {
            "single([2; 3])"
        } else {
            "single([2 3])"
        },
    );
    execute_source(&source.replace("$NUMERICAL$", &numerical), provider);
}

#[test]
fn inprocess_resident_single_execution_restores_double_learnable_gradients() {
    runmat_accelerate::simple_provider::register_inprocess_provider();
    let provider = runmat_accelerate_api::provider().expect("in-process provider registered");
    mixed_precision_parameter_gradient_witness(provider, false);
}

#[test]
#[ignore = "requires actual WGPU adapter with shader-f64 support"]
fn resident_single_execution_restores_double_learnable_gradients_on_f64_device() {
    let provider = required_provider(ProviderPrecision::F64);
    mixed_precision_parameter_gradient_witness(provider, false);
    mixed_precision_parameter_gradient_witness(provider, true);
}

#[test]
#[ignore = "requires an actual F32 WGPU adapter"]
fn f32_device_derives_input_without_unused_double_learnable_cast() {
    let provider = required_provider(ProviderPrecision::F32);
    let source = r#"
        net = dlnetwork({featureInputLayer(2); fullyConnectedLayer(1)});
        learnables = net.Learnables;
        learnables.Value{1} = dlarray([2 3]);
        learnables.Value{2} = dlarray(5);
        net.Learnables = learnables;
        x = dlarray(gpuArray(single([1 4])));
        [loss, dx] = dlfeval(@objective, net, x);
        assert(isgpuarray(loss.Data) && isgpuarray(dx.Data));
        $NUMERICAL$
        function [loss, dx] = objective(net, x)
            loss = forward(net, x);
            dx = dlgradient(loss, x);
        end
    "#;
    provider.reset_telemetry();
    execute_source(&source.replace("$NUMERICAL$", ""), provider);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    execute_source(
        &source.replace(
            "$NUMERICAL$",
            "values = gather(dx.Data); assert(strcmp(class(values), 'single') && isequal(size(values), [1 2]) && isequal(values, single([2 3])));",
        ),
        provider,
    );
}

#[test]
#[ignore = "requires an actual F32 WGPU adapter"]
fn f32_device_rejects_requested_double_learnable_cast_before_reverse_effects() {
    let provider = required_provider(ProviderPrecision::F32);
    let template = r#"
            net = dlnetwork({featureInputLayer(2); fullyConnectedLayer(1)});
            learnables = net.Learnables;
            learnables.Value{1} = dlarray([2 3]);
            learnables.Value{2} = dlarray(5);
            net.Learnables = learnables;
            x = dlarray(gpuArray(single([1 4])));
            loss = dlfeval(@objective, net, x);
            function [loss, derivative] = objective(net, x)
                loss = forward(net, x);
                derivative = $DERIVATIVE$;
            end
        "#;
    provider.reset_telemetry();
    execute_source(&template.replace("$DERIVATIVE$", "x"), provider);
    let forward_only = provider.telemetry_snapshot();
    provider.reset_telemetry();
    let error = run_source(
        &template.replace("$DERIVATIVE$", "dlgradient(loss, net)"),
        provider,
    )
    .expect_err("requested double Learnables cannot be restored on an F32-only owner");
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:Unsupported"));
    let after = provider.telemetry_snapshot();
    assert_eq!(after.kernel_launches, forward_only.kernel_launches);
    assert_eq!(after.download_bytes, 0);
}
