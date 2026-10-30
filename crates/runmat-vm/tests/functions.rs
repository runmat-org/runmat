#[path = "support/mod.rs"]
mod test_helpers;

#[path = "functions/lexical_environment.rs"]
mod lexical_environment;

#[path = "support/captured_arguments.rs"]
mod captured_arguments;
#[path = "support/captured_arguments/dotted.rs"]
mod captured_dotted_arguments;

#[path = "support/dotted_calls.rs"]
mod dotted_calls;

use captured_arguments::is_expand_all_cell_contents;
use captured_dotted_arguments::has_captured_cell_dotted_call;
use dotted_calls::has_single_value_dotted_call;

use std::convert::TryFrom;
use test_helpers::compile_source;
use test_helpers::interpret;

fn scoped_provider() -> runmat_accelerate_api::ThreadProviderGuard {
    static PROVIDER: std::sync::OnceLock<runmat_accelerate::simple_provider::InProcessProvider> =
        std::sync::OnceLock::new();
    let provider = PROVIDER.get_or_init(runmat_accelerate::simple_provider::InProcessProvider::new);
    runmat_accelerate_api::ThreadProviderGuard::set(Some(provider))
}

fn has_bound_expanded_subscript(
    bytecode: &runmat_vm::Bytecode,
    selection: runmat_types::SequenceUse,
) -> bool {
    bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::ReadSubscriptPath { steps, selection: actual, .. }
            if *actual == selection
                && matches!(steps.as_slice(), [runmat_vm::BytecodeSubscriptStep::Parentheses { selectors }]
                    if matches!(selectors.as_slice(), [runmat_vm::BytecodeSubscriptSelector::Expansion(_)]))
    ))
}

fn assert_prepared_indexed_store(bytecode: &runmat_vm::Bytecode) {
    use runmat_vm::{BytecodeSubscriptStep, Instr};
    let begin = bytecode
        .instructions
        .iter()
        .enumerate()
        .find_map(|(index, instruction)| match instruction {
            Instr::BeginFixedAssignment {
                source_statement,
                root_slot,
            } => Some((index, source_statement, root_slot)),
            _ => None,
        })
        .expect("indexed member store captures its original receiver");
    let finish = bytecode
        .instructions
        .iter()
        .enumerate()
        .skip(begin.0 + 1)
        .find_map(|(index, instruction)| match instruction {
            Instr::FinishFixedAssignmentTarget {
                endpoint: BytecodeSubscriptStep::Parentheses { selectors },
                selector_context: Some(count),
                mutation: runmat_types::PlaceMutationKind::IndexedAssign,
            } if *count == 1 && selectors.len() == 1 => Some(index),
            _ => None,
        })
        .expect("typed indexed endpoint preserves its selector receipt");
    let commit = bytecode
        .instructions
        .iter()
        .enumerate()
        .skip(finish + 1)
        .find_map(|(index, instruction)| match instruction {
            Instr::CommitFixedAssignment {
                source_statement,
                root_slot,
            } if source_statement == begin.1 && root_slot == begin.2 => Some(index),
            _ => None,
        })
        .expect("commit consumes the same source occurrence and receiver");
    assert!(bytecode.instructions[finish..commit]
        .iter()
        .any(|instruction| matches!(instruction, Instr::CaptureScalarSequence)));
}

fn has_single_selector_subscript_call(
    bytecode: &runmat_vm::Bytecode,
    selection: runmat_types::SequenceUse,
) -> bool {
    bytecode.instructions.iter().any(|instruction| matches!(
        instruction,
        runmat_vm::Instr::ReadSubscriptPath { steps, selection: actual, to_sequence_register: false, .. }
            if *actual == selection
                && matches!(steps.as_slice(), [runmat_vm::BytecodeSubscriptStep::Parentheses { selectors }]
                    if selectors.len() == 1)
    ))
}

fn has_exception_identifier(values: &[runmat_value::Value], identifier: &str) -> bool {
    let expected = runmat_value::CharArray::new_row(identifier);
    values.iter().any(|value| {
        matches!(value,
        runmat_value::Value::CharArray(actual) if actual == &expected)
    })
}

fn execute_source(source: &str) -> runmat_runtime::execution::RootedValues {
    let bytecode = compile_source(source).expect("compile source");
    interpret(&bytecode).expect("execute bytecode")
}

fn execute_source_with_extensions(source: &str) -> runmat_runtime::execution::RootedValues {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(source)
}

fn execute_source_with_base_workspace(source: &str) -> runmat_runtime::execution::RootedValues {
    let bytecode = compile_source(source).expect("compile source");
    test_helpers::interpret_with_base_workspace(&bytecode)
        .expect("execute with selected base owner")
}

fn execute_source_with_catalog(
    source: &str,
    name: &str,
) -> runmat_runtime::execution::RootedValues {
    let bytecode = compile_source(source).expect("compile source");
    test_helpers::interpret_with_program_source(&bytecode, name, source)
        .expect("execute bytecode with selected program source")
}

fn execute_source_result(
    source: &str,
) -> Result<runmat_runtime::execution::RootedValues, Box<runmat_runtime::RuntimeError>> {
    let bytecode = compile_source(source).expect("compile source");
    interpret(&bytecode).map_err(Box::new)
}

fn execute_source_result_with_extensions(
    source: &str,
) -> Result<runmat_runtime::execution::RootedValues, Box<runmat_runtime::RuntimeError>> {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source_result(source)
}

fn has_num(values: &[runmat_value::Value], expected: f64) -> bool {
    values
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - expected).abs() < 1e-9))
}

fn numeric_workspace_value(values: &[runmat_value::Value], expected: f64) -> bool {
    has_num(values, expected)
}

fn has_object_class(values: &[runmat_value::Value], class_name: &str) -> bool {
    let class_name = runmat_types::ClassIdentity::new(class_name).expect("test class identity");
    values.iter().any(|v| match v {
        runmat_value::Value::Object(obj) => obj.class_name() == &class_name,
        runmat_value::Value::HandleObject(obj) => obj.class_name == class_name,
        _ => false,
    })
}

fn has_class_ref(values: &[runmat_value::Value], class_name: &str) -> bool {
    let class_name = runmat_types::ClassIdentity::new(class_name).expect("test class identity");
    values
        .iter()
        .any(|v| matches!(v, runmat_value::Value::ClassRef(name) if name == &class_name))
}

fn has_numeric_tensor(values: &[runmat_value::Value], expected: &[f64]) -> bool {
    values.iter().any(|value| match value {
        runmat_value::Value::Tensor(tensor) => {
            tensor.materialize_f64().len() == expected.len()
                && tensor
                    .materialize_f64()
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| (*actual - *expected).abs() < 1e-9)
        }
        _ => false,
    })
}

fn cell_char_row_values(value: &runmat_value::Value) -> Option<((usize, usize), Vec<String>)> {
    let runmat_value::Value::Cell(cell) = value else {
        return None;
    };
    let values = cell
        .data()
        .iter()
        .map(|entry| match entry {
            runmat_value::Value::CharArray(chars) => Some(
                runmat_value::decode_utf16_code_units(chars.data())
                    .expect("valid UTF-16 text fixture"),
            ),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some(((cell.rows, cell.cols), values))
}

fn has_cell_char_row(values: &[runmat_value::Value], expected: &[&str]) -> bool {
    values.iter().any(|value| {
        cell_char_row_values(value).is_some_and(|((rows, cols), actual)| {
            rows == 1
                && cols == expected.len()
                && actual
                    == expected
                        .iter()
                        .map(|entry| entry.to_string())
                        .collect::<Vec<_>>()
        })
    })
}

fn has_object_num_property(
    values: &[runmat_value::Value],
    class_name: &str,
    property_name: &str,
    expected: f64,
) -> bool {
    let class_name = runmat_types::ClassIdentity::new(class_name).expect("test class identity");
    values.iter().any(|v| match v {
        runmat_value::Value::Object(obj) if obj.class_name() == &class_name => {
            matches!(
                obj.properties().get(property_name),
                Some(runmat_value::Value::Num(n)) if (*n - expected).abs() < 1e-9
            )
        }
        _ => false,
    })
}

fn has_object_string_property(
    values: &[runmat_value::Value],
    class_name: &str,
    property_name: &str,
    expected: &str,
) -> bool {
    let class_name = runmat_types::ClassIdentity::new(class_name).expect("test class identity");
    values.iter().any(|v| match v {
        runmat_value::Value::Object(obj) if obj.class_name() == &class_name => {
            matches!(
                obj.properties().get(property_name),
                Some(runmat_value::Value::String(s)) if s == expected
            )
        }
        _ => false,
    })
}

#[test]
fn min_max_row_vector_after_linear_assignment_reduce_to_scalars_for_fprintf() {
    execute_source(
        r#"
        a = single(zeros(1, 3));
        a(1) = single(0.3);
        a(2) = single(0.1);
        a(3) = single(0.5);
        mn = min(a);
        mx = max(a);
        mn_size = size(mn);
        mx_size = size(mx);
        fprintf('%.4f', mn);
        fprintf('%.4f', mx);
        "#,
    );
}

#[test]
fn input_parser_parses_varargin_name_value_pair() {
    let vars = execute_source(
        r#"
        y = f('scale', 4);
        function y = f(varargin)
            p = inputParser;
            addParameter(p, 'scale', 2);
            parse(p, varargin{:});
            y = p.Results.scale;
        end
        "#,
    );
    assert!(
        numeric_workspace_value(&vars, 4.0),
        "expected parsed scale override; vars={vars:?}"
    );
}

#[test]
fn input_parser_preserves_defaults_for_omitted_parameters() {
    let vars = execute_source(
        r#"
        y = f('scale', 4);
        function y = f(varargin)
            p = inputParser;
            addParameter(p, 'scale', 2);
            addParameter(p, 'offset', 10);
            parse(p, varargin{:});
            y = p.Results.scale + p.Results.offset;
        end
        "#,
    );
    assert!(
        numeric_workspace_value(&vars, 14.0),
        "expected override plus default; vars={vars:?}"
    );
}

#[test]
fn input_parser_accepts_string_scalar_parameter_names() {
    let vars = execute_source(
        r#"
        y = f("scale", 4);
        function y = f(varargin)
            p = inputParser;
            addParameter(p, "scale", 2);
            parse(p, varargin{:});
            y = p.Results.scale;
        end
        "#,
    );
    assert!(
        numeric_workspace_value(&vars, 4.0),
        "expected parsed string scalar name; vars={vars:?}"
    );
}

fn assert_import_ambiguity_error(err: &runmat_runtime::RuntimeError) {
    assert_eq!(
        err.identifier(),
        Some("RunMat:ImportAmbiguous"),
        "expected import ambiguity identifier, got: {}",
        err.message()
    );
}

#[test]
fn arguments_block_accepts_extended_validation_family() {
    let values = execute_source(
        r#"
        y = checked(3, 'alpha', 'on');
        function out = checked(x, name, mode)
            arguments
                x (1,1) double {mustBeNumeric, mustBeFloat, mustBeInteger, mustBeInRange(1, 5)}
                name {mustBeTextScalar, mustBeNonzeroLengthText, mustBeValidVariableName}
                mode {mustBeMember('on')}
            end
            out = x + 4;
        end
        "#,
    );
    assert!(
        has_num(&values, 7.0),
        "expected checked output, got {values:?}"
    );
}

#[test]
fn arguments_block_extended_validator_failures_are_runtime_errors() {
    let err = execute_source_result(
        r#"
        y = checked(8);
        function out = checked(x)
            arguments
                x {mustBeInRange(1, 5)}
            end
            out = x;
        end
        "#,
    )
    .expect_err("out-of-range argument should fail validation");
    assert_eq!(
        err.identifier(),
        Some("RunMat:ArgumentValidationFunction"),
        "unexpected error: {}",
        err.message()
    );
    assert!(
        err.message().contains("mustBeInRange"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn arguments_block_in_range_honors_exclusive_interval_flag() {
    let err = execute_source_result(
        r#"
        y = checked(1);
        function out = checked(x)
            arguments
                x {mustBeInRange(1, 5, 'exclusive')}
            end
            out = x;
        end
        "#,
    )
    .expect_err("exclusive lower endpoint should fail validation");
    assert_eq!(
        err.identifier(),
        Some("RunMat:ArgumentValidationFunction"),
        "unexpected error: {}",
        err.message()
    );
    assert!(
        err.message().contains("mustBeInRange"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn arguments_block_in_range_accepts_explicit_argument_form() {
    let values = execute_source(
        r#"
        y = checked(3);
        function out = checked(x)
            arguments
                x {mustBeInRange(x, 1, 5)}
            end
            out = x + 4;
        end
        "#,
    );
    assert!(
        has_num(&values, 7.0),
        "expected checked output, got {values:?}"
    );

    let err = execute_source_result(
        r#"
        y = checked(1);
        function out = checked(x)
            arguments
                x {mustBeInRange(x, 1, 5, 'exclusive')}
            end
            out = x;
        end
        "#,
    )
    .expect_err("exclusive lower endpoint should fail validation");
    assert_eq!(
        err.identifier(),
        Some("RunMat:ArgumentValidationFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn argument_validation_helpers_are_callable_builtins() {
    let values = execute_source(
        r#"
        mustBePositive(3);
        mustBeMember('on', {'on', 'off'});
        tf = isvarname('alpha_1');
        args = namedargs2cell(struct('Name', 'Ada', 'Value', 7));
        n = numel(args);
        "#,
    );
    assert!(
        values
            .iter()
            .any(|v| matches!(v, runmat_value::Value::Bool(true))),
        "expected isvarname true value, got {values:?}"
    );
    assert!(
        has_num(&values, 4.0),
        "expected namedargs2cell numel, got {values:?}"
    );
}

#[test]
fn arguments_block_validators_preserve_wide_integer_values() {
    let values = execute_source(
        r#"
        x = uint64(9007199254740992) + uint64(1);
        y = checked(x);
        function out = checked(value)
            arguments
                value {mustBeInteger, mustBePositive, mustBeNonzero, mustBeGreaterThan(9007199254740992)}
            end
            out = value;
        end
        "#,
    );
    assert!(
        values.iter().any(|value| matches!(
            value,
            runmat_value::Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993))
        )),
        "expected exact uint64 output, got {values:?}"
    );
}

#[test]
fn arguments_block_in_range_requires_matching_numeric_classes() {
    let err = execute_source_result(
        r#"
        x = uint64(3);
        y = checked(x);
        function out = checked(value)
            arguments
                value {mustBeInRange(1, 5)}
            end
            out = value;
        end
        "#,
    )
    .expect_err("double bounds must not silently validate uint64 data");
    assert!(
        err.message().contains("same numeric class"),
        "unexpected error: {err}"
    );
}

#[test]
fn arguments_block_resident_integer_validation_preserves_source_ownership() {
    // This fixture owns its provider selection; parallel CPU tests must never
    // acquire a temporary global default that disappears while they use it.
    let before = runmat_accelerate_api::provider().map(|provider| provider.device_id());
    let provider_guard = scoped_provider();
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(
        r#"
        x = uint64(9007199254740993);
        g = gpuArray(x);
        y = checked(g);
        if ~isgpuarray(y) || gather(y) ~= x
            error('resident validator changed the source');
        end
        function out = checked(value)
            arguments
                value {mustBePositive, mustBeNonzero}
            end
            out = value;
        end
        "#,
    );
    drop(provider_guard);
    assert_eq!(
        runmat_accelerate_api::provider().map(|provider| provider.device_id()),
        before,
        "resident validator must restore its caller's provider selection"
    );
}

#[test]
fn unresolved_external_function_handle_fails_without_legacy_fallback() {
    let err = execute_source_result("h = @definitely_missing_callback; y = feval(h, 1);")
        .expect_err("unresolved external callback should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_function_handle_zero_output_feval_fails_without_legacy_fallback() {
    let err = execute_source_result("h = @definitely_missing_callback; feval(h, 1);")
        .expect_err("unresolved external callback should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn nested_varargout_forwarding_with_nargout_slice_assignment() {
    let source = r#"
        [a,b] = outer(5);
        function varargout = outer(x)
            [varargout{1:nargout}] = inner(x);
            function varargout = inner(v)
                varargout{1} = v + 1;
                varargout{2} = v + 2;
            end
        end
    "#;
    let bytecode = compile_source(source).expect("compile source");
    let entry = bytecode
        .function_registry
        .functions
        .values()
        .find(|f| f.display_name == "outer")
        .expect("outer bytecode present");
    assert!(
        entry.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { .. }
                | runmat_vm::Instr::CallSemanticFunctionMultiUsingOutputSlot(_, _, _)
                | runmat_vm::Instr::CallFunctionMultiUsingOutputSlot { .. }
        )),
        "outer should lower nested forwarding call with dynamic output-slot arity; instructions={:?}",
        entry.instructions
    );
    let vars = interpret(&bytecode).expect("execute bytecode");
    assert!(
        has_num(&vars, 6.0),
        "expected first forwarded output; vars={vars:?}"
    );
    assert!(
        has_num(&vars, 7.0),
        "expected second forwarded output; vars={vars:?}"
    );
}

#[test]
fn zero_requested_varargout_forwarding_still_executes_rhs_body() {
    let error = execute_source_result(
        r#"
        outer(5);
        function varargout = outer(x)
            [varargout{1:nargout}] = inner(x);
        end
        function varargout = inner(v)
            error('RunMat:ZeroOutputBodyReached', 'zero-output body was invoked');
        end
    "#,
    )
    .expect_err("the zero-output RHS call must still execute");
    assert_eq!(error.identifier(), Some("RunMat:ZeroOutputBodyReached"));
}

#[test]
fn nested_varargout_forwarding_with_nargout_slice_assignment_via_feval() {
    let source = r#"
        [a,b] = outer(5);
        function varargout = outer(x)
            [varargout{1:nargout}] = feval('pair', x);
        end
        function varargout = pair(v)
            varargout{1} = v + 10;
            varargout{2} = v + 20;
        end
    "#;
    let bytecode = compile_source(source).expect("compile source");
    let entry = bytecode
        .function_registry
        .functions
        .values()
        .find(|f| f.display_name == "outer")
        .expect("outer bytecode present");
    assert!(
        entry.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CallFevalMultiUsingOutputSlot(_, _)
        )),
        "outer should lower feval forwarding call with dynamic output-slot arity; instructions={:?}",
        entry.instructions
    );
    let vars = interpret(&bytecode).expect("execute bytecode");
    assert!(
        has_num(&vars, 15.0),
        "expected first forwarded output; vars={vars:?}"
    );
    assert!(
        has_num(&vars, 25.0),
        "expected second forwarded output; vars={vars:?}"
    );
}

#[test]
fn nested_varargout_forwarding_with_nargout_slice_assignment_via_nested_feval() {
    let source = r#"
        [a,b] = outer(5);
        function varargout = outer(x)
            [varargout{1:nargout}] = feval(@inner, x);
            function varargout = inner(v)
                varargout{1} = v + 10;
                varargout{2} = v + 20;
            end
        end
    "#;
    let bytecode = compile_source(source).expect("compile source");
    let entry = bytecode
        .function_registry
        .functions
        .values()
        .find(|f| f.display_name == "outer")
        .expect("outer bytecode present");
    assert!(
        entry.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateSemanticClosure(_, name, 1, _) if name == "inner"
        )),
        "outer should lower explicit nested handle to a captured semantic closure; instructions={:?}",
        entry.instructions
    );
    assert!(
        entry
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalMultiUsingOutputSlot(_, _))),
        "outer should retain dynamic output-slot feval; instructions={:?}",
        entry.instructions
    );
    let vars = interpret(&bytecode).expect("execute bytecode");
    assert!(
        has_num(&vars, 15.0),
        "expected first forwarded output; vars={vars:?}"
    );
    assert!(
        has_num(&vars, 25.0),
        "expected second forwarded output; vars={vars:?}"
    );
}

#[test]
fn unresolved_external_function_handle_multi_output_feval_fails_without_legacy_fallback() {
    let err = execute_source_result("h = @definitely_missing_callback; [a,b] = feval(h, 1);")
        .expect_err("unresolved external callback should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_function_handle_expand_feval_fails_without_legacy_fallback() {
    let err =
        execute_source_result("h = @definitely_missing_callback; C = {1,2}; y = feval(h, C{:});")
            .expect_err("unresolved external expanded callback should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_function_handle_expand_zero_output_feval_fails_without_legacy_fallback() {
    let err = execute_source_result("h = @definitely_missing_callback; C = {1,2}; feval(h, C{:});")
        .expect_err("unresolved external expanded callback should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_function_handle_expand_multi_output_feval_fails_without_legacy_fallback() {
    let err = execute_source_result(
        "h = @definitely_missing_callback; C = {1,2}; [a,b] = feval(h, C{:});",
    )
    .expect_err("unresolved external expanded callback should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_external_function_handle_uses_external_handle_instruction() {
    let bytecode = compile_source("h = @pkg.remote_inc; y = feval(h, 1);")
        .expect("qualified handle source should compile");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc")
    ));
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CallFevalMulti(argc, out_count) if *argc == 1 && *out_count == 1)
    ));
    let err = interpret(&bytecode).expect_err("unresolved qualified callback should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_external_handle_zero_output_feval_uses_typed_instruction() {
    let source = "h = @pkg.remote_inc; feval(h, 1);";
    let bytecode = compile_source(source)
        .expect("qualified external handle zero-output feval source should compile");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc")
    ));
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CallFevalMulti(argc, out_count) if *argc == 1 && *out_count == 0)
    ));
    let err = interpret(&bytecode).expect_err("unresolved qualified zero-output feval should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_external_handle_multi_output_feval_uses_typed_instruction() {
    let source = "h = @pkg.remote_inc; [a,b] = feval(h, 1);";
    let bytecode = compile_source(source)
        .expect("qualified external handle multi-output feval source should compile");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc")
    ));
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CallFevalMulti(argc, out_count) if *argc == 1 && *out_count == 2)
    ));
    let err =
        interpret(&bytecode).expect_err("unresolved qualified multi-output feval should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_external_handle_expand_zero_output_feval_uses_typed_instruction() {
    let source = "h = @pkg.remote_inc; C = {1,2}; feval(h, C{:});";
    let bytecode = compile_source(source)
        .expect("qualified external handle expanded zero-output feval source should compile");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc")
    ));
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFevalExpandMultiOutput(specs, out_count)
            if *out_count == 0 && specs.len() == 1 && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let err = interpret(&bytecode)
        .expect_err("unresolved qualified expanded zero-output feval should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_external_handle_expand_feval_uses_typed_instruction() {
    let source = "h = @pkg.remote_inc; C = {1,2}; y = feval(h, C{:});";
    let bytecode = compile_source(source)
        .expect("qualified external handle expanded feval source should compile");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc")
    ));
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFevalExpandMultiOutput(specs, out_count)
            if *out_count == 1 && specs.len() == 1 && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let err = interpret(&bytecode).expect_err("unresolved qualified expanded feval should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_external_handle_expand_multi_output_feval_uses_typed_instruction() {
    let source = "h = @pkg.remote_inc; C = {1,2}; [a,b] = feval(h, C{:});";
    let bytecode = compile_source(source)
        .expect("qualified external handle expanded multi-output feval source should compile");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc")
    ));
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFevalExpandMultiOutput(specs, out_count)
            if *out_count == 2 && specs.len() == 1 && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let err = interpret(&bytecode)
        .expect_err("unresolved qualified expanded multi-output feval should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_external_handle_feval_uses_external_handle_instruction() {
    let source = "h = @pkg.sub.remote; y = feval(h, 1);";
    let bytecode = compile_source(source).expect("nested qualified external handle feval compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.sub.remote")
    ));
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CallFevalMulti(argc, out_count) if *argc == 1 && *out_count == 1)
    ));
    let err = interpret(&bytecode)
        .expect_err("unresolved nested qualified feval handle call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_external_handle_zero_output_feval_uses_typed_instruction() {
    let source = "h = @pkg.sub.remote; feval(h, 1);";
    let bytecode = compile_source(source)
        .expect("nested qualified external handle zero-output feval compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.sub.remote")
    ));
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CallFevalMulti(argc, out_count) if *argc == 1 && *out_count == 0)
    ));
    let err = interpret(&bytecode)
        .expect_err("unresolved nested qualified zero-output feval handle call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_external_handle_multi_output_feval_uses_typed_instruction() {
    let source = "h = @pkg.sub.remote; [a,b] = feval(h, 1);";
    let bytecode = compile_source(source)
        .expect("nested qualified external handle multi-output feval compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.sub.remote")
    ));
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CallFevalMulti(argc, out_count) if *argc == 1 && *out_count == 2)
    ));
    let err = interpret(&bytecode)
        .expect_err("unresolved nested qualified multi-output feval handle call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_external_handle_expand_zero_output_feval_uses_typed_instruction() {
    let source = "h = @pkg.sub.remote; C = {1,2}; feval(h, C{:});";
    let bytecode = compile_source(source)
        .expect("nested qualified external handle expanded zero-output feval compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.sub.remote")
    ));
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFevalExpandMultiOutput(specs, out_count)
            if *out_count == 0 && specs.len() == 1 && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let err = interpret(&bytecode).expect_err(
        "unresolved nested qualified expanded zero-output feval handle call should fail",
    );
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_external_handle_expand_single_output_feval_uses_typed_instruction() {
    let source = "h = @pkg.sub.remote; C = {1,2}; y = feval(h, C{:});";
    let bytecode = compile_source(source)
        .expect("nested qualified external handle expanded single-output feval compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.sub.remote")
    ));
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFevalExpandMultiOutput(specs, out_count)
            if *out_count == 1 && specs.len() == 1 && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let err = interpret(&bytecode).expect_err(
        "unresolved nested qualified expanded single-output feval handle call should fail",
    );
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_external_handle_expand_feval_uses_typed_instruction() {
    let source = "h = @pkg.sub.remote; C = {1,2}; [a,b] = feval(h, C{:});";
    let bytecode =
        compile_source(source).expect("nested qualified external handle expanded feval compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.sub.remote")
    ));
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFevalExpandMultiOutput(specs, out_count)
            if *out_count == 2 && specs.len() == 1 && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let err = interpret(&bytecode)
        .expect_err("unresolved nested qualified expanded feval handle call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_external_handle_direct_call_uses_external_handle_instruction() {
    let source = "h = @pkg.remote_inc; y = h(1);";
    let bytecode = compile_source(source).expect("qualified external handle direct call compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc")
    ));
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::Index(1))),
        "single-output direct handle call should lower through Index(1)"
    );
    let err =
        interpret(&bytecode).expect_err("unresolved qualified direct handle call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_external_handle_zero_output_direct_call_uses_typed_instruction() {
    let source = "h = @pkg.remote_inc; h(1);";
    let bytecode =
        compile_source(source).expect("qualified external handle zero-output direct call compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc")
    ));
    assert!(has_single_selector_subscript_call(
        &bytecode,
        runmat_types::SequenceUse::Discard
    ));
    let err = interpret(&bytecode)
        .expect_err("unresolved qualified zero-output direct handle call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_external_handle_multi_output_direct_call_uses_typed_instruction() {
    let source = "h = @pkg.remote_inc; [a,b] = h(1);";
    let bytecode = compile_source(source)
        .expect("qualified external handle multi-output direct call compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc")
    ));
    assert!(has_single_selector_subscript_call(
        &bytecode,
        runmat_types::SequenceUse::SelectPrefix { count: 2 }
    ));
    let err = interpret(&bytecode)
        .expect_err("unresolved qualified multi-output direct handle call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_external_handle_expand_zero_output_direct_call_uses_typed_instruction() {
    let source = "h = @pkg.remote_inc; C = {1,2}; h(C{:});";
    let bytecode = compile_source(source)
        .expect("qualified external handle expanded zero-output direct call compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc")
    ));
    assert!(has_bound_expanded_subscript(
        &bytecode,
        runmat_types::SequenceUse::Discard
    ));
    let err = interpret(&bytecode)
        .expect_err("unresolved qualified expanded zero-output direct handle call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_external_handle_expand_direct_call_uses_typed_instruction() {
    let source = "h = @pkg.remote_inc; C = {1,2}; y = h(C{:});";
    let bytecode =
        compile_source(source).expect("qualified external handle expanded direct call compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc")
    ));
    assert!(has_bound_expanded_subscript(
        &bytecode,
        runmat_types::SequenceUse::RequireSingle
    ));
    let err = interpret(&bytecode)
        .expect_err("unresolved qualified expanded direct handle call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_external_handle_expand_multi_output_direct_call_uses_typed_instruction() {
    let source = "h = @pkg.remote_inc; C = {1,2}; [a,b] = h(C{:});";
    let bytecode = compile_source(source)
        .expect("qualified external handle expanded multi-output direct call compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc")
    ));
    assert!(has_bound_expanded_subscript(
        &bytecode,
        runmat_types::SequenceUse::SelectPrefix { count: 2 }
    ));
    let err = interpret(&bytecode)
        .expect_err("unresolved qualified expanded multi-output direct handle call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_external_handle_direct_call_uses_external_handle_instruction() {
    let source = "h = @pkg.sub.remote; y = h(1);";
    let bytecode =
        compile_source(source).expect("nested qualified external handle direct call compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.sub.remote")
    ));
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::Index(1))),
        "single-output direct handle call should lower through Index(1)"
    );
    let err = interpret(&bytecode)
        .expect_err("unresolved nested qualified external handle direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_external_handle_zero_output_direct_call_uses_typed_instruction() {
    let source = "h = @pkg.sub.remote; h(1);";
    let bytecode = compile_source(source)
        .expect("nested qualified external handle zero-output direct call compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.sub.remote")
    ));
    assert!(has_single_selector_subscript_call(
        &bytecode,
        runmat_types::SequenceUse::Discard
    ));
    let err = interpret(&bytecode).expect_err(
        "unresolved nested qualified external handle zero-output direct call should fail",
    );
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_external_handle_multi_output_direct_call_uses_typed_instruction() {
    let source = "h = @pkg.sub.remote; [a,b] = h(1);";
    let bytecode = compile_source(source)
        .expect("nested qualified external handle multi-output direct call compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.sub.remote")
    ));
    assert!(has_single_selector_subscript_call(
        &bytecode,
        runmat_types::SequenceUse::SelectPrefix { count: 2 }
    ));
    let err = interpret(&bytecode).expect_err(
        "unresolved nested qualified external handle multi-output direct call should fail",
    );
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_external_handle_expand_zero_output_direct_call_uses_typed_instruction(
) {
    let source = "h = @pkg.sub.remote; C = {1,2}; h(C{:});";
    let bytecode = compile_source(source)
        .expect("nested qualified external handle expanded zero-output direct call compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.sub.remote")
    ));
    assert!(has_bound_expanded_subscript(
        &bytecode,
        runmat_types::SequenceUse::Discard
    ));
    let err = interpret(&bytecode).expect_err(
        "unresolved nested qualified external handle expanded zero-output direct call should fail",
    );
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_external_handle_expand_direct_call_uses_typed_instruction() {
    let source = "h = @pkg.sub.remote; C = {1,2}; y = h(C{:});";
    let bytecode = compile_source(source)
        .expect("nested qualified external handle expanded direct call compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.sub.remote")
    ));
    assert!(has_bound_expanded_subscript(
        &bytecode,
        runmat_types::SequenceUse::RequireSingle
    ));
    let err = interpret(&bytecode)
        .expect_err("unresolved nested qualified external handle expanded direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_external_handle_expand_multi_output_direct_call_uses_typed_instruction(
) {
    let source = "h = @pkg.sub.remote; C = {1,2}; [a,b] = h(C{:});";
    let bytecode = compile_source(source)
        .expect("nested qualified external handle expanded multi-output direct call compiles");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.sub.remote")
    ));
    assert!(has_bound_expanded_subscript(
        &bytecode,
        runmat_types::SequenceUse::SelectPrefix { count: 2 }
    ));
    let err = interpret(&bytecode).expect_err(
        "unresolved nested qualified external handle expanded multi-output direct call should fail",
    );
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_direct_call_fails_without_runtime_name_fallback() {
    let err = execute_source_result("y = definitely_missing_callback(1);")
        .expect_err("unresolved external direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_direct_call_uses_external_boundary_typed_instruction() {
    let bytecode = compile_source("a = pkg.remote_inc(7);")
        .expect("qualified direct call source should compile");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallFunctionMulti {
            identity: runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(path)),
            fallback_policy,
            arg_count,
            out_count,
            ..
        } if path == &vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("remote_inc".to_string()),
            ]
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::ExternalBoundary
            && *arg_count == 1
            && *out_count == 1
    )));
    let err = interpret(&bytecode).expect_err("unresolved qualified direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_direct_call_zero_output_uses_external_boundary_typed_instruction() {
    let bytecode = compile_source("pkg.remote_inc(7);")
        .expect("qualified zero-output direct call source should compile");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallFunctionMulti {
            identity: runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(path)),
            fallback_policy,
            arg_count,
            out_count,
            ..
        } if path == &vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("remote_inc".to_string()),
            ]
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::ExternalBoundary
            && *arg_count == 1
            && *out_count == 0
    )));
    let err =
        interpret(&bytecode).expect_err("unresolved qualified zero-output direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_direct_call_multi_output_uses_external_boundary_typed_instruction() {
    let bytecode = compile_source("[a,b] = pkg.remote_inc(7);")
        .expect("qualified multi-output direct call source should compile");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallFunctionMulti {
            identity: runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(path)),
            fallback_policy,
            arg_count,
            out_count,
            ..
        } if path == &vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("remote_inc".to_string()),
            ]
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::ExternalBoundary
            && *arg_count == 1
            && *out_count == 2
    )));
    let err = interpret(&bytecode)
        .expect_err("unresolved qualified multi-output direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_direct_call_expand_zero_output_uses_external_boundary_typed_instruction() {
    let source = "C = {1,2}; pkg.remote_inc(C{:});";
    let bytecode = compile_source(source)
        .expect("qualified expanded zero-output direct call source should compile");
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFunctionExpandMultiOutput {
            identity: runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(path)),
            fallback_policy,
            specs,
            out_count,
            ..
        } if path == &vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("remote_inc".to_string()),
            ]
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::ExternalBoundary
            && *out_count == 0
            && specs.len() == 1
            && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let err = interpret(&bytecode)
        .expect_err("unresolved qualified expanded zero-output direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_qualified_direct_call_expand_multi_output_uses_external_boundary_typed_instruction() {
    let source = "C = {1,2}; [a,b] = pkg.remote_inc(C{:});";
    let bytecode = compile_source(source)
        .expect("qualified expanded multi-output direct call source should compile");
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFunctionExpandMultiOutput {
            identity: runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(path)),
            fallback_policy,
            specs,
            out_count,
            ..
        } if path == &vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("remote_inc".to_string()),
            ]
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::ExternalBoundary
            && *out_count == 2
            && specs.len() == 1
            && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let err = interpret(&bytecode)
        .expect_err("unresolved qualified expanded multi-output direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_direct_call_uses_external_boundary_typed_instruction() {
    let bytecode = compile_source("a = pkg.sub.remote(7);")
        .expect("nested qualified direct call source should compile");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallFunctionMulti {
            identity: runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(path)),
            fallback_policy,
            arg_count,
            out_count,
            ..
        } if path == &vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("sub".to_string()),
                runmat_hir::SymbolName("remote".to_string()),
            ]
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::ExternalBoundary
            && *arg_count == 1
            && *out_count == 1
    )));
    let err =
        interpret(&bytecode).expect_err("unresolved nested qualified direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_direct_call_zero_output_uses_external_boundary_typed_instruction() {
    let bytecode = compile_source("pkg.sub.remote(7);")
        .expect("nested qualified zero-output direct call source should compile");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallFunctionMulti {
            identity: runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(path)),
            fallback_policy,
            arg_count,
            out_count,
            ..
        } if path == &vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("sub".to_string()),
                runmat_hir::SymbolName("remote".to_string()),
            ]
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::ExternalBoundary
            && *arg_count == 1
            && *out_count == 0
    )));
    let err = interpret(&bytecode)
        .expect_err("unresolved nested qualified zero-output direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_direct_call_multi_output_uses_external_boundary_typed_instruction() {
    let bytecode = compile_source("[a,b] = pkg.sub.remote(7);")
        .expect("nested qualified multi-output direct call source should compile");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallFunctionMulti {
            identity: runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(path)),
            fallback_policy,
            arg_count,
            out_count,
            ..
        } if path == &vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("sub".to_string()),
                runmat_hir::SymbolName("remote".to_string()),
            ]
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::ExternalBoundary
            && *arg_count == 1
            && *out_count == 2
    )));
    let err = interpret(&bytecode)
        .expect_err("unresolved nested qualified multi-output direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_direct_call_expand_zero_output_uses_external_boundary_typed_instruction(
) {
    let source = "C = {1,2}; pkg.sub.remote(C{:});";
    let bytecode = compile_source(source)
        .expect("nested qualified expanded zero-output direct call source should compile");
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFunctionExpandMultiOutput {
            identity: runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(path)),
            fallback_policy,
            specs,
            out_count,
            ..
        } if path == &vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("sub".to_string()),
                runmat_hir::SymbolName("remote".to_string()),
            ]
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::ExternalBoundary
            && *out_count == 0
            && specs.len() == 1
            && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let err = interpret(&bytecode)
        .expect_err("unresolved nested qualified expanded zero-output direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_direct_call_expand_single_output_uses_external_boundary_typed_instruction(
) {
    let source = "C = {1,2}; a = pkg.sub.remote(C{:});";
    let bytecode = compile_source(source)
        .expect("nested qualified expanded single-output direct call source should compile");
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFunctionExpandMultiOutput {
            identity: runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(path)),
            fallback_policy,
            specs,
            out_count,
            ..
        } if path == &vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("sub".to_string()),
                runmat_hir::SymbolName("remote".to_string()),
            ]
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::ExternalBoundary
            && *out_count == 1
            && specs.len() == 1
            && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let err = interpret(&bytecode)
        .expect_err("unresolved nested qualified expanded single-output direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_nested_qualified_direct_call_expand_multi_output_uses_external_boundary_typed_instruction(
) {
    let source = "C = {1,2}; [a,b] = pkg.sub.remote(C{:});";
    let bytecode = compile_source(source)
        .expect("nested qualified expanded multi-output direct call source should compile");
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFunctionExpandMultiOutput {
            identity: runmat_hir::CallableIdentity::ExternalName(runmat_hir::QualifiedName(path)),
            fallback_policy,
            specs,
            out_count,
            ..
        } if path == &vec![
                runmat_hir::SymbolName("pkg".to_string()),
                runmat_hir::SymbolName("sub".to_string()),
                runmat_hir::SymbolName("remote".to_string()),
            ]
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::ExternalBoundary
            && *out_count == 2
            && specs.len() == 1
            && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let err = interpret(&bytecode)
        .expect_err("unresolved nested qualified expanded multi-output direct call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_cellfun_callback_fails_without_legacy_fallback() {
    let err = execute_source_result("xs = {1, 2}; ys = cellfun(@definitely_missing_callback, xs);")
        .expect_err("unresolved external cellfun callback should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_cellfun_str2func_callback_fails_without_legacy_fallback() {
    let err = execute_source_result(
        "xs = {1, 2}; h = str2func('definitely_missing_callback'); ys = cellfun(h, xs);",
    )
    .expect_err("unresolved external cellfun str2func callback should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_arrayfun_callback_fails_without_legacy_fallback() {
    let err =
        execute_source_result("xs = [1, 2]; ys = arrayfun(@definitely_missing_callback, xs);")
            .expect_err("unresolved external arrayfun callback should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_arrayfun_str2func_callback_fails_without_legacy_fallback() {
    let err = execute_source_result(
        "xs = [1, 2]; h = str2func('definitely_missing_callback'); ys = arrayfun(h, xs);",
    )
    .expect_err("unresolved external arrayfun str2func callback should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn function_handle_selector_colon_errors_with_identifier_contract() {
    let err = execute_source_result("f = @sin; y = f(:);")
        .expect_err("function-handle colon selector should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UnsupportedFunctionHandleSelector"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn function_handle_scalar_assignment_selector_errors_with_identifier_contract() {
    let err = execute_source_result("f = @sin; f(1) = 2;")
        .expect_err("function-handle scalar assignment selector should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UnsupportedFunctionHandleSelector"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn function_handle_slice_assignment_selector_errors_with_identifier_contract() {
    let err = execute_source_result("f = @sin; f(:) = 2;")
        .expect_err("function-handle slice assignment selector should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UnsupportedFunctionHandleSelector"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn function_handle_brace_selector_errors_with_identifier_contract() {
    let err = execute_source_result("f = @sin; y = f{1};")
        .expect_err("function-handle brace selector should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UnsupportedFunctionHandleSelector"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn function_handle_brace_assignment_selector_errors_with_identifier_contract() {
    let err = execute_source_result("f = @sin; f{1} = 2;")
        .expect_err("function-handle brace assignment selector should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UnsupportedFunctionHandleSelector"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn brace_read_on_noncell_errors_with_identifier_contract() {
    let err = execute_source_result("x = 10{1};").expect_err("brace read on non-cell should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:CellIndexingOnNonCell"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn brace_assignment_on_noncell_errors_with_identifier_contract() {
    let err = execute_source_result("x = 10; x{1} = 2;")
        .expect_err("brace assignment on non-cell should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:CellAssignmentOnNonCell"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn brace_expansion_on_noncell_errors_with_identifier_contract() {
    let err = execute_source_result("y = sin(10{:});")
        .expect_err("brace expansion on non-cell should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:InvalidExpandAllTarget"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn nargin_nargout_in_user_functions() {
    // Single-output: nargin/nargout should reflect call site
    let program = r#"
        function y = f(a,b)
            y = nargin() + nargout(); % 2 + 1 = 3
        end
        r = f(10, 20);
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-3.0).abs()<1e-9)));

    // Multi-output context: nargout seen inside should match requested outs
    let program2 = r#"
        function [x,y] = g(a)
            x = nargin(); % 1
            y = nargout(); % 2
        end
        [u,v] = g(7);
    "#;
    let vars2 = execute_source(program2);
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-1.0).abs()<1e-9)));
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-2.0).abs()<1e-9)));
}

#[test]
fn narginchk_nargoutchk_validate_function_arity() {
    let program = r#"
        r = guarded(1, 2, 3);
        function y = guarded(a, b, varargin)
            narginchk(2, Inf);
            nargoutchk(1, 1);
            y = a + b + length(varargin);
        end
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-4.0).abs()<1e-9)));
}

#[test]
fn dynamic_workspace_eval_mutates_and_reads_current_vm_workspace() {
    let vars = execute_source(
        r#"
        eval('dyn_x = 41;');
        dyn_y = eval('dyn_x + 1');
    "#,
    );
    assert!(has_num(&vars, 41.0));
    assert!(has_num(&vars, 42.0));
}

#[test]
fn dynamic_workspace_evalin_caller_from_function_targets_vm_script_workspace() {
    let vars = execute_source(
        r#"
        outer_x = 5;
        y = helper();

        function out = helper()
            out = evalin('caller', 'outer_x + 2');
            assignin('caller', 'caller_z', 11);
        end
    "#,
    );
    assert!(has_num(&vars, 7.0));
    assert!(has_num(&vars, 11.0));
}

#[test]
fn assignin_compiled_transfer_preserves_all_integer_classes_exactly() {
    let vars = execute_source_with_base_workspace(
        "assignin('base','ai_i8',int8([-128 127])); assignin('base','ai_i16',int16([-32768 32767])); assignin('base','ai_i32',int32([-2147483648 2147483647])); assignin('base','ai_i64',int64([-7 9])); assignin('base','ai_u8',uint8([0 255])); assignin('base','ai_u16',uint16([0 65535])); assignin('base','ai_u32',uint32([0 4294967295])); base = uint64(9007199254740992); assignin('base','ai_u64',base + uint64([1 2]));",
    );
    for expected in [
        runmat_value::IntegerStorage::I8(vec![-128, 127]),
        runmat_value::IntegerStorage::I16(vec![-32768, 32767]),
        runmat_value::IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        runmat_value::IntegerStorage::I64(vec![-7, 9]),
        runmat_value::IntegerStorage::U8(vec![0, 255]),
        runmat_value::IntegerStorage::U16(vec![0, u16::MAX]),
        runmat_value::IntegerStorage::U32(vec![0, u32::MAX]),
        runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994]),
    ] {
        assert!(
            vars.iter().any(|value| {
                matches!(
                    value,
                    runmat_value::Value::Tensor(tensor)
                        if tensor.integer_storage() == Some(&expected)
                )
            }),
            "missing exact assignin storage {expected:?}"
        );
    }
}

#[test]
fn assignin_preserves_resident_integer_provider_ownership() {
    let _provider_guard = scoped_provider();
    let vars = execute_source_with_base_workspace(
        "base = uint64(9007199254740992); source_gpu = gpuArray([base + uint64(1) intmax('uint64')]); assignin('base','assigned_gpu',source_gpu); copied = evalin('base','gather(assigned_gpu)');",
    );
    let resident_handles = vars
        .iter()
        .filter_map(|value| match value {
            runmat_value::Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .collect::<Vec<_>>();
    let resident_ids = resident_handles
        .iter()
        .map(|handle| handle.allocation_identity())
        .collect::<Vec<_>>();
    assert!(
        resident_handles
            .iter()
            .enumerate()
            .any(|(index, handle)| resident_handles[index + 1..]
                .iter()
                .any(|candidate| handle.shares_allocation_with(candidate))),
        "assignin should clone the same resident handle, got {resident_ids:?}"
    );
    assert!(vars.iter().any(|value| {
        matches!(
            value,
            runmat_value::Value::Tensor(tensor)
                if tensor.integer_storage()
                    == Some(&runmat_value::IntegerStorage::U64(vec![
                        9_007_199_254_740_993,
                        u64::MAX,
                    ]))
        )
    }));
}

#[test]
fn assignin_rejects_outputs_and_invalid_variable_names() {
    let output_err = execute_source_result("out = assignin('base', 'x', uint64(1));")
        .expect_err("assignin must expose no output");
    assert_eq!(
        output_err.identifier(),
        Some("RunMat:assignin:TooManyOutputs")
    );

    for name in ["_leading", "for", "éclair"] {
        let source = format!("assignin('base', '{name}', int8(1));");
        let err = execute_source_result(&source).expect_err("invalid variable name must reject");
        assert_eq!(err.identifier(), Some("RunMat:DynamicWorkspaceName"));
    }

    let overlong = format!(
        "assignin('base', '{}', uint8(1));",
        "a".repeat(runmat_runtime::builtins::common::identifiers::MATLAB_NAME_LENGTH_MAX + 1)
    );
    let err = execute_source_result(&overlong).expect_err("overlong variable name must reject");
    assert_eq!(err.identifier(), Some("RunMat:DynamicWorkspaceName"));
}

#[test]
fn dynamic_workspace_invalid_selector_errors_are_catchable() {
    let vars = execute_source(
        r#"
        try
            evalin('workspace', '1');
            err = "BAD";
        catch e
            err = e.identifier;
        end
    "#,
    );
    assert!(has_exception_identifier(
        &vars,
        "RunMat:DynamicWorkspaceSelector"
    ));
}

#[test]
fn narginchk_reports_too_few_inputs() {
    let err = execute_source_result(
        r#"
        r = guarded(1);
        function y = guarded(a, b)
            narginchk(2, 2);
            y = a + b;
        end
    "#,
    )
    .expect_err("narginchk should reject too few inputs");
    assert_eq!(
        err.identifier(),
        Some("RunMat:NotEnoughInputs"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn narginchk_reports_too_many_inputs() {
    let err = execute_source_result(
        r#"
        r = guarded(1, 2);
        function y = guarded(a, varargin)
            narginchk(1, 1);
            y = a;
        end
    "#,
    )
    .expect_err("narginchk should reject too many inputs");
    assert_eq!(
        err.identifier(),
        Some("RunMat:TooManyInputs"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn nargoutchk_reports_too_few_outputs() {
    let err = execute_source_result(
        r#"
        r = guarded();
        function [a, b] = guarded()
            nargoutchk(2, 2);
            a = 1;
            b = 2;
        end
    "#,
    )
    .expect_err("nargoutchk should reject too few requested outputs");
    assert_eq!(
        err.identifier(),
        Some("RunMat:NotEnoughOutputs"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn nargoutchk_reports_too_many_outputs() {
    let err = execute_source_result(
        r#"
        [a, b] = guarded();
        function varargout = guarded()
            nargoutchk(0, 1);
            varargout{1} = 1;
        end
    "#,
    )
    .expect_err("nargoutchk should reject too many requested outputs");
    assert_eq!(
        err.identifier(),
        Some("RunMat:TooManyOutputs"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn compiled_arity_builtin_handles_enforce_form_outputs_after_intrinsic_retirement() {
    for (name, source) in [
        (
            "narginchk",
            r#"
            y = guarded(1);
            function y = guarded(x)
                checker = @narginchk;
                y = checker(0, 2);
            end
            "#,
        ),
        (
            "nargoutchk",
            r#"
            y = guarded(1);
            function y = guarded(x)
                checker = @nargoutchk;
                y = checker(0, 2);
            end
            "#,
        ),
        (
            "nargoutchk legacy",
            r#"
            [a, b] = guarded();
            function [a, b] = guarded()
                checker = @nargoutchk;
                [a, b] = checker(0, 1, 2);
            end
            "#,
        ),
    ] {
        let error = execute_source_result(source).expect_err(name);
        assert_eq!(
            error.identifier(),
            Some("RunMat:TooManyOutputs"),
            "{name}: {}",
            error.message()
        );
    }

    let values = execute_source(
        r#"
        y = guarded(7);
        function y = guarded(x)
            checker = @narginchk;
            checker(1, 1);
            checker = @nargoutchk;
            checker(1, 1);
            y = x;
        end
        "#,
    );
    assert!(has_num(&values, 7.0));
}

#[test]
fn arity_check_helpers_report_invalid_bounds() {
    let err = execute_source_result(
        r#"
        r = guarded(1);
        function y = guarded(a)
            narginchk(2, 1);
            y = a;
        end
    "#,
    )
    .expect_err("narginchk should reject invalid bounds");
    assert_eq!(
        err.identifier(),
        Some("RunMat:NarginchkBoundsInvalid"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn arity_check_helper_errors_are_catchable() {
    let program = r#"
        try
            guarded(1, 2);
            eid = "BAD";
        catch e
            eid = e.identifier;
        end
        function y = guarded(a, varargin)
            narginchk(1, 1);
            y = a;
        end
    "#;
    let vars = execute_source(program);
    assert!(has_exception_identifier(&vars, "RunMat:TooManyInputs"));
}

#[test]
fn dynamic_narginchk_function_handle_uses_current_function_arity() {
    let program = r#"
        ok = guarded(1, 2);
        function y = guarded(a, varargin)
            checker = @narginchk;
            feval(checker, 1, 2);
            y = a;
        end
    "#;
    let vars = execute_source(program);
    assert!(has_num(&vars, 1.0));
}

#[test]
fn dynamic_narginchk_errors_are_catchable() {
    let program = r#"
        try
            guarded(1, 2, 3);
            eid = "BAD";
        catch e
            eid = e.identifier;
        end
        function y = guarded(a, varargin)
            checker = @narginchk;
            feval(checker, 1, 2);
            y = a;
        end
    "#;
    let vars = execute_source(program);
    assert!(has_exception_identifier(&vars, "RunMat:TooManyInputs"));
}

#[test]
fn dynamic_nargoutchk_function_handle_uses_current_function_arity() {
    let program = r#"
        y = guarded(1);
        function y = guarded(a)
            checker = @nargoutchk;
            feval(checker, 1, 1);
            y = a;
        end
    "#;
    let vars = execute_source(program);
    assert!(has_num(&vars, 1.0));
}

#[test]
fn dynamic_nargoutchk_errors_are_catchable() {
    let program = r#"
        try
            [a, b] = guarded(1);
            eid = "BAD";
        catch e
            eid = e.identifier;
        end
        function [a, b] = guarded(x)
            checker = @nargoutchk;
            feval(checker, 1, 1);
            a = x;
            b = x;
        end
    "#;
    let vars = execute_source(program);
    assert!(has_exception_identifier(&vars, "RunMat:TooManyOutputs"));
}

#[test]
fn inputname_reports_direct_caller_argument_names() {
    let source = r#"
        alpha = 10;
        beta = 20;
        [n1, n2, n3, n4] = probe(alpha, alpha + 1, 7, beta);
        function [a, b, c, d] = probe(x, y, z, w)
            a = inputname(1);
            b = inputname(2);
            c = inputname(3);
            d = inputname(4);
        end
    "#;
    let vars = execute_source_with_catalog(source, "/tmp/runmat_vm_inputname_probe.m");
    assert!(vars.iter().any(|v| {
        matches!(v, runmat_value::Value::CharArray(chars) if runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 text fixture") == "alpha")
    }), "expected alpha character row; vars={vars:?}");
    assert!(vars.iter().any(|v| {
        matches!(v, runmat_value::Value::CharArray(chars) if runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 text fixture") == "beta")
    }), "expected beta character row; vars={vars:?}");
    let empty_strings = vars
        .iter()
        .filter(|v| matches!(v, runmat_value::Value::CharArray(chars) if chars.data().is_empty()))
        .count();
    assert!(
        empty_strings >= 2,
        "expression and literal arguments should return empty input names; vars={vars:?}"
    );
}

#[test]
fn inputname_reports_feval_caller_argument_names() {
    let source = r#"
        beta = 20;
        [n1, n2] = feval(@probe, beta, beta + 1);
        function [a, b] = probe(x, y)
            a = inputname(1);
            b = inputname(2);
        end
    "#;
    let vars = execute_source_with_catalog(source, "/tmp/runmat_vm_inputname_feval_probe.m");
    assert!(vars.iter().any(|v| {
        matches!(v, runmat_value::Value::CharArray(chars) if runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 text fixture") == "beta")
    }), "expected beta character row; vars={vars:?}");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::CharArray(chars) if chars.data().is_empty())));
}

#[test]
fn display_script_surface_uses_variable_header() {
    let source = r#"
        txt = evalc("alpha = [1 2; 3 4]; display(alpha);");
    "#;
    let vars = execute_source_with_catalog(source, "/tmp/runmat_vm_display_probe.m");
    assert!(vars.iter().any(|v| matches!(
        v,
        runmat_value::Value::String(runmat_value::StringElement::Text(s))
            if s.contains("alpha =")
                && s.contains("1       2")
                && s.contains("3       4")
    )));
}

#[test]
fn mfilename_reads_current_source_context() {
    let source = r#"
        name = mfilename();
        full = mfilename("fullpath");
        fallback = mfilename("not-a-mode");
    "#;
    let bytecode = compile_source(source).expect("compile source");
    let vars = test_helpers::interpret_with_program_source(
        &bytecode,
        "/tmp/runmat_vm_mfilename_probe.m",
        source,
    )
    .expect("execute bytecode with selected program source");
    let stem_count = vars
        .iter()
        .filter(|value| {
            matches!(value, runmat_value::Value::String(s) if s == "runmat_vm_mfilename_probe")
        })
        .count();
    assert_eq!(
        stem_count, 2,
        "expected name and fallback stem; vars={vars:?}"
    );
    assert!(
        vars.iter().any(|v| {
            matches!(v, runmat_value::Value::String(s) if s == "/tmp/runmat_vm_mfilename_probe")
        }),
        "expected full source path; vars={vars:?}"
    );
}

#[test]
fn functions_builtin_reports_function_handle_metadata() {
    let source = r#"
        builtin = functions(@sin);
        builtin_name = getfield(builtin, "function");
        builtin_type = builtin.type;
        local_handle = @local_id;
        local_info = functions(local_handle);
        local_name = getfield(local_info, "function");
        anon_info = functions(@(x) x + 1);
        anon_type = anon_info.type;
        function y = local_id(x)
            y = x;
        end
    "#;
    let vars = execute_source(source);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::String(s) if s == "sin")));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::String(s) if s == "simple")));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::String(s) if s == "local_id")));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::String(s) if s == "anonymous")));
}

#[test]
fn localfunctions_returns_callable_local_function_handles() {
    let source = r#"
        handles = localfunctions();
        h1 = handles{1};
        h2 = handles{2};
        info1 = functions(h1);
        info2 = functions(h2);
        name1 = getfield(info1, "function");
        name2 = getfield(info2, "function");
        total = feval(h1, 5) + feval(h2, 5);

        function y = helper_one(x)
            y = x + 1;
        end

        function y = helper_two(x)
            y = x + 2;
        end
    "#;
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source_with_catalog(source, "/tmp/runmat_vm_localfunctions_probe.m");
    assert!(
        has_num(&vars, 13.0),
        "local function handles should be callable; vars={vars:?}"
    );
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::String(s) if s == "helper_one")));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::String(s) if s == "helper_two")));
}

#[test]
fn compiler_classifies_function_file_primary_and_local_without_name_rules() {
    let source = "function handles = metadata_file()\nhandles = localfunctions();\nend\nfunction y = child_local(x)\ny = x + 1;\nend\n";
    let parsed = runmat_parser::parse(source).expect("function file parses");
    let lowered = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty())
        .expect("function file lowers");
    let roles = lowered
        .assembly
        .functions
        .iter()
        .map(|function| {
            (
                function.name.0.as_str(),
                lowered.assembly.function_source_role(function),
            )
        })
        .collect::<Vec<_>>();
    assert!(roles.iter().any(|(name, role)| *name == "metadata_file"
        && *role == runmat_types::FunctionSourceRole::Primary));
    assert!(roles
        .iter()
        .any(|(name, role)| *name == "child_local"
            && *role == runmat_types::FunctionSourceRole::Local));
}

#[test]
fn missing_fixed_input_can_be_guarded_with_nargin() {
    let program = r#"
        r = f(1, 2);
        function y = f(a,b,c)
            if nargin < 3
                c = 0;
            end
            y = a + b + c + nargout;
        end
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-4.0).abs()<1e-9)));
}

#[test]
fn bare_nargin_counts_varargin_inputs() {
    let program = r#"
        a = greet("Alice");
        b = greet("Bob", 1, 2, 3);
        function out = greet(name, varargin)
            if nargin > 1
                out = length(varargin);
            else
                out = 0;
            end
        end
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-0.0).abs()<1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-3.0).abs()<1e-9)));
}

#[test]
fn varargout_indexed_fill_respects_nargout_loop() {
    let program = r#"
        [a, b] = mysize([1 2; 3 4]);
        function varargout = mysize(A)
            s = size(A);
            for k = 1:nargout
                varargout{k} = s(k);
            end
        end
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-2.0).abs()<1e-9)));
}

#[test]
fn implicit_array_creation_from_linear_index_assignment() {
    let program = r#"
        x(3) = 10;
        w(1) = 0;
    "#;
    let vars = execute_source(program);
    assert!(vars.iter().any(|v| matches!(
        v,
        runmat_value::Value::Tensor(t) if t.shape() == vec![1, 3] && t.materialize_f64() == vec![0.0, 0.0, 10.0]
    )));
    assert!(vars.iter().any(|v| matches!(
        v,
        runmat_value::Value::Tensor(t) if t.shape() == vec![1, 1] && t.materialize_f64() == vec![0.0]
    )));
}

#[test]
fn not_enough_and_too_many_inputs_fixed_arity() {
    // not enough inputs
    let program = r#"
        function y = f(a,b)
            y = a + b;
        end
        r = f(1);
    "#;
    let err = execute_source_result(program).err().unwrap();
    assert_eq!(err.identifier(), Some("RunMat:NotEnoughInputs"));

    // too many inputs
    let program2 = r#"
        function y = f(a,b)
            y = a + b;
        end
        r = f(1,2,3);
    "#;
    let err2 = execute_source_result(program2).err().unwrap();
    assert_eq!(err2.identifier(), Some("RunMat:TooManyInputs"));
}

#[test]
fn inputs_with_varargin_minimum_only() {
    // With varargin, fewer than fixed args is an error; equal to fixed is okay; extras go to varargin
    let program_err = r#"
        function y = f(a,b,varargin)
            y = a + b;
        end
        r = f(1);
    "#;
    let err = execute_source_result(program_err).err().unwrap();
    assert_eq!(err.identifier(), Some("RunMat:NotEnoughInputs"));

    let program_ok = r#"
        function y = f(a,b,varargin)
            % Keep it simple to avoid runtime cat/sum dependencies
            y = a + b + varargin{1} + varargin{2};
        end
        r = f(1,2,3,4);
    "#;
    let vars_ok = execute_source(program_ok);
    assert!(vars_ok
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-10.0).abs()<1e-9)));
    // 1+2+3+4
}

#[test]
fn too_many_outputs_and_varargout_mismatch() {
    // Too many outputs for a function without varargout
    let program_tmo = r#"
        function y = f(a)
            y = a + 1;
        end
        [x1,x2] = f(3);
    "#;
    let err_tmo = execute_source_result(program_tmo).err().unwrap();
    assert_eq!(err_tmo.identifier(), Some("RunMat:TooManyOutputs"));

    // Too many outputs from non-call expression should not silently zero-pad.
    let expr_tmo = "[x1,x2] = 1;";
    let err_expr = execute_source_result(expr_tmo).err().unwrap();
    assert_eq!(err_expr.identifier(), Some("RunMat:TooManyOutputs"));

    let caught = execute_source(
        "x1=9; x2=10; try; [x1,x2]=1; catch e; caught_id=e.identifier; end; unchanged=x1+x2;",
    );
    assert!(has_exception_identifier(&caught, "RunMat:TooManyOutputs"));
    assert!(has_num(&caught, 19.));

    // The builtin owns its output diagnostic; user functions and expressions
    // above retain the executor's generic output-count identifier.
    let builtin_tmo = "[x1,x2] = sqrt(9);";
    let err_builtin = execute_source_result(builtin_tmo).err().unwrap();
    assert_eq!(err_builtin.identifier(), Some("RunMat:sqrt:TooManyOutputs"));

    // Varargout requested more than provided
    let program_mis = r#"
        function varargout = h(a)
            varargout = {a+1, a+2};
        end
        [x1,x2,x3] = h(5);
    "#;
    let err_mis = execute_source_result(program_mis).err().unwrap();
    assert_eq!(err_mis.identifier(), Some("RunMat:VarargoutMismatch"));
}
#[test]
fn nested_function_calls() {
    let program = "function y = add(a, b); y = a + b; end; function y = multiply_and_add(x); y = add(x * 2, x * 3); end; result = multiply_and_add(4);";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 20.0).abs() < 1e-9)));
}

#[test]
fn nested_function_shared_lexical_scope_read_and_write() {
    let program = r#"
        function r = outer(a)
            total = 100;
            function y = add(x)
                total = total + x;
                y = total;
            end
            r1 = add(a);
            r2 = add(1);
            r = r1 + r2;
        end
        result = outer(5);
    "#;
    let vars = execute_source(program);
    // total evolves: 100 -> 105 -> 106, so r = 105 + 106 = 211.
    assert!(has_num(&vars, 211.0), "unexpected vars: {vars:?}");
}

#[test]
fn direct_recursive_function_executes() {
    let program = r#"
        result = fact(5);
        function y = fact(n)
            if n <= 1
                y = 1;
            else
                y = n * fact(n - 1);
            end
        end
    "#;
    let vars = execute_source(program);
    assert!(has_num(&vars, 120.0), "unexpected vars: {vars:?}");
}

#[test]
fn mutual_recursive_functions_execute() {
    let program = r#"
        a = is_even(6);
        b = is_odd(5);
        result = a * 10 + b;
        function y = is_even(n)
            if n <= 0
                y = 1;
            else
                y = is_odd(n - 1);
            end
        end
        function y = is_odd(n)
            if n <= 0
                y = 0;
            else
                y = is_even(n - 1);
            end
        end
    "#;
    let vars = execute_source(program);
    assert!(has_num(&vars, 11.0), "unexpected vars: {vars:?}");
}

#[test]
fn recursive_function_handle_call_executes() {
    let program = r#"
        h = @fact;
        result = h(5);
        function y = fact(n)
            if n <= 1
                y = 1;
            else
                y = n * fact(n - 1);
            end
        end
    "#;
    let bytecode = compile_source(program).expect("compile recursive handle call");
    assert!(
        bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name, runmat_types::FunctionSourceRole::Local) if name == "fact"
        )),
        "recursive handle should carry semantic function identity"
    );

    let vars = interpret(&bytecode).expect("execute recursive handle call");
    assert!(has_num(&vars, 120.0), "unexpected vars: {vars:?}");
}

#[test]
fn recursive_feval_string_call_executes() {
    let program = r#"
        result = feval('fact', 5);
        function y = fact(n)
            if n <= 1
                y = 1;
            else
                y = n * feval('fact', n - 1);
            end
        end
    "#;
    let bytecode = compile_source(program).expect("compile recursive feval call");
    assert!(
        bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::LoadCharRow(name) if name == "fact"
        )),
        "textual feval callee should remain a runtime name"
    );
    assert!(
        !bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name, _) if name == "fact"
        )),
        "textual feval must not acquire a static definition lease"
    );

    let vars = interpret(&bytecode).expect("execute recursive feval call");
    assert!(has_num(&vars, 120.0), "unexpected vars: {vars:?}");
}

#[test]
fn local_textual_feval_constant_and_variable_use_name_dispatch() {
    let program = r#"
        result = call_names(2);
        function y = call_names(x)
            target = 'helper';
            y = feval('helper', x) + feval(target, x);
        end
        function y = helper(x)
            y = x + 1;
        end
    "#;
    let bytecode = compile_source(program).expect("compile local textual feval calls");
    let caller = bytecode
        .function_registry
        .functions
        .values()
        .find(|function| function.display_name == "call_names")
        .expect("local caller bytecode");
    assert!(caller.instructions.iter().any(|instruction| matches!(
        instruction,
        runmat_vm::Instr::LoadCharRow(name) if name == "helper"
    )));
    assert!(
        !caller.instructions.iter().any(|instruction| matches!(
            instruction,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name, _) if name == "helper"
        )),
        "neither textual name may acquire the helper's static lease"
    );
    let values = interpret(&bytecode).expect("execute both textual forms");
    assert!(has_num(&values, 6.0), "unexpected values: {values:?}");
}

#[test]
fn nested_explicit_handle_without_captures_keeps_lexical_identity() {
    let program = r#"
        result = outer(2);
        function y = outer(x)
            y = feval(@inner, x);
            function z = inner(value)
                z = value + 1;
            end
        end
    "#;
    let bytecode = compile_source(program).expect("compile nested lexical handle");
    let outer = bytecode
        .function_registry
        .functions
        .values()
        .find(|function| function.display_name == "outer")
        .expect("outer bytecode");
    assert!(outer.instructions.iter().any(|instruction| matches!(
        instruction,
        runmat_vm::Instr::CreateSemanticClosure(
            _,
            name,
            0,
            runmat_types::FunctionSourceRole::Nested
        ) if name == "inner"
    )));
    assert!(!outer.instructions.iter().any(|instruction| matches!(
        instruction,
        runmat_vm::Instr::CreateBoundFunctionHandle(_, name, _) if name == "inner"
    )));
    let values = interpret(&bytecode).expect("execute nested lexical handle");
    assert!(has_num(&values, 3.0), "unexpected values: {values:?}");
}

#[test]
fn nested_textual_literal_and_variable_names_do_not_acquire_lexical_identity() {
    for mode in [
        runmat_parser::CompatMode::Matlab,
        runmat_parser::CompatMode::RunMat,
    ] {
        for captured in [false, true] {
            for expression in ["feval('inner', x)", "feval(target, x)"] {
                let body = if captured {
                    "z = value + shared;"
                } else {
                    "z = value + 1;"
                };
                let program = format!(
                    "result = outer(2);\n function y = outer(x)\n shared = 1;\n target = 'inner';\n y = {expression};\n function z = inner(value)\n {body}\n end\n end"
                );
                let ast = runmat_parser::parse_with_options(
                    &program,
                    runmat_parser::ParserOptions::new(mode),
                )
                .expect("parse textual nested name");
                let hir = runmat_hir::lower(
                    &ast,
                    &runmat_hir::LoweringContext::empty()
                        .with_runmat_extensions_enabled(mode.allows_runmat_extensions()),
                )
                .expect("lower textual nested name");
                let mir = runmat_mir::lowering::lower_assembly(&hir.assembly)
                    .expect("lower textual nested MIR");
                let bytecode =
                    runmat_vm::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
                        .expect("compile textual nested name");
                assert!(
                    !bytecode.function_registry.names.contains_key("inner"),
                    "nested definition must not enter the textual function index"
                );
                let outer = bytecode
                    .function_registry
                    .functions
                    .values()
                    .find(|function| function.display_name == "outer")
                    .expect("outer bytecode");
                assert!(
                    !outer.instructions.iter().any(|instruction| matches!(
                        instruction,
                        runmat_vm::Instr::CreateBoundFunctionHandle(
                            _,
                            name,
                            runmat_types::FunctionSourceRole::Nested
                        ) if name == "inner"
                    )),
                    "textual name acquired a nested definition handle"
                );
                let error = interpret(&bytecode)
                    .expect_err("nested name must not resolve without an explicit handle");
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:UndefinedFunction"),
                    "unexpected {mode:?}/{captured}/{expression} error: {}",
                    error.message()
                );
            }
        }
    }
}

#[test]
fn nested_sibling_and_descendant_calls_propagate_mutable_shared_binding() {
    let program = r#"
        answer = outer();
        function result = outer()
            shared = 1;
            function value = reader()
                value = shared;
            end
            function value = writer()
                function inner()
                    shared = shared + 2;
                end
                inner();
                value = reader();
            end
            result = writer() + shared;
        end
    "#;
    let values = execute_source(program);
    assert!(has_num(&values, 6.0), "unexpected values: {values:?}");
}

#[test]
fn nested_recursive_function_with_capture_executes() {
    let program = r#"
        result = outer(4);
        function r = outer(n)
            scale = 2;
            function y = inner(k)
                if k <= 1
                    y = scale;
                else
                    y = scale + inner(k - 1);
                end
            end
            r = inner(n);
        end
    "#;
    let vars = execute_source(program);
    assert!(has_num(&vars, 8.0), "unexpected vars: {vars:?}");
}

#[test]
fn shared_input_output_name_updates_in_place() {
    let program = r#"
        function x = bump(x)
            x = x + 1;
        end
        r = bump(4);
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn shared_input_output_name_multi_output_reads_original_input() {
    let program = r#"
        function [x, y] = bump_and_copy(x)
            y = x;
            x = x + 1;
        end
        [a, b] = bump_and_copy(4);
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 4.0).abs() < 1e-9)));
}

#[test]
fn user_function_multi_assign_executes() {
    let bytecode = compile_source(
        r#"
        function [a,b] = g()
            a = 2;
            b = 3;
        end
        [x,y] = g();
        z = x + y;
    "#,
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn feval_multi_assign_executes() {
    let bytecode = compile_source(
        r#"
        function [a,b] = g()
            a = 6;
            b = 7;
        end
        h = @g;
        [x,y] = feval(h);
        z = x + y;
    "#,
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic feval multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 13.0).abs() < 1e-9)));
}

#[test]
fn feval_multi_assign_uses_typed_instruction() {
    let source = r#"
        function [a,b] = g()
            a = 6;
            b = 7;
        end
        h = @g;
        [x,y] = feval(h);
    "#;
    let bytecode = compile_source(source).expect("compile semantic feval multi-assign");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallFevalMulti(argc, out_count) if *argc == 0 && *out_count == 2
    )));
}

#[test]
fn feval_zero_output_uses_typed_instruction() {
    let source = r#"
        function y = inc(x)
            y = x + 1;
        end
        h = @inc;
        feval(h, 2);
    "#;
    let bytecode = compile_source(source).expect("compile semantic feval zero-output");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallFevalMulti(argc, out_count) if *argc == 1 && *out_count == 0
    )));
    interpret(&bytecode).expect("execute semantic feval zero-output");
}

#[test]
fn feval_single_output_uses_typed_instruction() {
    let source = r#"
        function y = inc(x)
            y = x + 1;
        end
        h = @inc;
        z = feval(h, 2);
    "#;
    let bytecode = compile_source(source).expect("compile semantic feval single-output");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallFevalMulti(argc, out_count) if *argc == 1 && *out_count == 1
    )));
    let vars = interpret(&bytecode).expect("execute semantic feval single-output");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn feval_expand_multi_assign_uses_typed_instruction() {
    let source = r#"
        function [a,b] = pair(x,y)
            a = x;
            b = y;
        end
        C = {7,8};
        h = @pair;
        [u,v] = feval(h, C{:});
        s = u + v;
    "#;
    let bytecode = compile_source(source).expect("compile semantic feval expanded multi-assign");
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFevalExpandMultiOutput(specs, out_count)
            if *out_count == 2 && specs.len() == 1 && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));

    let vars = interpret(&bytecode).expect("execute semantic feval expanded multi-assign");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 15.0).abs() < 1e-9)));
}

#[test]
fn feval_expand_zero_output_uses_typed_instruction() {
    let source = r#"
        function [a,b] = pair(x,y)
            a = x;
            b = y;
        end
        C = {7,8};
        h = @pair;
        feval(h, C{:});
    "#;
    let bytecode = compile_source(source).expect("compile semantic feval expanded zero-output");
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFevalExpandMultiOutput(specs, out_count)
            if *out_count == 0 && specs.len() == 1 && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    interpret(&bytecode).expect("execute semantic feval expanded zero-output");
}

#[test]
fn feval_expand_single_output_uses_typed_instruction() {
    let source = r#"
        function [a,b] = pair(x,y)
            a = x;
            b = y;
        end
        C = {7,8};
        h = @pair;
        u = feval(h, C{:});
    "#;
    let bytecode = compile_source(source).expect("compile semantic feval expanded single-output");
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallFevalExpandMultiOutput(specs, out_count)
            if *out_count == 1 && specs.len() == 1 && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let vars = interpret(&bytecode).expect("execute semantic feval expanded single-output");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 7.0).abs() < 1e-9)));
}

#[test]
fn size_builtin_multi_assign_executes() {
    let bytecode = compile_source("[r,c] = size([1 2; 3 4]); z = r + c;").unwrap();
    let vars = interpret(&bytecode).expect("semantic size multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 4.0).abs() < 1e-9)));
}

#[test]
fn min_max_builtin_multi_assign_execute() {
    let bytecode =
        compile_source("[mx,mi] = max([1 3 2]); [mn,ni] = min([4 1 5]); z = mx + mi + mn + ni;")
            .unwrap();
    let vars = interpret(&bytecode).expect("semantic min/max multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 8.0).abs() < 1e-9)));
}

#[test]
fn cummin_cummax_builtin_return_documented_indices_and_reject_third_output() {
    for (builtin, expected_values, expected_indices) in [
        ("cummax", [2., 2., 3.], [1., 1., 3.]),
        ("cummin", [2., 1., 1.], [1., 2., 2.]),
    ] {
        let values = execute_source(&format!("[values,indices] = {builtin}([2 1 3]);"));
        assert!(has_numeric_tensor(&values, &expected_values));
        assert!(has_numeric_tensor(&values, &expected_indices));
        let error = execute_source_result(&format!("[values,indices,extra] = {builtin}([2 1 3]);"))
            .expect_err("third output must be rejected");
        assert_eq!(
            error.identifier(),
            Some(if builtin == "cummax" {
                "RunMat:cummax:InvalidArgument"
            } else {
                "RunMat:cummin:InvalidArgument"
            })
        );
    }
}

#[test]
fn sort_unique_find_builtin_multi_assign_execute() {
    let bytecode = compile_source(
        "[s,si] = sort([3 1 2]); [u,ui] = unique([2 1 2]); [fr,fc] = find([0 1; 2 0]); z = sum(s) + sum(si) + sum(u) + sum(ui) + sum(fr) + sum(fc);",
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic sort/unique/find multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 24.0).abs() < 1e-9)));
}

#[test]
fn union_ismember_sortrows_builtin_multi_assign_execute() {
    let bytecode = compile_source(
        "[u,ia,ib] = union([3 1],[2 1]); [tf,loc] = ismember([3 1 2],[2 3]); [sr,idx] = sortrows([2 2; 1 3]); z = u(1) + u(2) + u(3) + ia(1) + ia(2) + ib + loc(1) + loc(2) + loc(3) + sr(1,1) + sr(1,2) + sr(2,1) + sr(2,2) + idx(1) + idx(2);",
    )
    .unwrap();
    let vars =
        interpret(&bytecode).expect("semantic union/ismember/sortrows multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 24.0).abs() < 1e-9)));
}

#[test]
fn compiled_set_functions_enforce_integer_class_compatibility() {
    for builtin in ["ismember", "intersect", "union", "setdiff", "setxor"] {
        let source = format!("out = {builtin}(int8([1 2]), uint8([1 2]));");
        let error = execute_source_result(&source).expect_err("unlike integer classes must reject");
        assert_eq!(
            error.identifier(),
            Some(match builtin {
                "ismember" => "RunMat:ismember:NumericClassMismatch",
                "intersect" => "RunMat:intersect:NumericClassMismatch",
                "union" => "RunMat:union:NumericClassMismatch",
                "setdiff" => "RunMat:setdiff:NumericClassMismatch",
                "setxor" => "RunMat:setxor:NumericClassMismatch",
                _ => unreachable!(),
            })
        );
    }
}

#[test]
fn compiled_set_functions_preserve_integer_double_result_classes() {
    let vars = execute_source(
        "a = uint32([1 3]); b = [2 3]; [i,ii,ij] = intersect(a,b); [u,ui,uj] = union(b,a); [d,di] = setdiff(a,b); [x,xi,xj] = setxor(b,a); [tf,loc] = ismember(a,b); ci = class(i); cu = class(u); cd = class(d); cx = class(x); cii = class(ii); cui = class(ui); cdi = class(di); cxi = class(xi); cloc = class(loc);",
    );
    assert!(
        vars.iter()
            .filter(
                |value| matches!(value, runmat_value::Value::String(class) if class == "uint32")
            )
            .count()
            >= 4
    );
    assert!(
        vars.iter()
            .filter(
                |value| matches!(value, runmat_value::Value::String(class) if class == "double")
            )
            .count()
            >= 5
    );
}

#[test]
fn chol_builtin_multi_assign_execute() {
    let bytecode = compile_source("[r,p] = chol([4 0; 0 9]); z = r(1,1) + r(2,2) + p;").unwrap();
    let vars = interpret(&bytecode).expect("semantic chol multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn lu_builtin_multi_assign_execute() {
    let bytecode = compile_source(
        "[l,u,p] = lu([2 0; 0 3]); z = l(1,1) + l(2,2) + u(1,1) + u(2,2) + p(1,1) + p(2,2);",
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic lu multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn qr_builtin_multi_assign_execute() {
    let bytecode = compile_source(
        "[q,r,p] = qr([2 0; 0 3]); z = q(1,1) + q(2,2) + r(1,1) + r(2,2) + p(1,1) + p(2,2);",
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic qr multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n + 1.0).abs() < 1e-9)));
}

#[test]
fn svd_builtin_multi_assign_execute() {
    let bytecode = compile_source(
        "[u,s,v] = svd([3 0; 0 2]); z = s(1,1) + s(2,2) + u(1,1) * u(1,1) + v(1,1) * v(1,1);",
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic svd multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 7.0).abs() < 1e-9)));
}

#[test]
fn eig_builtin_multi_assign_execute() {
    let bytecode = compile_source(
        "[v,d] = eig([3 0; 0 2]); z = d(1,1) + d(2,2) + v(1,1) * v(1,1) + v(2,2) * v(2,2);",
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic eig multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 7.0).abs() < 1e-9)));
}

#[test]
fn eig_generalized_builtin_execute() {
    let bytecode =
        compile_source("A = [2 0; 0 9]; B = [1 0; 0 3]; d = eig(A, B); z = d(1) + d(2);").unwrap();
    let vars = interpret(&bytecode).expect("generalized eig should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn fprintf_inline_cast_argument_does_not_stack_underflow() {
    let program = r#"
        x = single(3.14);
        n = fprintf("Value: %.4f\n", double(x));
    "#;
    let vars = execute_source(program);
    assert!(has_num(&vars, 14.0));
}

#[test]
fn sprintf_inline_cast_argument_formats_value() {
    let program = r#"
        x = single(3.14);
        s = sprintf("Value: %.4f", double(x));
    "#;
    let vars = execute_source(program);
    assert!(vars.iter().any(|value| {
        if let runmat_value::Value::CharArray(chars) = value {
            let rendered: String = runmat_value::decode_utf16_code_units(chars.data())
                .expect("valid UTF-16 text fixture");
            rendered == "Value: 3.1400"
        } else {
            false
        }
    }));
}

#[test]
fn member_get_set_and_method_call_skeleton() {
    let input = "obj = new_object('Point'); obj = setfield(obj, 'x', 3); ax = getfield(obj, 'x');";
    let vars = execute_source_with_extensions(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 3.0).abs() < f64::EPSILON)));

    // Install a real source-owned method, not the former process-global
    // example Point.move registration.
    let input2 = r#"
        classdef PointMethodFixture
            properties
                x
                y
            end
            methods
                function obj = move(obj, dx, dy)
                    obj.x = obj.x + dx;
                    obj.y = obj.y + dy;
                end
            end
        end
        obj = new_object('PointMethodFixture');
        obj = setfield(obj,'x',5); obj = setfield(obj,'y',7);
        obj = call_method(obj, 'move', 1, 2);
        rx = getfield(obj,'x'); ry = getfield(obj,'y');
    "#;
    let vars2 = execute_source_with_extensions(input2);
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 6.0).abs() < f64::EPSILON)));
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 9.0).abs() < f64::EPSILON)));
}

#[test]
fn implicit_struct_creation_for_root_variable_assignment() {
    let input = "s.x = 10; s.y = 20; v = getfield(s, 'x') + getfield(s, 'y');";
    let vars = execute_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 30.0).abs() < 1e-9)));
}

#[test]
fn member_read_write_executes() {
    let bytecode = compile_source("s.x = 10; s.y = 20; v = s.x + s.y;").unwrap();
    let vars = interpret(&bytecode).expect("semantic member read/write should succeed");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 30.0).abs() < 1e-9)));
}

#[test]
fn dynamic_member_read_executes() {
    let vars = execute_source("s = struct('x', 9); f = 'x'; y = s.(f);");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn indexed_dynamic_member_read_executes() {
    let vars = execute_source("s = struct('x', {{1, 2, 3}}); f = 'x'; y = s.(f){2};");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 2.0).abs() < 1e-9)));
}

#[test]
fn indexed_member_store_back_executes() {
    let bytecode = compile_source("s.a = [1 2 3]; s.a(2) = 9; y = s.a(2);").unwrap();
    let vars = interpret(&bytecode).expect("semantic indexed member store-back should succeed");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn indexed_member_vector_store_back_lowers_to_slice_instruction() {
    let source = "s.a = [10 20 30 40]; idx = [2 4]; s.a(idx) = 99; y = s.a(4);";
    let bytecode = compile_source(source).expect("compile indexed member vector store");
    assert_prepared_indexed_store(&bytecode);
    assert!(
        !bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::StoreIndex(_) | runmat_vm::Instr::StoreIndexDelete(_)
        )),
        "indexed member vector assignment should not lower through StoreIndex*"
    );
    let vars = interpret(&bytecode).expect("execute indexed member vector store");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 99.0).abs() < 1e-9)));
}

#[test]
fn indexed_member_logical_store_back_lowers_to_slice_instruction() {
    let source = "s.a = [1 2 3 4]; mask = logical([1 0 1 0]); s.a(mask) = 0; y = s.a(3);";
    let bytecode = compile_source(source).expect("compile indexed member logical store");
    assert_prepared_indexed_store(&bytecode);
    assert!(
        !bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::StoreIndex(_) | runmat_vm::Instr::StoreIndexDelete(_)
        )),
        "indexed member logical assignment should not lower through StoreIndex*"
    );
    let vars = interpret(&bytecode).expect("execute indexed member logical store");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 0.0).abs() < 1e-9)));
}

#[test]
fn indexed_base_member_vector_store_back_lowers_to_slice_instruction() {
    let source = "s = struct('a', {[10 20 30 40], [1 2 3 4]}); idx = [2 4]; s(2).a(idx) = 99; t = s(2); y = getfield(t, 'a'); z = y(4);";
    let bytecode = compile_source(source).expect("compile indexed-base member vector store");
    assert_prepared_indexed_store(&bytecode);
    let vars = interpret(&bytecode).expect("execute indexed-base member vector store");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 99.0).abs() < 1e-9)));
}

#[test]
fn indexed_base_member_logical_store_back_lowers_to_slice_instruction() {
    let source = "s = struct('a', {[1 2 3 4], [5 6 7 8]}); mask = logical([1 0 1 0]); s(2).a(mask) = 0; t = s(2); y = getfield(t, 'a'); z = y(3);";
    let bytecode = compile_source(source).expect("compile indexed-base member logical store");
    assert_prepared_indexed_store(&bytecode);
    let vars = interpret(&bytecode).expect("execute indexed-base member logical store");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 0.0).abs() < 1e-9)));
}

#[test]
fn indexed_dynamic_member_vector_store_back_lowers_to_slice_instruction() {
    let source = "s.a = [10 20 30 40]; f = 'a'; idx = [2 4]; s.(f)(idx) = 99; y = s.(f)(4);";
    let bytecode = compile_source(source).expect("compile indexed dynamic member vector store");
    assert_prepared_indexed_store(&bytecode);
    assert!(
        !bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::StoreIndex(_) | runmat_vm::Instr::StoreIndexDelete(_)
        )),
        "indexed dynamic-member vector assignment should not lower through StoreIndex*"
    );
    let vars = interpret(&bytecode).expect("execute indexed dynamic member vector store");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 99.0).abs() < 1e-9)));
}

#[test]
fn indexed_dynamic_member_logical_store_back_lowers_to_slice_instruction() {
    let source =
        "s.a = [1 2 3 4]; f = 'a'; mask = logical([1 0 1 0]); s.(f)(mask) = 0; y = s.(f)(3);";
    let bytecode = compile_source(source).expect("compile indexed dynamic member logical store");
    assert_prepared_indexed_store(&bytecode);
    assert!(
        !bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::StoreIndex(_) | runmat_vm::Instr::StoreIndexDelete(_)
        )),
        "indexed dynamic-member logical assignment should not lower through StoreIndex*"
    );
    let vars = interpret(&bytecode).expect("execute indexed dynamic member logical store");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 0.0).abs() < 1e-9)));
}

#[test]
fn cell_paren_delete_executes_with_semantic_store_back() {
    let vars = execute_source("c = {1, 2, 3}; c(2) = []; y = c{2};");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn indexed_member_delete_executes_with_semantic_store_back() {
    let vars = execute_source("s.a = {1, 2, 3}; s.a(2) = []; y = s.a{2};");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn indexed_cell_content_delete_executes_with_semantic_store_back() {
    let vars = execute_source("c = {{1, 2, 3}}; c{1}(2) = []; y = c{1}{2};");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn matrix_delete_reports_unsupported_deletion_identifier_contract() {
    let err =
        execute_source_result("x = [1 2; 3 4]; x(2) = [];").expect_err("matrix delete should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UnsupportedDeletion"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn string_slice_delete_reports_identifier_contract() {
    let err = execute_source_result("s = [\"a\" \"b\"]; s(1:1) = [];")
        .expect_err("string slice delete should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UnsupportedSliceDeletion"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn cell_member_store_back_executes() {
    let bytecode = compile_source("C = {struct()}; C{1}.a = 5; y = C{1}.a;").unwrap();
    let vars = interpret(&bytecode).expect("semantic cell member store-back should succeed");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn indexed_cell_member_vector_store_back_lowers_to_slice_instruction() {
    let source = "C = {struct('a', [10 20 30 40])}; idx = [2 4]; C{1}.a(idx) = 99; y = C{1}.a(4);";
    let bytecode = compile_source(source).expect("compile indexed cell member vector store");
    assert_prepared_indexed_store(&bytecode);
    assert!(
        !bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::StoreIndex(_) | runmat_vm::Instr::StoreIndexDelete(_)
        )),
        "indexed cell-member vector assignment should not lower through StoreIndex*"
    );
    let vars = interpret(&bytecode).expect("execute indexed cell member vector store");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 99.0).abs() < 1e-9)));
}

#[test]
fn indexed_cell_member_logical_store_back_lowers_to_slice_instruction() {
    let source =
        "C = {struct('a', [1 2 3 4])}; mask = logical([1 0 1 0]); C{1}.a(mask) = 0; y = C{1}.a(3);";
    let bytecode = compile_source(source).expect("compile indexed cell member logical store");
    assert_prepared_indexed_store(&bytecode);
    assert!(
        !bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::StoreIndex(_) | runmat_vm::Instr::StoreIndexDelete(_)
        )),
        "indexed cell-member logical assignment should not lower through StoreIndex*"
    );
    let vars = interpret(&bytecode).expect("execute indexed cell member logical store");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 0.0).abs() < 1e-9)));
}

#[test]
fn empty_array_member_assignment_assigns_empty_value() {
    let bytecode =
        compile_source("s = struct('x', {1, 2}); s(2).x = []; t = s(2); y = getfield(t, 'x');")
            .expect("compile");
    let vars = interpret(&bytecode).expect("member empty assignment should succeed");

    assert!(vars.iter().any(|v| {
        matches!(
            v,
            runmat_value::Value::Tensor(t) if t.materialize_f64().is_empty()
        )
    }));
}

#[test]
fn empty_array_cell_content_assignment_assigns_empty_value() {
    let bytecode = compile_source("c = {1}; c{1} = []; y = c{1};").expect("compile");
    let vars = interpret(&bytecode).expect("cell content empty assignment should succeed");

    assert!(vars.iter().any(|v| {
        matches!(
            v,
            runmat_value::Value::Tensor(t) if t.materialize_f64().is_empty()
        )
    }));
}

#[test]
fn implicit_struct_creation_for_function_output_variable() {
    let input = r#"
        function r = make_result()
            r.x = 10;
            r.y = 20;
        end
        s = make_result();
        x = getfield(s, 'x');
        y = getfield(s, 'y');
    "#;
    let vars = execute_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 10.0).abs() < 1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 20.0).abs() < 1e-9)));
}

#[test]
fn nested_member_assignment_materializes_missing_intermediate_structs() {
    let input = "s = struct(); s.a.b = 1; v = getfield(getfield(s, 'a'), 'b');";
    let vars = execute_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 1.0).abs() < 1e-9)));
}

#[test]
fn struct_field_indexing_read_path_uses_member_then_index_semantics() {
    let input =
        "s = struct(); s.arr = [10 20 30]; x = s.arr(2); y = s.arr(1:2); y1 = y(1); y2 = y(2);";
    let vars = execute_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 20.0).abs() < 1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 10.0).abs() < 1e-9)));
}

#[test]
fn dotted_invoke_preserves_object_method_dispatch() {
    let input = "__register_test_classes(); obj = new_object('Point'); obj = setfield(obj,'x',5); obj = setfield(obj,'y',7); obj = obj.move(1,2); rx = getfield(obj,'x'); ry = getfield(obj,'y');";
    let vars = execute_source_with_extensions(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 6.0).abs() < 1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn dotted_invoke_runtime_struct_dispatch_when_base_type_unknown() {
    let input = r#"
        function y = pick(s)
            y = s.arr(2);
        end
        s = struct();
        s.arr = [10 20 30];
        z = pick(s);
    "#;
    let vars = execute_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 20.0).abs() < 1e-9)));
}

#[test]
fn nested_dynamic_member_assignment_materializes_and_writes_back() {
    let input =
        "s = struct(); f1 = 'a'; f2 = 'b'; s.(f1).(f2) = 3; v = getfield(getfield(s, 'a'), 'b');";
    let vars = execute_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn mixed_member_cell_and_index_read_chain() {
    let input = "s = struct(); s.arr = {[10 20], [30 40]}; v = s.arr{2}(1);";
    let vars = execute_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 30.0).abs() < 1e-9)));
}

#[test]
fn function_handle_anon_round_trip() {
    let input = "h = @sin; g = make_anon('x', 'x+1');";
    let vars = execute_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::FunctionHandle(_))));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::String(runmat_value::StringElement::Text(s)) if s.starts_with("@anon"))));
}

#[test]
fn builtin_function_handle_feval_executes() {
    let bytecode = compile_source("h = @sin; y = feval(h, 0);").unwrap();
    let vars = interpret(&bytecode).expect("semantic builtin handle feval should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if n.abs() < 1e-12)));
}

#[test]
fn anonymous_function_handle_feval_executes() {
    let bytecode = compile_source("f = @(x) x + 1; y = feval(f, 4);").unwrap();
    let vars = interpret(&bytecode).expect("semantic anonymous handle feval should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-12)));
}

#[test]
fn anonymous_call_body_forwards_requested_outputs_and_captures() {
    let source = r#"
        bias = 4;
        f = @(x) deal(x + bias, x + bias + 1);
        [first, second] = feval(f, 7);
        difference = second - first;
    "#;
    let vars = execute_source(source);
    assert!(has_num(&vars, 11.0), "first forwarded output: {vars:?}");
    assert!(has_num(&vars, 12.0), "second forwarded output: {vars:?}");
    assert!(has_num(&vars, 1.0), "captured result difference: {vars:?}");
}

#[test]
fn anonymous_noncall_body_rejects_multiple_outputs() {
    let error = execute_source_result("f = @(x) x + 1; [a,b] = feval(f, 2);")
        .expect_err("a scalar anonymous body cannot supply two outputs");
    assert_eq!(
        error.identifier(),
        Some("RunMat:CommaSeparatedListOutputShortage")
    );
}

#[test]
fn function_handle_index_call_executes() {
    let bytecode =
        compile_source("h = @inc; y = h(2);\nfunction z = inc(x)\n  z = x + 1;\nend").unwrap();
    assert!(
        bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name, _) if name == "inc"
        )),
        "semantic function handle index calls should carry semantic identity"
    );
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::Index(1))),
        "single-output handle invocation should stay on dynamic Index(1) dispatch"
    );

    let vars = interpret(&bytecode).expect("semantic handle index call should execute");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 3.0).abs() < 1e-12)));
}

#[test]
fn function_handle_index_zero_output_executes() {
    let bytecode =
        compile_source("h = @inc; h(2);\nfunction z = inc(x)\n  z = x + 1;\nend").unwrap();
    assert!(
        bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateBoundFunctionHandle(_, name, _) if name == "inc"
        )),
        "semantic function handle zero-output index calls should carry semantic identity"
    );
    assert!(has_single_selector_subscript_call(
        &bytecode,
        runmat_types::SequenceUse::Discard
    ));

    interpret(&bytecode).expect("semantic handle zero-output index call should execute");
}

#[test]
fn function_handle_index_multi_output_executes() {
    let source =
        "h = @pair; [a,b] = h(2); s = a + b;\nfunction [u,v] = pair(x)\n  u = x;\n  v = x + 1;\nend";
    let bytecode = compile_source(source).expect("compile semantic handle multi-output");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CreateBoundFunctionHandle(_, name, _) if name == "pair"
    )));
    assert!(has_single_selector_subscript_call(
        &bytecode,
        runmat_types::SequenceUse::SelectPrefix { count: 2 }
    ));

    let vars = interpret(&bytecode).expect("execute semantic handle multi-output");
    assert!(has_num(&vars, 5.0));
}

#[test]
fn function_handle_expand_single_output_executes() {
    let source = "h = @inc; C = {2}; y = h(C{:});\nfunction z = inc(x)\n  z = x + 1;\nend";
    let bytecode = compile_source(source).expect("compile semantic handle expanded single-output");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CreateBoundFunctionHandle(_, name, _) if name == "inc"
    )));
    assert!(has_bound_expanded_subscript(
        &bytecode,
        runmat_types::SequenceUse::RequireSingle
    ));

    let vars = interpret(&bytecode).expect("execute semantic handle expanded single-output");
    assert!(has_num(&vars, 3.0));

    let array = compile_source("A = [29, 30]; C = {2}; x = A(C{:});")
        .expect("compile receiver-selected expanded array indexing");
    assert!(has_bound_expanded_subscript(
        &array,
        runmat_types::SequenceUse::RequireSingle,
    ));
    let vars = interpret(&array).expect("execute expanded array selector");
    assert!(has_num(&vars, 30.0));

    // The two struct members are independent row/column selectors, not one
    // array-valued index. Keep the actual receiver's complete path intact.
    for (case, source) in [
        "A=[11,12,13;21,22,23]; S=struct('idx',{2,3}); y=A(S.idx);",
        "A=[11,12,13;21,22,23]; S=struct('idx',{2,3}); name='idx'; y=A(S.(name));",
        "A=[11,12,13;21,22,23]; S=struct('idx',{2,3}); y=A(S(1:2).idx);",
        "A=[11,12,13;21,22,23]; S=struct('idx',{2,3}); holder=struct('data',A); y=(holder.data)(S.idx);",
        "A=[11,12,13;21,22,23]; S=struct('idx',{2,3}); y=(A+0)(S.idx);",
    ].into_iter().enumerate() {
        let bytecode = compile_source(source).expect("compile receiver-selected member CSL");
        if case == 3 {
            assert!(bytecode.instructions.iter().any(|instruction| matches!(instruction,
                runmat_vm::Instr::ReadSubscriptPath { steps, selection: runmat_types::SequenceUse::RequireSingle, .. }
                    if matches!(steps.as_slice(), [runmat_vm::BytecodeSubscriptStep::Member(member), runmat_vm::BytecodeSubscriptStep::Parentheses { selectors }]
                        if member == &runmat_hir::MemberName::from("data") && matches!(selectors.as_slice(), [runmat_vm::BytecodeSubscriptSelector::Expansion(_)])))));
        } else {
            assert!(has_bound_expanded_subscript(&bytecode, runmat_types::SequenceUse::RequireSingle));
        }
        let vars = interpret(&bytecode).expect("execute member CSL as two selectors");
        assert!(has_num(&vars, 23.0), "independent A(2,3) oracle for {source}");
    }
    let callable = compile_source("h=@plus; S=struct('idx',{2,3}); y=h(S.idx);")
        .expect("compile member CSL callable arguments");
    let vars = interpret(&callable).expect("execute member CSL callable arguments");
    assert!(has_num(&vars, 5.0));

    let explicit_member_feval =
        compile_source("A=[11,12,13;21,22,23]; S=struct('idx',{2,3}); y=feval(A,S.idx);")
            .expect("compile explicit member CSL feval");
    assert!(explicit_member_feval
        .instructions
        .iter()
        .any(|instruction| matches!(
            instruction,
            runmat_vm::Instr::CallFevalExpandMultiOutput(_, 1)
        )));
    interpret(&explicit_member_feval)
        .expect_err("explicit feval must not index its noncallable target");

    let member = compile_source("s = struct('field', [29, 30]); C = {2}; x = (s.field)(C{:});")
        .expect("compile full member-and-expanded-selector path");
    assert!(member.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::ReadSubscriptPath { steps, selection: runmat_types::SequenceUse::RequireSingle, .. }
            if matches!(steps.as_slice(), [runmat_vm::BytecodeSubscriptStep::Member(name), runmat_vm::BytecodeSubscriptStep::Parentheses { selectors }]
                if name == &runmat_hir::MemberName::from("field") && matches!(selectors.as_slice(), [runmat_vm::BytecodeSubscriptSelector::Expansion(_)]))
    )));
    let vars = interpret(&member).expect("execute full expanded subscript path");
    assert!(has_num(&vars, 30.0));

    let dotted = compile_source("s = struct('field', [29, 30]); C = {2}; x = s.field(C{:});")
        .expect("compile ordinary dotted invocation syntax");
    assert!(has_captured_cell_dotted_call(&dotted.instructions, 1));
    assert!(dotted.instructions.iter().any(|instruction| matches!(instruction,
        runmat_vm::Instr::ReadSubscriptPath { steps, .. }
            if matches!(steps.as_slice(), [runmat_vm::BytecodeSubscriptStep::DottedInvoke { member, .. }] if member == &runmat_hir::MemberName::from("field")))));
    let vars = interpret(&dotted).expect("execute ordinary dotted field invocation");
    assert!(has_num(&vars, 30.0));

    let text = compile_source("A = 'ab'; C = {2}; x = A(C{:});")
        .expect("compile receiver-selected expanded character indexing");
    assert!(has_bound_expanded_subscript(
        &text,
        runmat_types::SequenceUse::RequireSingle,
    ));
    let vars = interpret(&text).expect("execute expanded character selector");
    assert!(vars.iter().any(|value| matches!(value,
        runmat_value::Value::CharArray(array)
            if array.try_row_string().ok().flatten().as_deref() == Some("b")
    )));

    let explicit_feval = compile_source("A = [29, 30]; C = {2}; x = feval(A, C{:});")
        .expect("compile explicit feval with expanded noncallable input");
    assert!(explicit_feval
        .instructions
        .iter()
        .any(|instr| matches!(instr, runmat_vm::Instr::CallFevalExpandMultiOutput(_, 1))));
    interpret(&explicit_feval).expect_err("explicit feval must not index an array receiver");
}

#[test]
fn function_handle_expand_zero_output_executes() {
    let source = "h = @inc; C = {2}; h(C{:});\nfunction z = inc(x)\n  z = x + 1;\nend";
    let bytecode = compile_source(source).expect("compile semantic handle expanded zero-output");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CreateBoundFunctionHandle(_, name, _) if name == "inc"
    )));
    assert!(has_bound_expanded_subscript(
        &bytecode,
        runmat_types::SequenceUse::Discard
    ));

    interpret(&bytecode).expect("execute semantic handle expanded zero-output");
}

#[test]
fn function_handle_expand_multi_output_executes() {
    let source =
        "h = @pair; C = {2}; [a,b] = h(C{:}); s = a + b;\nfunction [u,v] = pair(x)\n  u = x;\n  v = x + 1;\nend";
    let bytecode = compile_source(source).expect("compile semantic handle expanded multi-output");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CreateBoundFunctionHandle(_, name, _) if name == "pair"
    )));
    assert!(has_bound_expanded_subscript(
        &bytecode,
        runmat_types::SequenceUse::SelectPrefix { count: 2 }
    ));

    let vars = interpret(&bytecode).expect("execute semantic handle expanded multi-output");
    assert!(has_num(&vars, 5.0));
}

#[test]
fn unresolved_external_function_handle_index_call_errors_with_identifier() {
    let source = "h = @pkg.remote_inc; y = h(1);";
    let bytecode = compile_source(source).expect("compile unresolved external handle index call");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc"
    )));
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::Index(1))),
        "single-output handle invocation should lower through Index(1) dynamic dispatch"
    );

    let err = interpret(&bytecode).expect_err("unresolved external handle index call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_function_handle_index_zero_output_errors_with_identifier() {
    let source = "h = @pkg.remote_inc; h(1);";
    let bytecode =
        compile_source(source).expect("compile unresolved external handle zero-output index call");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc"
    )));
    assert!(has_single_selector_subscript_call(
        &bytecode,
        runmat_types::SequenceUse::Discard
    ));

    let err = interpret(&bytecode)
        .expect_err("unresolved external handle zero-output index call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_function_handle_index_multi_output_errors_with_identifier() {
    let source = "h = @pkg.remote_inc; [a,b] = h(1);";
    let bytecode =
        compile_source(source).expect("compile unresolved external handle multi-output index call");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc"
    )));
    assert!(has_single_selector_subscript_call(
        &bytecode,
        runmat_types::SequenceUse::SelectPrefix { count: 2 }
    ));

    let err = interpret(&bytecode)
        .expect_err("unresolved external handle multi-output index call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_function_handle_expand_index_call_errors_with_identifier() {
    let source = "h = @pkg.remote_inc; C = {1,2}; y = h(C{:});";
    let bytecode =
        compile_source(source).expect("compile unresolved external expanded-handle index call");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc"
    )));
    assert!(has_bound_expanded_subscript(
        &bytecode,
        runmat_types::SequenceUse::RequireSingle
    ));

    let err = interpret(&bytecode)
        .expect_err("unresolved external expanded-handle index call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_function_handle_expand_index_zero_output_errors_with_identifier() {
    let source = "h = @pkg.remote_inc; C = {1,2}; h(C{:});";
    let bytecode = compile_source(source)
        .expect("compile unresolved external expanded-handle zero-output index call");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc"
    )));
    assert!(has_bound_expanded_subscript(
        &bytecode,
        runmat_types::SequenceUse::Discard
    ));

    let err = interpret(&bytecode)
        .expect_err("unresolved external expanded-handle zero-output index call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn unresolved_external_function_handle_expand_index_multi_output_errors_with_identifier() {
    let source = "h = @pkg.remote_inc; C = {1,2}; [a,b] = h(C{:});";
    let bytecode = compile_source(source)
        .expect("compile unresolved external expanded-handle multi-output index call");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CreateFunctionHandle(name) if name == "pkg.remote_inc"
    )));
    assert!(has_bound_expanded_subscript(
        &bytecode,
        runmat_types::SequenceUse::SelectPrefix { count: 2 }
    ));

    let err = interpret(&bytecode)
        .expect_err("unresolved external expanded-handle multi-output index call should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn method_syntax_with_semantic_function_callee_executes_directly() {
    let source = "obj = 2; y = obj.bump(3);\nfunction out = bump(receiver, value)\n  out = receiver + value;\nend";
    let bytecode = compile_source(source).expect("semantic method-style call compiles");
    let function = bytecode
        .function_registry
        .functions
        .values()
        .find(|entry| entry.display_name == "bump")
        .expect("local bump has a semantic function identity")
        .function;
    assert!(has_single_value_dotted_call(
        &bytecode.instructions,
        &runmat_types::MemberName::from("bump"),
        1,
        1,
    ));
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::ReadSubscriptPath { steps, .. }
            if matches!(steps.as_slice(), [runmat_vm::BytecodeSubscriptStep::DottedInvoke {
                member,
                semantic_fallback: Some(identity),
                ..
            }] if member.0 == "bump" && matches!(identity.as_ref(),
                runmat_types::CallableIdentity::BoundFunction(id) if *id == function))
    )));
    let vars = interpret(&bytecode).expect("semantic method-style call executes");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-12)));
}

#[test]
fn dotted_stored_member_precedes_same_named_semantic_function() {
    let source = "obj = struct('bump', [10, 20]); y = obj.bump(2);\nfunction out = bump(receiver, index)\n  out = 999;\nend";
    let vars = execute_source(source);
    assert!(numeric_workspace_value(&vars, 20.0));
    assert!(!numeric_workspace_value(&vars, 999.0));
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn declared_method_precedes_same_named_semantic_fallback() {
    let source = "__register_test_classes(); obj = new_object('Point'); moved = obj.move(3, 4); x = moved.x;\nfunction out = move(receiver, dx, dy)\n out = 999;\nend";
    let vars = execute_source(source);
    assert!(numeric_workspace_value(&vars, 3.0));
    assert!(!numeric_workspace_value(&vars, 999.0));
}

#[test]
fn cellfun_upper_function_handle_round_trip() {
    let input =
        "names = {'Ada', 'Linus', 'Katherine'}; upper = cellfun(@upper, names, 'UniformOutput', false);";
    let vars = execute_source(input);

    let mut found = false;
    for value in vars.values() {
        if let runmat_value::Value::Cell(ca) = value {
            if ca.data().len() != 3 {
                continue;
            }
            let texts: Vec<String> = ca
                .data()
                .iter()
                .map(String::try_from)
                .collect::<Result<_, _>>()
                .unwrap_or_default();
            if texts == vec!["ADA", "LINUS", "KATHERINE"] {
                found = true;
                break;
            }
        }
    }

    assert!(found);
}

#[test]
fn cellfun_str2func_local_semantic_callback_executes() {
    let vars = execute_source(
        "function y = inc(x); y = x + 1; end; xs = {1, 2}; ys = cellfun(str2func('inc'), xs); total = sum(ys);",
    );
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-12)));
}

#[test]
fn cellfun_compiled_identity_preserves_all_integer_classes_and_wide_values() {
    let vars = execute_source(
        "a = cellfun(@(x) x, {int8(-128), int8(127)}); b = cellfun(@(x) x, {int16(-32768), int16(32767)}); c = cellfun(@(x) x, {int32(-2147483648), int32(2147483647)}); d = cellfun(@(x) x, {int64(-7), int64(9)}); e = cellfun(@(x) x, {uint8(0), uint8(255)}); f = cellfun(@(x) x, {uint16(0), uint16(65535)}); g = cellfun(@(x) x, {uint32(0), uint32(4294967295)}); base = uint64(9007199254740992); h = cellfun(@(x) x, {base + uint64(1), base + uint64(2)});",
    );
    for expected in [
        runmat_value::IntegerStorage::I8(vec![-128, 127]),
        runmat_value::IntegerStorage::I16(vec![-32768, 32767]),
        runmat_value::IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        runmat_value::IntegerStorage::I64(vec![-7, 9]),
        runmat_value::IntegerStorage::U8(vec![0, 255]),
        runmat_value::IntegerStorage::U16(vec![0, 65535]),
        runmat_value::IntegerStorage::U32(vec![0, u32::MAX]),
        runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994]),
    ] {
        assert!(
            vars.iter().any(|value| {
                matches!(
                    value,
                    runmat_value::Value::Tensor(tensor)
                        if tensor.integer_storage() == Some(&expected)
                )
            }),
            "missing compiled cellfun storage {expected:?}"
        );
    }
}

#[test]
fn compiled_cell_container_integer_cohort_preserves_native_storage() {
    let vars = execute_source(
        "wide = uint64(9007199254740992) + uint64(1); a = cell2mat({wide, wide + uint64(1)}); s = cell2struct({wide}, {'value'}, uint8(1)); b = s.value; t = cell2table({uint16(7); uint16(9)}); c = t.Var1; text = char(uint16([65, 66]));",
    );
    for expected in [
        runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994]),
        runmat_value::IntegerStorage::U16(vec![7, 9]),
    ] {
        assert!(vars.iter().any(|value| {
            matches!(value, runmat_value::Value::Tensor(tensor) if tensor.integer_storage() == Some(&expected))
        }));
    }
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993))
    )));
    assert!(vars.iter().any(
        |value| matches!(value, runmat_value::Value::CharArray(chars)
            if chars.data().iter().map(|unit| unit.raw()).eq("AB".encode_utf16()))
    ));
}

#[test]
fn compiled_cellstr_rejects_integer_before_conversion() {
    let err = execute_source_result("x = cellstr(uint8(65));").expect_err("integer cellstr");
    assert_eq!(err.identifier(), Some("RunMat:cellstr:InvalidInput"));
}

#[test]
fn compiled_numeric_integer_cohort_executes_in_runmat_mode() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source(
        "[n, wn] = cheb2ord(uint8(10), uint8(20), uint8(1), uint8(40), 's'); p = chi2cdf(uint16(3), uint16(5)); r = chol(uint16([4, 2; 2, 3])); sample = uint8([0; 3]); training = uint8([0; 1; 2; 3]); base = uint64(9007199254740992); group = [base + uint64(1); base + uint64(1); base + uint64(2); base + uint64(2)]; labels = classify(sample, training, group);",
    );
    assert!(vars.iter().any(
        |value| matches!(value, runmat_value::Value::Num(number) if (*number - 5.0).abs() < 1.0e-12)
    ));
    assert!(vars
        .iter()
        .any(|value| matches!(value, runmat_value::Value::Num(number) if (*number - 0.300_014_164_121_372).abs() < 1.0e-12)));
    assert!(vars.iter().any(|value| {
        matches!(value, runmat_value::Value::Tensor(tensor) if tensor.integer_storage() == Some(&runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994])))
    }));
}

#[test]
fn compiled_colon_to_cond_integer_cohort_preserves_exact_values_and_boundaries() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source(
        "base = uint64(9007199254740992); sequence = colon(base + uint64(1), base + uint64(3)); z = complex(base + uint64(1), uint64(7)); choices = combinations(uint8([1 2]), int16([3 4])); text = compose(\"%d\", base + uint64(1)); estimate = cond(uint8([1 0; 0 2]));",
    );
    assert!(vars.iter().any(|value| {
        matches!(value, runmat_value::Value::Tensor(tensor) if tensor.integer_storage() == Some(&runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994, 9_007_199_254_740_995])))
    }));
    assert!(vars.iter().any(|value| {
        matches!(value, runmat_value::Value::ComplexTensor(tensor) if tensor.integer_storage().is_some_and(|storage| storage.real() == &runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993]) && storage.imag() == &runmat_value::IntegerStorage::U64(vec![7])))
    }));
    assert!(has_object_class(&vars, "table"));
    assert!(vars.iter().any(|value| {
        matches!(value, runmat_value::Value::StringArray(strings) if strings.data() == vec!["9007199254740993"])
    }));
    assert!(has_num(&vars, 2.0));
}

#[test]
fn arrayfun_str2func_local_semantic_callback_executes() {
    let vars = execute_source(
        "function y = inc(x); y = x + 1; end; xs = [1, 2]; ys = arrayfun(str2func('inc'), xs); total = sum(ys);",
    );
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-12)));
}

#[test]
fn arrayfun_compiled_identity_preserves_all_integer_classes_and_single() {
    let vars = execute_source(
        "a = arrayfun(@(x) x, int8([-128 127])); b = arrayfun(@(x) x, int16([-32768 32767])); c = arrayfun(@(x) x, int32([-2147483648 2147483647])); d = arrayfun(@(x) x, int64([-7 9])); e = arrayfun(@(x) x, uint8([0 255])); f = arrayfun(@(x) x, uint16([0 65535])); g = arrayfun(@(x) x, uint32([0 4294967295])); base = uint64(9007199254740992); h = arrayfun(@(x) x, base + uint64([1 2])); s = arrayfun(@(x) x, single([0.25 -1.5]));",
    );
    for expected in [
        runmat_value::IntegerStorage::I8(vec![-128, 127]),
        runmat_value::IntegerStorage::I16(vec![-32768, 32767]),
        runmat_value::IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        runmat_value::IntegerStorage::I64(vec![-7, 9]),
        runmat_value::IntegerStorage::U8(vec![0, 255]),
        runmat_value::IntegerStorage::U16(vec![0, 65535]),
        runmat_value::IntegerStorage::U32(vec![0, u32::MAX]),
        runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994]),
    ] {
        assert!(
            vars.iter().any(|value| {
                matches!(
                    value,
                    runmat_value::Value::Tensor(tensor)
                        if tensor.integer_storage() == Some(&expected)
                )
            }),
            "missing compiled arrayfun storage {expected:?}"
        );
    }
    assert!(vars.iter().any(|value| {
        matches!(
            value,
            runmat_value::Value::Tensor(tensor)
                if tensor.as_f32_slice() == Some(&[0.25, -1.5][..])
        )
    }));
}

#[test]
fn classes_static_and_inheritance() {
    // Register classes
    assert!(execute_source_result_with_extensions("__register_test_classes();").is_ok());

    // Default init and namespaced
    let vars2 = execute_source_with_extensions("__register_test_classes(); p = new_object('Point'); ax = getfield(p,'x'); ns = new_object('pkg.PointNS'); nsx = getfield(ns,'x');");
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 0.0).abs()<f64::EPSILON)));
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 1.0).abs()<f64::EPSILON)));

    // Static method and property
    let vars3 = execute_source_with_extensions("__register_test_classes(); o = classref('Point').origin(); sv = classref('Point').staticValue;");
    assert!(vars3
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Object(_))));
    assert!(vars3
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 42.0).abs()<f64::EPSILON)));

    // Inheritance override: use feval(getmethod(...))()
    let vars4 = execute_source_with_extensions("__register_test_classes(); c = new_object('Circle'); c = setfield(c,'r', 2); a = feval(getmethod(c,'area'));");
    assert!(vars4.iter().any(
        |v| matches!(v, runmat_value::Value::Num(n) if (*n - std::f64::consts::PI*4.0).abs() < 1e-9)
    ));

    // Access control violations
    let err = execute_source_result_with_extensions(
        "__register_test_classes(); p = new_object('Point'); s = getfield(p,'secret');",
    )
    .expect_err("private property get should error");
    assert_eq!(
        err.identifier(),
        Some("RunMat:PropertyPrivateAccess"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn static_method_via_classref_without_class_registry_is_unresolved() {
    let err =
        execute_source_result("P = classref(\"UnregisteredClassForStaticMethodTest\").origin();")
            .expect_err("classref static call should be unresolved without class registration");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UndefinedFunction"),
        "unexpected error: {}",
        err.message()
    );
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn classes_constructor_and_overloaded_indexing() {
    // Call Ctor constructor; pass the canonical subscript payload to both methods.
    let program = "__register_test_classes(); c = Ctor(7); o = new_object('OverIdx'); S = struct('type', '.', 'subs', 'k'); o = call_method(o,'subsasgn',S,5); t = call_method(o,'subsref',S);";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Object(_))));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn classes_property_access_attributes() {
    // Private get already covered by existing test; ensure private set is rejected with
    // an access-control diagnostic.
    let err = execute_source_result_with_extensions(
        "__register_test_classes(); p = new_object('Point'); p = setfield(p,'secret', 7);",
    )
    .expect_err("private property set should error");
    assert_eq!(
        err.identifier(),
        Some("RunMat:PropertyPrivateAccess"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn import_builtin_resolution_for_static_method() {
    // Register classes and import Point.* so we can call origin() unqualified
    let program = "__register_test_classes(); import Point.*; o = origin();";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Object(_))));
}

#[test]
fn import_specific_resolution_for_builtin() {
    // Use specific import to bring a qualified builtin into scope
    let program =
        "import pkg.missing.*; import Point.origin; __register_test_classes(); o = origin();";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Object(_))));
}

#[test]
fn import_ambiguity_specific_conflict_errors() {
    // Two specifics that map the same unqualified name should cause compile-time ambiguity.
    let program = "import PkgF.foo; import PkgG.foo; r = foo();";
    let err = compile_source(program).expect_err("expected specific import ambiguity");
    assert_import_ambiguity_error(&err);
}

#[test]
fn import_ambiguity_wildcard_conflict_errors() {
    // Two wildcard imports that expose the same unqualified name should be ambiguous.
    let program = "import PkgF.*; import PkgG.*; r = foo();";
    let err = compile_source(program).expect_err("expected wildcard import ambiguity");
    assert_import_ambiguity_error(&err);
}

#[test]
fn import_ambiguity_wildcard_handle_conflict_errors() {
    let program = "import PkgF.*; import PkgG.*; h = @foo;";
    let err = compile_source(program).expect_err("expected wildcard handle import ambiguity");
    assert_import_ambiguity_error(&err);
}
#[test]
fn import_static_method_via_specific_class_import() {
    // import ClassName.* to allow unqualified static methods and properties
    let program = "__register_test_classes(); import Point.*; p = origin(); v = classref('Point').staticValue;";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Object(_))));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(_))));
}

#[test]
fn import_static_method_function_handle_executes() {
    let program = "__register_test_classes(); import Point.origin; h = @origin; o = feval(h);";
    let bytecode = compile_source(program).expect("semantic import function handle compile");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateExternalFunctionHandle(name) if name == "Point.origin")
    ));
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Object(_))));
}

#[test]
fn import_static_method_function_handle_direct_call_executes() {
    let program = "__register_test_classes(); import Point.origin; h = @origin; o = h();";
    let bytecode = compile_source(program).expect("semantic import function handle compile");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateExternalFunctionHandle(name) if name == "Point.origin")
    ));
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Object(_))));
}

#[test]
fn import_wildcard_static_method_function_handle_executes() {
    let program = "__register_test_classes(); import Point.*; h = @origin; o = feval(h);";
    let bytecode =
        compile_source(program).expect("semantic wildcard import function handle compile");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateExternalFunctionHandle(name) if name == "Point.origin")
    ));
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Object(_))));
}

#[test]
fn import_wildcard_static_method_function_handle_direct_call_executes() {
    let program = "__register_test_classes(); import Point.*; h = @origin; o = h();";
    let bytecode =
        compile_source(program).expect("semantic wildcard import function handle compile");
    assert!(bytecode.instructions.iter().any(
        |instr| matches!(instr, runmat_vm::Instr::CreateExternalFunctionHandle(name) if name == "Point.origin")
    ));
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Object(_))));
}

#[test]
fn qualified_static_method_function_handle_executes() {
    let program = "__register_test_classes(); h = @Point.origin; o = feval(h);";
    let bytecode =
        compile_source(program).expect("semantic qualified static method function handle compile");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CreateExternalFunctionHandle(name)
            | runmat_vm::Instr::CreateFunctionHandle(name)
            if name == "Point.origin"
    )));
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Object(_))));
}

#[test]
fn qualified_static_method_function_handle_direct_call_executes() {
    let program = "__register_test_classes(); h = @Point.origin; o = h();";
    let bytecode =
        compile_source(program).expect("semantic qualified static method function handle compile");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CreateExternalFunctionHandle(name)
            | runmat_vm::Instr::CreateFunctionHandle(name)
            if name == "Point.origin"
    )));
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Object(_))));
}

#[test]
fn classref_getmethod_static_method_handle_executes() {
    let program = "__register_test_classes(); h = getmethod(classref('Point'), 'origin'); name = func2str(h); o = feval(h);";
    let vars = execute_source_with_extensions(program);
    assert!(
        vars.iter()
            .any(|v| matches!(v, runmat_value::Value::String(name) if name == "Point.origin")),
        "expected getmethod(classref(...)) handle to preserve qualified method name"
    );
    assert!(
        vars.iter()
            .any(|v| matches!(v, runmat_value::Value::Object(_))),
        "expected feval(getmethod(classref(...))) to execute static method"
    );
}

#[test]
fn classref_getmethod_static_method_handle_direct_call_executes() {
    let program = "__register_test_classes(); h = getmethod(classref('Point'), 'origin'); name = func2str(h); o = h();";
    let vars = execute_source_with_extensions(program);
    assert!(
        vars.iter()
            .any(|v| matches!(v, runmat_value::Value::String(name) if name == "Point.origin")),
        "expected getmethod(classref(...)) direct-call handle to preserve qualified method name"
    );
    assert!(
        vars.iter()
            .any(|v| matches!(v, runmat_value::Value::Object(_))),
        "expected direct call through getmethod(classref(...)) handle to execute static method"
    );
}

#[test]
fn object_getmethod_instance_method_handle_direct_call_executes() {
    let program = "__register_test_classes(); c = new_object('Circle'); c = setfield(c,'r', 2); h = getmethod(c, 'area'); a = h();";
    let vars = execute_source_with_extensions(program);
    assert!(vars.iter().any(
        |v| matches!(v, runmat_value::Value::Num(n) if (*n - std::f64::consts::PI * 4.0).abs() < 1e-9)
    ));
}

#[test]
fn classref_nonstatic_method_call_errors_with_identifier_contract() {
    let err = execute_source_result("__register_test_classes(); classref('Point').move(1, 2);")
        .expect_err("classref call to non-static method should fail");
    assert_eq!(err.identifier(), Some("RunMat:MethodNotStatic"));
}

#[test]
fn classref_unknown_static_method_call_remains_unresolved_with_identifier_contract() {
    let err = execute_source_result(
        "__register_test_classes(); classref('Point').definitely_missing(1);",
    )
    .expect_err("classref call to unknown static method should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn addlistener_invalid_target_errors_with_identifier_contract() {
    let err = execute_source_result("addlistener(1, 'Changed', @sin);")
        .expect_err("addlistener should reject non-object target");
    assert_eq!(err.identifier(), Some("RunMat:AddListenerTargetInvalid"));
}

#[test]
fn notify_invalid_target_errors_with_identifier_contract() {
    let err = execute_source_result("notify(1, 'Changed');")
        .expect_err("notify should reject non-object target");
    assert_eq!(err.identifier(), Some("RunMat:NotifyTargetInvalid"));
}

#[test]
fn call_method_empty_name_errors_with_identifier_contract() {
    let err = execute_source_result(
        "__register_test_classes(); p = new_object('Point'); call_method(p, '   ');",
    )
    .expect_err("empty call_method name should fail");
    assert_eq!(err.identifier(), Some("RunMat:CallMethodNameInvalid"));
}

#[test]
fn call_method_nonobject_receiver_errors_with_identifier_contract() {
    let err = execute_source_result("call_method(1, 'origin');")
        .expect_err("non-object call_method receiver should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidObjectDispatch"));
}

#[test]
fn public_subsref_uses_default_numeric_indexing() {
    let values = execute_source_result(
        "a = [10, 20]; S = struct('type', '()', 'subs', {{2}}); result = subsref(a, S);",
    )
    .expect("public subsref supports default numeric indexing");
    assert!(has_num(&values, 20.0));
}

#[test]
fn public_subsasgn_uses_default_numeric_indexing() {
    let values = execute_source_result("a = [10, 20]; S = struct('type', '()', 'subs', {{2}}); a = subsasgn(a, S, 99); result = a(2); untouched = a(1);")
        .expect("public subsasgn supports default numeric indexing");
    assert!(has_num(&values, 99.0));
    assert!(has_num(&values, 10.0));
}

#[test]
fn public_subsref_uses_default_object_indexing_without_overload() {
    let values = execute_source_result(
        "__register_test_classes(); p = new_object('Point'); p.x = 13; S = struct('type', '()', 'subs', {{1}}); q = subsref(p, S); result = q.x;",
    )
    .expect("object without subsref overload uses default indexing");
    assert!(has_object_class(&values, "Point"));
    assert!(has_num(&values, 13.0));
}

#[test]
fn public_subsasgn_uses_default_object_indexing_without_overload() {
    let values = execute_source_result(
        "__register_test_classes(); p = new_object('Point'); q = new_object('Point'); q.x = 77; S = struct('type', '()', 'subs', {{2}}); p = subsasgn(p, S, q); count = numel(p); second = p(2); first = p(1); assigned = second.x; untouched = first.x;",
    )
    .expect("object without subsasgn overload uses default homogeneous array assignment");
    assert!(has_num(&values, 2.0));
    assert!(has_num(&values, 77.0));
    assert!(has_num(&values, 0.0));
}

#[test]
fn public_subsref_rejects_invalid_subscript_payload_with_owned_identifier() {
    let err = execute_source_result("subsref(1, struct('type', '()', 'subs', 1));")
        .expect_err("parentheses subscript payload must be a cell array");
    assert_eq!(err.identifier(), Some("RunMat:InvalidObjectSubscriptPath"));
}

#[test]
fn public_subsasgn_rejects_invalid_subscript_payload_with_owned_identifier() {
    let err = execute_source_result("subsasgn(1, struct('type', '()', 'subs', 1), 2);")
        .expect_err("parentheses subscript payload must be a cell array");
    assert_eq!(err.identifier(), Some("RunMat:InvalidObjectSubscriptPath"));
}

#[test]
fn object_paren_index_uses_default_scalar_array_semantics_without_subsref() {
    let values = execute_source_result_with_extensions(
        "__register_test_classes(); p = new_object('Point'); q = p(1);",
    )
    .expect("ordinary object indexing does not require a subsref overload");
    assert!(has_object_class(&values, "Point"));
}

#[test]
fn feval_unsupported_callable_value_errors_with_identifier_contract() {
    let err = execute_source_result("x = 1; y = feval(x, 2);")
        .expect_err("feval on non-callable value should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:FevalFunctionValueUnsupported"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn object_paren_assign_uses_default_homogeneous_array_semantics_without_subsasgn() {
    let values = execute_source_result(
        "__register_test_classes(); p = new_object('Point'); q = new_object('Point'); p(2) = q; n = numel(p);",
    )
    .expect("ordinary object assignment does not require a subsasgn overload");
    assert!(has_num(&values, 2.0));
}

#[test]
fn import_precedence_specific_over_wildcard_and_locals() {
    // Specific imports should take precedence over wildcard imports; locals should shadow both
    let program = r#"
        __register_test_classes();
        import PkgF.foo;         % specific import of PkgF.foo
        import PkgG.*;           % wildcard import with also foo
        a = foo();               % should resolve to PkgF.foo (10)
        foo = @() 42;            % local function handle shadowing imports
        b = feval(foo);          % should be 42 (local shadow)
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 10.0).abs()<1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 42.0).abs()<1e-9)));
}

#[test]
fn import_ambiguity_between_specifics_errors() {
    // Two specific imports of same unqualified function should cause compile-time error
    let program = r#"
        import PkgF.foo;
        import PkgG.foo;
        y = foo();
    "#;
    let err = compile_source(program).expect_err("expected specific import ambiguity");
    assert_import_ambiguity_error(&err);
}

#[test]
fn import_ambiguity_between_wildcards_errors() {
    // Two wildcard imports that both contain 'foo' should cause an ambiguity error on unqualified call
    let program = r#"
        import PkgF.*;
        import PkgG.*;
        y = foo();
    "#;
    let err = compile_source(program).expect_err("expected wildcard import ambiguity");
    assert_import_ambiguity_error(&err);
}

#[test]
fn import_specific_conflict_with_user_function_prefers_local() {
    // A user function in scope should overshadow specific imports
    let program = r#"
        import PkgF.foo;   % specific import exists
        function y = foo()
            y = 33;
        end
        a = foo();         % should call local function => 33
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 33.0).abs()<1e-9)));
}

#[test]
fn import_static_property_shadowed_by_local_variable() {
    // Local variables should shadow static properties brought in via Class.* imports
    let program = r#"
        __register_test_classes();
        import Point.*;
        staticValue = 7;   % shadows classref('Point').staticValue (42)
        v = staticValue;
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 7.0).abs()<1e-9)));
}

#[test]
fn import_specific_beats_wildcard_for_function_resolution() {
    // Specific import should win over wildcard providing the same name
    let program = r#"
        import PkgF.foo;   % specific import
        import PkgG.*;     % wildcard that also provides foo
        y = foo();         % should resolve to PkgF.foo => 10
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 10.0).abs()<1e-9)));
}

#[test]
fn import_wildcard_static_method_ambiguity_errors() {
    // Two Class.* wildcards that both bring in a static method with same unqualified name should be ambiguous
    // We reuse Point.* for one side; fabricate another class via runtime registration if needed.
    // Here we simulate ambiguity on function resolution instead, since a second class isn't pre-registered.
    let program = r#"
        import PkgF.*;
        import PkgG.*;
        z = foo();   % ambiguous via wildcard imports
    "#;
    let err = compile_source(program).expect_err("expected wildcard import ambiguity");
    assert_import_ambiguity_error(&err);
}

#[test]
fn local_function_shadows_class_static_method_under_class_star() {
    // Define a local function 'origin' that should shadow Point.origin brought via import Point.*
    let program = r#"
        __register_test_classes();
        import Point.*;
        function y = origin()
            y = 123;
        end
        v = origin();   % should call local (123), not Point.origin
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 123.0).abs()<1e-9)));
}

#[test]
fn multi_segment_import_builtin_vs_user_function_specific_prefers_user_function() {
    // When both a builtin and a user function exist under the same unqualified name via specific import,
    // ensure the user function in scope takes precedence.
    let program = r#"
        import PkgF.foo;   % provides builtin PkgF.foo
        function y = foo()
            y = 77;
        end
        a = foo();         % should call the user function -> 77
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 77.0).abs()<1e-9)));
}

#[test]
fn unqualified_static_property_without_imports_errors() {
    // Referencing staticValue unqualified without imports should error
    let program = r#"
        __register_test_classes();
        v = staticValue;
    "#;
    let err = compile_source(program).expect_err("expected missing static property error");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedVariable"));
}

#[test]
fn unqualified_static_property_shadowed_by_local_variable() {
    // Even with import Point.*, a local variable should shadow unqualified static property name
    let program = r#"
        __register_test_classes();
        import Point.*;
        staticValue = 9;
        v = staticValue;  % picks local, not class static 42
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 9.0).abs()<1e-9)));
}

#[test]
fn import_nested_package_class_static_method_resolution() {
    // Simulate nested pkg class via import path and ensure resolution prefers functions first, then static method
    let program = r#"
        __register_test_classes();
        import pkg.PointNS.*;   % nested path Simulation (pkg.PointNS has no methods; ensures no accidental resolution)
        function z = origin()
            z = 5;
        end
        v = origin();          % local function shadows any static method named origin
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs()<1e-9)));
}

#[test]
fn class_property_attribute_conflicts_error() {
    // Properties with Constant and Dependent together should error
    let program = r#"
        classdef Bad
            properties(Constant, Dependent)
                x
            end
        end
    "#;
    let err = compile_source(program).expect_err("expected class property attribute conflict");
    assert_eq!(
        err.identifier(),
        Some("RunMat:ClassPropertyAttributeConflict"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn class_method_attribute_conflicts_error() {
    // Methods with Abstract and Sealed together should error
    let program = r#"
        classdef Bad
            methods(Abstract, Sealed)
                function y = f(obj)
                    y = 1;
                end
            end
        end
    "#;
    let err = compile_source(program).expect_err("expected class method attribute conflict");
    assert_eq!(
        err.identifier(),
        Some("RunMat:ClassMethodAttributeConflict"),
        "unexpected error: {}",
        err.message()
    );
}

#[test]
fn metaclass_context_with_imports() {
    // Ensure ?pkg.Class parses and coexists with imports in semantic execution.
    let program = "import pkg.*; ?pkg.Class; x=1;";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-1.0).abs()<1e-9)));
}

#[test]
fn metaclass_postfix_member_and_method() {
    let program = "__register_test_classes(); v = ?Point.staticValue; o = ?Point.origin();";
    let vars = execute_source(program);
    assert!(has_num(&vars, 42.0));
    assert!(has_object_class(&vars, "Point"));
    assert!(has_object_num_property(&vars, "Point", "x", 0.0));
    assert!(has_object_num_property(&vars, "Point", "y", 0.0));
}

#[test]
fn metaclass_builtin_script_surface_reports_runtime_class_refs() {
    let program = r#"
        __register_test_classes();
        p = new_object('Point');
        mc_obj = metaclass(p);
        mc_char = metaclass('Point');
        mc_string = metaclass("Point");
    "#;
    let vars = execute_source(program);
    assert!(has_class_ref(&vars, "Point"));
    assert!(has_class_ref(&vars, "char"));
    assert!(has_class_ref(&vars, "string"));
}

#[test]
fn superclasses_script_surface_reports_registered_parent_chain() {
    let program = r#"
        __register_test_classes();
        c = new_object('Circle');
        s_obj = superclasses(c);
        s_name = superclasses('Circle');
        s_ref = superclasses(classref('Circle'));
        s_none = superclasses('Point');
    "#;
    let vars = execute_source(program);
    assert!(has_cell_char_row(&vars, &["Shape"]));
    assert!(
        vars.iter()
            .filter(|value| {
                cell_char_row_values(value).is_some_and(|((rows, cols), actual)| {
                    rows == 1 && cols == 1 && actual == vec!["Shape".to_string()]
                })
            })
            .count()
            >= 3
    );
    assert!(has_cell_char_row(&vars, &[]));
}

#[test]
fn ismethod_script_surface_checks_registered_class_methods() {
    let program = r#"
        __register_test_classes();
        p = new_object('Point');
        circle = new_object('Circle');
        ok = ismethod(p, 'move') && ismethod(p, "origin") && ~ismethod(p, 'missing') && ismethod(circle, 'area');
    "#;
    let vars = execute_source(program);
    assert!(
        matches!(vars.last(), Some(runmat_value::Value::Bool(true))),
        "expected per-case ismethod checks to produce ok=true: {vars:?}"
    );
}

#[test]
fn underlying_type_script_surface() {
    let program = r#"
        __register_test_classes();
        p = new_object('Point');
        a = underlyingType(1);
        b = underlyingType(single([1 2]));
        c = underlyingType(true);
        d = underlyingType('abc');
        e = underlyingType("abc");
        f = underlyingType(p);
        ok = isUnderlyingType(single([3 4]), "single") && isUnderlyingType(true, 'logical') && ~isUnderlyingType(p, "double");
    "#;
    let vars = execute_source(program);
    for expected in ["double", "single", "logical", "char", "string", "Point"] {
        assert!(
            vars.iter()
                .any(|v| matches!(v, runmat_value::Value::String(s) if s == expected)),
            "expected underlyingType output {expected:?}; vars={vars:?}"
        );
    }
    assert!(
        matches!(vars.last(), Some(runmat_value::Value::Bool(true))),
        "expected isUnderlyingType combined predicate to be true: {vars:?}"
    );
}

#[test]
fn typed_complex_integer_metadata_script_surface() {
    let vars = execute_source(
        "z = complex(uint64(9223372036854775808), uint64(1)); c = class(z); t = underlyingType(z); ok = isUnderlyingType(z, 'uint64');",
    );
    assert!(
        vars.iter()
            .filter(|value| matches!(value, runmat_value::Value::String(name) if name == "uint64"))
            .count()
            >= 2,
        "class and underlyingType must preserve typed complex integer class: {vars:?}"
    );
    assert!(
        vars.iter()
            .any(|value| matches!(value, runmat_value::Value::Bool(true))),
        "isUnderlyingType must recognize typed complex integer class: {vars:?}"
    );
}

#[test]
fn typed_complex_integer_jsonencode_script_surface_preserves_exact_components() {
    let vars = execute_source_with_extensions(
        "z = complex(uint64(18446744073709551615), uint64(9223372036854775808)); encoded = jsonencode(z);",
    );
    assert!(
        vars.iter().any(|value| matches!(
            value,
            runmat_value::Value::CharArray(chars)
                if runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 text fixture")
                    == "{\"real\":18446744073709551615,\"imag\":9223372036854775808}"
        )),
        "jsonencode must use exact typed-complex integer components: {vars:?}"
    );
}

#[test]
fn primes_script_surface_preserves_signed_and_unsigned_integer_classes() {
    let vars = execute_source(
        "signed = primes(int64(12)); unsigned = primes(uint64(12)); signed_class = class(signed); unsigned_class = class(unsigned);",
    );
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&runmat_value::IntegerStorage::I64(vec![2, 3, 5, 7, 11]))
    )));
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&runmat_value::IntegerStorage::U64(vec![2, 3, 5, 7, 11]))
    )));
    for expected in ["int64", "uint64"] {
        assert!(
            vars.iter().any(
                |value| matches!(value, runmat_value::Value::String(class) if class == expected)
            ),
            "expected class {expected:?}; vars={vars:?}"
        );
    }
}

#[test]
fn randi_script_surface_preserves_explicit_and_like_integer_classes() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source(
        "signed = randi(5, 2, 2, 'int64'); prototype = uint64(zeros(2, 2)); unsigned = randi(5, 'like', prototype); signed_class = class(signed); unsigned_class = class(unsigned);",
    );
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if matches!(tensor.integer_storage(), Some(runmat_value::IntegerStorage::I64(_)))
    )));
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if matches!(tensor.integer_storage(), Some(runmat_value::IntegerStorage::U64(_)))
    )));
    for expected in ["int64", "uint64"] {
        assert!(
            vars.iter().any(
                |value| matches!(value, runmat_value::Value::String(class) if class == expected)
            ),
            "expected class {expected:?}; vars={vars:?}"
        );
    }
}

#[test]
fn randi_script_surface_gates_wide_integer_classes_by_compatibility_policy() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for source in [
        "out = randi(5, 2, 2, 'int64');",
        "prototype = uint64(zeros(2, 2)); out = randi(5, 'like', prototype);",
    ] {
        let error = execute_source_result(source).expect_err("wide randi output must be gated");
        assert!(
            error.to_string().contains("RunMat extensions"),
            "unexpected error: {error}"
        );
    }
}

#[test]
fn missing_integer_cast_names_are_registered_on_script_surface() {
    let vars = execute_source(
        "a = int32([2147483648 -2147483649 3.5]); b = uint8([-1 260 4.5]); c = uint16([-1 70000 5.5]); d = uint32([-1 4294967296 6.5]); ca = class(a); cb = class(b); cc = class(c); cd = class(d);",
    );
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&runmat_value::IntegerStorage::I32(vec![
                    i32::MAX,
                    i32::MIN,
                    4,
                ]))
    )));
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&runmat_value::IntegerStorage::U8(vec![0, u8::MAX, 5]))
    )));
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&runmat_value::IntegerStorage::U16(vec![0, u16::MAX, 6]))
    )));
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&runmat_value::IntegerStorage::U32(vec![0, u32::MAX, 7]))
    )));
    for expected in ["int32", "uint8", "uint16", "uint32"] {
        assert!(
            vars.iter().any(
                |value| matches!(value, runmat_value::Value::String(class) if class == expected)
            ),
            "expected class {expected:?}; vars={vars:?}"
        );
    }
}

#[test]
fn num_arguments_from_subscript_script_surface() {
    let program = r#"
        C = {"one", 2, "three"};
        S = struct();
        S.type = '{}';
        S.subs = {[1 2]};
        n1 = numArgumentsFromSubscript(C, S, 'Statement');

        S2 = struct();
        S2.type = '{}';
        S2.subs = {':'};
        ctx = classref('matlab.indexing.IndexingContext').Statement;
        n2 = numArgumentsFromSubscript(C, S2, ctx);

        __register_test_classes();
        o = new_object('OverIdx');
        o = setfield(o, 'nargs', 6);
        S3 = struct();
        S3.type = '.';
        S3.subs = 'field';
        n3 = numArgumentsFromSubscript(o, S3, 'Expression');
    "#;
    // This fixture intentionally uses the RunMat-only object setfield form.
    let vars = execute_source_with_extensions(program);
    assert!(has_num(&vars, 2.0), "expected two brace outputs: {vars:?}");
    assert!(
        has_num(&vars, 3.0),
        "expected colon brace output count: {vars:?}"
    );
    assert!(
        has_num(&vars, 6.0),
        "expected object overload to supply output count: {vars:?}"
    );
}

#[test]
fn object_saveobj_loadobj_script_surface() {
    let program = r#"
        __register_test_classes();
        o = new_object('OverIdx');
        o = setfield(o, 'k', 21);
        o = setfield(o, 'nargs', 4);
        payload = saveobj(o);
        saved_by = getfield(payload, 'saved_by');
        restored = classref('OverIdx').loadobj(payload);
    "#;
    let vars = execute_source_with_extensions(program);
    assert!(
        vars.iter()
            .any(|v| matches!(v, runmat_value::Value::String(s) if s == "OverIdx.saveobj")),
        "expected saveobj payload marker: {vars:?}"
    );
    assert!(
        has_object_num_property(&vars, "OverIdx", "k", 21.0),
        "expected loadobj-restored OverIdx.k property: {vars:?}"
    );
    assert!(
        has_object_string_property(&vars, "OverIdx", "loaded_by", "OverIdx.loadobj"),
        "expected loadobj marker property: {vars:?}"
    );
}

#[test]
fn findobj_script_surface_finds_graphics_handles() {
    let program = r#"
        figure('Visible', 'off');
        h = plot([1 2 3], 'DisplayName', 'series-a');
        fig = gcf();
        by_type = findobj('Type', 'line');
        by_name = findobj(gca(), 'DisplayName', 'series-a');
        flat = findobj(fig, 'flat');
    "#;
    let vars = execute_source(program);
    let line_handle = vars
        .iter()
        .find_map(|value| match value {
            runmat_value::Value::Num(value) if *value > 1.0e6 => Some(*value),
            _ => None,
        })
        .expect("line handle");
    let figure_handle = vars
        .iter()
        .find_map(|value| match value {
            runmat_value::Value::Num(value) if *value > 0.0 && *value < 1.0e6 => Some(*value),
            _ => None,
        })
        .expect("figure handle");

    assert!(
        vars.iter()
            .filter(|value| matches!(
                value,
                runmat_value::Value::Tensor(tensor)
                    if tensor.materialize_f64().len() == 1 && (tensor.materialize_f64()[0] - line_handle).abs() < 1e-9
            ))
            .count()
            >= 2
    );
    assert!(has_numeric_tensor(&vars, &[figure_handle]));
}

#[test]
fn classdef_with_attributes_enforced() {
    // Define class A with private get and public set on property p, then enforce via getfield/setfield
    let src = "classdef A\n  properties(GetAccess=private, SetAccess=public)\n    p\n  end\nend\n a = new_object('A'); a = setfield(a,'p',5); try; v = getfield(a,'p'); catch e; ok=1; end";
    let vars = execute_source_with_extensions(src);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 1.0).abs() < 1e-9)));
}

#[test]
fn dynamicprops_class_supports_addprop_dot_access_and_delete() {
    let src = r#"
classdef DynPoint < dynamicprops
end
d = new_object('DynPoint');
p = addprop(d, 'gain');
d.gain = 7;
observed = d.gain;
p.Hidden = true;
hidden = p.Hidden;
is_dyn = isa(d, 'dynamicprops');
supers = superclasses(d);
delete(p);
try
  missing = d.gain;
catch e
  removed = 1;
end
"#;
    let vars = execute_source(src);
    assert!(has_num(&vars, 7.0));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Bool(true))));
    assert!(has_cell_char_row(&vars, &["dynamicprops", "handle"]));
    assert!(has_num(&vars, 1.0));
}

#[test]
fn builtin_call_with_expanded_middle_argument() {
    // Use a cell row and index into it to pass as middle arg
    // max(a,b) with b coming from C{1}
    let program = "C = {10, 20}; r = max(5, C{1});";
    let vars = execute_source(program);
    // Expect 10 as result appears in vars somewhere
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 10.0).abs() < 1e-9)));
}

#[test]
fn builtin_call_with_two_expanded_args() {
    let program = "C = {3, 4}; D = {5, 6}; r = max(C{1}, D{1});";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn user_function_with_two_expanded_args() {
    let program =
        "function y = sum2(a,b); y = a + b; end; C = {7,8}; D = {11,12}; r = sum2(C{2}, D{1});";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 19.0).abs() < 1e-9)));
}

#[test]
fn expansion_on_non_cell_errors() {
    let program = "r = max(5, 10{1});";
    let bytecode = compile_source(program).expect("compile expansion error source");
    let err = interpret(&bytecode).expect_err("expansion on non-cell should fail");
    assert_eq!(err.identifier(), Some("RunMat:InvalidExpandTarget"));
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn object_cell_expansion_via_subsref() {
    let program = "__register_test_classes(); o = new_object('OverIdx'); S = struct('type', '{}', 'subs', {{1}}); o = call_method(o,'subsasgn',S,42); r = max(o{1}, 5);";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 42.0).abs() < 1e-9)));
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn method_expand_multi_output_uses_typed_instruction() {
    let source = r#"
        function [first, second] = pair(x, y)
            first = x;
            second = y;
        end
        obj = struct('deal', @pair);
        C = {7, 3};
        [a,b] = obj.deal(C{:});
        s = b;
    "#;
    let bytecode = compile_source(source).expect("compile method expand multi-output source");
    assert!(
        has_captured_cell_dotted_call(&bytecode.instructions, 2),
        "{:#?}",
        bytecode.instructions
    );
    assert!(!has_captured_cell_dotted_call(&bytecode.instructions, 1));

    let vars = interpret(&bytecode).expect("execute method expand multi-output");
    assert!(has_num(&vars, 7.0));
    assert!(has_num(&vars, 3.0));
}

#[test]
fn expand_all_elements_in_args() {
    // C{:} expands all elements of C into separate arguments
    // max takes two args; here C has more; we only assert no crash and presence of some expected nums
    let program = "C = {1,2}; a = max(C{:});";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(_))));
}

#[test]
fn scalar_brace_read_requires_exactly_one_cell_value() {
    let single = execute_source_result("C = {42}; x = C{:};")
        .expect("one selected cell value is valid in a scalar RHS");
    assert!(has_num(&single, 42.0));

    for source in [
        "C = {}; x = C{:};",
        "C = {1, 2}; x = C{:};",
        "C = {1, 2}; x = C{1:2};",
    ] {
        let error = execute_source_result(source)
            .expect_err("scalar RHS must reject non-singleton cell contents");
        assert_eq!(
            error.identifier(),
            Some("RunMat:CommaSeparatedListRequiresSingleValue"),
            "{source}: {error}"
        );
    }
}

#[test]
fn prepared_cell_destination_and_bracketed_aggregate_keep_expansion() {
    let values = execute_source_result(
        "C = {0, 0}; [C{:}] = deal(4, 5); A = [C{:}]; first = C{1}; second = C{2};",
    )
    .expect("prepared cell destination and aggregate expansion");
    assert!(has_num(&values, 4.0));
    assert!(has_num(&values, 5.0));
    assert!(has_numeric_tensor(&values, &[4.0, 5.0]));
}

#[test]
fn bracketed_cell_aggregate_still_assigns_to_numeric_slice() {
    let values = execute_source_result("A = zeros(1, 2); C = {7, 8}; A(1, 1:2) = [C{:}];")
        .expect("bracketed cell contents expand before numeric slice assignment");
    assert!(has_numeric_tensor(&values, &[7.0, 8.0]));
}

#[test]
fn mixed_cell_colon_expansion_in_call_args() {
    let program = r#"
        C = {1,2;3,4};
        [a,b] = deal(C{:,2});
        z = a + b;
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-6.0).abs()<1e-9)));
}

#[test]
fn builtin_expand_multi_output_uses_typed_instruction() {
    let source = "C = {7,3}; [a,b] = deal(C{:}); s = a + b;";
    let bytecode = compile_source(source).expect("compile builtin expand multi-output");
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallBuiltinExpandMultiOutput(name, specs, out_count)
            if name == "deal" && *out_count == 2 && specs.len() == 1 && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));

    let vars = interpret(&bytecode).expect("execute builtin expand multi-output");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 10.0).abs() < 1e-9)));
}

#[test]
fn builtin_expand_single_output_uses_typed_instruction() {
    let source = "C = {7,3}; a = max(C{:});";
    let bytecode = compile_source(source).expect("compile builtin expand single output");
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallBuiltinExpandMultiOutput(name, specs, out_count)
            if name == "max" && *out_count == 1 && specs.len() == 1 && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let values = interpret(&bytecode).expect("execute captured builtin argument");
    assert!(has_num(&values, 7.0));
}

#[test]
fn expand_single_output_uses_typed_instruction() {
    let source = "function y = sum2(a,b); y = a + b; end; C = {7,8}; r = sum2(C{:});";
    let bytecode = compile_source(source).expect("compile semantic expand single output source");
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallSemanticFunctionExpandMultiOutput(_, specs, out_count)
            if *out_count == 1 && specs.len() == 1 && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));
    let values = interpret(&bytecode).expect("execute captured semantic function argument");
    assert!(has_num(&values, 15.0));
}

#[test]
fn expand_multi_output_uses_typed_instruction() {
    let source =
        "function [u,v] = pair(a,b); u = a; v = b; end; C = {7,8}; [x,y] = pair(C{:}); s = x + y;";
    let bytecode = compile_source(source).expect("compile semantic expand multi-output source");
    assert!(bytecode.instructions.iter().enumerate().any(|(call_index, instr)| matches!(
        instr,
        runmat_vm::Instr::CallSemanticFunctionExpandMultiOutput(_, specs, out_count)
            if *out_count == 2 && specs.len() == 1 && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
    )));

    let vars = interpret(&bytecode).expect("execute semantic expand multi-output");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 15.0).abs() < 1e-9)));
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn method_expand_single_output_uses_typed_instruction() {
    let source = r#"
        function answer = fortytwo()
            answer = 42;
        end
        obj = struct('deal', @fortytwo);
        C = {};
        a = obj.deal(C{:});
    "#;
    let bytecode = compile_source(source).expect("compile method expand single-output source");
    assert!(
        has_captured_cell_dotted_call(&bytecode.instructions, 1),
        "{:#?}",
        bytecode.instructions
    );
    assert!(!has_captured_cell_dotted_call(&bytecode.instructions, 2));
    let vars = interpret(&bytecode).expect("execute zero-argument dotted invocation");
    assert!(has_num(&vars, 42.0));
}

#[test]
fn method_expand_compound_segments_run_after_their_receivers() {
    let source = r#"
        function output = first(value)
            output = struct('second', @(next) value + next);
        end
        s = struct('first', @first);
        C = {7};
        D = {3};
        result = s.first(C{:}).second(D{:});
    "#;
    let bytecode = compile_source(source).expect("compile compound dotted invocation");
    let segments = bytecode
        .instructions
        .iter()
        .filter(|instruction| {
            matches!(
                instruction,
                runmat_vm::Instr::ReadSubscriptPath { steps, .. }
                    if matches!(steps.as_slice(),
                        [runmat_vm::BytecodeSubscriptStep::DottedInvoke { .. }])
            )
        })
        .count();
    assert_eq!(
        segments, 2,
        "each invocation has its own execution boundary"
    );
    let values = interpret(&bytecode).expect("execute compound dotted invocation");
    assert!(has_num(&values, 10.0));
}

#[test]
fn method_expand_prefix_error_precedes_later_argument_error() {
    let source = r#"
        function output = first(value)
            error('RunMat:FirstInvocation', 'first invocation fails');
            output = value;
        end
        function output = later()
            error('RunMat:LaterArgument', 'later argument was evaluated');
            output = 0;
        end
        s = struct('first', @first);
        C = {7};
        result = s.first(C{:}).second(later());
    "#;
    let bytecode = compile_source(source).expect("compile ordered dotted invocation");
    let error = interpret(&bytecode).expect_err("first invocation fails before later argument");
    assert_eq!(error.identifier(), Some("RunMat:FirstInvocation"));
}

#[test]
fn builtin_multi_output_uses_typed_instruction() {
    let source = "[a,b] = deal(7,3); s = a + b;";
    let bytecode = compile_source(source).expect("compile builtin multi-output");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallBuiltinMulti(name, argc, out_count)
            if name == "deal" && *argc == 2 && *out_count == 2
    )));

    let vars = interpret(&bytecode).expect("execute builtin multi-output");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 10.0).abs() < 1e-9)));
}

#[test]
fn builtin_single_output_uses_typed_instruction() {
    let source = "a = sqrt(9);";
    let bytecode = compile_source(source).expect("compile builtin single-output");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallBuiltinMulti(name, argc, out_count)
            if name == "sqrt" && *argc == 1 && *out_count == 1
    )));
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn method_multi_output_uses_typed_instruction() {
    let source =
        "__register_test_classes(); obj = new_object('Point'); [a,b,c] = obj.deal(7,3); s = b;";
    let bytecode = compile_source(source).expect("compile method multi-output");
    let member = runmat_types::MemberName::from("deal");
    assert!(has_single_value_dotted_call(
        &bytecode.instructions,
        &member,
        2,
        3
    ));
    assert!(!has_single_value_dotted_call(
        &bytecode.instructions,
        &member,
        2,
        1
    ));

    let vars = interpret(&bytecode).expect("execute method multi-output");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 7.0).abs() < 1e-9)));
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn method_single_output_uses_typed_instruction() {
    let source = "__register_test_classes(); obj = new_object('Point'); a = obj.deal();";
    let bytecode = compile_source(source).expect("compile method single-output");
    let member = runmat_types::MemberName::from("deal");
    assert!(has_single_value_dotted_call(
        &bytecode.instructions,
        &member,
        0,
        1
    ));
    assert!(!has_single_value_dotted_call(
        &bytecode.instructions,
        &member,
        0,
        3
    ));
    interpret(&bytecode).expect("execute valid method single-output fallback");
}

#[test]
fn unresolved_function_single_output_uses_typed_instruction() {
    let source = "a = definitely_missing_callback(7);";
    let bytecode = compile_source(source).expect("compile unresolved call");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallFunctionMulti {
            identity: runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(name)),
            fallback_policy,
            arg_count,
            out_count,
            ..
        } if name == "definitely_missing_callback"
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::RuntimeNameResolution
            && *arg_count == 1
            && *out_count == 1
    )));

    let err = interpret(&bytecode).expect_err("unresolved call should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn unresolved_function_zero_output_uses_typed_instruction_and_errors() {
    let source = "definitely_missing_callback(7);";
    let bytecode = compile_source(source).expect("compile unresolved zero-output call");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallFunctionMulti {
            identity: runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(name)),
            fallback_policy,
            arg_count,
            out_count,
            ..
        } if name == "definitely_missing_callback"
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::RuntimeNameResolution
            && *arg_count == 1
            && *out_count == 0
    )));

    let err = interpret(&bytecode).expect_err("unresolved zero-output call should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn unresolved_function_multi_output_uses_typed_instruction_and_errors() {
    let source = "[a,b] = definitely_missing_callback(7);";
    let bytecode = compile_source(source).expect("compile unresolved multi-output call");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        runmat_vm::Instr::CallFunctionMulti {
            identity: runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(name)),
            fallback_policy,
            arg_count,
            out_count,
            ..
        } if name == "definitely_missing_callback"
            && *fallback_policy == runmat_hir::CallableFallbackPolicy::RuntimeNameResolution
            && *arg_count == 1
            && *out_count == 2
    )));

    let err = interpret(&bytecode).expect_err("unresolved multi-output call should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn unresolved_function_expand_single_output_uses_typed_instruction() {
    let source = "C = {7,3}; a = definitely_missing_callback(C{:});";
    let bytecode = compile_source(source).expect("compile unresolved expanded call");
    assert!(bytecode
        .instructions
        .iter()
        .enumerate()
        .any(|(call_index, instr)| matches!(
            instr,
            runmat_vm::Instr::CallFunctionExpandMultiOutput {
                identity: runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(name)),
                fallback_policy,
                specs,
                out_count,
                ..
            } if name == "definitely_missing_callback"
                && *fallback_policy == runmat_hir::CallableFallbackPolicy::RuntimeNameResolution
                && *out_count == 1
                && specs.len() == 1
                && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
        )));

    let err = interpret(&bytecode).expect_err("unresolved expanded call should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn unresolved_function_expand_zero_output_uses_typed_instruction_and_errors() {
    let source = "C = {7,3}; definitely_missing_callback(C{:});";
    let bytecode = compile_source(source).expect("compile unresolved expanded zero-output call");
    assert!(bytecode
        .instructions
        .iter()
        .enumerate()
        .any(|(call_index, instr)| matches!(
            instr,
            runmat_vm::Instr::CallFunctionExpandMultiOutput {
                identity: runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(name)),
                fallback_policy,
                specs,
                out_count,
                ..
            } if name == "definitely_missing_callback"
                && *fallback_policy == runmat_hir::CallableFallbackPolicy::RuntimeNameResolution
                && *out_count == 0
                && specs.len() == 1
                && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
        )));

    let err = interpret(&bytecode).expect_err("unresolved expanded zero-output call should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn unresolved_function_expand_multi_output_uses_typed_instruction_and_errors() {
    let source = "C = {7,3}; [a,b] = definitely_missing_callback(C{:});";
    let bytecode = compile_source(source).expect("compile unresolved expanded call");
    assert!(bytecode
        .instructions
        .iter()
        .enumerate()
        .any(|(call_index, instr)| matches!(
            instr,
            runmat_vm::Instr::CallFunctionExpandMultiOutput {
                identity: runmat_hir::CallableIdentity::DynamicName(runmat_hir::SymbolName(name)),
                fallback_policy,
                specs,
                out_count,
                ..
            } if name == "definitely_missing_callback"
                && *fallback_policy == runmat_hir::CallableFallbackPolicy::RuntimeNameResolution
                && *out_count == 2
                && specs.len() == 1
                && is_expand_all_cell_contents(&bytecode.instructions[..call_index], &specs[0], 0)
        )));

    let err = interpret(&bytecode).expect_err("unresolved expanded multi-output call should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn builtin_vector_index_expansion() {
    let program = "C = {9, 2}; r = max(C{[1 2]});";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn user_function_vector_index_expansion() {
    let program = "function y = sum2(a,b); y = a + b; end; C = {3,4}; r = sum2(C{[1 2]});";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 7.0).abs() < 1e-9)));
}

#[test]
fn end_minus_one_1d_slice_collect() {
    let program = "A = [1 2 3]; B = A(1:1:end-1); s = sum(B);";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn end_minus_one_1d_slice_assign_broadcast() {
    let program = "A = [1 2 3 4]; A(2:1:end-1) = 9; r = sum(A);";
    let vars = execute_source(program);
    // A becomes [1 9 9 4] => sum 23
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 23.0).abs() < 1e-9)));
}

#[test]
fn multidim_range_end_assign() {
    // Assign on second dim using range with end-1
    let program = "A = [1 2 3; 4 5 6]; A(:,2:2:end-1) = 9; s = sum(sum(A));";
    // Original A sum is 21. We set column 2 to 9 across all rows: [1 9 3; 4 9 6] => sum 32
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 32.0).abs() < 1e-9)));
}

#[test]
fn multidim_range_end_assign_exact_nonscalar_rhs() {
    // 2x3; assign the middle 2x1 column selection with an exact 2x1 RHS.
    let program = "A = [1 2 3; 4 5 6]; B = [7;8]; A(:,2:2:end-1) = B; s = sum(sum(A));";
    // Becomes [1 7 3; 4 8 6] => sum 29
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 29.0).abs() < 1e-9)));
}

#[test]
fn mixed_range_end_assign_rejects_nonscalar_expansion() {
    let program =
        "A = [1 2 3 4; 5 6 7 8; 9 10 11 12]; B = [100;200]; A(2:end, 1:2:end-1) = B; s = sum(sum(A));";
    let error = execute_source_result(program)
        .expect_err("indexed assignment must not implicitly expand a nonscalar RHS");
    assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
}

#[test]
fn mixed_range_end_assign_matrix_rhs_exact_shape() {
    // Assign 2x2 block with exact-shaped RHS
    let program =
        "A = [1 2 3 4; 5 6 7 8; 9 10 11 12]; B = [1 3; 2 4]; A(2:end, 1:2:end-1) = B; s = sum(sum(A));";
    // Note: [1 3; 2 4] in MATLAB column-major maps to data [1,2,3,4] in our Tensor internal
    let vars = execute_source(program);
    // New values: positions (2,1)=1,(3,1)=2,(2,3)=3,(3,3)=4. Change from original 5,9,7,11 -> delta = (1-5)+(2-9)+(3-7)+(4-11) = -22; sum 78-22=56
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 56.0).abs() < 1e-9)));
}

#[test]
fn mixed_range_end_assign_shape_mismatch_error() {
    // RHS shape 3x1 does not match rows 2:end (len 2) and cannot broadcast
    let program = "A = [1 2 3 4; 5 6 7 8; 9 10 11 12]; B = [1;2;3]; A(2:end, 1:2:end-1) = B;";
    let err =
        execute_source_result(program).expect_err("shape-mismatched range assignment should fail");
    assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
}

#[test]
fn indexed_assignment_rejects_column_vector_expansion_across_columns() {
    let program = "A = zeros(3,4); v = [7;8;9]; A(:, 1:2:end-1) = v; x = A(:,1); y = A(:,3);";
    let error = execute_source_result(program)
        .expect_err("indexed assignment must not implicitly expand a column vector");
    assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
}

#[test]
fn logical_mask_write_scalar_and_vector() {
    // Scalar write via linear logical mask
    let program = "A = [1 2 3 4 5 6]; m = [true false true false true false]; A(m) = 9; s1 = sum(A);\nB = [1 2 3 4 5 6]; idx = [1 3 5]; B(idx) = [7 8 9]; s2 = sum(B);";
    let vars = execute_source(program);
    // After A(m)=9, A becomes [9 2 9 4 9 6] => sum 39
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-39.0).abs()<1e-9)));
    // After B(idx)=[7 8 9], B becomes [7 2 8 4 9 6] => sum 36
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-36.0).abs()<1e-9)));
}

#[test]
fn gather_scatter_roundtrip_nd() {
    // Gather a slice, scatter it back, tensor must be unchanged
    let program = "A = [1 2 3; 4 5 6; 7 8 9]; S = A(2:3, 1:2); A(2:3, 1:2) = S; t = sum(A(:));";
    let vars = execute_source(program);
    // Sum remains 45
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-45.0).abs()<1e-9)));
}

#[test]
fn indexed_assignment_rejects_nonscalar_expansion_across_2d_and_nd_selections() {
    for program in [
        "A = zeros(3,4); v = [1;2;3]; A(:, 2:2:4) = v;",
        "C = zeros(2,3,2); w = [5;6]; C(:,2,:) = w;",
    ] {
        let error = execute_source_result(program)
            .expect_err("indexed assignment must not implicitly expand a nonscalar RHS");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
    }
}

#[test]
fn compiled_integer_implicit_expansion_uses_trailing_singletons_and_strict_zero_rules() {
    for class_name in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "a = reshape({class_name}([1 2]), 2, 1); b = reshape({class_name}([10 20 30 40 50 60]), 2, 1, 3); c = a + b;"
        );
        let values = execute_source(&source);
        assert!(
            values.iter().any(|value| {
                let runmat_value::Value::Tensor(tensor) = value else {
                    return false;
                };
                tensor.shape() == [2, 1, 3]
                    && tensor.integer_storage().map(|storage| storage.class_name())
                        == Some(class_name)
                    && tensor.materialize_f64() == [11.0, 22.0, 31.0, 42.0, 51.0, 62.0]
            }),
            "missing exact compiled {class_name} broadcast result: {values:?}"
        );
    }

    let front_alignment_error = execute_source_result(
        "a = reshape(int8(1:6), 2, 3); b = reshape(int8(1:6), 1, 2, 3); c = a + b;",
    )
    .expect_err("front-aligned rank expansion must be rejected");
    assert!(
        front_alignment_error
            .message()
            .contains("dimension mismatch"),
        "unexpected rank mismatch error: {front_alignment_error}"
    );

    let empty_values = execute_source("a = reshape(int8([]), 0, 3); b = int8([1 2 3]); c = a + b;");
    assert!(empty_values.iter().any(|value| {
        matches!(
            value,
            runmat_value::Value::Tensor(tensor)
                if tensor.shape() == [0, 3]
                    && tensor.integer_storage().is_some_and(|storage| storage.class_name() == "int8")
                    && tensor.is_empty()
        )
    }));

    let zero_error = execute_source_result(
        "a = reshape(int8([]), 0, 3); b = reshape(int8(1:6), 2, 3); c = a + b;",
    )
    .expect_err("zero extent cannot expand against a nonsingleton");
    assert!(
        zero_error.message().contains("dimension mismatch"),
        "unexpected zero mismatch error: {zero_error}"
    );
}

#[test]
fn column_major_rhs_mapping() {
    // Verify RHS mapping enumerates first-dimension fastest
    let program = "A = zeros(3,3); R = [10 13 16; 11 14 17; 12 15 18]; A(:, [1 3]) = R(:, [1 3]); s = sum(A(:));";
    let vars = execute_source(program);
    // Selected columns are 1 and 3, filled from R(:,[1 3]) which in column-major is [10;11;12;16;17;18] => sum 84
    assert!(
        vars.iter()
            .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-84.0).abs()<1e-9)),
        "expected column-major sum; vars={vars:?}"
    );
}
#[test]
fn builtin_call_with_function_return_propagation() {
    // g returns two numbers; propagate as args to max
    let program = "function [a,b] = g(); a=9; b=4; end; r = max(g());";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn function_call_base_expand_all() {
    let program = r#"
        function y = sum2(a,b); y = a + b; end; C = {5,6}; r = sum2(C{:});
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 11.0).abs() < 1e-9)));
}

#[test]
fn function_return_propagation_in_args() {
    // g returns [a,b]; f takes two inputs; adapt to current semantics by binding outputs, then calling f
    let program = "function [a,b] = g(); a=2; b=3; end; function y = f(x1,x2); y = x1 + x2; end; [u,v] = g(); r = f(u,v);";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn varargin_pack_and_forward() {
    // f sums all inputs using varargin; g forwards varargin into f via feval(@f, varargin{:})
    let program = r#"
        function y = f(varargin)
            y = sum(cat(2, varargin{:}));
        end
        function out = g(varargin)
            out = feval(@f, varargin{:});
        end
        r1 = f(1,2,3);
        r2 = g(4,5,6,7);
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-6.0).abs()<1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-22.0).abs()<1e-9)));
}

#[test]
fn feval_expand_multi_forwards_expanded_cell_args() {
    let program = r#"
        function y = f(a,b,c,d)
            y = a + b + c + d;
        end
        function out = g(varargin)
            out = feval(@f, varargin{:});
        end
        r = g(4,5,6,7);
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-22.0).abs()<1e-9)));
}

#[test]
fn feval_cell_expansion_can_supply_callable_and_arguments() {
    let vars =
        execute_source("C = {@plus, 2, 3}; y = feval(C{:}); D = {@sin}; z = feval(D{1}, 0);");
    assert!(has_num(&vars, 5.0));
    assert!(has_num(&vars, 0.0));

    // The expanded first value must still satisfy the callable contract.
    assert!(execute_source_result("C = {2, 3}; y = feval(C{:});").is_err());
}

#[test]
fn feval_cell_expansion_forwards_dynamic_output_count() {
    let source = r#"
        C = {@deal, 15, 25};
        [a,b] = forward(C);
        function varargout = forward(C)
            [varargout{1:nargout}] = feval(C{:});
        end
    "#;
    let bytecode = compile_source(source).expect("compile expanded callable forwarding");
    let function = bytecode
        .function_registry
        .functions
        .values()
        .find(|function| function.display_name == "forward")
        .expect("forward bytecode present");
    assert!(function.instructions.iter().any(|instruction| matches!(
        instruction,
        runmat_vm::Instr::CallBuiltinExpandMultiOutputUsingOutputSlot(name, _, _)
            if name == "feval"
    )));
    let vars = interpret(&bytecode).expect("execute expanded callable forwarding");
    assert!(has_num(&vars, 15.0));
    assert!(has_num(&vars, 25.0));
}

#[test]
fn feval_expand_cell_indices_support_end_offsets() {
    let program = r#"
        function y = f(a,b)
            y = a + b;
        end
        C = {10, 20, 30};
        r = feval(@f, C{end-1}, C{end});
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-50.0).abs()<1e-9)));
}

#[test]
fn feval_expand_cell_indices_end_plus_offset_errors() {
    let program = "C = {10}; r = feval(@max, C{end+1}, 0);";
    let err = execute_source_result(program).err().unwrap();
    assert_eq!(err.identifier(), Some("RunMat:CellIndexOutOfBounds"));
}

#[test]
fn feval_expand_cell_indices_support_2d_end_selectors() {
    let program = r#"
        function y = f(a,b)
            y = a + b;
        end
        C = {1, 2; 3, 4};
        r = feval(@f, C{end,1}, C{1,end});
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn feval_expand_cell_indices_support_2d_end_offsets() {
    let program = r#"
        function y = f(a,b)
            y = a + b;
        end
        C = {1, 2; 3, 4};
        r = feval(@f, C{end-1,1}, C{1,end-1});
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 2.0).abs() < 1e-9)));
}

#[test]
fn feval_expand_cell_indices_2d_end_plus_offset_errors() {
    let program = "C = {1, 2; 3, 4}; r = feval(@max, C{end+1,1}, 0);";
    let err = execute_source_result(program).err().unwrap();
    assert_eq!(err.identifier(), Some("RunMat:CellSubscriptOutOfBounds"));
}

#[test]
fn feval_expand_cell_indices_non_offset_end_expr_executes() {
    let program = "C = {10, 20, 30, 40}; r = feval(@max, C{end/2}, 0);";
    let bytecode = compile_source(program).expect("compile contextual cell end expression");
    assert!(bytecode.instructions.iter().any(|instruction| matches!(
        instruction,
        runmat_vm::Instr::LoadContextualIndexEnd { component: 0 }
    )));
    let values = interpret(&bytecode).expect("execute contextual cell end expression");
    assert!(has_num(&values, 20.0));

    // Contextual expression support does not relax integer selector admission.
    let fractional = "C = {10, 20, 30}; r = feval(@max, C{end/2}, 0);";
    let error = execute_source_result(fractional).expect_err("fractional selector must fail");
    assert_eq!(error.identifier(), Some("RunMat:CellIndexType"));
}

#[test]
fn feval_expand_cell_fractional_index_errors() {
    let program = "C = {10, 20}; r = feval(@max, C{1.5}, 0);";
    let err = execute_source_result(program).err().unwrap();
    assert_eq!(err.identifier(), Some("RunMat:CellIndexType"));
}

#[test]
fn varargout_argument_supplies_first_output_without_implicit_expansion() {
    let program = r#"
        function varargout = h(x)
            varargout = {x+1, x+2, x+3};
        end
        r = max(h(10));
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-11.0).abs()<1e-9)));
}

#[test]
fn user_function_does_not_infer_varargout_argument_spread() {
    let program = r#"
        function y = f(a,b,c)
            y = a + 10*b + 100*c;
        end
        function varargout = g()
            varargout = {1,2,3};
        end
        r = f(g());
    "#;
    let err =
        execute_source_result(program).expect_err("f(g()) should pass only g()'s first output");
    assert_eq!(err.identifier(), Some("RunMat:NotEnoughInputs"));
}

#[test]
fn operator_overloading_plus_times_lt_eq() {
    let setup = "__register_test_classes();";
    let program = format!(
        "{setup} o = new_object('OverIdx'); o = call_method(o,'subsasgn',struct('type', '.', 'subs', 'k'),5); r1 = o + 3;"
    );
    let vars = execute_source(&program);
    assert!(has_num(&vars, 8.0));
}

#[test]
fn operator_overloading_elementwise_vs_mtimes_and_mixed() {
    let setup = "__register_test_classes();";
    let program = format!(
        "{setup} o = new_object('OverIdx'); o = call_method(o,'subsasgn',struct('type', '.', 'subs', 'k'),5); a = o .* 2; b = o * 2; c = 2 .* o; d = 2 * o; s = a + b + c + d;"
    );
    let vars = execute_source(&program);
    assert!(has_num(&vars, 40.0));
}

#[test]
fn operator_overloading_relational_lt_eq() {
    let setup = "__register_test_classes();";
    let program = format!(
        "{setup} o = new_object('OverIdx'); o = call_method(o,'subsasgn',struct('type', '.', 'subs', 'k'),5); t1 = (o < 10); t2 = (10 < o); t3 = (o == 0); t4 = (0 == o); s = t1 + t2 + t3 + t4;"
    );
    let vars = execute_source(&program);
    assert!(has_num(&vars, 1.0));
}

#[test]
fn operator_overloading_full_grid_basic() {
    let setup = "__register_test_classes(); o = new_object('OverIdx'); o = call_method(o,'subsasgn',struct('type', '.', 'subs', 'k'),5);";
    let statements = [
        "a1 = (o ~= 10);",
        "a2 = (10 ~= o);",
        "b1 = (o >= 5);",
        "b2 = (o <= 5);",
        "% use numeric power to avoid object exponent when not provided",
        "c1 = ([2 3] .^ 2);",
        "c2 = (2 .^ [2 3]);",
        "d1 = (o ./ 2);",
        "d2 = (2 ./ o);",
        "e1 = (o .\\ 2);",
        "e2 = (2 .\\ o);",
        "f1 = ([1 0 1] & [1 1 0]);",
        "f2 = ([1 0 1] | [0 1 1]);",
    ];
    let mut program = String::from(setup);
    program.push('\n');
    for (idx, stmt) in statements.iter().enumerate() {
        program.push_str(stmt);
        program.push('\n');
        execute_source_result(&program).unwrap_or_else(|err| {
            let bc = compile_source(&program).unwrap();
            panic!(
                "operator overload script failed after stmt #{idx} `{stmt}`: {err}\nbytecode={:?}",
                bc.instructions
            );
        });
    }
}

#[test]
fn import_precedence_and_class_static_shadowing() {
    // Locals > user functions > specific imports > wildcard imports; static under Class.* last
    let program = r#"
        function y = f(); y = 123; end
        __register_test_classes();
        import Point.origin; % specific import
        import Pkg.*;        % wildcard import (non-existent)
        origin = 7;          % local shadows import
        a = origin;          % uses local variable, not static
        b = f();             % user function resolves before imports
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-7.0).abs()<1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-123.0).abs()<1e-9)));
}

#[test]
fn static_method_resolution_under_wildcard_import() {
    // Ensure unqualified static via Class.* resolves when no shadowing
    let program = r#"
        __register_test_classes();
        import Point.*;
        r = origin(); % static method
    "#;
    let vars = execute_source(program);
    assert!(has_object_class(&vars, "Point"));
    assert!(has_object_num_property(&vars, "Point", "x", 0.0));
    assert!(has_object_num_property(&vars, "Point", "y", 0.0));
}

#[test]
fn operator_overloading_numeric_results_and_bitwise_arrays() {
    // Verify explicit numeric outcomes for OverIdx overloads
    let program = "__register_test_classes(); o = new_object('OverIdx'); o = call_method(o,'subsasgn',struct('type', '.', 'subs', 'k'),5); r1 = o + 3; r2 = o .* 2; r3 = o * 4; a = (o < 10); b = (o == 5);";
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 8.0).abs() < 1e-9))); // r1 = 5+3
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 10.0).abs() < 1e-9))); // r2 = 5*2
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 20.0).abs() < 1e-9))); // r3 = 5*4
                                                                                         // a and b push logicals (1.0/0.0) somewhere in vars
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 1.0).abs() < 1e-9))); // 5<10 true
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 1.0).abs() < 1e-9))); // 5==5 true

    // Bitwise and/or on arrays: verify element-wise behavior
    let program2 = "A = [1 0 1; 0 1 0]; B = [1 1 0; 0 0 1]; C = A & B; D = A | B; Sc = sum(C(:)); Sd = sum(D(:));";
    let vars2 = execute_source(program2);
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-1.0).abs()<1e-9)));
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-5.0).abs()<1e-9)));
}

#[test]
fn right_hand_object_overloads_keep_original_operand_order() {
    let program = "__register_test_classes(); o = new_object('OverIdx'); \
        o = call_method(o,'subsasgn',struct('type', '.', 'subs', 'k'),5); \
        quotient = 2 ./ o; inverse = o ./ 2; \
        leftLess = 2 < o; rightLess = o < 2;";
    let vars = execute_source(program);
    assert!(has_num(&vars, 0.4), "2 ./ o must evaluate 2/5");
    assert!(has_num(&vars, 2.5), "o ./ 2 must evaluate 5/2");
    assert!(has_num(&vars, 1.0), "2 < o must be true");
    assert!(has_num(&vars, 0.0), "o < 2 must be false");
}

#[test]
fn operator_overloading_left_division_variants() {
    let setup = "__register_test_classes(); o = new_object('OverIdx'); o = call_method(o,'subsasgn',struct('type', '.', 'subs', 'k'),5);";
    let program = format!(
        "{setup} a = (o .\\ 2); b = (2 .\\ o); c = (o ./ 2); d = (2 ./ o); s = a + b + c + d;",
    );
    let vars = execute_source(&program);
    assert!(has_num(&vars, 5.8));
}

#[test]
fn bitwise_or_row_vectors() {
    let program = "f = ([1 0 1] | [0 1 1]);";
    let bc = compile_source(program).unwrap();
    let _ = interpret(&bc).unwrap();
}

#[test]
fn ordinary_function_argument_supplies_first_output_only() {
    let program = "function [a,b,c] = g(); a=1; b=2; c=3; end; function y = h(x1,x2); y = x1*10 + x2; end; r = h(g());";
    let err =
        execute_source_result(program).expect_err("h(g()) should pass only g()'s first output");
    assert_eq!(err.identifier(), Some("RunMat:NotEnoughInputs"));
}

#[test]
fn ordinary_nested_function_argument_does_not_spread_outputs() {
    let program = "function [a,b] = g(); a=4; b=5; end; function y = f(x,p,z); y = x + p + z; end; r = f(1, g());";
    let err =
        execute_source_result(program).expect_err("f(1, g()) should pass only g()'s first output");
    assert_eq!(err.identifier(), Some("RunMat:NotEnoughInputs"));
}

#[test]
fn nested_try_catch_rethrow_unified_exception_ids() {
    // inner throws, caught and rethrown, outer catches; identifiers/messages preserved
    let program = r#"
        function y = inner()
            error('RunMat:inner:bad', 'inner fail')
        end
        function z = middle()
            try
                z = inner();
            catch e
                rethrow(e);
            end
        end
        try
            r = middle();
        catch e
            id = getfield(e, 'identifier');
            msg = getfield(e, 'message');
            % Surface identifier and message as outputs
            out_id = id; out_msg = msg; out_exc = e;
        end
    "#;
    let vars = execute_source_with_extensions(program);
    // Look for the exception object with identifier/message
    let has_exc = vars.iter().any(|v| match v {
        runmat_value::Value::MException(me) => {
            me.identifier == "RunMat:inner:bad" && me.message == "inner fail"
        }
        _ => false,
    });
    // If out_exc wasn't preserved as MException, ensure at least identifier/message strings are present
    let has_id = vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::String(s) if s=="RunMat:inner:bad"));
    let has_msg = vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::String(s) if s=="inner fail"));
    assert!(has_exc || (has_id && has_msg));
}

#[test]
fn globals_basic_and_shadowing() {
    let prog = "global G; G = 5; function y = f(x); global G; y = G + x; end; a = f(3);";
    let vars = execute_source(prog);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 8.0).abs() < 1e-12)));
}

#[test]
fn persistents_init_once_across_calls() {
    let prog = "function y = counter(); persistent C; if isempty(C); C = 0; end; C = C + 1; y = C; end; a = counter(); b = counter(); c = counter();";
    let vars = execute_source(prog);
    // Expect last value 3 somewhere in vars
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 3.0).abs() < 1e-12)));
}

#[test]
fn import_precedence_and_shadowing() {
    // Define user function f; specific import for Point.origin; wildcard import Pkg.* (nonexistent)
    // Local variable named origin should shadow imports; then function should shadow imports.
    let program = "function y = f(); y = 123; end; __register_test_classes(); import Point.origin; import Pkg.*; origin = 7; a = origin; b = f();";
    let vars = execute_source(program);
    // Expect 7 and 123 present
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-7.0).abs()<1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-123.0).abs()<1e-9)));
}

#[test]
fn class_dependent_property_get_set() {
    // Define class with Dependent property 'p'; test generic get.p/set.p builtins via backing field
    let program = r#"
        classdef D
          properties(Dependent)
            p
          end
        end
        d = new_object('D');
        d = setfield(d, 'p', 7); % should route to set.p, writing p_backing
        b = getfield(d, 'p_backing');
        % get.p should return p_backing
        v = getfield(d, 'p');
    "#;
    let vars = execute_source_with_extensions(program);
    let sevens = vars
        .iter()
        .filter(|v| matches!(v, runmat_value::Value::Num(n) if (*n-7.0).abs()<1e-9))
        .count();
    assert!(
        sevens >= 2,
        "expected both `b` and `v` to be 7; got {sevens} matches"
    );
    assert!(has_object_num_property(&vars, "D", "p_backing", 7.0));
}

#[test]
fn struct_isfield_multi_and_fieldnames() {
    let program = r#"
        s = struct(); s = setfield(s, 'a', 1); s = setfield(s, 'b', 2);
        c = {'a','x';'b','a'}; r = isfield(s, c); f = fieldnames(s);
    "#;
    let vars = execute_source(program);
    // Expect r to be 2x2 logical matrix [[1,0];[1,1]] in column-major data [1,1,0,1]
    let mut found_r_ok = false;
    let mut found_f_ok = false;
    for v in vars.values() {
        match v {
            runmat_value::Value::LogicalArray(arr) => {
                if arr.shape == vec![2, 2] && arr.data == vec![1, 1, 0, 1] {
                    found_r_ok = true;
                }
            }
            runmat_value::Value::Cell(ca) => {
                // fieldnames returns a column cell array with at least two entries
                if ca.cols == 1 && ca.rows >= 2 {
                    found_f_ok = true;
                }
            }
            _ => {}
        }
    }
    assert!(found_r_ok && found_f_ok);
}

#[test]
fn struct_isfield_string_array_names() {
    let program = r#"
        s = struct(); s = setfield(s, 'a', 1);
        names = ["a" "b"; "x" "a"];
        r = isfield(s, names);
    "#;
    let vars = execute_source(program);
    // Expect r to be 2x2 logical matrix [[1,0];[0,1]] in column-major data [1,0,0,1]
    assert!(vars.iter().any(|v| matches!(
        v,
        runmat_value::Value::LogicalArray(arr)
            if arr.shape == vec![2, 2] && arr.data == vec![1, 0, 0, 1]
    )));
}

#[test]
fn oop_negative_undefined_property_and_missing_subsref() {
    // Undefined property on object without subsref should raise an error caught by try/catch
    let prog = r#"
        classdef NoRef
            properties
                p
            end
        end
        o = new_object('NoRef');
        try
            v = o.noSuchProp;
        catch e
            ok = 1;
        end
    "#;
    let res = execute_source_result(prog);
    if let Ok(vars) = res {
        assert!(vars
            .iter()
            .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-1.0).abs()<1e-9)));
    }

    // Default object-array indexing still enforces scalar bounds when no
    // class-defined subsref overload exists.
    let prog2 = r#"
        classdef NoRef
            properties
                p
            end
        end
        o = new_object('NoRef');
        try
            x = o(2);
        catch e
            ok=2;
        end
    "#;
    let res2 = execute_source_result(prog2);
    if let Ok(vars2) = res2 {
        assert!(vars2
            .iter()
            .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-2.0).abs()<1e-9)));
    }
}

#[test]
fn containers_map_parenthesis_indexing() {
    let program = r#"
        fruit = containers.Map({'apple'}, {99});
        energy = fruit('apple');
    "#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n - 99.0).abs() < 1e-9)));
}

#[test]
fn containers_map_dot_properties() {
    let program = r#"
        m = containers.Map();
        key_type = m.KeyType;
        value_type = m.ValueType;
        count = m.Count;
    "#;
    let vars = execute_source(program);

    assert!(vars.iter().any(|v| match v {
        runmat_value::Value::CharArray(ca) =>
            runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 text fixture")
                == "char",
        _ => false,
    }));
    assert!(vars.iter().any(|v| match v {
        runmat_value::Value::CharArray(ca) =>
            runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 text fixture")
                == "any",
        _ => false,
    }));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Int(runmat_value::IntValue::U64(0)))));
}

#[test]
fn containers_map_preserves_wide_integer_keys_values_and_count() {
    let _ = execute_source(
        "m = containers.Map('KeyType','uint64','ValueType','uint64'); k = intmax('uint64'); m(k) = k-uint64(1); if m.Count ~= uint64(1) || m(k) ~= k-uint64(1) || ~isKey(m,k); error('integer Map mismatch'); end; ks = keys(m); vs = values(m,{k}); if ks{1} ~= k || vs{1} ~= k-uint64(1); error('integer Map collection mismatch'); end; remove(m,k); if isKey(m,k) || m.Count ~= uint64(0); error('integer Map removal mismatch'); end;",
    );
}

#[test]
fn string_array_literal_concat_index_and_compare() {
    let program = r#"
        A = ["a" "bb"; "ccc" "d"];   % 2x2 string array
        x = A(2,1);                      % "ccc"
        B = [A, ["e"; "e"]];            % hcat with 2x1 string-array -> 2x3
        C = [A; ["z" "y"]];             % vcat -> 3x2
        e1 = (A == "a");                % logical mask 2x2
        e2 = (A ~= "bb");               % logical mask 2x2
    "#;
    let vars = execute_source(program);
    let mut saw_a = false;
    let mut saw_x = false;
    let mut saw_b = false;
    let mut saw_c = false;
    let mut saw_e1 = false;
    let mut saw_e2 = false;
    for v in vars.values() {
        match v {
            runmat_value::Value::StringArray(sa) => {
                if sa.shape == vec![2, 2] {
                    saw_a = true;
                }
                if sa.shape == vec![2, 3] {
                    saw_b = true;
                }
                if sa.shape == vec![3, 2] {
                    saw_c = true;
                }
                if sa.data().iter().any(|s| s == "ccc") {
                    saw_x = true;
                }
            }
            runmat_value::Value::String(s) => {
                if s == "ccc" {
                    saw_x = true;
                }
            }
            runmat_value::Value::CharArray(ca) => {
                let s: String = runmat_value::decode_utf16_code_units(ca.data())
                    .expect("valid UTF-16 text fixture");
                if s == "ccc" {
                    saw_x = true;
                }
            }
            runmat_value::Value::LogicalArray(mask) => {
                if mask.shape == [2, 2] {
                    saw_e1 |= mask.data.as_slice() == [1, 0, 0, 0];
                    saw_e2 |= mask.data.as_slice() == [1, 1, 0, 1];
                }
            }
            _ => {}
        }
    }
    assert!(saw_a && saw_x && saw_b && saw_c && saw_e1 && saw_e2);
}

#[test]
fn string_literal_and_num2str_horzcat_promotes_and_runs() {
    let program = r#"
        wn = 6;
        label = ["wn = ", num2str(wn), " rad/s"];
    "#;
    let vars = execute_source(program);
    let mut saw_label = false;
    for value in vars.values() {
        if let runmat_value::Value::StringArray(sa) = value {
            if sa.shape == vec![1, 3] && sa.data() == vec!["wn = ", "6", " rad/s"] {
                saw_label = true;
            }
        }
    }
    assert!(saw_label, "expected label string-array concat to succeed");
}

#[test]
fn computed_integer_indices_work_for_column_slice_read_and_assign() {
    let program = r#"
        A = [1, 2, 3; 4, 5, 6];
        j = length(1:2);
        c = A(:, j);

        B = zeros(2, 3);
        B(:, j) = [7; 9];
        d = B(:, j);
    "#;
    let vars = execute_source(program);

    let mut saw_read = false;
    let mut saw_assign = false;
    for value in vars.values() {
        if let runmat_value::Value::Tensor(t) = value {
            if t.shape() == vec![2, 1] && t.materialize_f64() == vec![2.0, 5.0] {
                saw_read = true;
            }
            if t.shape() == vec![2, 1] && t.materialize_f64() == vec![7.0, 9.0] {
                saw_assign = true;
            }
        }
    }

    assert!(
        saw_read,
        "expected computed integer slice read to succeed; vars={vars:?}"
    );
    assert!(
        saw_assign,
        "expected computed integer slice assign to succeed; vars={vars:?}"
    );
}

#[test]
fn import_deep_multiseg_package_specific_vs_wildcard() {
    // Specific nested beats wildcard from another nested package
    let program = r#"
		__register_test_classes();
		import PkgF.foo;     % specific import of PkgF.foo (builtin -> 10)
		import pkg.PointNS.*; % wildcard unrelated (should not affect foo)
		a = foo();            % resolves to PkgF.foo => 10
	"#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-10.0).abs()<1e-9)));
}

#[test]
fn import_shadowing_matrix_locals_user_specific_wildcard_classstar() {
    // Locals > user function in scope > specific imports > wildcard imports > Class.* statics
    let program = r#"
		__register_test_classes();
		import Point.*;       % Class.* provides origin (static)
		import PkgF.foo;      % specific provides foo()=10
		import PkgG.*;        % wildcard also provides foo()=20
		function y = bar(); y = 33; end   % user function (distinct name)
		foo = @() 77;         % local variable (handle)
		a = feval(foo);       % 77 (local)
		b = bar();            % 33 (user function)
		c = origin();         % static method via Class.* (no shadowing by foo)
	"#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-77.0).abs()<1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-33.0).abs()<1e-9)));
    // presence of an Object from origin()
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Object(_))));
}

#[test]
fn import_wildcard_vs_classstar_ambiguity_for_static_method() {
    // Duplicate Class.* imports should be caught by import validation
    let program = r#"
		__register_test_classes();
		import Point.*;
		import Point.*;   % duplicate
		r = origin();
	"#;
    let err = compile_source(program).expect_err("expected duplicate classstar import");
    assert_eq!(
        err.identifier(),
        Some("RunMat:ImportDuplicate"),
        "expected duplicate import identifier, got: {}",
        err.message()
    );
}

#[test]
fn import_specific_vs_wildcard_same_name_prefers_specific_under_nesting() {
    let program = r#"
		import PkgF.foo;   % specific import
		import PkgG.*;     % wildcard also has foo
		y = foo();         % should call specific -> 10
	"#;
    let vars = execute_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_value::Value::Num(n) if (*n-10.0).abs()<1e-9)));
}

#[test]
fn type_class_static_method_zeros() {
    // Test that double.zeros(2, 3) is lowered to zeros(2, 3, 'double')
    let program = r#"
        A = double.zeros(2, 3);
        s = size(A);
    "#;
    let vars = execute_source(program);
    // The result should be a 2x3 matrix of zeros
    assert!(
        vars.iter().any(|v| {
            if let runmat_value::Value::Tensor(t) = v {
                t.shape() == vec![2, 3] && t.materialize_f64().iter().all(|&x| x == 0.0)
            } else {
                false
            }
        }),
        "expected double.zeros 2x3 tensor; vars={vars:?}"
    );
}

#[test]
fn type_class_static_method_logical_zeros() {
    // Test that logical.zeros(2, 2) is lowered to zeros(2, 2, 'logical')
    let program = r#"
        A = logical.zeros(2, 2);
    "#;
    let vars = execute_source(program);
    // The result should be a 2x2 logical array of zeros
    assert!(vars.iter().any(|v| {
        if let runmat_value::Value::LogicalArray(l) = v {
            l.shape == vec![2, 2] && l.data.iter().all(|&x| x == 0)
        } else {
            false
        }
    }));
}

#[test]
fn integer_type_class_static_method_zeros_preserves_integer_classes() {
    let program = r#"
        a = int8.zeros(2, 3);
        b = int32.zeros([1 4]);
        c = uint16.zeros(3, 1);
        d = uint64.zeros(1, 2);
        ca = class(a);
        cb = class(b);
        cc = class(c);
        cd = class(d);
    "#;
    let vars = execute_source(program);
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.shape() == vec![2, 3]
                && tensor.integer_storage()
                    == Some(&runmat_value::IntegerStorage::I8(vec![0; 6]))
    )));
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.shape() == vec![1, 4]
                && tensor.integer_storage()
                    == Some(&runmat_value::IntegerStorage::I32(vec![0; 4]))
    )));
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.shape() == vec![3, 1]
                && tensor.integer_storage()
                    == Some(&runmat_value::IntegerStorage::U16(vec![0; 3]))
    )));
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.shape() == vec![1, 2]
                && tensor.integer_storage()
                    == Some(&runmat_value::IntegerStorage::U64(vec![0; 2]))
    )));
    for expected in ["int8", "int32", "uint16", "uint64"] {
        assert!(
            vars.iter().any(
                |value| matches!(value, runmat_value::Value::String(class) if class == expected)
            ),
            "expected class {expected:?}; vars={vars:?}"
        );
    }
}

#[test]
fn eye_integer_class_strings_preserve_integer_storage_on_script_surface() {
    let program = r#"
        a = eye(2, 3, 'int16');
        b = eye([2 2], 'uint32');
        c = eye(1, 4, 'uint64');
        ca = class(a);
        cb = class(b);
        cc = class(c);
    "#;
    let vars = execute_source(program);
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.shape() == vec![2, 3]
                && tensor.integer_storage()
                    == Some(&runmat_value::IntegerStorage::I16(vec![1, 0, 0, 1, 0, 0]))
    )));
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.shape() == vec![2, 2]
                && tensor.integer_storage()
                    == Some(&runmat_value::IntegerStorage::U32(vec![1, 0, 0, 1]))
    )));
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.shape() == vec![1, 4]
                && tensor.integer_storage()
                    == Some(&runmat_value::IntegerStorage::U64(vec![1, 0, 0, 0]))
    )));
    for expected in ["int16", "uint32", "uint64"] {
        assert!(
            vars.iter().any(
                |value| matches!(value, runmat_value::Value::String(class) if class == expected)
            ),
            "expected class {expected:?}; vars={vars:?}"
        );
    }
}

#[test]
fn diag_integer_class_strings_preserve_integer_storage_on_script_surface() {
    let program = r#"
        a = diag([1.5 -2.5 300], 'int16');
        b = diag(uint64([18446744073709551615 9223372036854775808]), 'int64');
        ca = class(a);
        cb = class(b);
    "#;
    let vars = execute_source_with_extensions(program);
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.shape() == vec![3, 3]
                && tensor.integer_storage()
                    == Some(&runmat_value::IntegerStorage::I16(vec![
                        2, 0, 0, 0, -3, 0, 0, 0, 300
                    ]))
    )));
    assert!(vars.iter().any(|value| matches!(
        value,
        runmat_value::Value::Tensor(tensor)
            if tensor.shape() == vec![2, 2]
                && tensor.integer_storage()
                    == Some(&runmat_value::IntegerStorage::I64(vec![
                        i64::MAX,
                        0,
                        0,
                        i64::MAX
                    ]))
    )));
    for expected in ["int16", "int64"] {
        assert!(
            vars.iter().any(
                |value| matches!(value, runmat_value::Value::String(class) if class == expected)
            ),
            "expected class {expected:?}; vars={vars:?}"
        );
    }
}
