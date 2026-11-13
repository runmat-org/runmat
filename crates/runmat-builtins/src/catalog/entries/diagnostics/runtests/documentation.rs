use crate::*;
mod examples;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("runtests"),
    slug: Some("runtests"),
    summary: "Discover and run MATLAB-style tests through RunMat's shared test engine.",
    description: "`runtests` freezes the selected sources, uses Core's semantic test discovery, executes exact test procedures through the shared lifecycle engine, and returns homogeneous `matlab.unittest.TestResult`-compatible object arrays.",
    keywords: &[
        "runtests",
        "unit testing",
        "test",
        "developer tools",
        "matlab.unittest"
    ],
    related: &[
        "assert",
        "run",
        "evalc"
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`results = runtests` discovers tests in the current folder.",
                "`results = runtests(target)` accepts a test file, folder, function name, string array, or cell array of string targets. Bare names are resolved like MATLAB `.m` files on the active search path.",
                "`BaseFolder` constrains relative explicit targets to one or more base folders. When no explicit target is supplied, the base folders act as discovery roots.",
                "Folder discovery includes files whose stem starts with `test` or ends with `Test`. `IncludeSubfolders` recursively includes subfolders.",
                "Script, `functiontests(localfunctions)`, and class-based tests use semantic procedure identities rather than generated source calls. Function, method, class, and shared fixtures retain declared setup/teardown ordering.",
                "Each test executes in an isolated Core test context. Variables assigned by a test do not leak into the caller workspace.",
                "Each result object includes `Name`, `TestFile`, `Passed`, `Failed`, `Incomplete`, `Duration`, `Details`, and qualification data. Multiple results use a homogeneous TestResult object array.",
                "A failing test is captured as a failed result rather than as a `runtests` call failure. Discovery, invalid option, and unsupported infrastructure errors still surface as diagnostics.",
                "`Name` and `ProcedureName` filters currently keep cases whose semantic display name contains the supplied text. They do not yet implement the full MATLAB selector model. `OutputDetail` and `LoggingLevel` are accepted for compatibility but do not affect execution output yet.",
                "`Coverage=true` enables the shared interpreter/JIT/WASM coverage collector. `CodeCoveragePlugin.forFolder` selects recursive or non-recursive coverage roots through the same policy.",
                "`UseParallel=false` is accepted. Product and CLI runners provide parallel process or Web Worker execution; in-function `UseParallel=true` remains unsupported."
            ]
        },
        BuiltinDocumentationSection {
            heading: "Host and accelerator behavior",
            paragraphs: &[
                "`runtests` delegates semantic discovery and lifecycle execution to Core. Test bodies may call GPU-capable builtins normally, but `runtests` itself has no provider kernel.",
                "`runtests` is a Core test-service boundary. It does not create GPU kernels and does not participate in fusion. Individual test bodies may exercise GPU-capable operations normally."
            ]
        },
        BuiltinDocumentationSection {
            heading: "Authority, resource policy, and limits",
            paragraphs: &[
                "Runtime retains the original argument owner while validating options. Resident selectors and option payloads reject as controls without a gather or provider lookup. Integer flags accept only exact scalar zero or one.",
                "Each selected file is read once by Runtime; Core freezes those exact bytes. Runtime never infers test functions by scanning source lines and never appends generated calls. Core semantic discovery supplies procedure identities, setup/teardown, and result names.",
                "A finite host-memory policy rejects before directory discovery or source reads because source, descriptor, execution, and result storage do not yet have a certified physical bound. This is not a claim of finite-budget completion."
            ]
        },
        BuiltinDocumentationSection {
            heading: "MATLAB API boundary",
            paragraphs: &[
                "This adapter returns at most one output; the MATLAB coverage-result second output is not implemented. The RunMat Coverage option is distinct from MATLAB ReportCoverageFor.",
                "Name and ProcedureName are substring filters, not wildcard selectors. BaseFolder supplies path-resolution roots rather than the MATLAB wildcard filter. Direct file/procedure and namespace/project selection, advanced dependency and parameter filters, and newer string-valued UseParallel modes are not implemented by this adapter. Unknown options reject explicitly."
            ]
        }
    ],
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `runtests` support Parallel Computing Toolbox execution?",
            answer: "No. `UseParallel=true` is rejected and tracked with the separate parallel/GPU effort. `UseParallel=false` is accepted."
        },
        BuiltinDocumentationFaq {
            question: "Does `runtests` support coverage?",
            answer: "Yes. Use `Coverage=true` or `matlab.unittest.plugins.CodeCoveragePlugin.forFolder`; RunMat collects stable function and statement sites across interpreter, JIT, and WASM execution."
        },
        BuiltinDocumentationFaq {
            question: "Does `runtests` return MATLAB object arrays?",
            answer: "Yes. Results are projected as homogeneous `matlab.unittest.TestResult`-compatible object arrays."
        },
        BuiltinDocumentationFaq {
            question: "Does `runtests` support the full `matlab.unittest` framework?",
            answer: "RunMat supports script, function, and class suites; `functiontests(localfunctions)`; common TestCase qualifications; fixtures; parameters; tags; plugins; and coverage. Unsupported MATLAB APIs fail explicitly rather than selecting a different execution path."
        },
        BuiltinDocumentationFaq {
            question: "Do test variables leak into the caller workspace?",
            answer: "No. Core runs the frozen test snapshot in a child test session. The caller workspace is not copied, mutated, or restored by this builtin."
        },
        BuiltinDocumentationFaq {
            question: "Can tests use GPU arrays?",
            answer: "Yes, when the test body calls GPU-capable builtins. `runtests` itself is not a numeric operation and does not own GPU dispatch."
        }
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "assert",
            target: BuiltinDocumentationLinkTarget::Builtin("assert")
        },
        BuiltinDocumentationLink {
            label: "run",
            target: BuiltinDocumentationLinkTarget::Builtin("run")
        },
        BuiltinDocumentationLink {
            label: "evalc",
            target: BuiltinDocumentationLinkTarget::Builtin("evalc")
        },
        BuiltinDocumentationLink {
            label: "MATLAB runtests reference",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/runtests.html")
        }
    ],
    media: &[
    ],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime runtests retained admission",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/diagnostics/runtests.rs")
            },
            BuiltinDocumentationLink {
                label: "Core frozen-source semantic executor",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-core/src/testing/runtime_adapter.rs")
            }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "builtins::diagnostics::runtests::tests",
                location: "builtins::diagnostics::runtests::tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "runtests_runs_script_test_file_and_restores_workspace",
                location: "runtests_runs_script_test_file_and_restores_workspace"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "runtests_records_failing_script_test_as_result",
                location: "runtests_records_failing_script_test_as_result"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "runtests_discovers_subfolder_tests_when_requested",
                location: "runtests_discovers_subfolder_tests_when_requested"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "runtests_no_arg_discovers_current_folder_without_recursing",
                location: "runtests_no_arg_discovers_current_folder_without_recursing"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "runtests_basefolder_constrains_relative_target_resolution",
                location: "runtests_basefolder_constrains_relative_target_resolution"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "runtests_accepts_useparallel_false",
                location: "runtests_accepts_useparallel_false"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "runtests_runs_zero_argument_function_tests",
                location: "runtests_runs_zero_argument_function_tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "runtests_rejects_parallel_execution_option",
                location: "runtests_rejects_parallel_execution_option"
            }
        ],
        notes: &[
            "The old workspace-snapshot FAQ was inaccurate: isolation uses the existing Core child-session executor, not a caller workspace copy. The old VM source label is replaced by the current Core adapter evidence.",
            "Source migration only: listed legacy test locations and independently asserted standalone examples are evidence pointers, not claims of fresh execution. Central integration must run focused canonical/runtime/Core and example qualification.",
            "Contract maturity remains Incomplete and documentation status Partial. Host-only placement and fusion boundaries do not certify full MATLAB compatibility, browser execution, positive Windows registry qualification, or finite resource completion."
        ]
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
