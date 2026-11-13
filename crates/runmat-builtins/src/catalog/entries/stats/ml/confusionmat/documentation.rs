use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation{
authority:BuiltinDocumentationAuthority::Catalog,title:Some("confusionmat"),slug:Some("confusionmat"),
summary:"Compute a confusion matrix from true and predicted class labels.",description:"`confusionmat(group,grouphat)` compares true class labels in `group` with predicted labels in `grouphat` and returns a matrix whose rows are true classes and columns are predicted classes.",keywords:&["confusionmat","confusion matrix","classification","statistics","machine learning"],related:&["classify","fitctree"],
sections:&[
BuiltinDocumentationSection{heading:"Behavior",paragraphs:&["`group` and `grouphat` must contain the same number of labels.",
"MATLAB-compatible inputs are single, double, logical, string, character-row, categorical, and cell-string labels. In RunMat mode, all eight typed-integer classes are also accepted as an explicit extension, without conversion through floating point. Text label families can be mixed between string, char, cell-string, and categorical inputs.",
"Without an explicit order, numeric labels are sorted ascending, text labels follow first appearance across `group` then `grouphat`, categorical labels follow category order for observed categories, and logical labels are ordered false then true.",
"`confusionmat(group,grouphat,'Order',order)` uses the supplied class order, requires every nonmissing label in `group` and `grouphat` to appear in `order`, and includes zero rows or columns for extra order labels that do not appear in the data.",
"Rows with missing true or predicted labels are omitted. Numeric `NaN`, empty text, and typed missing strings are treated as missing labels. Literal `<missing>` text is an ordinary label.",
"`[C,order] = confusionmat(...)` returns the class labels corresponding to the rows and columns of `C`. Typed-integer extension inputs preserve their exact class and values in `order`, including int64 and uint64 values above flintmax."]},
BuiltinDocumentationSection{heading:"Execution and limits",paragraphs:&[
"MATLAB does not document full gpuArray support for confusionmat. RunMat mode accepts resident inputs as an explicit extension, gathers them to host, and performs exact label matching and count accumulation on CPU.",
"Counts are host double arrays. Floating label order is double; integer order retains its class. Resident execution gathers inputs and does not fuse into a device kernel.",
"At most two outputs are supported. Zero-output calls still validate inputs. The output matrix is limited to 25,000,000 cells.",
"Character labels are two-dimensional rows. Numeric, logical, string, and cell-string inputs must be vectors. Duplicate explicit Order labels are removed while retaining their first appearance."
]},
],
examples:&[BuiltinExample{id:"example-1",title:"Build a numeric confusion matrix",program:"truth = [1; 1; 2; 2];\npred = [1; 2; 2; 1];\n[C,order] = confusionmat(truth, pred)",display_output:Some("C is [1 1; 1 1] and order is [1; 2]."),
compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{source:"assert(isequal(C,[1 1;1 1])); assert(isequal(order,[1;2]));"},
},
BuiltinExample{id:"example-2",title:"Use an explicit class order",program:"C = confusionmat([\"cat\";\"dog\";\"dog\"], [\"dog\";\"dog\";\"cat\"], 'Order', [\"dog\";\"cat\"])",display_output:Some("Rows and columns are ordered as dog, then cat."),
compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
verification:BuiltinExampleVerification::Assertions{source:"assert(isequal(C,[1 1;1 0]));"},
}],
example_exemption:None,
faqs:&[BuiltinDocumentationFaq{question:"Does `confusionmat` normalize counts?",answer:"No. It returns raw counts. Normalize rows or columns explicitly after computing `C`."},
BuiltinDocumentationFaq{question:"How are labels not present in an explicit Order handled?",answer:"When `Order` is supplied, it must contain every nonmissing true and predicted label. Extra order labels are allowed and produce zero rows or columns."}],
links:&[BuiltinDocumentationLink{label:"classify",target:BuiltinDocumentationLinkTarget::Builtin("classify")},
BuiltinDocumentationLink{label:"fitctree",target:BuiltinDocumentationLinkTarget::Builtin("fitctree")},
BuiltinDocumentationLink{label:"Source",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/confusionmat/mod.rs")}],
media:&[],
evidence:BuiltinDocumentationEvidence{implementation:&[],verification:&[
BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"Label classes and count semantics",location:"builtins::stats::ml::confusionmat::tests"},
BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"VM statistics integration",location:"crates/runmat-vm/tests/stats_ml.rs"}
],notes:&["All legacy descriptive fields, examples, FAQ, links and test references retained. GPU/fusion notes are expressed as host gathering; null requires_feature omitted. Source link follows modular implementation."]},
introduced:None,status:Some(BuiltinDocumentationStatus::Stable)
};
