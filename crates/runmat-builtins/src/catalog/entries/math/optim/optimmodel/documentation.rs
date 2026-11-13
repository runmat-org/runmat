use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation{
    authority:BuiltinDocumentationAuthority::Catalog,title:Some("optimexpr"),slug:Some("optimexpr"),
    summary:"Create and evaluate arrays of optimization expressions.",
    description:"Build symbolic expressions from original optimization variables. Expressions retain their graph and variable definitions through ordinary copies and indexing; they do not contain numerical decision values.",
    keywords:&["optimexpr","OptimizationExpression","evaluate","OptimizationEquality","OptimizationInequality"],
    related:&["optimvar","optimproblem","evaluate","linprog","quadprog","fmincon"],
    sections:&[
        BuiltinDocumentationSection{heading:"Construction and assignment",paragraphs:&[
            "optimexpr with no arguments creates an empty expression accumulator. Adding or subtracting the first expression initializes its shape and original variable references. This empty initializer is distinct from a scalar numerical zero and from arrays of unassigned slots.",
            "zeros(dimensions...,'like',prototype) creates a zero-valued OptimizationExpression when prototype is an original expression or variable. Dimensions follow ordinary zeros: no dimensions gives 1-by-1, one scalar gives a square array, and a size vector or separate dimensions gives the requested N-D shape. An empty size vector gives 0-by-0. The result can be assigned symbolically without converting variable values to double.",
            "optimexpr(n) creates n-by-1 unassigned expression slots. Separate positive double dimensions, a dimension vector, named string/cell axes, and a cell of mixed axes create N-D arrays. A lone named vector preserves its orientation. Assign expressions into these slots with numeric, logical, colon or named selectors; repeated indices follow normal assignment order.",
            "Unassigned coordinates are not numeric zeros. Evaluating an expression that still depends on an unassigned coordinate reports an error. An assigned subset can be evaluated without first assigning unrelated slots. Expression edits create a new expression value; the retained original variable definitions remain shared.",
        ]},
        BuiltinDocumentationSection{heading:"Arithmetic and numerical functions",paragraphs:&[
            "Explicit modeling constants must be finite and real. Complex, Inf and NaN inputs are rejected during expression construction, including numeric function metadata. This restriction applies to construction constants. Numerical evaluation can produce nonfinite results, and variable bounds may be infinite.",
            "Addition, subtraction, unary signs, transpose, conjugate transpose, matrix/pointwise multiplication, constant division and constant scalar powers construct expression nodes. Matrix powers require a constant scalar exponent. Pointwise division requires a constant denominator (or numerator for left division). Black-box function conversion with fcn2optimexpr is unavailable.",
            "Supported numerical functions include sum, mean, prod, trace, cumsum, cumprod, diff, reshape, diag, dot, default or numeric-two norm, sqrt, exp, log, and the trigonometric, inverse-trigonometric, hyperbolic and inverse-hyperbolic functions. The norm is Euclidean for scalar/vector inputs and spectral for 2-D matrices; a matrix spectral norm square is not a squared-residual or quadratic certificate. cat, horzcat and vertcat concatenate expressions. Function options and array dimensions follow the corresponding numerical function. Construction rejects unsupported forms and forms whose shape cannot be established.",
        ]},
        BuiltinDocumentationSection{heading:"Properties and variable references",paragraphs:&[
            "The RunMat functional setfield extension accepts contextual end inside a native property index path. The actual property child is retained and indexed once, with the same native class and access checks as ordinary assignment.",
            "IndexNames contains one cell entry per axis; empty entries use numerical indexing. IndexNames can be edited directly or through an indexed cell property path. Variables is a read-only scalar structure containing complete original variable definitions reachable from the expression, including variables used only through subsets.",
            "Copy Variables into a separate structure to edit the bounds of its live variable handles. Different original definitions with the same name cannot be merged into one evaluation point. An unrelated user object or callback capture is not converted into a numeric model operand.",
        ]},
        BuiltinDocumentationSection{heading:"Evaluation and comparisons",paragraphs:&[
            "evaluate(expr,point) returns a double array with the expression's shape. point is a scalar structure whose selected fields have the names and complete original shapes of the reachable variables. Real host numeric point values use the canonical double conversion; each original definition is converted once per evaluation. Extra point fields are not inspected.",
            "Comparisons ==, <= and >= require equal shapes or one scalar side. Equality results have class optim.problemdef.OptimizationEquality; inequality results have class optim.problemdef.OptimizationInequality. evaluate(L<=R,point) is L-R, evaluate(L>=R,point) is R-L, and evaluate(L==R,point) is abs(L-R). IndexNames and Variables are available on comparison values.",
        ]},
        BuiltinDocumentationSection{heading:"Execution and current limits",paragraphs:&[
            "Model construction and evaluation use the original invocation context and cooperate with cancellation. Invalid structural carriers are rejected before property/capture traversal or numeric conversion. Variable definitions and expression dependencies remain managed GC roots throughout construction, evaluation and property edits.",
            "Finite host-memory graph construction/evaluation, model transport, expression growth, and constraint-array element indexing/assignment are unsupported. Virtual index metadata uses checked platform-sized offsets and allocates only the addressed selection, subject to the existing per-selection materialization limit. Evaluation of a selected uniform zero source materializes only the requested output. A nonempty unassigned selection remains an error; empty selections have no missing element. General arithmetic and full-array property results can still require physical storage for their complete inputs or outputs. optimconstr, optimeq, optimineq, OptimizationValues, expression display/export and fcn2optimexpr are unavailable. The problem-based solve interface supports continuous scalar affine problems; other model solver forms remain unavailable.",
        ]},
    ],
    examples:&[
        BuiltinExample{
            id:"zero-prototype-and-accumulator",title:"Initialize zero expressions and accumulate a scalar",
            program:"x=optimvar('x',2);\nz=zeros(size(x),'like',x);\nz(2)=sum(x);\ne=optimexpr;\nfor k=1:2, e=e+x(k); end;\np=struct('x',[3;4]);\nv=evaluate(z,p);\ns=evaluate(e,p)",
            display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(isequal(v,[0;7])); assert(s==7);"},
        },
        BuiltinExample{
            id:"expression-loop",title:"Fill expression slots and evaluate a point",
            program:"x=optimvar('x',2);\ne=optimexpr(2);\ne(1)=2*x(1)+x(2);\ne(2)=x(1)-3*x(2);\np=struct('x',[4;1]);\nv=evaluate(e,p)",
            display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(isequal(size(e),[2 1])); assert(isequal(v,[9;1]));"},
        },
        BuiltinExample{
            id:"comparison-directions",title:"Evaluate directed comparisons",
            program:"x=optimvar('x',2);\np=struct('x',[3;1]);\na=evaluate(x>=2,p);\nb=evaluate(x==2,p)",
            display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{source:"assert(isequal(a,[-1;1])); assert(isequal(b,[1;1]));"},
        },
    ],
    example_exemption:None,faqs:&[],
    links:&[
        BuiltinDocumentationLink{label:"Initialize optimization expressions",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/initialize-optimization-expressions.html")},
        BuiltinDocumentationLink{label:"Optimization expression construction",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/optimization-expressions.html")},
        BuiltinDocumentationLink{label:"MATLAB optimexpr",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/optimexpr.html")},
        BuiltinDocumentationLink{label:"Supported expression operations",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/supported-operations-on-optimization-variables-expressions.html")},
        BuiltinDocumentationLink{label:"MATLAB evaluate",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/optim.problemdef.optimizationexpression.evaluate.html")},
    ],
    media:&[],evidence:BuiltinDocumentationEvidence{
        implementation:&[BuiltinDocumentationLink{label:"Runtime expression owner",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/optimmodel.rs")}],
        verification:&[BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"Expression graph, assignment and evaluation controls",location:"builtins::math::optim::optimmodel::tests"}],notes:&[],
    },
    introduced:None,status:Some(BuiltinDocumentationStatus::Partial),
};
pub(super) const EVALUATE:BuiltinDocumentation=BuiltinDocumentation{
    title:Some("evaluate"),slug:Some("evaluate"),
    summary:"Evaluate an optimization expression or comparison at a point.",
    description:"Evaluate retained original variable references using a scalar point structure and the canonical numerical operations in the expression graph.",
    ..DOCUMENTATION
};
pub(super) const METHOD:BuiltinDocumentation=BuiltinDocumentation{
    title:None,slug:None,
    summary:"Operate on retained optimization expressions and comparisons.",
    description:"Standard optimization-model class methods construct expressions, project properties or evaluate the same original model graph. See optimexpr for operations, point fields, comparison directions and current scope.",
    examples:&[],example_exemption:Some("Covered by the optimexpr public expression and comparison examples."),
    ..DOCUMENTATION
};

pub(super) const fn method(title: &'static str, slug: &'static str) -> BuiltinDocumentation {
    BuiltinDocumentation {
        title: Some(title),
        slug: Some(slug),
        ..METHOD
    }
}
