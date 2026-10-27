use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation{
    authority:BuiltinDocumentationAuthority::Catalog,title:Some("optimproblem"),slug:Some("optimproblem"),
    summary:"Create an optimization problem from original symbolic objectives and constraints.",
    description:"Store symbolic objective and constraint arrays with their original variable definitions. Problem property edits produce a new problem value while contained variable references remain live aliases of their original definitions.",
    keywords:&["optimproblem","OptimizationProblem","Objective","Constraints","ObjectiveSense","Variables"],
    related:&["optimvar","optimexpr","evaluate","linprog","quadprog","fmincon"],
    sections:&[
        BuiltinDocumentationSection{heading:"Construction and properties",paragraphs:&[
            "optimproblem creates a scalar OptimizationProblem. The defaults are an empty character Description, minimize ObjectiveSense, an empty OptimizationExpression Objective, and empty 0-by-0 structures for Constraints and Variables. Property names in constructor name/value pairs are case-insensitive; direct property names have their declared spelling.",
            "Description accepts scalar text and retains its accepted text carrier. Objective accepts an expression, a variable reference, a finite real numeric value, or a scalar structure of named objectives. Each named objective must be scalar; an unnamed objective can be a vector or N-D expression array. Constraints accepts an OptimizationConstraint comparison array or a scalar structure of named comparison arrays. The constructor accepts property pairs in either order and rejects duplicate pairs.",
            "ObjectiveSense accepts minimize/min or maximize/max. A uniform sense applies to all objectives. A structure of senses requires the same field names as a named Objective structure. Editing a named sense through a property path first expands the current uniform sense over the existing objective names.",
        ]},
        BuiltinDocumentationSection{heading:"Named edits and original definitions",paragraphs:&[
            "Use problem.Objective.name and problem.Constraints.name to add or replace named members. Assigning an empty double array to a named objective or constraint removes that member. Objective and constraint names must be valid identifiers and must not duplicate each other or any original variable name.",
            "Variables is a read-only structure containing the complete original variable definitions reachable from all objectives and both sides of every constraint. A subset reference does not introduce a new variable or copied bound authority. Different definitions with the same public name cannot be combined into one problem.",
            "Copy Variables to a separate structure before editing a contained variable's bounds. Those edits affect the original variable handles retained by every expression. Replacing a problem's Description, Objective, Constraints, or ObjectiveSense creates a new problem generation and does not overwrite another copied problem value.",
        ]},
        BuiltinDocumentationSection{heading:"Execution and supported scope",paragraphs:&[
            "Construction and property access use the original invocation context, retain model GC roots, and cooperate with cancellation. Only selected concrete collection fields are normalized. Invalid structural carriers are rejected before numeric conversion or rich property/capture inspection.",
            "Finite-memory graph/header/collection backing, model transport, arbitrary huge virtual references, expression growth, and native end selectors are unsupported. The shared physical index planner retains its materialization limits. The solve interface supports continuous scalar affine problems through linprog, with original shaped solution/multiplier fields and OptimizationExitflag conversions. Other solver families, OptimizationValues, display/export, and broader solver selection are unavailable. Use eqnproblem for equation problems and their separate residual outputs.",
            "A problem can retain an empty objective for a feasibility formulation, or named/vector objectives for a multiobjective formulation. This constructor does not invoke a solver or turn an empty objective into a scalar expression. Numerical solving and multiobjective result production are separate interfaces.",
        ]},
    ],
    examples:&[
        BuiltinExample{
            id:"problem-property-generation",title:"Retain original variables through a problem",
            program:"x=optimvar('decision',2,'LowerBound',0);\np=optimproblem('Description','allocation');\np.Objective=2*x(1)+x(2);\np.Constraints.capacity=sum(x)<=5;\nv=p.Variables;\nv.decision.UpperBound=4;\na=evaluate(p.Objective,struct('decision',[1;2]));\nb=evaluate(p.Constraints.capacity,struct('decision',[1;2]))",
            display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{
                source:"assert(a==4); assert(b==-2); assert(isequal(x.UpperBound,[4;4]));"},
        },
        BuiltinExample{
            id:"named-objective-senses",title:"Give named objectives distinct senses",
            program:"x=optimvar('x');\np=optimproblem;\np.Objective.first=x;\np.Objective.second=2*x;\np.ObjectiveSense.first='min';\np.ObjectiveSense.second='max';\ns=p.ObjectiveSense;",
            display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{
                source:"assert(strcmp(s.first,'minimize')); assert(strcmp(s.second,'maximize'));"},
        },
    ],
    example_exemption:None,faqs:&[],
    links:&[
        BuiltinDocumentationLink{label:"MATLAB optimproblem",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/optimproblem.html")},
        BuiltinDocumentationLink{label:"OptimizationProblem properties",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/optim.problemdef.optimizationproblem.html")},
    ],
    media:&[],evidence:BuiltinDocumentationEvidence{
        implementation:&[BuiltinDocumentationLink{label:"Native problem property owner",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/optimmodel/problem.rs")}],
        verification:&[BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,label:"Problem generation and property controls",location:"builtins::math::optim::optimmodel::problem::tests"}],
        notes:&[],
    },introduced:None,status:Some(BuiltinDocumentationStatus::Partial),
};
pub(super) const READ:BuiltinDocumentation=BuiltinDocumentation{
    title:Some("OptimizationProblem.subsref"),slug:Some("optimization-problem-subsref"),
    summary:"Read a selected original optimization problem property.",
    description:"Resolve the declared problem property once and delegate its remaining subscript path through the original-context native object protocol.",
    examples:&[],example_exemption:Some("Covered by the optimproblem public property examples."),
    ..DOCUMENTATION
};
pub(super) const WRITE:BuiltinDocumentation=BuiltinDocumentation{
    title:Some("OptimizationProblem.subsasgn"),slug:Some("optimization-problem-subsasgn"),
    summary:"Publish an optimization problem with a selected property edit.",
    description:"Validate the changed property and original variable/name contracts before publishing its successor problem generation.",
    examples:&[],example_exemption:Some("Covered by the optimproblem public property examples."),
    ..DOCUMENTATION
};
