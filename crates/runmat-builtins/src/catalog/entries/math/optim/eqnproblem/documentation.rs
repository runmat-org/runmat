use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("eqnproblem"), slug: Some("eqnproblem"),
    summary: "Create an equation problem from symbolic equality expressions.",
    description: "Store equation arrays and their original optimization-variable definitions in a scalar EquationProblem value.",
    keywords: &["eqnproblem", "EquationProblem", "Equations", "Variables", "solve"],
    related: &["optimvar", "optimexpr", "solve", "lsqlin", "fzero", "fsolve", "lsqnonlin"],
    sections: &[
        BuiltinDocumentationSection { heading: "Construction and properties", paragraphs: &[
            "p=eqnproblem creates an empty equation problem. Description defaults to an empty character row and Equations to []. Constructor property names are case-insensitive; direct property access uses Description, Equations and Variables. Duplicate constructor properties are rejected.",
            "Equations accepts an OptimizationEquality array or a scalar structure of named equality arrays. Inequalities are rejected. Assigning [] to a named equation removes that field. Equation field names must be valid identifiers and have a separate namespace from variable names.",
            "Variables is a read-only structure of the complete original variable definitions reachable from both sides of every equation. Distinct variable definitions with the same name cannot be combined. Variable references remain aliases of their original definitions; problem property edits create a new problem value without changing copied problem values.",
        ] },
        BuiltinDocumentationSection { heading: "Solving equations", paragraphs: &[
            "Use sol=solve(p), solve(p,x0), or [sol,fval,exitflag,output]=solve(p,x0,'Solver',name,'Options',options). x0 is a structure whose fields use original variable names and shapes. Solver and Options are the supported solve name/value parameters. Equation problems return at most four outputs and do not return lambda.",
            "The default solver is lsqlin for affine equations, fzero for one nonlinear residual in one unbounded scalar variable, fsolve for an unbounded nonlinear system, and lsqnonlin for nonlinear equations with variable bounds. Solver can explicitly select lsqlin, fzero, fsolve or lsqnonlin; incompatible geometry or options from another solver are rejected. Integer variable domains are not silently relaxed.",
            "Linear equations use the sparse residual matrix directly, without forming normal equations. Nonlinear systems use the selected residual controller and finite-difference Jacobians. fzero uses scalar bracket expansion and Brent interpolation/bisection. Nonlinear solvers require x0. Linear interior-point solving does not inspect an inactive x0; reflective and active-set methods use their selected starting-point contract.",
            "sol contains original variable names and shapes. fval is left minus right in the original equation-array shape when there is one equation entry, even when that entry has a name. Multiple named entries return a structure with those names and shapes. The reported residual comes from the retained source evaluation, without a final function reevaluation.",
            "Equation convergence code 1 returns EquationSolved (double value 1). StepSizeBelowTolerance and FunctionChangeBelowTolerance retain their solver-specific numeric reasons. The scalar fzero controller retains its numerical bracket/step convergence, including a small nonzero rounded residual; it does not require exact floating-point zero. Its output.rootcriterion identifies numerical-bracket-or-zero. This numerical status is not an exact-zero or global certificate.",
            "A least-squares stationary point with a nonzero residual is not an equation solution. A positive residual-solver stop is qualified against its equation tolerance; a nonzero residual returns NoFeasiblePointFound, meaning no solution was found, not a proof that no root exists. Limits preserve the solver's stopping reason and available candidate. output identifies the selected solver and algorithm, its counts and message; requesting output also requires the invocation's Clock service.",
        ] },
        BuiltinDocumentationSection { heading: "Resources and supported boundaries", paragraphs: &[
            "Model compilation and solver workspaces require an unbounded host-memory policy. Cancellation and operational failures propagate through the original invocation. EquationProblem uses the native value-class, rooting and copy-on-write owners; model DAG persistence and worker transport are not supported. No displayed-property serialization substitutes for the equation graph.",
            "EquationDerivative controls, automatic differentiation, global/multiple-start equation solving and additional equation solver selections are not implemented. Numerical and resource limits of the selected raw solver remain applicable.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "affine-equation", title: "Solve an affine equation",
            program: "x=optimvar('x');\np=eqnproblem('Equations',2*x==6);\n[s,r,flag]=solve(p);",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(abs(s.x-3)<1e-6); assert(abs(r)<1e-6); assert(double(flag)>0);" },
        },
        BuiltinExample { id: "scalar-equation", title: "Solve a scalar nonlinear equation",
            program: "x=optimvar('x');\np=eqnproblem('Equations',x^2==2);\n[s,r,flag]=solve(p,struct('x',1));",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(abs(s.x-sqrt(2))<1e-6); assert(abs(r)<1e-6); assert(double(flag)>0);" },
        },
    ],
    example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "MATLAB eqnproblem", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/eqnproblem.html") },
        BuiltinDocumentationLink { label: "MATLAB solve", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/optim.problemdef.optimizationproblem.solve.html") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Equation problem owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/optimmodel/equation.rs") }],
        verification: &[], notes: &[],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
pub(super) const READ: BuiltinDocumentation = BuiltinDocumentation {
    title: Some("EquationProblem.subsref"),
    slug: Some("equation-problem-subsref"),
    summary: "Read an original equation-problem property.",
    examples: &[],
    example_exemption: Some("Covered by eqnproblem property and solve examples."),
    ..DOCUMENTATION
};
pub(super) const WRITE: BuiltinDocumentation = BuiltinDocumentation {
    title: Some("EquationProblem.subsasgn"),
    slug: Some("equation-problem-subsasgn"),
    summary: "Publish an equation problem with a validated property edit.",
    examples: &[],
    example_exemption: Some("Covered by eqnproblem property and solve examples."),
    ..DOCUMENTATION
};
