use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("optimwarmstart"), slug: Some("optimwarmstart"),
    summary: "Create an immutable active-set least-squares or quadratic warm start.",
    description: "Retain an initial point, selected solver options and the active constraints returned by a previous solve.",
    keywords: &["optimwarmstart", "LsqlinWarmStart", "QuadprogWarmStart", "lsqlin", "quadprog", "active-set"],
    related: &["lsqlin", "quadprog", "optimoptions"],
    sections: &[
        BuiltinDocumentationSection { heading: "Construction and properties", paragraphs: &[
            "ws = optimwarmstart(x0,options) creates a scalar LsqlinWarmStart or QuadprogWarmStart according to the options Solver profile. options must identify lsqlin or quadprog with Algorithm active-set. x0 is a finite real host double array and X stores its elements as a column in column-major order. Options retains the supplied scalar options structure.",
            "X, Options, MaxLinearEqualities and MaxLinearInequalities are read-only, including edits below a property. Recreate the object to change them. Copying a warm start shares immutable storage; solving returns a new object and never mutates existing aliases.",
            "Optionally supply both MaxLinearEqualities and MaxLinearInequalities as name/value pairs in either order. Each is a positive scalar count or Inf, and both default to Inf. These properties describe code-generation memory limits; they do not restrict ordinary solver allocations or enable code generation.",
        ]},
        BuiltinDocumentationSection { heading: "Solving and reuse", paragraphs: &[
            "[wsout,resnorm,residual,exitflag,output,lambda] = lsqlin(C,d,A,b,Aeq,beq,lb,ub,ws) uses ws.Options and ws.X. Supply all eight numeric positions, using [] for absent constraints or bounds. Do not append options or put ws in a problem structure. Read the new solution from wsout.X.",
            "[wsout,fval,exitflag,output,lambda] = quadprog(H,f,A,b,Aeq,beq,lb,ub,ws) uses a QuadprogWarmStart in the same ninth position. Its objective is 0.5*X'*H*X + f'*X, not a least-squares residual norm. The two warm classes are not interchangeable.",
            "The variable count must match X. The solver projects the new initial point and establishes feasibility before reuse. Saved equalities must match; a changed equality discards saved active information. Other saved rows are reused only when their original row identity, coefficients and right-hand side match and they remain tight at the actual new seed. Changed rows or bounds are rebuilt through the ordinary active-set algorithm.",
            "The retained state is an active constraint set, not a cached numerical factorization or proof of optimality. Every call uses its current matrices, options, cancellation, clock and numerical validation. An unavailable numerical result produces an empty X and no reusable state; inspect exitflag before another solve.",
        ]},
        BuiltinDocumentationSection { heading: "Supported scope", paragraphs: &[
            "This interface supports host double lsqlin and quadprog active-set solves. Single-precision code generation, generated-code memory-limit enforcement, finite-workspace warm storage and warm-object transport are not implemented. The ordinary raw lsqlin calling forms and other algorithms remain separate interfaces.",
        ]},
    ],
    examples: &[BuiltinExample {
        id: "warm-least-squares", title: "Reuse an active lower bound",
        program: "options=optimoptions('lsqlin','Algorithm','active-set');\nws=optimwarmstart(0,options);\nold=ws;\n[first,r1]=lsqlin(1,-1,[],[],[],[],0,[],ws);\n[second,r2]=lsqlin(1,-2,[],[],[],[],0,[],first);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(first.X==0); assert(second.X==0); assert(r1==1); assert(r2==4); assert(old.X==0);",
        },
    }],
    example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB optimwarmstart",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/optimwarmstart.html") }],
    media: &[], evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Warm-start owner",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/optimwarmstart.rs") }],
        verification: &[], notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
pub(super) const READ: BuiltinDocumentation = BuiltinDocumentation {
    title: Some("LsqlinWarmStart.subsref"), slug: Some("lsqlin-warm-start-subsref"),
    summary: "Read an immutable warm-start property.",
    description: "Read X, Options or a code-generation memory-limit property using the original invocation context.",
    examples: &[], example_exemption: Some("Covered by the optimwarmstart example."), ..DOCUMENTATION
};
pub(super) const WRITE: BuiltinDocumentation = BuiltinDocumentation {
    title: Some("LsqlinWarmStart.subsasgn"),
    slug: Some("lsqlin-warm-start-subsasgn"),
    summary: "Reject edits to a read-only warm-start property.",
    description: "Recreate the object with optimwarmstart to change a property.",
    examples: &[],
    example_exemption: Some("The class properties are read-only."),
    ..DOCUMENTATION
};
pub(super) const QUADRATIC_READ: BuiltinDocumentation = BuiltinDocumentation {
    title: Some("QuadprogWarmStart.subsref"),
    slug: Some("quadprog-warm-start-subsref"),
    ..READ
};
pub(super) const QUADRATIC_WRITE: BuiltinDocumentation = BuiltinDocumentation {
    title: Some("QuadprogWarmStart.subsasgn"),
    slug: Some("quadprog-warm-start-subsasgn"),
    ..WRITE
};
