use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation {
    authority:BuiltinDocumentationAuthority::Catalog,title:Some("optimvar"),slug:Some("optimvar"),
    summary:"Create named optimization-variable arrays and indexed references.",
    description:"optimvar creates symbolic variable definitions, not numerical decision values. Copies and indexed references retain the same original definition. Bound edits through a reference update the corresponding original coordinates.",
    keywords:&["optimvar","OptimizationVariable","bounds","integer","named dimensions"],
    related:&["intlinprog","linprog","quadprog","fmincon"],
    sections:&[
        BuiltinDocumentationSection {heading:"Dimensions and names",paragraphs:&[
            "optimvar(name) creates a scalar; optimvar(name,n) creates n-by-1. Supply separate positive double dimensions, one dimension vector, or a cell of numeric and named axes for N-D arrays. Named axes use nonempty string vectors or cell vectors of text, with unique nonempty labels per axis. A lone named vector preserves its row/column orientation. A scalar string is not an axis-name vector; wrap a single label in a cell.",
            "The first name is a valid variable identifier. It remains read-only. Different definitions can have the same displayed name; model assembly must resolve conflicting definition names rather than merging their identities.",
        ]},
        BuiltinDocumentationSection {heading:"Types and bounds",paragraphs:&[
            "Type defaults to continuous; integer, semi-continuous and semi-integer are also supported. The original array has one type. Semi domains admit zero or values inside positive active bounds, both at most 1e5. Their mixed-integer solver interpretation remains distinct from an ordinary convex interval.",
            "LowerBound and UpperBound default to -Inf and Inf. Supply a real double scalar or an elementwise double array; row/column vectors of matching length are accepted for vector variables. Scalars stay compact until an array presentation or partial edit needs element storage. NaN, complex, single, logical, native integer, sparse and device bound carriers are not admitted.",
        ]},
        BuiltinDocumentationSection {heading:"Copies, references and properties",paragraphs:&[
            "Numeric/logical selectors, colon selectors and named axes create references to original coordinates. Chained indexing composes that original mapping. No new variable definition is created. LowerBound and UpperBound edits, including edits on an indexed reference, update the original definition atomically. Repeated indices use normal assignment ordering.",
            "The RunMat getfield/setfield extensions also support native variable properties. In a setfield index cell, textual end is resolved against the current retained property value, once that child is reached; it is not a named-axis label or a second getter call.",
            "Name is read-only on all views. Type and IndexNames can be changed on an original definition view, but are read-only on an indexing-created reference, including an identity selection. IndexNames has one cell entry per axis; empty entries select numeric indexing. Reference labels project the current original labels through their stored source-axis mapping.",
        ]},
        BuiltinDocumentationSection {heading:"Execution and current scope",paragraphs:&[
            "Construction and property operations use host double data and the original invocation context. Numeric work cooperates with cancellation. Invalid carriers do not trigger numeric gathering or callback-capture conversion. Native variable references retain their GC definition through normal rooted publication. No random, filesystem, network or user-object callback is used to create a definition.",
            "With a finite managed host workspace, construction is available when the original session has already registered the canonical variable class. Names, named-axis metadata, shapes, native definition/reference headers and root leases use retained allocation owners; bound arrays share their original double storage. Cold class registration, property/index operations, edits, expression transformations, numerical lowering and native-object transport still require their separate unbounded workspace paths.",
            "Reference shapes and selected offsets use checked platform-sized extents. A selection retains only its addressed offsets, without allocating the whole virtual variable array or imposing a separate 32-bit physical-offset limit. The per-selection materialization limit still applies. Reading full bounds or replacing a physical bound generation requires storage for that actual result; a small reference does not make full-array property operations cheap.",
            "Arithmetic and supported numerical functions build retained optimization expressions; optimexpr creates expression slots and evaluate obtains numerical values at a supplied point. Problem construction, solver lowering and export remain separate families; a variable constructor is not a numerical solver. Public handle superclass/deletion behavior is not inferred from alias-copy behavior.",
        ]},
    ],
    examples:&[
        BuiltinExample {
            id:"reference-bound-alias",title:"Change selected original bounds through a reference",
            program:"x=optimvar('x',3,'LowerBound',0);\ny=x([1,3]);\ny.LowerBound=2;\nb=x.LowerBound",
            display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions {source:"assert(isequal(size(x),[3 1])); assert(isequal(b,[2;0;2])); assert(strcmp(x.Name,'x'));"},
        },
        BuiltinExample {
            id:"named-integer-array",title:"Use named coordinates in an integer array",
            program:"x=optimvar('stock',{'red','blue'},2,'Type','integer','LowerBound',0,'UpperBound',10);\nx('blue',2).UpperBound=4;\nb=x.UpperBound",
            display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions {source:"assert(isequal(size(x),[2 2])); assert(b(2,2)==4); assert(b(1,2)==10); assert(strcmp(x.Type,'integer'));"},
        },
    ],example_exemption:None,faqs:&[],
    links:&[BuiltinDocumentationLink {label:"MATLAB optimvar",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/optim/ug/optimvar.html")}],
    media:&[],evidence:BuiltinDocumentationEvidence {
        implementation:&[BuiltinDocumentationLink {label:"Runtime variable/reference owner",target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/optimvar.rs")}],
        verification:&[BuiltinEvidenceReference {kind:BuiltinEvidenceKind::UnitTest,label:"Variable grammar, aliases and GC controls",location:"builtins::math::optim::optimvar::tests"}],notes:&[],
    },introduced:None,status:Some(BuiltinDocumentationStatus::Partial),
};
pub(super) const READ:BuiltinDocumentation=BuiltinDocumentation {
    title:Some("OptimizationVariable.subsref"),slug:Some("optimization-variable-subsref"),
    summary:"Read optimization-variable properties and original-definition references.",
    description:"The class subscript protocol creates indexed aliases and reads current native variable properties.",
    examples:&[],example_exemption:Some("Covered through optimvar public subscript examples."),
    ..DOCUMENTATION
};
pub(super) const WRITE:BuiltinDocumentation=BuiltinDocumentation {
    title:Some("OptimizationVariable.subsasgn"),slug:Some("optimization-variable-subsasgn"),
    summary:"Edit original optimization-variable properties through a rooted reference.",
    description:"The class assignment protocol updates selected original bounds without replacing the original variable by its subset.",
    examples:&[],example_exemption:Some("Covered through optimvar public alias-edit examples."),
    ..DOCUMENTATION
};
