use super::*;
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label:"Datetime typed host operator kernel",
    target:BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/operators/binary.rs"),
};
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading:"Actual result and operand classes", paragraphs:&[
        "The left receiver must be a datetime value object. Comparisons accept another datetime or host real numeric/logical serial day values. They currently publish double 0/1, not logical arrays. Text comparisons, duration/calendarDuration comparison operands and handle-object wrappers are not admitted by these methods.",
        "Addition accepts duration, calendarDuration or host real numeric/logical day counts and returns datetime. It does not accept datetime on the right. Subtracting datetime returns double elapsed days, not a duration object; subtracting duration, calendarDuration or numeric/logical day counts returns datetime. Arithmetic preserves the left Format property and constructs results through the actual selected Runtime context.",
        "Native single and integer numeric operands enter binary64 host serial arithmetic. Native integer RHS values are checked exactly against [-1000000000,1000000000] before conversion. Logical RHS values use the declared datetime logical-input extension; compatibility mode rejects that extension. The existing raw numeric method forms are RunMat behavior and are not gated by the constructor-only implicit-datenum extension.",
    ] },
    BuiltinDocumentationSection { heading:"Geometry and comparisons", paragraphs:&[
        "A one-element operand expands to the other operand. Otherwise full non-scalar shapes must match exactly, including empty axes and low-rank storage. General dimension-wise implicit expansion is not implemented here. A one-element numeric comparison or datetime difference publishes a scalar double; empty results retain their complete array geometry.",
        "Finite equality retains the existing absolute tolerance of 1e-12 serial days. Inequality is the complement of that equality relation. Missing/NaN serials are unequal to everything including themselves, while equal-sign infinities compare equal. Ordering uses the stored binary64 ordering and is false when either operand is NaN.",
        "Calendar-relative shifts apply integral month offsets with month-end clamping, then day offsets. Calendar month/day range validation and civil calendar limits remain those of the existing host calendar owner. Time-zone/time-standard and full MATLAB NaT construction/display parity are not established by this method migration.",
    ] },
    BuiltinDocumentationSection { heading:"Execution boundary and remaining frontiers", paragraphs:&[
        "These are MethodOnly entries reached by the existing typed datetime class declaration and ordinary operator syntax, not independent command-style functions. The actual default binding retains BuiltinArguments and selected context, checks output demand before work and admits the existing UnboundedCalendarWork owner before legacy temporary arrays or object assembly.",
        "All results are host values and the operation is a fusion boundary. Resident arguments are rejected without a provider lookup, download, gather, kernel, input retirement or replay. No new accelerator, distributed or native integer provider coverage is claimed.",
        "The current synchronous host kernels remain Incomplete: complete finite scratch/output ownership and bounded browser cooperation are not implemented. Managed calls fail closed. NativeAndWasm is portable source/link eligibility, not a claim that fresh browser or AOT qualification has run.",
    ] },
];
macro_rules! doc {
    ($id:ident,$name:literal,$summary:literal,$program:literal,$oracle:literal,$second:literal,$second_oracle:literal) => {
        pub(super) const $id: BuiltinDocumentation = BuiltinDocumentation {
            authority:BuiltinDocumentationAuthority::Catalog,title:Some($name),slug:Some($name),summary:$summary,
            description:concat!("The datetime class method ",$name," applies the existing typed host serial-date operation. It retains operation-specific RHS dispatch, scalar-or-exact geometry and native result-class behavior rather than assuming generic numeric comparison or duration arithmetic."),
            keywords:&["datetime",$name,"operator","serial date","calendar arithmetic"],
            related:&["datetime","duration","calendarDuration","datenum"],sections:SECTIONS,
            examples:&[
                BuiltinExample { id:"operator",title:$summary,program:$program,display_output:None,
                    compatibility:BuiltinExampleCompatibility::RunMat,harness:BuiltinExampleHarness::Portable,
                    fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
                    verification:BuiltinExampleVerification::Assertions {source:$oracle} },
                BuiltinExample { id:"class-and-shape",title:"Check the actual result class and geometry",program:$second,display_output:None,
                    compatibility:BuiltinExampleCompatibility::RunMat,harness:BuiltinExampleHarness::Portable,
                    fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,
                    verification:BuiltinExampleVerification::Assertions {source:$second_oracle} },
            ],
            example_exemption:None,faqs:&[
                BuiltinDocumentationFaq {question:"Do datetime comparisons return logical in this implementation?",answer:"No. These methods currently publish double 0/1. The canonical result facts and example assertions intentionally retain that implementation boundary."},
                BuiltinDocumentationFaq {question:"Does subtracting two dates construct duration?",answer:"No. This Runtime path currently returns a double day difference. Subtracting a duration or calendarDuration instead returns a shifted datetime."},
            ],
            links:&[
                BuiltinDocumentationLink {label:"datetime",target:BuiltinDocumentationLinkTarget::Builtin("datetime")},
                BuiltinDocumentationLink {label:"duration",target:BuiltinDocumentationLinkTarget::Builtin("duration")},
                BuiltinDocumentationLink {label:"calendarDuration",target:BuiltinDocumentationLinkTarget::Builtin("calendarDuration")},
                BuiltinDocumentationLink {label:"MathWorks relational missing-value rules",target:BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/matlab_prog/array-comparison-with-relational-operators.html")},
                SOURCE,
            ],
            media:&[],evidence:BuiltinDocumentationEvidence {
                implementation:&[SOURCE],verification:&[
                    BuiltinEvidenceReference {kind:BuiltinEvidenceKind::UnitTest,label:"Typed datetime method inference and class declaration linkage",location:"catalog::entries::datetime::operators::tests"},
                    BuiltinEvidenceReference {kind:BuiltinEvidenceKind::UnitTest,label:"Runtime method classes, full shapes, missing-value comparisons and authority",location:"builtins::datetime::operators::binary::tests"},
                ],notes:&[
                    "First-class documentation for a previously undocumented MethodOnly Runtime registration; no JSON legacy baseline is fabricated or retired.",
                    "Missing-value inequality and equal-sign infinity equality are repaired against primary MathWorks relational comparison rules, retaining finite tolerance and actual double publication. Native class, shape, placement and finite-allocation limitations remain explicit.",
                    "Examples contain independent assertions. Source authoring and primary-reference inspection are not fresh compiler, exporter, VM/JIT/AOT or browser execution evidence.",
                ],
            },introduced:None,status:Some(BuiltinDocumentationStatus::Partial),
        };
    }
}
doc!(
    EQ_DOC,
    "datetime.eq",
    "Compare datetime values for equality.",
    "a = datetime(2024,1,1); b = datetime(2024,1,2); tf = a == b;",
    "assert(isa(tf,'double')); assert(tf==0);",
    "a = datetime(2024,1,[1 2]); b = datetime(2024,1,2); tf = a == b;",
    "assert(isa(tf,'double')); assert(isequal(size(tf),[1 2])); assert(isequal(tf,[0 1]));"
);
doc!(
    NE_DOC,
    "datetime.ne",
    "Compare datetime values for inequality.",
    "a = datetime(2024,1,1); b = datetime(2024,1,2); tf = a ~= b;",
    "assert(isa(tf,'double')); assert(tf==1);",
    "a = datetime(2024,1,[1 2]); b = datetime(2024,1,2); tf = a ~= b;",
    "assert(isa(tf,'double')); assert(isequal(size(tf),[1 2])); assert(isequal(tf,[1 0]));"
);
doc!(
    LT_DOC,
    "datetime.lt",
    "Compare datetime values with less-than.",
    "a = datetime(2024,1,1); b = datetime(2024,1,2); tf = a < b;",
    "assert(isa(tf,'double')); assert(tf==1);",
    "a = datetime(2024,1,[1 2]); b = datetime(2024,1,2); tf = a < b;",
    "assert(isa(tf,'double')); assert(isequal(size(tf),[1 2])); assert(isequal(tf,[1 0]));"
);
doc!(
    LE_DOC,
    "datetime.le",
    "Compare datetime values with less-than-or-equal.",
    "a = datetime(2024,1,1); b = datetime(2024,1,2); tf = a <= b;",
    "assert(isa(tf,'double')); assert(tf==1);",
    "a = datetime(2024,1,[1 2]); b = datetime(2024,1,2); tf = a <= b;",
    "assert(isa(tf,'double')); assert(isequal(size(tf),[1 2])); assert(isequal(tf,[1 1]));"
);
doc!(
    GT_DOC,
    "datetime.gt",
    "Compare datetime values with greater-than.",
    "a = datetime(2024,1,1); b = datetime(2024,1,2); tf = a > b;",
    "assert(isa(tf,'double')); assert(tf==0);",
    "a = datetime(2024,1,[1 2]); b = datetime(2024,1,2); tf = a > b;",
    "assert(isa(tf,'double')); assert(isequal(size(tf),[1 2])); assert(isequal(tf,[0 0]));"
);
doc!(
    GE_DOC,
    "datetime.ge",
    "Compare datetime values with greater-than-or-equal.",
    "a = datetime(2024,1,1); b = datetime(2024,1,2); tf = a >= b;",
    "assert(isa(tf,'double')); assert(tf==0);",
    "a = datetime(2024,1,[1 2]); b = datetime(2024,1,2); tf = a >= b;",
    "assert(isa(tf,'double')); assert(isequal(size(tf),[1 2])); assert(isequal(tf,[0 1]));"
);
doc!(PLUS_DOC,"datetime.plus","Shift datetime values by fixed or calendar-relative amounts.","a = datetime(2024,1,31); b = a + calmonths(1); c = a + days(1);","assert(isdatetime(b)); assert(year(b)==2024); assert(month(b)==2); assert(day(b)==29); assert(isdatetime(c)); assert(day(c)==1); assert(month(c)==2);","a = datetime(2024,1,1); b = a + int32([1 2]);","assert(isdatetime(b)); assert(isequal(size(b),[1 2])); assert(isequal(day(b),[2 3]));");
doc!(MINUS_DOC,"datetime.minus","Subtract dates or shift a datetime backwards.","a = datetime(2024,1,3); b = datetime(2024,1,1); elapsed = a - b; shifted = a - days(1);","assert(isa(elapsed,'double')); assert(elapsed==2); assert(isdatetime(shifted)); assert(day(shifted)==2);","a = datetime(2024,3,31); b = a - calmonths(1); c = a - [1 2];","assert(isdatetime(b)); assert(year(b)==2024); assert(month(b)==2); assert(day(b)==29); assert(isdatetime(c)); assert(isequal(size(c),[1 2])); assert(isequal(day(c),[30 29]));");
