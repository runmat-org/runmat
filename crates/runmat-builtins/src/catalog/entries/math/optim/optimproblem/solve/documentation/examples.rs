//! Executable optimization examples and their independent assertions.
use super::{residual, unconstrained};
use crate::*;

pub(super) const EXAMPLES: &[BuiltinExample] = &[BuiltinExample{
        id:"affine-original-layout",title:"Solve with original array and multiplier shapes",
        program:"x=optimvar('allocation',2,1,'LowerBound',0);\np=optimproblem('ObjectiveSense','max');\np.Objective=3*x(1)+x(2)+7;\np.Constraints.capacity=sum(x)<=4;\no=optimoptions('linprog','Display','off');\n[s,f,e,out,l]=solve(p,'Options',o);",
        display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
        harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
        requirements:BuiltinExampleRequirements::NONE,
        verification:BuiltinExampleVerification::Assertions{
            source:"assert(isequal(size(s.allocation),[2 1])); assert(abs(s.allocation(1)-4)<1e-6); assert(abs(f-19)<1e-6); assert(double(e)==1); assert(strcmp(string(e),'OptimalSolution')); assert(isequal(size(l.Variables.allocation.Lower),[2 1])); assert(abs(l.Constraints.capacity-3)<1e-5);"},
    }, BuiltinExample {
        id: "affine-linear-functions",
        title: "Combine reductions, diagonal structure and reverse accumulation",
        program: concat!(
            "x=optimvar('rate',2,1,'LowerBound',0,'UpperBound',2);\n",
            "p=optimproblem('Objective',mean(x)+trace(diag(x))+sum(cumsum(x,'reverse')));\n",
            "p.Constraints.minimum=sum(x)>=2;\n",
            "[s,f,e]=solve(p);"
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(s.rate(1)-2)<1e-5); assert(abs(s.rate(2))<1e-5); assert(abs(f-5)<1e-5); assert(double(e)==1);",
        },
    }, BuiltinExample {
        id: "affine-factor-functions",
        title: "Combine rowwise factors, differences and identity powers",
        program: concat!(
            "x=optimvar('amount','LowerBound',0,'UpperBound',2);\n",
            "p=optimproblem;\n",
            "p.Objective=dot([x;2],[3;x])+prod([x;2])+sum(cumprod([x;2]))+diff([x;2;x],2)+x.^0;\n",
            "p.Constraints.minimum=x>=1;\n",
            "[s,f,e]=solve(p);"
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(s.amount-1)<1e-5); assert(abs(f-9)<1e-5); assert(double(e)==1);",
        },
    }, BuiltinExample {
        id:"linear-least-squares-original-units",
        title:"Solve weighted affine residuals with original objective and multiplier units",
        program:concat!(
            "x=optimvar('allocation',2,1,'LowerBound',0);\n",
            "r=x-[2;4];\n",
            "p=optimproblem('Objective',2*sum(r.^2)+norm(r)^2+7);\n",
            "p.Constraints.capacity=sum(x)<=3;\n",
            "o=optimoptions('lsqlin','Display','off');\n",
            "[s,f,e,out,l]=solve(p,'Options',o);"
        ),
        display_output:None,
        compatibility:BuiltinExampleCompatibility::Matlab,
        harness:BuiltinExampleHarness::Portable,
        fixture:BuiltinExampleFixture::None,
        requirements:BuiltinExampleRequirements::NONE,
        verification:BuiltinExampleVerification::Assertions{
            source:concat!(
                "assert(isequal(size(s.allocation),[2 1])); ",
                "assert(max(abs(s.allocation-[.5;2.5]))<1e-5); ",
                "assert(abs(f-20.5)<1e-5); assert(double(e)==1); ",
                "assert(strcmp(out.Solver,'lsqlin')); ",
                "assert(abs(l.Constraints.capacity-9)<1e-4);"
            )
        },
    }, BuiltinExample {
        id:"quadratic-original-objective",
        title:"Solve a quadratic model without dropping its constant",
        program:concat!(
            "x=optimvar('amount','LowerBound',0,'UpperBound',1);\n",
            "p=optimproblem('Objective',3*x*x-2*x+5);\n",
            "o=optimoptions('quadprog','Display','off');\n",
            "[s,f,e,out]=solve(p,'Options',o);"
        ),
        display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
        harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
        requirements:BuiltinExampleRequirements::NONE,
        verification:BuiltinExampleVerification::Assertions{
            source:"assert(abs(s.amount-1/3)<1e-5); assert(abs(f-14/3)<1e-6); assert(double(e)==1); assert(strcmp(out.Solver,'quadprog')); assert(abs(evaluate(p.Objective,s)-f)<1e-12);"
        },
    }, BuiltinExample {
        id:"quadratic-vector-norm-affine",title:"Combine a vector norm square and an affine term",
        program:concat!(
            "x=optimvar('rate',2,1);\n",
            "p=optimproblem('Objective',norm(x-[1;2])^2+sum(x));\n",
            "[s,f,e,out]=solve(p);"
        ),
        display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
        harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
        requirements:BuiltinExampleRequirements::NONE,
        verification:BuiltinExampleVerification::Assertions{
            source:"assert(norm(s.rate-[0.5;1.5])<1e-5); assert(abs(f-2.5)<1e-6); assert(double(e)==1); assert(strcmp(out.Solver,'quadprog')); assert(abs(evaluate(p.Objective,s)-f)<1e-12);"
        },
    }, unconstrained::EXAMPLE, BuiltinExample {
        id:"integer-original-layout",title:"Solve with integer domains, a constant and a named start",
        program:concat!(
            "x=optimvar('count',2,1,'Type','integer','LowerBound',0,'UpperBound',1);\n",
            "p=optimproblem('Objective',3*sum(x)+5,'ObjectiveSense','max');\n",
            "p.Constraints.capacity=2*sum(x)<=3; x0.count=[1;0];\n",
            "o=optimoptions('intlinprog','Display','off','MaxNodes',100);\n",
            "[s,f,e,out,lambda]=solve(p,x0,'Options',o);"
        ),
        display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
        harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
        requirements:BuiltinExampleRequirements::NONE,
        verification:BuiltinExampleVerification::Assertions{
            source:"assert(isequal(size(s.count),[2 1])); assert(all(s.count==round(s.count))); assert(sum(s.count)==1); assert(abs(f-8)<1e-6); assert(double(e)==1); assert(strcmp(out.Solver,'intlinprog')); assert(out.numfeaspoints>=1); assert(out.lowerbound<=out.certifiedobjective); assert(isempty(lambda)); assert(abs(evaluate(p.Objective,s)-f)<1e-12);"
        },
    }, residual::EXAMPLE];
