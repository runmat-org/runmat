function tests = arithmeticTest()
  tests = functiontests(localfunctions);
end

function testPasses(testCase)
  testCase.verifyEqual(helper() + testDependencyValue(), 42);
  testCase.verifyTrue(abs(corr(uint16([1;2;3;4]), int32([2;1;4;3])) - 0.6) < 1e-12);
end

function testFails(testCase)
  testCase.verifyEqual(1 + 1, 3);
end

function testHangs(testCase)
  while true
  end
  testCase.verifyTrue(false);
end
