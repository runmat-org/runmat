function tests = recoveryTest()
  tests = functiontests(localfunctions);
end

function testRecoveryHangs(testCase)
  while true
  end
  testCase.verifyTrue(false);
end

function testRecoveryPasses(testCase)
  testCase.verifyEqual(6 * 7, 42);
end
