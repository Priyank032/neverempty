/**
 * Exercise the exporter's pure functions against YojanaKhoj's real ruleEvaluator.
 *
 * No LLM, no database, no network: this checks that the three-way rule result
 * survives, that the degraded fallback is detected, and that --no-cache is
 * mandatory.
 */
'use strict';

const assert = require('node:assert');
const path = require('node:path');

// Paths assume this file sits in the YojanaKhoj repo at scripts/, beside the
// exporter. Adjust REPO if you keep it elsewhere.
const REPO = process.env.YOJANAKHOJ_REPO || path.resolve(__dirname, '..');
const exporter = require(path.join(__dirname, 'neverempty-export.js'));
const ruleEvaluator = require(path.join(REPO, 'backend/shared/services/ruleEvaluator.js'));
const SCHEMES = path.join(REPO, 'data/schemes');

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log(`  ok  ${name}`);
}

console.log('1. --no-cache is mandatory');
check('omitting --no-cache throws', () => {
  assert.throws(
    () => exporter.parseArgs(['node', 'x']),
    /--no-cache is required/,
  );
});
check('passing --no-cache is accepted', () => {
  const args = exporter.parseArgs(['node', 'x', '--no-cache']);
  assert.strictEqual(args.noCache, true);
});
check('the refusal names the bucketing hazard', () => {
  try {
    exporter.parseArgs(['node', 'x']);
  } catch (e) {
    assert.match(e.message, /bucketed age and income/);
  }
});

console.log('\n2. The real rule engine, three-way results preserved');
const scheme = (id, rules) => ({ schemeId: id, eligibilityRules: rules });

check('all criteria true -> rule_result true', () => {
  const out = exporter.evaluateScheme(
    scheme('pm-kisan', [
      { field: 'occupation', operator: 'eq', value: 'farmer' },
      { field: 'is_income_taxpayer', operator: 'eq', value: false },
    ]),
    ruleEvaluator.deriveDerivedFields({ occupation: 'farmer', is_income_taxpayer: false }),
    ruleEvaluator,
  );
  assert.strictEqual(out.rule_result, true);
});

check('a false criterion -> rule_result false', () => {
  const out = exporter.evaluateScheme(
    scheme('pm-kisan', [{ field: 'occupation', operator: 'eq', value: 'farmer' }]),
    ruleEvaluator.deriveDerivedFields({ occupation: 'teacher' }),
    ruleEvaluator,
  );
  assert.strictEqual(out.rule_result, false);
});

check('a MISSING field -> rule_result null, not false', () => {
  // The doc's most interesting case: "cannot evaluate" is a different claim
  // from "ineligible", and collapsing them would destroy it.
  const out = exporter.evaluateScheme(
    scheme('pm-kisan', [
      { field: 'occupation', operator: 'eq', value: 'farmer' },
      { field: 'is_income_taxpayer', operator: 'eq', value: false },
    ]),
    ruleEvaluator.deriveDerivedFields({ occupation: 'farmer' }),  // tax status absent
    ruleEvaluator,
  );
  assert.strictEqual(out.rule_result, null, 'missing field must yield null');
  const step = out.rule_trace.find((s) => s.criterion === 'is_income_taxpayer');
  assert.strictEqual(step.result, null);
  assert.strictEqual(step.missing_field, 'is_income_taxpayer');
});

check('false dominates a later null', () => {
  const out = exporter.evaluateScheme(
    scheme('x', [
      { field: 'occupation', operator: 'eq', value: 'farmer' },
      { field: 'never_set', operator: 'eq', value: 1 },
    ]),
    ruleEvaluator.deriveDerivedFields({ occupation: 'teacher' }),
    ruleEvaluator,
  );
  assert.strictEqual(out.rule_result, false);
});

check('null dominates a later true', () => {
  const out = exporter.evaluateScheme(
    scheme('x', [
      { field: 'never_set', operator: 'eq', value: 1 },
      { field: 'occupation', operator: 'eq', value: 'farmer' },
    ]),
    ruleEvaluator.deriveDerivedFields({ occupation: 'farmer' }),
    ruleEvaluator,
  );
  assert.strictEqual(out.rule_result, null);
});

console.log('\n3. The LLM-degraded fallback is detected (rule 2)');
check("the real fallback string is caught", () => {
  assert.strictEqual(
    exporter.isDegraded({ eligibility_reason: 'Verify at nearest Jan Seva Kendra.' }),
    true,
  );
});
check('a genuine model answer is not', () => {
  assert.strictEqual(
    exporter.isDegraded({ eligibility_reason: 'You own 1.5 acres, within the limit.' }),
    false,
  );
});
check('the marker list matches the Python side', () => {
  assert.deepStrictEqual(exporter.DEGRADED_MARKERS, ['jan seva kendra']);
});

console.log('\n4. Scheme files are hashed (rule 4)');
check('hashing the real scheme directory produces sha256 per file', () => {
  const hashes = exporter.hashSchemeFiles(SCHEMES);
  const names = Object.keys(hashes);
  assert.ok(names.length > 0, 'found no scheme files');
  for (const name of names.slice(0, 3)) {
    assert.match(hashes[name], /^[0-9a-f]{64}$/);
  }
  console.log(`      (${names.length} scheme file(s) hashed)`);
});

console.log(`\n${passed} checks passed`);
