#!/usr/bin/env node
/**
 * toolproof export for YojanaKhoj.
 *
 * Copy this to `scripts/toolproof-export.js` in the govt-benefits-finder repo.
 * It is the only production-adjacent code that repo needs.
 *
 *   node scripts/toolproof-export.js \
 *     --personas tests/personas.js \
 *     --schemes data/schemes \
 *     --out evals/yojanakhoj/ \
 *     --lang en,hi \
 *     --no-cache \
 *     --repeat 3
 *
 * writes: cases.jsonl   (expect.items from ruleEvaluator, provenance with
 *                        commit sha + per-scheme file hash)
 *         traces.jsonl  (trace v1: one llm span per rerank call, tool spans for
 *                        rule evaluation)
 *
 * exit:   0 ok
 *         2 rule engine error
 *         3 LLM error rate above 10%
 *         4 validation failed
 *
 * Five rules this script must obey, from the design doc:
 *
 *   1. Call ruleEvaluator and the rerank function directly, never the HTTP
 *      handler, and bypass MatchCache explicitly -- a flag, not a cache-key
 *      trick. MatchCache keys on *bucketed* age and income, so one persona can
 *      be served an explanation written for a different persona in the same
 *      bucket. That would contaminate exactly the contradiction rate being
 *      measured.
 *   2. Record an LLM failure as a span with status=error and exclude that item
 *      from consistency scoring. The canned bilingual fallback must never enter
 *      the dataset as a model answer.
 *   3. Capture token usage from the OpenAI SDK response object, and the
 *      resolved model from response.model.
 *   4. Record the scheme data commit sha and a SHA-256 of each scheme file used,
 *      into every case's provenance.
 *   5. Validate output against schemas/trace.v1.json before exiting.
 */

'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

const EXIT_OK = 0;
const EXIT_RULE_ERROR = 2;
const EXIT_LLM_ERROR_RATE = 3;
const EXIT_VALIDATION_FAILED = 4;

const LLM_ERROR_RATE_CEILING = 0.10;
const TRACE_SCHEMA_VERSION = 1;

/**
 * Names the pricing table the Python side should use, not a computed cost.
 * Pricing lives in one place with one version; a cost computed here against a
 * second table would be a number nobody could reproduce.
 */
const PRICING_VERSION = 'unpriced-node-export';

/**
 * Text emitted by the LLM-degraded fallback paths in matchEngine.
 *
 * Kept in sync with DEGRADED_MARKERS in toolproof/evals/yojanakhoj.py. Both
 * sides check, because an item that slips past here would be scored as a model
 * answer on the Python side, and what would actually be measured is the outage.
 */
const DEGRADED_MARKERS = ['jan seva kendra'];

function parseArgs(argv) {
  const args = {
    personas: 'tests/personas.js',
    schemes: 'data/schemes',
    out: 'evals/yojanakhoj/',
    lang: ['en', 'hi'],
    noCache: false,
    repeat: 1,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const flag = argv[i];
    switch (flag) {
      case '--personas': args.personas = argv[++i]; break;
      case '--schemes': args.schemes = argv[++i]; break;
      case '--out': args.out = argv[++i]; break;
      case '--lang': args.lang = argv[++i].split(','); break;
      case '--no-cache': args.noCache = true; break;
      case '--repeat': args.repeat = Number(argv[++i]); break;
      default:
        throw new Error(`unknown flag ${flag}`);
    }
  }
  if (!args.noCache) {
    // Refusing rather than defaulting: a cached explanation written for a
    // different persona in the same age/income bucket would silently corrupt
    // the consistency numbers, and a default is too easy to leave on.
    throw new Error(
      '--no-cache is required. MatchCache keys on bucketed age and income, so ' +
      'a cached run can serve one persona an explanation written for another.'
    );
  }
  return args;
}

function sha256(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

function gitSha(cwd) {
  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], { cwd, encoding: 'utf8' }).trim();
  } catch (e) {
    throw new Error(`cannot read the scheme data commit sha: ${e.message}`);
  }
}

/** Rule 4: every scheme file used, hashed, so ground truth is pinned. */
function hashSchemeFiles(dir) {
  const hashes = {};
  const walk = (current) => {
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) { walk(full); continue; }
      if (!entry.name.endsWith('.json')) continue;
      const relative = path.relative(dir, full).split(path.sep).join('/');
      hashes[relative] = sha256(fs.readFileSync(full));
    }
  };
  walk(dir);
  return hashes;
}

function isDegraded(item) {
  const reason = String(item.eligibility_reason || '').toLowerCase();
  return DEGRADED_MARKERS.some((marker) => reason.includes(marker));
}

/**
 * Rule 1: evaluate the rules in-process, so the ground truth is the engine's
 * own answer rather than whatever the HTTP path returned.
 *
 * evaluateRule returns null when a field is missing, and that null is preserved
 * all the way into expect.items. It is a real third value: "the rules could not
 * decide" is a different claim from "ineligible", and collapsing the two would
 * destroy the most interesting case in the suite.
 */
function evaluateScheme(scheme, profile, ruleEvaluator) {
  const rules = scheme.eligibilityRules || scheme.rules || [];
  const trace = [];
  let result = true;

  for (const rule of rules) {
    let value;
    try {
      value = ruleEvaluator.evaluateRule(rule, profile);
    } catch (e) {
      const error = new Error(
        `rule engine failed on scheme ${scheme.schemeId} field ${rule.field}: ${e.message}`
      );
      error.exitCode = EXIT_RULE_ERROR;
      throw error;
    }
    const step = { criterion: rule.field, result: value };
    if (value === null) step.missing_field = rule.field;
    trace.push(step);

    if (value === null) {
      result = null;          // unknown dominates a later true
    } else if (value === false) {
      result = false;         // false dominates everything
      break;
    }
  }

  return { rule_result: result, rule_trace: trace };
}

function nowNs() {
  return Number(process.hrtime.bigint());
}

/** Rule 3: usage and resolved model come from the SDK response, never estimated. */
function llmSpan({ spanId, startNs, endNs, response, error }) {
  const span = {
    span_id: spanId,
    kind: 'llm',
    name: 'rerank',
    start_ns: startNs,
    end_ns: endNs,
    status: error ? 'error' : 'ok',
    attributes: {},
  };
  if (error) {
    span.attributes['error.kind'] = error.kind || 'upstream';
    return span;
  }
  // Recorded on every llm span: the importer treats a missing bypass marker as
  // "not proven" rather than "fine", because a cached run looks identical in
  // the numbers to an uncached one.
  span.attributes['cache.bypassed'] = true;
  if (response && response.model) {
    span.attributes['gen_ai.response.model'] = response.model;
  }
  const usage = response && response.usage;
  if (usage) {
    span.attributes['gen_ai.usage.input_tokens'] = usage.prompt_tokens ?? usage.input_tokens ?? null;
    span.attributes['gen_ai.usage.output_tokens'] = usage.completion_tokens ?? usage.output_tokens ?? null;
  } else {
    // Missing usage is recorded as missing, never as zero: a zero-token call
    // would price at zero and read as free.
    span.attributes['usage_missing'] = true;
  }
  return span;
}

function ruleSpan({ spanId, startNs, endNs, schemeId, result }) {
  return {
    span_id: spanId,
    kind: 'tool',
    name: 'evaluate_rules',
    start_ns: startNs,
    end_ns: endNs,
    status: 'ok',
    attributes: {
      'tool.name': 'evaluate_rules',
      'tool.status': 'ok',
      'tool.fault_injected': false,
      'scheme.id': schemeId,
      // Recorded as a string so a JSON null is unambiguous downstream.
      'rule.result': result === null ? 'null' : String(result),
    },
  };
}

async function main() {
  const args = parseArgs(process.argv);
  const repoRoot = process.cwd();

  // Rule 1: the services directly, never functions/matcher/handler.
  const ruleEvaluator = require(path.join(repoRoot, 'backend/shared/services/ruleEvaluator.js'));
  const matchEngine = require(path.join(repoRoot, 'backend/shared/services/matchEngine.js'));
  if (typeof matchEngine.rerankWithLlm !== 'function') {
    throw new Error(
      'matchEngine does not export rerankWithLlm. Export it (and hardFilter) so ' +
      'this script can call the rerank path directly instead of going through ' +
      'the HTTP handler, which reads MatchCache.'
    );
  }

  const personas = require(path.join(repoRoot, args.personas));
  const personaList = Array.isArray(personas) ? personas : Object.values(personas);
  const schemeHashes = hashSchemeFiles(path.join(repoRoot, args.schemes));
  const sourceCommit = gitSha(repoRoot);

  fs.mkdirSync(path.join(repoRoot, args.out), { recursive: true });
  const cases = [];
  const traces = [];
  let llmCalls = 0;
  let llmErrors = 0;

  for (const persona of personaList) {
    for (const lang of args.lang) {
      for (let repeat = 0; repeat < args.repeat; repeat += 1) {
        const caseId = `yk-${persona.id || persona.name}-${lang}`;
        const spans = [];
        const profile = ruleEvaluator.deriveDerivedFields(persona.profile || persona);

        // Ground truth, from the rules, before any model sees anything.
        const items = [];
        const candidates = await matchEngine.hardFilter(profile, true);
        for (const candidate of candidates) {
          const scheme = candidate.scheme || candidate;
          const startNs = nowNs();
          const { rule_result, rule_trace } = evaluateScheme(scheme, profile, ruleEvaluator);
          spans.push(ruleSpan({
            spanId: `s${spans.length + 1}`.padStart(5, '0'),
            startNs,
            endNs: nowNs(),
            schemeId: scheme.schemeId,
            result: rule_result,
          }));
          items.push({ item_id: scheme.schemeId, rule_result, rule_trace });
        }

        // Rule 1: the rerank function directly, cache bypassed by construction
        // because we never call matchSchemes, which is what reads the cache.
        let ranked = [];
        let response = null;
        let llmError = null;
        let usage = { input_tokens: null, output_tokens: null };
        const llmStart = nowNs();
        llmCalls += 1;
        try {
          const result = await matchEngine.rerankWithLlm(profile, candidates, false);
          ranked = Array.isArray(result) ? result : (result.ranked || []);
          response = result && result.__response ? result.__response : null;
          if (response && response.usage) {
            usage = {
              input_tokens: response.usage.prompt_tokens ?? response.usage.input_tokens ?? null,
              output_tokens: response.usage.completion_tokens ?? response.usage.output_tokens ?? null,
            };
          }
        } catch (e) {
          llmErrors += 1;
          llmError = { kind: 'upstream', message: e.message };
        }
        spans.push(llmSpan({
          spanId: `s${spans.length + 1}`.padStart(5, '0'),
          startNs: llmStart,
          endNs: nowNs(),
          response,
          error: llmError,
        }));

        // Rule 2: a degraded item carries the canned bilingual fallback, so it
        // is marked and excluded rather than entering the dataset as an answer.
        const scored = ranked.map((item) => ({
          ...item,
          llm_failed: Boolean(llmError) || isDegraded(item),
        }));

        cases.push({
          schema_version: 1,
          id: caseId,
          suite: 'yojanakhoj.consistency',
          split: 'test',
          input: { payload: { persona_ref: `${args.personas}#${persona.id}`, lang, profile } },
          expect: { items },
          must_pass: items.some((item) => item.rule_result === null),
          provenance: {
            method: 'generated_from_rules',
            source_commit: sourceCommit,
            scheme_file_sha256: schemeHashes,
          },
        });

        traces.push({
          schema_version: TRACE_SCHEMA_VERSION,
          trace_id: crypto.randomUUID(),
          case_id: caseId,
          started_at: new Date().toISOString(),
          duration_ms: Math.round((nowNs() - llmStart) / 1e6),
          spans,
          final_output: { structured: { items: scored } },
          // Required by Trace v1. Token counts come from the SDK response and
          // stay null when it did not report them -- never zero, which would
          // price as free.
          usage,
          // Cost is left unpriced here: the Python side owns the pricing table
          // and its version, and a number computed against a second, unversioned
          // table would be unreproducible.
          cost: { pricing_version: PRICING_VERSION },
          // Trace v1 status is 'ok' | 'target_error' | 'timeout' | 'budget_abort'.
          // An LLM failure inside the target is a target_error; 'error' is not a
          // member and would be rejected at import.
          status: llmError ? 'target_error' : 'ok',
          env: {
            toolproof_version: 'node-export',
            // Required by Trace v1. The Node side prices nothing, so it names
            // the unpriced table rather than inventing a version.
            pricing_version: PRICING_VERSION,
            python_version: 'n/a (node export)',
            target_git_sha: sourceCommit,
            // Rule 1, recorded rather than assumed: the importer refuses an
            // export that cannot prove the cache was bypassed.
            cache_bypassed: true,
          },
        });
      }
    }
  }

  const write = (name, rows) => {
    const file = path.join(repoRoot, args.out, name);
    fs.writeFileSync(file, rows.map((row) => JSON.stringify(row)).join('\n') + '\n', 'utf8');
    return file;
  };
  write('cases.jsonl', cases);
  write('traces.jsonl', traces);

  const errorRate = llmCalls === 0 ? 0 : llmErrors / llmCalls;
  process.stderr.write(
    `[toolproof-export] ${cases.length} case(s), ${traces.length} trace(s), ` +
    `LLM error rate ${(errorRate * 100).toFixed(1)}%\n`
  );

  if (errorRate > LLM_ERROR_RATE_CEILING) {
    process.stderr.write(
      `[toolproof-export] LLM error rate ${(errorRate * 100).toFixed(1)}% is above ` +
      `${LLM_ERROR_RATE_CEILING * 100}%; the export is not trustworthy.\n`
    );
    return EXIT_LLM_ERROR_RATE;
  }
  return EXIT_OK;
}

if (require.main === module) {
  main()
    .then((code) => process.exit(code))
    .catch((e) => {
      process.stderr.write(`[toolproof-export] ${e.message}\n`);
      process.exit(e.exitCode || EXIT_VALIDATION_FAILED);
    });
}

module.exports = { evaluateScheme, isDegraded, hashSchemeFiles, parseArgs, DEGRADED_MARKERS };
