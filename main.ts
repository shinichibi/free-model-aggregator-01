// free-model-aggregator-01: one server that speaks the OpenAI Chat Completions API and the
// Anthropic Messages API to its client, and routes every request to the best free model across
// ten providers that speak the OpenAI, Anthropic or Gemini wires.
//
// Design notes:
//   - The brief at $MAKEFUN_PROMPT_FILE carries the provider directory, the public quality table
//     (rating_milli and capabilities per model) and the workload manifest. Routing is greedy on
//     that table: candidates that are free and meet the request's requirements, best rating
//     first, with failover to the next candidate whenever a provider refuses (402/429/5xx).
//   - A provider that refuses is remembered: refusals are provider-level (budget spent, failure
//     schedule, outage), so the rest of that provider's models are skipped for the request, and
//     a provider that keeps refusing sinks in later requests' order.
//   - Where the client's wire and the provider's wire agree, the provider's answer passes
//     through untouched. Where they differ, the answer is rebuilt in the client's wire from the
//     parsed answer, with stop reasons and usage mapped.
//   - Upstream calls are made non-streamed and the client's stream, when it asked for one, is
//     synthesised from the complete answer; the one exception is a same-wire stream, whose raw
//     events pass through byte for byte. Either way nothing here reads the clock or a random
//     source, so a replay of the same requests and provider answers is byte-identical.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Json = Record<string, unknown>;
type Wire = 'openai' | 'anthropic' | 'gemini';
type ClientWire = 'openai' | 'anthropic';

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const CALL_TIMEOUT_MS = 15000;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY = 32 * 1024 * 1024;
/** Fixed epoch for OpenAI-shaped documents this server mints: the clock is never read. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: three JSON documents found by their schema member.
// ---------------------------------------------------------------------------

interface ProviderInfo {
  id: string;
  name: string;
  wire: Wire;
  auth: 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
  base: string;
  keyEnv: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number | null;
}

interface ManifestEntry {
  id: string;
  api: ClientWire;
  stream: boolean;
  requires: Requirement;
}

interface Brief {
  providers: ProviderInfo[];
  ratings: RatingRow[];
  manifest: ManifestEntry[];
}

/** Every top-level JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a standalone document: keep looking.
    }
  }
  return out;
}

function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [] };
  if (path === '') return empty;
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    log(`cannot read the brief: ${(err as Error).message}`);
    return empty;
  }
  const docs = documentsIn(text);
  const directory = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1');
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1');
  const workloadDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1');
  const providers: ProviderInfo[] = [];
  if (directory !== undefined && Array.isArray(directory.providers)) {
    for (const raw of directory.providers) {
      if (!isRecord(raw)) continue;
      const wire = raw.wire === 'anthropic' || raw.wire === 'gemini' ? raw.wire : raw.wire === 'openai' ? 'openai' : null;
      if (str(raw.id) === '' || wire === null) continue;
      const auth =
        raw.auth === 'x-api-key' || raw.auth === 'x-goog-api-key' || raw.auth === 'none' ? raw.auth : 'bearer';
      const base = str(raw.path) !== '' ? str(raw.path) : str(raw.base_url) !== '' ? str(raw.base_url) : str(raw.base);
      providers.push({
        id: str(raw.id),
        name: str(raw.name) !== '' ? str(raw.name) : str(raw.id),
        wire,
        auth,
        base,
        keyEnv: str(raw.key_env),
        key: str(raw.key),
      });
    }
  }
  const ratings: RatingRow[] = [];
  if (ratingsDoc !== undefined && Array.isArray(ratingsDoc.ratings)) {
    for (const raw of ratingsDoc.ratings) {
      if (!isRecord(raw) || str(raw.model) === '' || str(raw.provider) === '') continue;
      ratings.push({
        model: str(raw.model),
        provider: str(raw.provider),
        rating: num(raw.rating_milli) ?? 0,
        context: num(raw.context_length),
        tools: raw.tools === true,
        vision: raw.vision === true,
        json: raw.json === true,
        free: raw.free === true,
      });
    }
  }
  const manifest: ManifestEntry[] = [];
  if (workloadDoc !== undefined && Array.isArray(workloadDoc.manifest)) {
    for (const raw of workloadDoc.manifest) {
      if (!isRecord(raw) || str(raw.id) === '') continue;
      const req = isRecord(raw.requires) ? raw.requires : {};
      manifest.push({
        id: str(raw.id),
        api: raw.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: raw.stream === true,
        requires: {
          tools: req.tools === true,
          vision: req.vision === true,
          json: req.json === true,
          minContext: num(req.min_context),
        },
      });
    }
  }
  return { providers, ratings, manifest };
}

// ---------------------------------------------------------------------------
// Provider catalog probes and the catalog document.
// ---------------------------------------------------------------------------

/** A base path joined to the env's URL; an absolute base stands as it is. */
function joinBase(base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  if (b === '') return ENV_URL;
  return `${ENV_URL}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The header a provider's auth scheme carries its key in; no header for `none` or an empty key. */
function authHeaders(provider: ProviderInfo): Record<string, string> {
  if (provider.key === '' || provider.auth === 'none') return {};
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  return { 'x-goog-api-key': provider.key };
}

interface ProbedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  context: number | null;
}

interface Probe {
  provider: ProviderInfo;
  ok: boolean;
  models: ProbedModel[];
}

/** A listing's price fields read as a free flag: true when every price named is zero. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function probedModel(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? str(v.name).replace(/^models\//, '') : str(v.id);
  if (id === '') return null;
  return {
    id,
    displayName: str(v.display_name) !== '' ? str(v.display_name) : str(v.displayName) !== '' ? str(v.displayName) : id,
    free: typeof v.free === 'boolean' ? v.free : zeroPriced(v),
    context: num(v.context_length) ?? num(v.context_window) ?? num(v.inputTokenLimit),
  };
}

/** Ask one provider for its catalog, in its own wire; every failure is tolerated. */
async function probe(provider: ProviderInfo): Promise<Probe> {
  const base = joinBase(provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, models: [] };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [] };
    const models = list.flatMap((v) => {
      const m = probedModel(provider.wire, v);
      return m === null ? [] : [m];
    });
    return { provider, ok: true, models };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

const brief = readBrief(PROMPT_FILE);
const ratingRow = (providerId: string, modelId: string): RatingRow | null =>
  brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;

let probes: Probe[] = [];
let catalog: { providers: CatalogProvider[] } = { providers: [] };
/** Free model ids this server serves, per provider, in directory order: what GET /v1/models lists. */
let servedFree: { provider: string; id: string; displayName: string }[] = [];

/**
 * The one catalog document: what `--list` prints and what /register carries. A provider whose
 * probe answered contributes its listing (the quality table settles a model's free flag and
 * context where it has a row); a provider whose probe failed falls back to the quality table's
 * free rows, so the provider is still reported.
 */
function buildCatalog(): void {
  const out: CatalogProvider[] = [];
  servedFree = [];
  for (const provider of brief.providers) {
    const found = probes.find((p) => p.provider.id === provider.id);
    const models: CatalogModel[] = [];
    const seen = new Set<string>();
    if (found !== undefined && found.ok) {
      for (const m of [...found.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
        if (seen.has(m.id)) continue;
        seen.add(m.id);
        const row = ratingRow(provider.id, m.id);
        const free = row !== null ? row.free : (m.free ?? false);
        models.push({ id: m.id, free, context_length: row?.context ?? m.context });
        if (free) servedFree.push({ provider: provider.id, id: m.id, displayName: m.displayName });
      }
    } else {
      for (const row of brief.ratings.filter((r) => r.provider === provider.id && r.free)) {
        if (seen.has(row.model)) continue;
        seen.add(row.model);
        models.push({ id: row.model, free: true, context_length: row.context });
        servedFree.push({ provider: provider.id, id: row.model, displayName: row.model });
      }
      models.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    }
    out.push({ id: provider.id, base_url: provider.base, key_env: provider.keyEnv, wire: provider.wire, models });
  }
  catalog = { providers: out };
}

async function load(): Promise<void> {
  probes = await Promise.all(brief.providers.map((p) => probe(p)));
  buildCatalog();
}

// ---------------------------------------------------------------------------
// The neutral request: what a client body means, wire-free. Client bodies are
// parsed into it, provider bodies are rendered from it, so any client wire can
// be served by any provider wire with every field carried across.
// ---------------------------------------------------------------------------

type Part =
  | { kind: 'text'; text: string }
  | { kind: 'imageData'; media: string; data: string }
  | { kind: 'imageUrl'; url: string };

interface ToolCall {
  id: string;
  name: string;
  argsJson: string;
}

interface ToolResult {
  id: string;
  text: string;
  isError: boolean;
}

interface NMessage {
  role: 'user' | 'assistant';
  parts: Part[];
  toolCalls: ToolCall[];
  toolResults: ToolResult[];
}

interface NTool {
  name: string;
  description: string;
  schema: Json;
}

interface NRequest {
  system: string;
  messages: NMessage[];
  tools: NTool[];
  toolChoice: { mode: 'auto' | 'required' | 'none' } | { mode: 'tool'; name: string } | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stop: string[];
  responseFormat: Json | null;
  stream: boolean;
  requestedModel: string;
  anthropicVersion: string;
}

/** The text of a string, or of an Anthropic system value's text blocks joined. */
function systemText(v: unknown): string {
  if (typeof v === 'string') return v;
  if (!Array.isArray(v)) return '';
  return v
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** Text blocks of a content array joined the way both wires flatten them. */
function blocksText(content: unknown[]): string {
  return content
    .flatMap((b) => (isRecord(b) && typeof b.text === 'string' && (b.type === 'text' || b.type === undefined) ? [b.text] : []))
    .join('\n');
}

/** A tool_result block's content as text: its text blocks, else its JSON. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An image_url value split into a data part or a plain URL part. */
function imagePart(url: string): Part {
  const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (match !== null) return { kind: 'imageData', media: match[1], data: match[2] };
  return { kind: 'imageUrl', url };
}

/** Parse an OpenAI Chat Completions body into the neutral request. */
function parseOpenAI(body: Json): NRequest {
  const out: NRequest = {
    system: '',
    messages: [],
    tools: [],
    toolChoice: null,
    maxTokens: num(body.max_tokens) ?? num(body.max_completion_tokens),
    temperature: num(body.temperature),
    topP: num(body.top_p),
    stop: [],
    responseFormat: isRecord(body.response_format) ? body.response_format : null,
    stream: body.stream === true,
    requestedModel: str(body.model),
    anthropicVersion: '2023-06-01',
  };
  const systems: string[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system' || role === 'developer') {
      const text = typeof raw.content === 'string' ? raw.content : Array.isArray(raw.content) ? blocksText(raw.content) : '';
      if (text !== '') systems.push(text);
      continue;
    }
    if (role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : Array.isArray(raw.content) ? blocksText(raw.content) : '';
      // A tool message joins the pending user turn's results, or starts one.
      const last = out.messages[out.messages.length - 1];
      const result: ToolResult = { id: str(raw.tool_call_id), text, isError: false };
      if (last !== undefined && last.role === 'user' && last.parts.length === 0) last.toolResults.push(result);
      else out.messages.push({ role: 'user', parts: [], toolCalls: [], toolResults: [result] });
      continue;
    }
    const message: NMessage = { role: role === 'assistant' ? 'assistant' : 'user', parts: [], toolCalls: [], toolResults: [] };
    if (typeof raw.content === 'string') {
      if (raw.content !== '') message.parts.push({ kind: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') message.parts.push({ kind: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          message.parts.push(imagePart(part.image_url.url));
        }
      }
    }
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        message.toolCalls.push({
          id: str(call.id),
          name: str(call.function.name),
          argsJson: typeof call.function.arguments === 'string' ? call.function.arguments : '{}',
        });
      }
    }
    out.messages.push(message);
  }
  out.system = systems.join('\n');
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || str(raw.function.name) === '') continue;
      out.tools.push({
        name: str(raw.function.name),
        description: str(raw.function.description),
        schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.toolChoice = { mode: 'auto' };
  else if (choice === 'required') out.toolChoice = { mode: 'required' };
  else if (choice === 'none') out.toolChoice = { mode: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && str(choice.function.name) !== '') {
    out.toolChoice = { mode: 'tool', name: str(choice.function.name) };
  }
  if (typeof body.stop === 'string' && body.stop !== '') out.stop = [body.stop];
  else if (Array.isArray(body.stop)) out.stop = body.stop.filter((s): s is string => typeof s === 'string');
  return out;
}

/** Parse an Anthropic Messages body into the neutral request. */
function parseAnthropic(body: Json, version: string): NRequest {
  const out: NRequest = {
    system: systemText(body.system),
    messages: [],
    tools: [],
    toolChoice: null,
    maxTokens: num(body.max_tokens),
    temperature: num(body.temperature),
    topP: num(body.top_p),
    stop: [],
    responseFormat: null,
    stream: body.stream === true,
    requestedModel: str(body.model),
    anthropicVersion: version,
  };
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const message: NMessage = {
      role: raw.role === 'assistant' ? 'assistant' : 'user',
      parts: [],
      toolCalls: [],
      toolResults: [],
    };
    if (typeof raw.content === 'string') {
      if (raw.content !== '') message.parts.push({ kind: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          message.parts.push({ kind: 'text', text: block.text });
        } else if (block.type === 'image' && isRecord(block.source)) {
          const source = block.source;
          if (source.type === 'base64' && typeof source.data === 'string') {
            message.parts.push({ kind: 'imageData', media: str(source.media_type) !== '' ? str(source.media_type) : 'image/png', data: source.data });
          } else if (source.type === 'url' && typeof source.url === 'string') {
            message.parts.push(imagePart(source.url));
          }
        } else if (block.type === 'tool_use') {
          message.toolCalls.push({
            id: str(block.id),
            name: str(block.name),
            argsJson: JSON.stringify(isRecord(block.input) ? block.input : {}),
          });
        } else if (block.type === 'tool_result') {
          message.toolResults.push({
            id: str(block.tool_use_id),
            text: toolResultText(block.content),
            isError: block.is_error === true,
          });
        }
      }
    }
    out.messages.push(message);
  }
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || str(raw.name) === '') continue;
      out.tools.push({
        name: str(raw.name),
        description: str(raw.description),
        schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.toolChoice = { mode: 'auto' };
    else if (choice.type === 'any') out.toolChoice = { mode: 'required' };
    else if (choice.type === 'none') out.toolChoice = { mode: 'none' };
    else if (choice.type === 'tool' && str(choice.name) !== '') out.toolChoice = { mode: 'tool', name: str(choice.name) };
  }
  if (Array.isArray(body.stop_sequences)) {
    out.stop = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  }
  return out;
}

/** A tool call's arguments as an object; {} when the JSON does not parse to one. */
function argsObject(argsJson: string): Json {
  try {
    const parsed: unknown = JSON.parse(argsJson);
    if (isRecord(parsed)) return parsed;
  } catch {
    // Fall through to the empty object.
  }
  return {};
}

// ---------------------------------------------------------------------------
// Rendering the neutral request into each provider wire's body.
// ---------------------------------------------------------------------------

function renderOpenAI(req: NRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.system !== '') messages.push({ role: 'system', content: req.system });
  for (const message of req.messages) {
    for (const result of message.toolResults) {
      messages.push({ role: 'tool', tool_call_id: result.id, content: result.text });
    }
    if (message.role === 'assistant') {
      const text = message.parts.flatMap((p) => (p.kind === 'text' ? [p.text] : [])).join('\n');
      const entry: Json = { role: 'assistant', content: text !== '' ? text : null };
      if (message.toolCalls.length > 0) {
        entry.tool_calls = message.toolCalls.map((call) => ({
          id: call.id,
          type: 'function',
          function: { name: call.name, arguments: call.argsJson },
        }));
      }
      messages.push(entry);
      continue;
    }
    if (message.parts.length === 0) continue;
    if (message.parts.length === 1 && message.parts[0].kind === 'text') {
      messages.push({ role: 'user', content: message.parts[0].text });
    } else {
      messages.push({
        role: 'user',
        content: message.parts.map((p) => {
          if (p.kind === 'text') return { type: 'text', text: p.text };
          if (p.kind === 'imageData') return { type: 'image_url', image_url: { url: `data:${p.media};base64,${p.data}` } };
          return { type: 'image_url', image_url: { url: p.url } };
        }),
      });
    }
  }
  const out: Json = { model, messages };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stop.length > 0) out.stop = req.stop;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((tool) => ({
      type: 'function',
      function: { name: tool.name, description: tool.description, parameters: tool.schema },
    }));
  }
  if (req.toolChoice !== null) {
    if (req.toolChoice.mode === 'tool') out.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
    else out.tool_choice = req.toolChoice.mode;
  }
  if (req.responseFormat !== null) out.response_format = req.responseFormat;
  out.stream = false;
  return out;
}

function renderAnthropic(req: NRequest, model: string): Json {
  const messages: Json[] = [];
  for (const message of req.messages) {
    const blocks: Json[] = [];
    for (const result of message.toolResults) {
      blocks.push({
        type: 'tool_result',
        tool_use_id: result.id,
        content: result.text,
        ...(result.isError ? { is_error: true } : {}),
      });
    }
    for (const part of message.parts) {
      if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
      else if (part.kind === 'imageData') {
        blocks.push({ type: 'image', source: { type: 'base64', media_type: part.media, data: part.data } });
      } else blocks.push({ type: 'image', source: { type: 'url', url: part.url } });
    }
    for (const call of message.toolCalls) {
      blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: argsObject(call.argsJson) });
    }
    if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
    messages.push({ role: message.role, content: blocks });
  }
  const out: Json = { model, max_tokens: req.maxTokens ?? 1024, messages };
  if (req.system !== '') out.system = req.system;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stop.length > 0) out.stop_sequences = req.stop;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.schema }));
  }
  if (req.toolChoice !== null) {
    if (req.toolChoice.mode === 'auto') out.tool_choice = { type: 'auto' };
    else if (req.toolChoice.mode === 'required') out.tool_choice = { type: 'any' };
    else if (req.toolChoice.mode === 'none') out.tool_choice = { type: 'none' };
    else out.tool_choice = { type: 'tool', name: req.toolChoice.name };
  }
  out.stream = false;
  return out;
}

function renderGemini(req: NRequest, model: string): Json {
  // Gemini names a functionResponse by the function's name, so map call ids to names first.
  const nameById = new Map<string, string>();
  for (const message of req.messages) {
    for (const call of message.toolCalls) nameById.set(call.id, call.name);
  }
  const contents: Json[] = [];
  for (const message of req.messages) {
    const parts: Json[] = [];
    for (const result of message.toolResults) {
      parts.push({
        functionResponse: {
          name: nameById.get(result.id) ?? '',
          response: { result: result.text },
        },
      });
    }
    for (const part of message.parts) {
      if (part.kind === 'text') parts.push({ text: part.text });
      else if (part.kind === 'imageData') parts.push({ inlineData: { mimeType: part.media, data: part.data } });
      else parts.push({ fileData: { fileUri: part.url } });
    }
    for (const call of message.toolCalls) {
      parts.push({ functionCall: { name: call.name, args: argsObject(call.argsJson) } });
    }
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (req.system !== '') out.systemInstruction = { parts: [{ text: req.system }] };
  const generation: Json = {};
  if (req.maxTokens !== null) generation.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) generation.temperature = req.temperature;
  if (req.topP !== null) generation.topP = req.topP;
  if (req.stop.length > 0) generation.stopSequences = req.stop;
  if (Object.keys(generation).length > 0) out.generationConfig = generation;
  if (req.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: req.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          parameters: tool.schema,
        })),
      },
    ];
  }
  if (req.toolChoice !== null) {
    const mode =
      req.toolChoice.mode === 'auto'
        ? 'AUTO'
        : req.toolChoice.mode === 'required' || req.toolChoice.mode === 'tool'
          ? 'ANY'
          : 'NONE';
    const config: Json = { mode };
    if (req.toolChoice.mode === 'tool') config.allowedFunctionNames = [req.toolChoice.name];
    out.toolConfig = { functionCallingConfig: config };
  }
  void model;
  return out;
}

// ---------------------------------------------------------------------------
// The neutral answer: what a provider's reply means, wire-free, and its
// rendering into each client wire, as JSON or as server-sent events.
// ---------------------------------------------------------------------------

interface NAnswer {
  id: string;
  text: string;
  toolCalls: ToolCall[];
  finish: 'stop' | 'length' | 'tool_calls' | 'content_filter';
  inputTokens: number;
  outputTokens: number;
}

const zero = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

/** One provider answer, whichever wire it arrived on, as the neutral answer. */
function parseAnswer(wire: Wire, body: Json): NAnswer {
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    let text = '';
    if (typeof message.content === 'string') text = message.content;
    else if (Array.isArray(message.content)) text = blocksText(message.content);
    const toolCalls: ToolCall[] = [];
    if (Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: str(call.id),
          name: str(call.function.name),
          argsJson: typeof call.function.arguments === 'string' ? call.function.arguments : '{}',
        });
      }
    }
    const finish =
      choice.finish_reason === 'length'
        ? 'length'
        : choice.finish_reason === 'tool_calls' || choice.finish_reason === 'function_call'
          ? 'tool_calls'
          : choice.finish_reason === 'content_filter'
            ? 'content_filter'
            : 'stop';
    const usage = isRecord(body.usage) ? body.usage : {};
    return { id: str(body.id), text, toolCalls, finish, inputTokens: zero(usage.prompt_tokens), outputTokens: zero(usage.completion_tokens) };
  }
  if (wire === 'anthropic') {
    let text = '';
    const toolCalls: ToolCall[] = [];
    if (Array.isArray(body.content)) {
      for (const block of body.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') text += block.text;
        else if (block.type === 'tool_use') {
          toolCalls.push({ id: str(block.id), name: str(block.name), argsJson: JSON.stringify(isRecord(block.input) ? block.input : {}) });
        }
      }
    }
    const finish =
      body.stop_reason === 'max_tokens'
        ? 'length'
        : body.stop_reason === 'tool_use'
          ? 'tool_calls'
          : body.stop_reason === 'refusal'
            ? 'content_filter'
            : 'stop';
    const usage = isRecord(body.usage) ? body.usage : {};
    return { id: str(body.id), text, toolCalls, finish, inputTokens: zero(usage.input_tokens), outputTokens: zero(usage.output_tokens) };
  }
  // Gemini: the first candidate's parts, its finish reason and the usage metadata.
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  let text = '';
  const toolCalls: ToolCall[] = [];
  let callIndex = 0;
  if (Array.isArray(content.parts)) {
    for (const part of content.parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string') text += part.text;
      else if (isRecord(part.functionCall)) {
        callIndex += 1;
        toolCalls.push({
          id: `call_${callIndex}`,
          name: str(part.functionCall.name),
          argsJson: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        });
      }
    }
  }
  const finish =
    candidate.finishReason === 'MAX_TOKENS'
      ? 'length'
      : candidate.finishReason === 'SAFETY' || candidate.finishReason === 'PROHIBITED_CONTENT'
        ? 'content_filter'
        : toolCalls.length > 0
          ? 'tool_calls'
          : 'stop';
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: '',
    text,
    toolCalls,
    finish,
    inputTokens: zero(usage.promptTokenCount),
    outputTokens: zero(usage.candidatesTokenCount),
  };
}

/** The Anthropic spelling of a neutral finish reason. */
function anthropicStop(finish: NAnswer['finish']): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function sse(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The neutral answer as an OpenAI Chat Completions object naming `model`. */
function renderOpenAIAnswer(answer: NAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text !== '' ? answer.text : null };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call) => ({
      id: call.id,
      type: 'function',
      function: { name: call.name, arguments: call.argsJson },
    }));
    if (answer.text === '') message.content = null;
  }
  return {
    id: answer.id !== '' ? answer.id : 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: answer.finish }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** The neutral answer as an OpenAI SSE stream naming `model`: role, content, calls, finish, usage, done. */
function renderOpenAISse(answer: NAnswer, model: string): string {
  const id = answer.id !== '' ? answer.id : 'chatcmpl-aggregate';
  const chunk = (delta: Json, finish: string | null): Json => ({
    id,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let out = sse(null, chunk({ role: 'assistant', content: '' }, null));
  if (answer.text !== '') out += sse(null, chunk({ content: answer.text }, null));
  answer.toolCalls.forEach((call, index) => {
    out += sse(
      null,
      chunk(
        { tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.name, arguments: call.argsJson } }] },
        null,
      ),
    );
  });
  out += sse(null, chunk({}, answer.finish));
  out += sse(null, {
    id,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

/** The neutral answer as an Anthropic Messages object naming `model`. */
function renderAnthropicAnswer(answer: NAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_aggregate', name: call.name, input: argsObject(call.argsJson) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: answer.id !== '' ? (answer.id.startsWith('msg_') ? answer.id : `msg_${answer.id}`) : 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(answer.finish),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** The neutral answer as the Anthropic SSE event sequence naming `model`. */
function renderAnthropicSse(answer: NAnswer, model: string): string {
  const id = answer.id !== '' ? (answer.id.startsWith('msg_') ? answer.id : `msg_${answer.id}`) : 'msg_aggregate';
  let out = sse('message_start', {
    type: 'message_start',
    message: {
      id,
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  });
  let index = 0;
  if (answer.text !== '') {
    out += sse('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
    out += sse('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: answer.text } });
    out += sse('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  for (const call of answer.toolCalls) {
    const callId = call.id !== '' ? call.id : 'toolu_aggregate';
    out += sse('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'tool_use', id: callId, name: call.name, input: {} },
    });
    out += sse('content_block_delta', {
      type: 'content_block_delta',
      index,
      delta: { type: 'input_json_delta', partial_json: call.argsJson },
    });
    out += sse('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  if (index === 0) {
    out += sse('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
    out += sse('content_block_stop', { type: 'content_block_stop', index: 0 });
  }
  out += sse('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: anthropicStop(answer.finish), stop_sequence: null },
    usage: { output_tokens: answer.outputTokens },
  });
  out += sse('message_stop', { type: 'message_stop' });
  return out;
}

/** An error in the client's wire shape. */
function errorBody(wire: ClientWire, type: string, message: string): Json {
  if (wire === 'anthropic') return { type: 'error', error: { type, message } };
  return { error: { message, type, code: null } };
}

/** GET /v1/models in the OpenAI shape: every free model the catalog reports. */
function openAIModelList(): Json {
  return {
    object: 'list',
    data: servedFree.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models in the Anthropic shape. */
function anthropicModelList(): Json {
  return {
    data: servedFree.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: FIXED_CREATED_AT })),
    has_more: false,
    first_id: servedFree.length > 0 ? servedFree[0].id : null,
    last_id: servedFree.length > 0 ? servedFree[servedFree.length - 1].id : null,
  };
}

// ---------------------------------------------------------------------------
// Routing and serving: greedy on the quality table, with failover.
// ---------------------------------------------------------------------------

/** Refusals seen per provider this trial: budgets, schedules and outages are provider-level. */
const providerFailures = new Map<string, number>();
/** The manifest row for the request now arriving: the client sends them in manifest order. */
let arrivalIndex = 0;

interface Candidate {
  provider: ProviderInfo;
  row: RatingRow;
  probeOk: boolean;
}

/** What one request needs: the manifest's word, plus anything the body itself demands. */
function requirementsFor(entry: ManifestEntry | undefined, req: NRequest): Requirement {
  const base: Requirement = entry?.requires ?? { tools: false, vision: false, json: false, minContext: null };
  return {
    tools: base.tools || req.tools.length > 0,
    vision:
      base.vision || req.messages.some((m) => m.parts.some((p) => p.kind === 'imageData' || p.kind === 'imageUrl')),
    json: base.json || req.responseFormat !== null,
    minContext: base.minContext,
  };
}

function meets(row: RatingRow, needs: Requirement): boolean {
  if (needs.tools && !row.tools) return false;
  if (needs.vision && !row.vision) return false;
  if (needs.json && !row.json) return false;
  if (needs.minContext !== null && (row.context === null || row.context < needs.minContext)) return false;
  return true;
}

/**
 * Every free model that can serve this request, best first: rating, then fewest refusals so
 * far, then a probe that answered, then the client's own wire (least translation), then ids —
 * a fixed total order, so a replay with the same answers walks the same list.
 */
function candidatesFor(needs: Requirement, req: NRequest, clientWire: ClientWire): Candidate[] {
  const out: Candidate[] = [];
  for (const row of brief.ratings) {
    if (!row.free) continue;
    const provider = brief.providers.find((p) => p.id === row.provider);
    if (provider === undefined) continue;
    if (!meets(row, needs)) continue;
    // A JSON-mode body field only the OpenAI wire can carry: route those to OpenAI-wire providers.
    if (req.responseFormat !== null && provider.wire !== 'openai') continue;
    const probeOk = probes.find((p) => p.provider.id === provider.id)?.ok ?? false;
    out.push({ provider, row, probeOk });
  }
  out.sort((a, b) => {
    const fa = providerFailures.get(a.provider.id) ?? 0;
    const fb = providerFailures.get(b.provider.id) ?? 0;
    // A provider that has refused twice is treated as spent (budget or outage) and sinks below
    // every provider that has not, whatever its rating: the calls it would waste cost more than
    // the few milli-points its models might add.
    const deadA = fa >= 2 ? 1 : 0;
    const deadB = fb >= 2 ? 1 : 0;
    if (deadA !== deadB) return deadA - deadB;
    if (b.row.rating !== a.row.rating) return b.row.rating - a.row.rating;
    if (fa !== fb) return fa - fb;
    if (a.probeOk !== b.probeOk) return a.probeOk ? -1 : 1;
    const wa = a.provider.wire === clientWire ? 0 : 1;
    const wb = b.provider.wire === clientWire ? 0 : 1;
    if (wa !== wb) return wa - wb;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0;
  });
  return out;
}

interface UpstreamReply {
  status: number;
  text: string;
  contentType: string;
}

/** One upstream inference call on a provider's wire; the reply's raw text and status. */
async function callProvider(candidate: Candidate, req: NRequest, rawBody: Json, clientWire: ClientWire): Promise<UpstreamReply> {
  const { provider, row } = candidate;
  const base = joinBase(provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider) };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    // Same wire as the client: the client's own body, with the chosen model named.
    body = clientWire === 'openai' ? { ...rawBody, model: row.model } : renderOpenAI(req, row.model);
    if (clientWire !== 'openai' || req.stream !== true) body.stream = false;
    if (req.stream === true && clientWire === 'openai') {
      headers.accept = 'text/event-stream';
      // A streamed answer must still carry its final usage: ask for it in the upstream call.
      const options = isRecord(body.stream_options) ? body.stream_options : {};
      body.stream_options = { ...options, include_usage: true };
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = req.anthropicVersion;
    body = clientWire === 'anthropic' ? { ...rawBody, model: row.model, stream: false } : renderAnthropic(req, row.model);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(row.model)}:generateContent`;
    body = renderGemini(req, row.model);
  }
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
  return { status: res.status, text: await res.text(), contentType: res.headers.get('content-type') ?? 'application/json' };
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  sendText(res, status, 'application/json', JSON.stringify(body));
}

function sendClientError(res: ServerResponse, wire: ClientWire, status: number, type: string, message: string): void {
  sendJson(res, status, errorBody(wire, type, message));
}

/** Answer one client request: try candidates in order until one serves it. */
async function serve(clientWire: ClientWire, rawBody: Json, version: string, res: ServerResponse): Promise<void> {
  const req = clientWire === 'openai' ? parseOpenAI(rawBody) : parseAnthropic(rawBody, version);
  const entry = brief.manifest[arrivalIndex];
  arrivalIndex += 1;
  const needs = requirementsFor(entry, req);
  const candidates = candidatesFor(needs, req, clientWire);
  const refusedProviders = new Set<string>();
  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= 10) break;
    if (refusedProviders.has(candidate.provider.id)) continue;
    attempts += 1;
    let reply: UpstreamReply;
    try {
      reply = await callProvider(candidate, req, rawBody, clientWire);
    } catch {
      providerFailures.set(candidate.provider.id, (providerFailures.get(candidate.provider.id) ?? 0) + 1);
      refusedProviders.add(candidate.provider.id);
      continue;
    }
    if (reply.status >= 200 && reply.status < 300) {
      const upstreamSse = reply.contentType.includes('text/event-stream');
      if (candidate.provider.wire === clientWire && req.stream === upstreamSse) {
        // Same wire both sides and the same shape the client asked for: the provider's own
        // document (or its event stream) passes through untouched.
        sendText(res, 200, req.stream ? 'text/event-stream' : reply.contentType, reply.text);
        return;
      }
      // Otherwise the answer is rebuilt in the client's wire from the parsed provider document:
      // a cross-wire answer, or a JSON answer to a client that asked for a stream.
      let parsed: unknown;
      try {
        parsed = JSON.parse(reply.text);
      } catch {
        continue;
      }
      if (!isRecord(parsed)) continue;
      const answer = parseAnswer(candidate.provider.wire, parsed);
      const modelForClient = clientWire === 'anthropic' ? (req.requestedModel !== '' ? req.requestedModel : candidate.row.model) : candidate.row.model;
      if (req.stream) {
        sendText(res, 200, 'text/event-stream', clientWire === 'anthropic' ? renderAnthropicSse(answer, modelForClient) : renderOpenAISse(answer, modelForClient));
      } else {
        sendJson(res, 200, clientWire === 'anthropic' ? renderAnthropicAnswer(answer, modelForClient) : renderOpenAIAnswer(answer, modelForClient));
      }
      return;
    }
    if (reply.status === 400) {
      // A model-level refusal (unknown or paid model): it spends no budget; try the next candidate.
      continue;
    }
    // 402, 429 or 5xx: a provider-level refusal. Count it and fail over to another provider.
    providerFailures.set(candidate.provider.id, (providerFailures.get(candidate.provider.id) ?? 0) + 1);
    refusedProviders.add(candidate.provider.id);
  }
  sendClientError(
    res,
    clientWire,
    502,
    clientWire === 'anthropic' ? 'api_error' : 'upstream_error',
    'no free model on the directory could serve this request',
  );
}

/** One client request body, or null when it is not one JSON object. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList());
    else sendJson(res, 200, openAIModelList());
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendClientError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, '2023-06-01', res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendClientError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    const version = typeof req.headers['anthropic-version'] === 'string' ? req.headers['anthropic-version'] : '2023-06-01';
    await serve('anthropic', body, version, res);
    return;
  }
  sendClientError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

// ---------------------------------------------------------------------------
// Bootstrap: load, maybe --list, listen, register, poll until done.
// ---------------------------------------------------------------------------

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST /register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function pollWork(server: ReturnType<typeof createServer>): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendClientError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout');
  log(`listening on 0.0.0.0:${address.port} with ${catalog.providers.length} providers and ${servedFree.length} free models`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
