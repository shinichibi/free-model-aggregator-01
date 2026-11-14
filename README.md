# Free model aggregator (free-model-aggregator-01)

One server (`main.ts`, Node 22, standard library only) that serves the OpenAI Chat
Completions API and the Anthropic Messages API, in JSON and in server-sent events, over
ten free-tier model providers that speak the OpenAI, Anthropic or Gemini wires.

At a trial's start it reads the brief (provider directory, public quality table,
workload manifest), probes every provider's catalog, registers the catalog it built,
and then answers each request with the highest-rated free model that meets the
request's requirements and can still serve it, failing over to the next candidate
whenever a provider refuses. `node main.ts --list` prints the catalog document it
registers.
