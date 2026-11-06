"use strict";
const $ = (id) => document.getElementById(id);
let state = {},
  current = null,
  selected = "",
  token = "",
  activeJob = "";
const drafts = {};
const message = (text) => ($("message").textContent = text);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
async function api(path, body) {
  const response = await fetch(
    path,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Lab-Token": token },
          body: JSON.stringify(body),
        }
      : {},
  );
  const data = await response.json();
  if (!response.ok) throw Error(data.error || response.statusText);
  return data;
}
function fillSelect(id, values, label = "All") {
  const node = $(id),
    old = node.value;
  node.replaceChildren();
  if (label) node.add(new Option(label, "all"));
  for (const value of values) node.add(new Option(value, value));
  if ([...node.options].some((x) => x.value === old)) node.value = old;
}
function group(items, key) {
  const groups = {};
  for (const item of items) {
    const k = key(item);
    (groups[k] ??= []).push(item);
  }
  return groups;
}
function mean(items, key) {
  return items.length
    ? items.reduce((s, x) => s + Number(key(x) || 0), 0) / items.length
    : 0;
}
function repetitionRatio(item) {
  const ids = item.token_ids || [];
  if (ids.length < 4) return 0;
  const grams = [];
  for (let i = 0; i <= ids.length - 4; i++)
    grams.push(ids.slice(i, i + 4).join(","));
  return 1 - new Set(grams).size / grams.length;
}
function probeOutputs() {
  return (current?.outputs || []).filter(
    (x) => !x.phase || x.phase === "probe",
  );
}
function cohortOutputs() {
  const cohort = $("cohortFilter").value || "all";
  return probeOutputs().filter((x) => cohort === "all" || x.cohort === cohort);
}
async function refresh() {
  try {
    state = await api("/api/state");
    token = state.token;
    estimate();
    $("activityText").textContent = state.gpu_busy
      ? "Experiment running"
      : "GPU lab idle";
    $("runs").replaceChildren();
    for (const run of state.runs) {
      const b = el(
        "button",
        "run-item" + (run.id === selected ? " active" : ""),
      );
      b.append(
        el(
          "strong",
          "",
          run.kind === "exploration"
            ? "Channel exploration"
            : run.kind === "personas"
              ? "Persona discovery"
              : "Baseline check",
        ),
        el("span", "", run.id.split("-LFM")[0]),
        el("span", "", `${run.outputs} replies · ${run.status}`),
      );
      b.onclick = () => loadRun(run.id);
      $("runs").append(b);
    }
    renderJobs();
    if (!selected && state.runs.length) await loadRun(state.runs[0].id);
    else if (
      current &&
      state.runs.find((x) => x.id === selected)?.outputs !==
        current.outputs.length
    )
      await loadRun(selected, false);
  } catch (e) {
    message(e.message);
  }
}
async function loadRun(id, reset = true) {
  current = await api("/api/run?id=" + encodeURIComponent(id));
  selected = id;
  $("runTitle").textContent = id;
  $("runSubtitle").textContent =
    "Pinned checkpoint · raw outputs preserved · exploratory measurements";
  const out = probeOutputs();
  fillSelect(
    "promptFilter",
    [...new Set(out.map((x) => x.prompt_id || x.id))],
    null,
  );
  fillSelect("conditionFilter", [...new Set(out.map((x) => x.condition))]);
  fillSelect("roleFilter", [
    ...new Set(out.filter((x) => x.role).map((x) => x.role)),
  ]);
  fillSelect("cohortFilter", [
    ...new Set(out.filter((x) => x.cohort).map((x) => x.cohort)),
  ]);
  fillSelect(
    "layerFilter",
    [
      ...new Set(out.filter((x) => x.layer !== undefined).map((x) => x.layer)),
    ].sort((a, b) => a - b),
  );
  fillSelect(
    "doseFilter",
    [
      ...new Set(out.filter((x) => x.dose !== undefined).map((x) => x.dose)),
    ].sort((a, b) => a - b),
  );
  if (reset) {
    $("personaFilter").value = "all";
    $("roleFilter").value = "all";
    $("cohortFilter").value = "all";
    $("conditionFilter").value = "all";
    $("layerFilter").value = "all";
    $("doseFilter").value = "all";
  }
  $("provenance").textContent = JSON.stringify(
    { checks: current.checks, metadata: current.metadata },
    null,
    2,
  );
  renderMetrics(cohortOutputs());
  renderCharts(cohortOutputs());
  renderMap();
  renderOutputs();
  document
    .querySelectorAll(".run-item")
    .forEach((n, i) => n.classList.toggle("active", state.runs[i]?.id === id));
}
function renderMetrics(out) {
  $("metrics").replaceChildren();
  const pairs = [
    [current.outputs.length, "Recorded replies"],
    [
      out.filter((x) => x.terminated_with_eos).length + "/" + out.length,
      "Probe replies reaching EOS",
    ],
    [Math.round(mean(out, (x) => x.output_tokens)), "Mean probe output tokens"],
    [
      Object.values(current.annotations || {}).filter(
        (x) => x.label !== "unreviewed",
      ).length,
      "Human annotations",
    ],
  ];
  for (const [value, label] of pairs) {
    const c = el("div", "metric");
    c.append(el("strong", "", value), el("span", "", label));
    $("metrics").append(c);
  }
}
function renderCharts(out) {
  const groups = group(out, (x) =>
    x.condition === "channel-gain" ? `${x.candidate} ×${x.gain}` : x.condition,
  );
  $("lengthChart").replaceChildren();
  $("timingChart").replaceChildren();
  const max = Math.max(
    1,
    ...Object.values(groups).map((a) => mean(a, (x) => x.output_tokens)),
  );
  for (const [name, items] of Object.entries(groups)) {
    for (const [target, value, text, violet] of [
      [
        "lengthChart",
        mean(items, (x) => x.output_tokens) / max,
        mean(items, (x) => x.output_tokens).toFixed(1),
        false,
      ],
      [
        "timingChart",
        items.filter((x) => x.terminated_with_eos).length / items.length,
        `${items.filter((x) => x.terminated_with_eos).length}/${items.length} · ${mean(items, (x) => x.seconds).toFixed(2)}s`,
        true,
      ],
    ]) {
      const row = el("div", "bar-row"),
        track = el("div", "bar-track"),
        bar = el("div", "bar-fill" + (violet ? " violet" : ""));
      bar.style.width = value * 100 + "%";
      track.append(bar);
      row.append(el("span", "", name), track, el("span", "bar-value", text));
      $(target).append(row);
    }
  }
}
function renderMap() {
  const out = cohortOutputs().filter(
    (x) =>
      x.condition === "contrast-direction" || x.condition === "channel-gain",
  );
  $("heatmap").replaceChildren();
  if (!out.length) {
    $("heatmap").append(el("p", "muted", "No intervention sweep in this run."));
    return;
  }
  const measure = $("mapMetric").value;
  const gainMode = out.some((x) => x.condition === "channel-gain");
  for (const persona of [
    ...new Set(out.map((x) => x.persona || "Channel exploration")),
  ]) {
    const data = out.filter(
        (x) => (x.persona || "Channel exploration") === persona,
      ),
      layers = gainMode
        ? [...new Set(data.map((x) => x.candidate))].sort()
        : [...new Set(data.map((x) => x.layer))].sort((a, b) => a - b),
      doses = [...new Set(data.map((x) => (gainMode ? x.gain : x.dose)))].sort(
        (a, b) => a - b,
      ),
      table = el("table", "heat-table"),
      head = el("tr");
    head.append(el("th", "", persona));
    doses.forEach((d) =>
      head.append(el("th", "", `${gainMode ? "gain" : "dose"} ${d}`)),
    );
    table.append(head);
    for (const layer of layers) {
      const row = el("tr");
      row.append(el("th", "", gainMode ? layer : `Layer ${layer}`));
      for (const dose of doses) {
        const items = data.filter(
          (x) =>
            (gainMode ? x.candidate === layer : x.layer === layer) &&
            (gainMode ? x.gain : x.dose) === dose,
        );
        const measurementKey =
          measure === "think"
            ? "reasoning_boundary_present"
            : measure === "json"
              ? "exact_json_match"
              : null;
        const scoredItems = measurementKey
          ? items.filter(
              (x) => typeof x.measurements?.[measurementKey] === "boolean",
            )
          : ["hamster", "villain"].includes(measure)
            ? items.filter((x) =>
                Array.isArray(x.indicators?.[measure + "_terms_present"]),
              )
            : items;
        const value = mean(scoredItems, (x) =>
          measurementKey
            ? Number(x.measurements[measurementKey])
            : measure === "eos"
              ? Number(x.terminated_with_eos)
              : measure === "tokens"
                ? x.output_tokens
                : measure === "seconds"
                  ? x.seconds
                  : measure === "repetition"
                    ? repetitionRatio(x)
                    : (x.indicators?.[measure + "_terms_present"] || []).length,
        );
        const cell = el("td", "", scoredItems.length ? value.toFixed(2) : "—");
        cell.append(el("span", "", `n=${scoredItems.length}`));
        row.append(cell);
      }
      table.append(row);
    }
    $("heatmap").append(table);
  }
}
function renderOutputs() {
  if (!current) return;
  const prompt = $("promptFilter").value;
  const original = current.outputs.find(
    (x) => (x.prompt_id || x.id) === prompt && x.condition === "baseline",
  );
  $("baselinePrompt").textContent = original?.prompt || prompt;
  $("baselineText").textContent =
    original?.text || "No unsteered reference recorded for this prompt.";
  $("outputs").replaceChildren();
  const filtered = current.outputs
    .map((x, i) => ({ ...x, _index: i }))
    .filter(
      (x) =>
        (!x.phase || x.phase === "probe") &&
        (x.prompt_id || x.id) === prompt &&
        ($("cohortFilter").value === "all" ||
          x.cohort === $("cohortFilter").value) &&
        ($("roleFilter").value === "all" || x.role === $("roleFilter").value) &&
        x.condition !== "baseline" &&
        ($("personaFilter").value === "all" ||
          x.persona === $("personaFilter").value ||
          x.condition === "random-direction") &&
        ($("conditionFilter").value === "all" ||
          x.condition === $("conditionFilter").value) &&
        ($("layerFilter").value === "all" ||
          String(x.layer) === $("layerFilter").value) &&
        ($("doseFilter").value === "all" ||
          String(x.dose) === $("doseFilter").value),
    );
  for (const output of filtered) {
    const card = el("article", "panel"),
      top = el("div", "output-title");
    top.append(
      el(
        "h3",
        "",
        [output.persona, output.condition].filter(Boolean).join(" · "),
      ),
    );
    card.append(top);
    const tags = el("div", "tags");
    for (const text of [
      output.layer !== undefined ? `Layer ${output.layer}` : null,
      output.neuron !== undefined ? `Channel ${output.neuron}` : null,
      output.gain !== undefined ? `Gain ${output.gain}` : null,
      output.role || null,
      output.cohort || null,
      output.measurements?.reasoning_boundary_present ? "Think boundary" : null,
      output.measurements?.self_check_phrase_present
        ? "Self-check phrase"
        : null,
      typeof output.measurements?.exact_json_match === "boolean"
        ? output.measurements.exact_json_match
          ? "Exact JSON matched"
          : "Exact JSON failed"
        : null,
      typeof output.token_identical_to_baseline === "boolean"
        ? output.token_identical_to_baseline
          ? "Token-identical"
          : "Changed sequence"
        : null,
      output.dose !== undefined ? `Dose ${output.dose}` : null,
      `${output.output_tokens} tokens`,
      `${output.seconds.toFixed(2)}s`,
      `${(repetitionRatio(output) * 100).toFixed(0)}% repeated 4-grams`,
      output.terminated_with_eos ? "EOS" : "Token cap / no EOS",
    ].filter(Boolean))
      tags.append(el("span", "tag", text));
    card.append(tags, el("pre", "", output.text));
    const form = el("div", "review-fields"),
      label = el("select"),
      note = el("textarea"),
      save = el("button", "secondary", "Save review");
    label.setAttribute("aria-label", "Review label");
    note.setAttribute("aria-label", "Review notes");
    for (const x of [
      "unreviewed",
      "interesting",
      "no-effect",
      "damaged",
      "needs-review",
    ])
      label.add(new Option(x, x));
    const draftKey = selected + ":" + output._index;
    const annotation =
      drafts[draftKey] || current.annotations?.[String(output._index)];
    label.value = annotation?.label || "unreviewed";
    note.value = annotation?.note || "";
    note.oninput = label.onchange = () => {
      drafts[draftKey] = { label: label.value, note: note.value };
    };
    note.placeholder = "What changed? What failed? Include uncertainty.";
    save.onclick = async () => {
      try {
        await api("/api/annotations", {
          run: selected,
          index: output._index,
          label: label.value,
          note: note.value,
        });
        current.annotations ??= {};
        current.annotations[output._index] = {
          label: label.value,
          note: note.value,
        };
        delete drafts[draftKey];
        renderMetrics(cohortOutputs());
        message("Review saved locally.");
      } catch (e) {
        message(e.message);
      }
    };
    form.append(label, note, save);
    card.append(form);
    $("outputs").append(card);
  }
  if (!filtered.length)
    $("outputs").append(
      el(
        "p",
        "muted",
        "No comparison replies match these filters. Baseline-only runs contain no persona sweep.",
      ),
    );
}
function renderJobs() {
  $("jobs").replaceChildren();
  for (const job of state.jobs || []) {
    const row = el("div", "job"),
      info = el("div", "", `${job.mode} · ${job.status}`),
      view = el("button", "secondary", "View log");
    view.onclick = () => {
      activeJob = job.id;
      refreshLog();
    };
    row.append(info, view);
    if (job.status === "running") {
      const stop = el("button", "secondary", "Stop job");
      stop.onclick = async () => {
        try {
          await api("/api/stop", { id: job.id });
          await refresh();
          message("Job stopped; partial artifacts kept.");
        } catch (e) {
          message(e.message);
        }
      };
      row.append(stop);
    }
    $("jobs").append(row);
  }
  $("launchButton").disabled = Boolean(state.gpu_busy);
}
async function refreshLog() {
  if (!activeJob && state.jobs?.length) activeJob = state.jobs.at(-1).id;
  if (!activeJob) return;
  try {
    const result = await api("/api/log?id=" + encodeURIComponent(activeJob));
    $("jobLog").textContent = result.text || "Starting…";
  } catch (e) {
    message(e.message);
  }
}
function estimate() {
  if ($("mode").value === "mapping") {
    const selected = state.mapping_designs?.[$("mappingDesign").value];
    const cfg = selected?.config;
    $("designConfig").textContent = cfg
      ? JSON.stringify(cfg, null, 2)
      : "Loading versioned design…";
    $("launchNotice").textContent =
      "The selected versioned design is copied exactly into the job record. Fresh-prompt replication and exploratory mapping remain separate. No persona target is supplied.";
    if (!cfg) {
      $("estimate").textContent = "Loading design…";
      return;
    }
    $("estimate").textContent =
      `${selected.generation_count} generations · frozen design`;
    return;
  }
  $("launchNotice").textContent =
    $("mode").value === "smoke"
      ? "Baseline and neutral-hook checks only."
      : "Persona discovery uses paired prompts and exploratory probes. This is not a held-out benchmark.";
  const n = $("layers")
      .value.split(",")
      .filter((x) => x.trim()).length,
    d = $("doses")
      .value.split(",")
      .filter((x) => x.trim()).length,
    c = Number($("controls").value);
  $("estimate").textContent =
    $("mode").value === "smoke"
      ? "6 generations"
      : `${12 + 3 + 2 + 6 + 2 * n * d * 3 + n * c * 3} generations · duration depends on token cap`;
}
function configureMode() {
  const mode = $("mode").value;
  for (const id of ["layers", "doses", "tokens", "controls", "controlDose"]) {
    $(id).disabled = mode !== "personas";
    $(id)
      .closest("label")
      .classList.toggle("hidden", mode !== "personas");
  }
  $("mappingDesignField").classList.toggle("hidden", mode !== "mapping");
  $("designPreview").classList.toggle("hidden", mode !== "mapping");
  estimate();
}
document.querySelectorAll("[data-tab]").forEach(
  (button) =>
    (button.onclick = () => {
      document
        .querySelectorAll("[data-tab]")
        .forEach((b) => b.classList.toggle("selected", b === button));
      document
        .querySelectorAll(".tab-panel")
        .forEach((p) =>
          p.classList.toggle("hidden", p.id !== button.dataset.tab),
        );
      if (button.dataset.tab === "logs") refreshLog();
    }),
);
for (const id of [
  "promptFilter",
  "personaFilter",
  "conditionFilter",
  "layerFilter",
  "doseFilter",
  "roleFilter",
])
  $(id).onchange = renderOutputs;
$("cohortFilter").onchange = () => {
  fillSelect(
    "promptFilter",
    [...new Set(cohortOutputs().map((x) => x.prompt_id || x.id))],
    null,
  );
  renderMetrics(cohortOutputs());
  renderCharts(cohortOutputs());
  renderMap();
  renderOutputs();
};
$("mapMetric").onchange = renderMap;
$("refresh").onclick = () => refresh().catch((e) => message(e.message));
$("refreshLogs").onclick = refreshLog;
for (const id of ["layers", "doses", "tokens", "controls"])
  $(id).oninput = estimate;
$("mode").onchange = configureMode;
$("mappingDesign").onchange = estimate;
$("launchForm").onsubmit = async (event) => {
  event.preventDefault();
  try {
    const body =
      $("mode").value === "mapping"
        ? { mode: "mapping", design: $("mappingDesign").value }
        : $("mode").value === "smoke"
          ? { mode: "smoke" }
          : {
              mode: $("mode").value,
              layers: $("layers")
                .value.split(",")
                .map((x) => Number(x.trim())),
              doses: $("doses")
                .value.split(",")
                .map((x) => Number(x.trim())),
              max_new_tokens: Number($("tokens").value),
              random_controls_per_layer: Number($("controls").value),
              random_control_dose: Number($("controlDose").value),
            };
    const job = await api("/api/jobs", body);
    activeJob = job.id;
    message("Experiment started. Configuration and log saved locally.");
    await refresh();
  } catch (e) {
    message(e.message);
  }
};
$("export").onclick = () => {
  if (!current) return;
  const blob = new Blob([JSON.stringify(current, null, 2)], {
      type: "application/json",
    }),
    url = URL.createObjectURL(blob),
    a = el("a");
  a.href = url;
  a.download = selected + ".json";
  a.click();
  URL.revokeObjectURL(url);
};
configureMode();
refresh();
setInterval(async () => {
  await refresh();
  if (activeJob && !$("logs").classList.contains("hidden")) await refreshLog();
}, 5000);
