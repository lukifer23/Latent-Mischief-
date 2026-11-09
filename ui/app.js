"use strict";
const $ = (id) => document.getElementById(id);
let state = {},
  current = null,
  selected = "",
  token = "",
  activeJob = "";
const drafts = {};
function conditionLabel(x) {
  if (x.profile_id) return x.profile_id;
  if (x.condition === "channel-offset") return `${x.candidate} ${x.sign > 0 ? "+" : "−"} offset${x.target_fraction === undefined ? "" : ` · target ${100*x.target_fraction}%`}`;
  if (x.condition === "channel-gain") return `${x.candidate} ×${x.gain}`;
  return [x.condition, x.persona].filter(Boolean).join(" · ");
}
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
  node.disabled = !values.length;
  if (!values.length) node.add(new Option("No choices in this experiment", ""));
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
    fillSelect("calibrationRun", (state.calibrated_designs?.available || []).map((x) => x.id), null);
    estimate();
    $("activityText").textContent = state.read_only ? 'Review only · inference disabled' : state.gpu_busy
      ? "Experiment running"
      : "GPU lab idle";
    $("runs").replaceChildren();
    for (const run of modelRuns()) {
      const b = el(
        "button",
        "run-item" + (run.id === selected ? " active" : ""),
      );
      b.dataset.runId=run.id;
      b.append(
        el(
          "strong",
          "",
          run.kind === "qwen-pilot" ? "Qwen sampled pilot" : run.kind === "live" ? "Live comparison" : run.kind === "exploration"
            ? "Channel exploration"
            : run.kind === "performance"
              ? "Context performance"
            : run.kind === "characters"
              ? "Sparse character lab"
            : run.kind === "calibration"
              ? "Activation calibration"
            : run.kind === "personas"
              ? "Persona discovery"
              : "Baseline check",
        ),
        el("span", "", run.id.split("-LFM")[0]),
        el("span", "", run.progress?.planned?LabCompare.runProgress(run.progress):`${run.kind === "calibration" ? `${run.screen_conditions} screen conditions` : `${run.outputs} replies`} · ${run.status}`),
      );
      b.onclick = async () => {if(run.kind === "performance")showTab("performance");else{await loadRun(run.id);showTab('review');}};
      $("runs").append(b);
    }
    renderJobs();
    const progress=state.runs.find(x=>x.id===selected)?.progress;
    if(progress?.planned)$('runSubtitle').textContent=LabCompare.runProgress(progress);
    if (!selected && state.runs.length) {
      const first = modelRuns().find(x=>x.kind!=="performance" && x.outputs>0);
      if(first) await loadRun(first.id);
    }
    else if (
      current &&
      (state.runs.find((x) => x.id === selected)?.outputs !==
        current.outputs.length || (current.calibration && state.runs.find((x) => x.id === selected)?.screen_conditions !== current.calibration.screening.length))
    )
      await loadRun(selected, false);
  } catch (e) {
    $("activityText").textContent = "Connection error";
    message(e.message);
  }
}
async function loadRun(id, reset = true) {
  current = await api("/api/run?id=" + encodeURIComponent(id));
  selected = id;
  $("runTitle").textContent = (current.metadata?.experiment || "Experiment") + " · " + (current.metadata?.model?.model_id || "Pinned model");
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
      ...new Set(out.filter((x) => (x.target_fraction ?? x.dose) !== undefined).map((x) => x.target_fraction ?? x.dose)),
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
    { checks: current.checks, summary: current.summary, interruption: current.interruption, metadata: current.metadata },
    null,
    2,
  );
  renderMetrics(cohortOutputs());
  renderCharts(cohortOutputs());
  renderMap();
  renderFeatureMap();
  renderOutputs();
  document
    .querySelectorAll(".run-item")
    .forEach((n) => n.classList.toggle("active", n.dataset.runId === id));
}
function renderMetrics(out) {
  $("metrics").replaceChildren();
  const pairs = current.calibration ? [
    [current.calibration.activation_summary?.channels ?? "—", "Channels with activation statistics"],
    [current.calibration.screening.length, "Completed screen conditions"],
    [current.calibration.screening.filter((x) => x.hit_cap).length, "Conditions reaching offset cap"],
    [current.calibration.selected.length / 2, "Selected paired channels"],
  ] : [
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
  const groups = group(out, conditionLabel);
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
  if (current.calibration) {
    $("heatmap").replaceChildren();
    const table = el("table", "heat-table"), head = el("tr");
    for (const title of ["Frozen unit", "Offset sign", "Target local change", "Actual screen mean", "Mean next-token KL (nats)"])
      head.append(el("th", "", title));
    table.append(head);
    for (const item of current.calibration.selected) {
      const row = el("tr");
      for (const value of [item.candidate, item.sign > 0 ? "+" : "−", (100 * item.target_fraction).toFixed(1) + "%",
        (100 * item.mean_actual_fraction).toFixed(2) + "%", item.mean_kl.toPrecision(4)])
        row.append(el("td", "", value));
      table.append(row);
    }
    $("heatmap").append(el("p", "muted", "Calibration screen: frozen selection, not generated behavior. KL is not a quality score. Export includes all screen conditions."), table);
    return;
  }
  const out = cohortOutputs().filter(
    (x) =>
      ["contrast-direction", "channel-gain", "channel-offset", "sparse-profile", "joint-profile"].includes(x.condition),
  );
  $("heatmap").replaceChildren();
  if (!out.length) {
    $("heatmap").append(el("p", "muted", "No intervention sweep in this run."));
    return;
  }
  const measure = $("mapMetric").value;
  const gainMode = out.some((x) => x.condition === "channel-gain");
  const offsetMode = out.some((x) => x.condition === "channel-offset");
  const jointMode = out.some((x) => x.condition === "joint-profile");
  const profileMode = jointMode || out.some((x) => x.condition === "sparse-profile");
  const channelMode = gainMode || offsetMode || profileMode;
  for (const persona of [
    ...new Set(out.map((x) => x.persona || "Channel exploration")),
  ]) {
    const data = out.filter(
        (x) => (x.persona || "Channel exploration") === persona,
      ),
      layers = channelMode
        ? [...new Set(data.map((x) => x.candidate))].sort()
        : [...new Set(data.map((x) => x.layer))].sort((a, b) => a - b),
      doses = [...new Set(data.map((x) => (profileMode ? (x.target_fraction ?? x.dose) : offsetMode ? LabCompare.offsetSetting(x) : gainMode ? x.gain : x.dose)))].sort(
        (a, b) => a - b,
      ),
      table = el("table", "heat-table"),
      head = el("tr");
    head.append(el("th", "", persona));
    doses.forEach((d) =>
      head.append(el("th", "", `${jointMode ? "multiplier" : profileMode ? "local target" : offsetMode ? "sign × local target" : gainMode ? "gain" : "dose"} ${d}`)),
    );
    table.append(head);
    for (const layer of layers) {
      const row = el("tr");
      row.append(el("th", "", channelMode ? layer : `Layer ${layer}`));
      for (const dose of doses) {
        const items = data.filter(
          (x) =>
            (channelMode ? x.candidate === layer : x.layer === layer) &&
            (profileMode ? (x.target_fraction ?? x.dose) : offsetMode ? LabCompare.offsetSetting(x) : gainMode ? x.gain : x.dose) === dose,
        );
        const measurementKey =
          measure === "think"
            ? "reasoning_boundary_present"
            : measure === "json"
              ? "exact_json_match"
              : measure === "exactText" ? "exact_text_match" : null;
        const scoredItems = measurementKey
          ? items.filter(
              (x) => typeof x.measurements?.[measurementKey] === "boolean",
            )
          : measure === "strength"
            ? items.filter((x) => Number.isFinite(x.actual_prefill_fraction))
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
                  : measure === "strength"
                    ? x.actual_prefill_fraction
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
function renderFeatureMap() {
  const keys = Object.keys(current.rankings || {});
  $("featurePanel").classList.toggle("hidden", !keys.length);
  if (!keys.length) return;
  fillSelect("featurePersona", [...new Set(keys.map(x => x.split(":")[0]))], null);
  fillSelect("featureLayer", [...new Set(keys.map(x => Number(x.split(":")[1])))].sort((a,b)=>a-b), null);
  $("featureMap").replaceChildren();
  const rows = current.rankings[$("featurePersona").value+":"+$("featureLayer").value] || [];
  const maximum = Math.max(1, ...rows.map(x => Math.abs(x.standardized_difference)));
  rows.forEach((x,index) => {
    const row=el("div","bar-row"), track=el("div","bar-track"), bar=el("div","bar-fill"+(x.standardized_difference<0?" violet":""));
    bar.style.width=(100*Math.abs(x.standardized_difference)/maximum)+"%";
    track.append(bar);
    row.append(el("span","",`#${index+1} channel ${x.neuron}`),track,el("span","bar-value",x.standardized_difference.toFixed(2)));
    row.title=`Signed paired mean / neutral RMS; consistency ${(100*x.pair_consistency).toFixed(0)}%; neutral RMS ${x.neutral_rms.toPrecision(3)}. Only first 16 selected for tested layers.`;
    $("featureMap").append(row);
  });
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
  $("launchButton").disabled = Boolean(state.read_only||state.gpu_busy)||selectedModelKey!=='lfm3';
  if(state.read_only)$('sendChat').disabled=true;
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
  if(selectedModelKey!=='lfm3'){$('launchNotice').textContent='These frozen batch designs are LFM3-specific. Use Chat & Try to run Qwen default versus its published edit.';$('estimate').textContent='Qwen interactive comparisons are saved as runs.';return;}
  if (["characters", "replicate", "joint"].includes($("mode").value)) {
    const design=state.extra_designs?.[$("mode").value];
    $("designConfig").textContent=JSON.stringify(design?.config || {},null,2);
    $("launchNotice").textContent=$("mode").value==="joint"
      ? "Combine four saved profiles at amplitude multipliers 1/2/4. Multipliers are not calibrated percentages; measured layer changes include propagated effects. Source CHAR-001 must be present."
      : $("mode").value==="characters"
      ? "Frozen character discovery: paired text extraction, measured-strength sparse profiles, ordinary prompting, random and reversed controls. Vocabulary hits are not personality scores."
      : "Frozen fresh-prompt replication: two target channels, two random same-layer controls each, two strengths. Anchors and fresh prompts remain separate.";
    $("estimate").textContent=design?`${design.generation_count} generations plus forward measurements`:"Loading design…";
    return;
  }
  if (["calibration", "offsets"].includes($("mode").value)) {
    const screenMode = $("mode").value === "calibration";
    const available = state.calibrated_designs?.available || [];
    const chosen = available.find((x) => x.id === $("calibrationRun").value);
    const cfg = screenMode ? state.calibrated_designs?.config : chosen?.config;
    $("designConfig").textContent = JSON.stringify(cfg || {}, null, 2);
    $("launchNotice").textContent = screenMode
      ? "Measure activation statistics, calibrate signed offsets, and select units using only calibration forward measurements. No reply generation or persona target."
      : "Use the selected screen's frozen offsets and disjoint evaluation prompts. The runner verifies source hashes and runtime lock before generating. Repeating these prompts is not fresh validation.";
    $("estimate").textContent = screenMode
      ? "96 sampled channels · 384 conditions · 1,536 screen forwards plus calibration/checks"
      : chosen ? `${chosen.generation_count} generations · ${chosen.selected_channels} frozen channels` : "Complete a qualifying calibration first";
    return;
  }
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
  $("calibrationField").classList.toggle("hidden", mode !== "offsets");
  $("designPreview").classList.toggle("hidden", !["mapping", "calibration", "offsets", "characters", "replicate", "joint"].includes(mode));
  estimate();
}
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
$("featurePersona").onchange = renderFeatureMap;
$("featureLayer").onchange = renderFeatureMap;
$("refresh").onclick = () => refresh().catch((e) => message(e.message));
$("refreshLogs").onclick = refreshLog;
for (const id of ["layers", "doses", "tokens", "controls"])
  $(id).oninput = estimate;
$("mode").onchange = configureMode;
$("mappingDesign").onchange = estimate;
$("calibrationRun").onchange = estimate;
$("launchForm").onsubmit = async (event) => {
  event.preventDefault();
  try {
    const body =
      ["characters", "replicate", "joint"].includes($("mode").value)
        ? {mode:$("mode").value}
        :
      $("mode").value === "calibration"
        ? { mode: "calibration" }
        : $("mode").value === "offsets"
        ? { mode: "offsets", calibration: $("calibrationRun").value }
        :
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
    body.model_key=selectedModelKey;
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
initWorkflow();
refresh();
setInterval(async () => {
  if (document.visibilityState !== "visible") return;
  if (state.gpu_busy || Date.now() - (globalThis.lastLabPoll || 0) > 30000) { await refresh(); globalThis.lastLabPoll = Date.now(); }
  if (activeJob && !$("logs").classList.contains("hidden")) await refreshLog();
}, 5000);
