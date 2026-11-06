"use strict";
const $ = (id) => document.getElementById(id);
let state = {},
  current = null,
  selected = "",
  token = "",
  activeJob = "";
const drafts = {};
let comparisonPins = [];
function conditionLabel(x) {
  if (x.profile_id) return x.profile_id;
  if (x.condition === "channel-offset") return `${x.candidate} ${x.sign > 0 ? "+" : "−"} offset`;
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
            : run.kind === "characters"
              ? "Sparse character lab"
            : run.kind === "calibration"
              ? "Activation calibration"
            : run.kind === "personas"
              ? "Persona discovery"
              : "Baseline check",
        ),
        el("span", "", run.id.split("-LFM")[0]),
        el("span", "", `${run.kind === "calibration" ? `${run.screen_conditions} screen conditions` : `${run.outputs} replies`} · ${run.status}`),
      );
      b.onclick = () => loadRun(run.id);
      $("runs").append(b);
    }
    renderJobs();
    if (!selected && state.runs.length) await loadRun(state.runs[0].id);
    else if (
      current &&
      (state.runs.find((x) => x.id === selected)?.outputs !==
        current.outputs.length || (current.calibration && state.runs.find((x) => x.id === selected)?.screen_conditions !== current.calibration.screening.length))
    )
      await loadRun(selected, false);
  } catch (e) {
    message(e.message);
  }
}
async function loadRun(id, reset = true) {
  if (selected !== id) comparisonPins = [];
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
    .forEach((n, i) => n.classList.toggle("active", state.runs[i]?.id === id));
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
      ["contrast-direction", "channel-gain", "channel-offset", "sparse-profile"].includes(x.condition),
  );
  $("heatmap").replaceChildren();
  if (!out.length) {
    $("heatmap").append(el("p", "muted", "No intervention sweep in this run."));
    return;
  }
  const measure = $("mapMetric").value;
  const gainMode = out.some((x) => x.condition === "channel-gain");
  const offsetMode = out.some((x) => x.condition === "channel-offset");
  const profileMode = out.some((x) => x.condition === "sparse-profile");
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
      doses = [...new Set(data.map((x) => (profileMode ? x.target_fraction : offsetMode ? x.sign : gainMode ? x.gain : x.dose)))].sort(
        (a, b) => a - b,
      ),
      table = el("table", "heat-table"),
      head = el("tr");
    head.append(el("th", "", persona));
    doses.forEach((d) =>
      head.append(el("th", "", `${profileMode ? "local target" : offsetMode ? "offset sign" : gainMode ? "gain" : "dose"} ${d}`)),
    );
    table.append(head);
    for (const layer of layers) {
      const row = el("tr");
      row.append(el("th", "", channelMode ? layer : `Layer ${layer}`));
      for (const dose of doses) {
        const items = data.filter(
          (x) =>
            (channelMode ? x.candidate === layer : x.layer === layer) &&
            (profileMode ? x.target_fraction : offsetMode ? x.sign : gainMode ? x.gain : x.dose) === dose,
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
function renderTray() {
  $("comparisonTray").replaceChildren();
  const prompt=$("promptFilter").value;
  const reference=current.outputs.find(x=>(x.prompt_id||x.id)===prompt && x.condition==="baseline");
  const pins=comparisonPins.filter(i=>(current.outputs[i]?.prompt_id||current.outputs[i]?.id)===prompt);
  if (!reference || !pins.length) {
    $("comparisonTray").append(el("p","muted","Use Pin comparison on replies below to compare them here.")); return;
  }
  for(const output of [reference,...pins.map(i=>current.outputs[i])]) {
    const card=el("article","compare-card");
    card.append(el("h4","",conditionLabel(output)),el("p","muted",`${output.output_tokens} tokens · ${output.terminated_with_eos?"EOS":"No EOS"} · Δ ${output.output_tokens-reference.output_tokens} tokens`),el("pre","",output.text));
    $("comparisonTray").append(card);
  }
}
function renderOutputs() {
  if (!current) return;
  if (current.calibration) {
    $("baselinePrompt").textContent = "Calibration uses forward measurements, without reply generation.";
    $("baselineText").textContent = "Open the associated mapping run to compare generated replies. Frozen selection is shown in the table above.";
    $("outputs").replaceChildren();
    return;
  }
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
          x.condition === "random-direction" || x.role === "sparse-random") &&
        ($("conditionFilter").value === "all" ||
          x.condition === $("conditionFilter").value) &&
        ($("layerFilter").value === "all" ||
          String(x.layer) === $("layerFilter").value) &&
        ($("doseFilter").value === "all" ||
          String(x.target_fraction ?? x.dose) === $("doseFilter").value),
    );
  for (const output of filtered) {
    const card = el("article", "panel"),
      top = el("div", "output-title");
    top.append(
      el(
        "h3",
        "",
        conditionLabel(output),
      ),
    );
    card.append(top);
    const pin=el("button","secondary",comparisonPins.includes(output._index)?"Unpin comparison":"Pin comparison");
    pin.onclick=()=>{
      const samePrompt=comparisonPins.filter(i=>(current.outputs[i]?.prompt_id||current.outputs[i]?.id)===prompt);
      if(comparisonPins.includes(output._index)) comparisonPins=comparisonPins.filter(i=>i!==output._index);
      else if(samePrompt.length>=3) { message("Unpin a reply before adding a fourth comparison for this prompt."); return; }
      else comparisonPins.push(output._index);
      renderOutputs();
    };
    top.append(pin);
    const tags = el("div", "tags");
    for (const text of [
      output.layer !== undefined ? `Layer ${output.layer}` : null,
      output.neuron !== undefined ? `Channel ${output.neuron}` : null,
      output.gain !== undefined ? `Gain ${output.gain}` : null,
      output.target_fraction !== undefined ? `Local target ${(100*output.target_fraction).toFixed(0)}%` : null,
      output.hit_cap ? "Calibration cap reached" : null,
      output.offset !== undefined ? `Offset ${output.offset.toFixed(4)} (${output.offset_in_rms_units.toFixed(2)} calibration RMS)` : null,
      output.actual_prefill_fraction !== undefined ? `Measured prefill change ${(100 * output.actual_prefill_fraction).toFixed(1)}%` : null,
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
      typeof output.measurements?.exact_text_match === "boolean"
        ? output.measurements.exact_text_match ? "Exact text matched" : "Exact text failed"
        : null,
      output.dose !== undefined ? `Dose ${output.dose}` : null,
      `${output.output_tokens} tokens`,
      `${output.seconds.toFixed(2)}s`,
      `${(repetitionRatio(output) * 100).toFixed(0)}% repeated 4-grams`,
      output.terminated_with_eos ? "EOS" : "Token cap / no EOS",
    ].filter(Boolean))
      tags.append(el("span", "tag", text));
    card.append(tags, el("pre", "", output.text));
    if (original) {
      const details=el("details","diff-panel"), text=el("pre","");
      const comparison=LabCompare.tokenComparison(original.token_ids,output.token_ids);
      details.append(el("summary","",comparison?.identical?"Token-identical to baseline":`Word diff · shared token prefix ${comparison?.commonPrefix ?? "unknown"}`),el("p","muted","Green: added text. Strikethrough: removed text. Word diff is separate from tokenizer comparison."));
      for(const part of LabCompare.wordDiff(original.text,output.text)) text.append(el("span","diff-"+part.kind,part.text));
      details.append(text); card.append(details);
    }
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
  renderTray();
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
  if (["characters", "replicate"].includes($("mode").value)) {
    const design=state.extra_designs?.[$("mode").value];
    $("designConfig").textContent=JSON.stringify(design?.config || {},null,2);
    $("launchNotice").textContent=$("mode").value==="characters"
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
  $("designPreview").classList.toggle("hidden", !["mapping", "calibration", "offsets", "characters", "replicate"].includes(mode));
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
      ["characters", "replicate"].includes($("mode").value)
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
