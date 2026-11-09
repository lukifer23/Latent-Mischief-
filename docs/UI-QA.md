# Fresh-start workbench QA — 2026-11-06

**Mac update, 2026-11-09:** 51 Python tests and Node comparison/startup/model-isolation checks pass on M3 Pro, including startup against the real loopback API. Four new MPS/BF16 paired HTTP cases pass; see [BACKEND-001](../experiments/BACKEND-001-mac-setup.md). Missing memory is no longer displayed as zero, catalog rebuild retains public cards, and absent raw-evidence buttons are disabled. No isolated browser tool was available; visual/keyboard/responsive QA below remains incomplete. No native desktop control was used.

Status: clean server/API and script-initialization checks passed. **The requested visual end-to-end UI/UX test is incomplete.** No new workbench screenshot was accepted in this attempt, so this is a QA record rather than a completed screenshot-based design audit.

The native Chrome attempt changed desktop focus and disrupted the user's other work. The user explicitly objected; all desktop/browser automation stopped. Future testing must use an isolated surface that does not touch their existing windows, or remain CLI/API/DOM-fixture-only. This preference is recorded in AGENTS.md. Do not repeat a native desktop attempt simply to complete the screenshot checklist below.

## Requested flow and evidence

| Step | Health | Evidence / limit |
|---|---|---|
| 1. Stop project processes and restart | Passed for server/worker | Project-local viewer restarted; live worker stopped through API, including Windows launcher child tracking; GPU process query empty after tests. Other projects/apps were left alone. |
| 2. Open workbench in a fresh browser window | Blocked | In-app browser host unavailable. Native Chrome fresh window was created, address typed and Enter attempted. Computer Use then stopped because it could not confidently determine the Windows browser URL. No further browser input was issued. |
| 3. Initialize runs and dropdowns | Passed nonvisually | Scripts executed in HTML order against a fresh DOM fixture and the real restarted API. Prompt/group/Edit A/Edit B selectors populated and response cards rendered. Regression fixture puts an empty failed performance run first to verify it is skipped. Every referenced script is fetched by HTTP tests. |
| 4. Switch LFM3 ↔ Qwen | Passed nonvisually | Model selector handler exercised against fixture and live API. History filters, selected run model, LFM-only batch disablement, and absence of LFM performance under Qwen checked. Visual affordances remain unverified. |
| 5. Run default/edited conversations | GPU/API passed; visual blocked | Four saved paired cases: LFM neutral identity, LFM nonzero hook, Qwen base/edit arithmetic and a continued conversation. Source hashes and parameter-version checks passed. See LIVE-001. |
| 6. Library, charts, performance, naming, logs | Partial | API/boundary and DOM-target checks passed; reports/figures inspected separately. No fresh click-through, keyboard traversal, local-name form interaction or responsive screenshot verification. |
| 7. Stop inference and leave lab ready | Passed for backend | Final live worker stopped; viewer retained at port 8765 with GPU idle. Browser window closure/final page state could not be verified after the tool stopped. |

## Changes and risks

The workbench now separates Chat & Try, Saved Comparisons, Persona Library, Effect Atlas, Performance, Run Experiments and Logs. The shared model selector scopes results and recipes. Reply bodies have their own scroll regions; long charts/provenance are moved out of the primary comparison flow. These changes are implemented, **not visually validated improvements**.

The previous blank-menu failure was a missing `/workflow.js` route on an older running server. The route is served, all referenced scripts are tested, disabled loading placeholders replace blank initial menus, startup errors are visible, and startup selects a run with replies. Windows virtualenv worker children previously caused a false busy state; tracking now includes their PIDs. Cross-model recipes and LFM batch launches under Qwen are rejected server-side.

Accessibility/UX gaps: focus styling and labels exist in source, but keyboard order, focus after model changes, screen-reader announcements, contrast, 125%/200% zoom, narrow viewports, tab overflow and chat composer visibility need direct browser testing. Screenshots alone would not prove accessibility compliance. Do not describe this release as fully UX-tested.

## Repeatable checks

```powershell
uv run --locked python -m unittest discover -s tests
node tests/test_ui.js
node tests/test_ui_startup.js
# Start the local viewer first for this integration variant:
node tests/test_ui_startup.js --live
```

When a genuinely isolated browser surface is available, the visual pass should verify the visible model, send a prompt to both branches, continue the conversation, switch models, inspect a saved run, compare two edits, name a recipe, inspect performance/OOM status, and launch/stop a bounded LFM job. Capture and inspect a screenshot at each step, including narrower viewport and keyboard-only navigation. Native desktop control is excluded by the user's instruction; current nonvisual checks do not satisfy this checklist.
