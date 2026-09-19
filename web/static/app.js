/**
 * RAG Eval Suite - Pure Vanilla JavaScript Frontend
 * No dependencies, no build tools, standard ES6+.
 */

function initApp() {
  // --- Element Selectors ---
  const navButtons = document.querySelectorAll(".nav-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const pageTitle = document.getElementById("page-title");
  const pageSubtitle = document.getElementById("page-subtitle");

  // Playground Elements
  const playDocSelect = document.getElementById("play-doc");
  const playLoaderSelect = document.getElementById("play-loader");
  const playSplitterSelect = document.getElementById("play-splitter");
  const playStoreSelect = document.getElementById("play-store");
  const playTopkRange = document.getElementById("play-topk");
  const playTopkVal = document.getElementById("topk-val");
  const playProviderSelect = document.getElementById("play-provider");
  const playModelInput = document.getElementById("play-model");
  const playPromptSelect = document.getElementById("play-prompt");
  const btnTogglePrompts = document.getElementById("btn-toggle-prompts");
  const promptsEditorContainer = document.getElementById("prompts-editor-container");
  const playSystemPrompt = document.getElementById("play-system-prompt");
  const playHybridCheck = document.getElementById("play-hybrid");
  const playRerankCheck = document.getElementById("play-rerank");
  const playQueryText = document.getElementById("play-query");
  const btnRunQuery = document.getElementById("btn-run-query");
  const btnQueryText = document.getElementById("btn-query-text");
  const btnQuerySpinner = document.getElementById("btn-query-spinner");
  const playAnswerBox = document.getElementById("play-answer");
  const playChunksBox = document.getElementById("play-chunks");
  const chunkCountSpan = document.getElementById("chunk-count");
  const playTimingBox = document.getElementById("play-timing");
  const timeTotal = document.getElementById("time-total");
  const timeRetrieval = document.getElementById("time-retrieval");
  const timeGeneration = document.getElementById("time-generation");

  // Benchmark Runs Elements
  const selectRun = document.getElementById("select-run");
  const btnRefreshRuns = document.getElementById("btn-refresh-runs");
  const runMetaStrip = document.getElementById("run-metadata-strip");
  const metricsGrid = document.getElementById("metrics-summary-grid");
  const opsStatsSection = document.getElementById("ops-stats-section");
  const opsStatsGrid = document.getElementById("ops-stats-grid");
  const testCasesBody = document.getElementById("test-cases-body");
  const casesCountSpan = document.getElementById("cases-count");

  // Benchmark Evaluation Launcher Elements
  const evalDocSelect = document.getElementById("eval-doc");
  const evalDatasetSelect = document.getElementById("eval-dataset");
  const evalLoaderSelect = document.getElementById("eval-loader");
  const evalSplitterSelect = document.getElementById("eval-splitter");
  const evalStoreSelect = document.getElementById("eval-store");
  const evalCasesRange = document.getElementById("eval-cases");
  const evalCasesVal = document.getElementById("eval-cases-val");
  const evalHybridCheck = document.getElementById("eval-hybrid");
  const evalRerankCheck = document.getElementById("eval-rerank");
  const evalProviderSelect = document.getElementById("eval-provider");
  const evalModelInput = document.getElementById("eval-model");
  const evalPromptSelect = document.getElementById("eval-prompt");
  const btnToggleEvalPrompts = document.getElementById("btn-toggle-eval-prompts");
  const evalPromptsEditorContainer = document.getElementById("eval-prompts-editor-container");
  const evalSystemPrompt = document.getElementById("eval-system-prompt");
  const evalGeneratorConfigRow = document.getElementById("eval-generator-config-row");
  const btnStartEval = document.getElementById("btn-start-eval");
  const btnEvalText = document.getElementById("btn-eval-text");
  const btnEvalSpinner = document.getElementById("btn-eval-spinner");
  const evalStatusMsg = document.getElementById("eval-status-msg");
  const metricsCheckboxContainer = document.getElementById("metrics-checkbox-container");
  const scopeRadios = document.querySelectorAll('input[name="eval-scope"]');
  const scopeCards = document.querySelectorAll(".scope-card");

  // Compare Elements
  const compareRunA = document.getElementById("compare-run-a");
  const compareRunB = document.getElementById("compare-run-b");
  const btnDoCompare = document.getElementById("btn-do-compare");
  const compareContainer = document.getElementById("compare-results-container");
  const comparisonBody = document.getElementById("comparison-body");

  // Golden Datasets Elements
  const genDocSelect = document.getElementById("gen-doc");
  const genLoaderSelect = document.getElementById("gen-loader");
  const genSplitterSelect = document.getElementById("gen-splitter");
  const genSystemPrompt = document.getElementById("gen-system-prompt");
  const genInstructionText = document.getElementById("gen-instruction");
  const genCountRange = document.getElementById("gen-count");
  const genCountVal = document.getElementById("gen-count-val");
  const genStrategySelect = document.getElementById("gen-strategy");
  const genOutfileInput = document.getElementById("gen-outfile");
  const btnGenerateGoldens = document.getElementById("btn-generate-goldens");
  const btnGenText = document.getElementById("btn-gen-text");
  const btnGenSpinner = document.getElementById("btn-gen-spinner");
  const genStatusMsg = document.getElementById("gen-status-msg");
  const selectDataset = document.getElementById("select-dataset");
  const btnRefreshDatasets = document.getElementById("btn-refresh-datasets");
  const datasetPreview = document.getElementById("dataset-preview");
  const datasetCountSpan = document.getElementById("dataset-items-count");

  // Modal Elements
  const modal = document.getElementById("detail-modal");
  const modalTitle = document.getElementById("modal-title");
  const modalBody = document.getElementById("modal-body");
  const btnCloseModal = document.getElementById("btn-close-modal");

  // In-memory cache
  let currentRunData = null;
  let promptPresets = {
    default: {
      system: "You are a helpful assistant. Answer the question using ONLY the provided context. If the answer cannot be found in the context, say \"I don't have enough information to answer that.\"",
      user: "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    },
    concise: {
      system: "You are a helpful assistant. Provide a concise, direct answer to the question using ONLY the provided context. Keep your response brief and to the point. If the answer cannot be found in the context, say \"I don't have enough information to answer that.\"",
      user: "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    },
    reasoning: {
      system: "You are a helpful assistant. Answer the question using ONLY the provided context. First, break down your reasoning step-by-step based strictly on the facts in the context. Then provide your final answer. If the answer cannot be found in the context, say \"I don't have enough information to answer that.\"",
      user: "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    }
  };
  let defaultSynthesisPrompt = "";

  // --- Tab Titles & Descriptions ---
  const tabInfo = {
    playground: {
      title: "Live RAG Playground",
      subtitle: "Interactively query your document corpus, inspect retrieved chunks, and test generator prompts live."
    },
    benchmarks: {
      title: "Benchmark Results Viewer",
      subtitle: "Inspect quantitative evaluation scores, operational metrics, and per-case breakdowns."
    },
    compare: {
      title: "Side-by-Side Configuration Comparison",
      subtitle: "Compare retrieval, latency, and generation quality differences across two evaluation runs."
    },
    goldens: {
      title: "Golden Datasets & Synthesis",
      subtitle: "Review ground-truth QA datasets or synthesize new golden test cases with custom prompt styling."
    }
  };

  // --- Navigation Handling ---
  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const tabKey = btn.getAttribute("data-tab");
      navButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(`tab-${tabKey}`);
      if (targetPane) targetPane.classList.add("active");

      if (tabInfo[tabKey]) {
        pageTitle.textContent = tabInfo[tabKey].title;
        pageSubtitle.textContent = tabInfo[tabKey].subtitle;
      }
    });
  });

  // Range sliders
  if (playTopkRange && playTopkVal) {
    playTopkRange.addEventListener("input", (e) => {
      playTopkVal.textContent = e.target.value;
    });
  }

  if (genCountRange && genCountVal) {
    genCountRange.addEventListener("input", (e) => {
      genCountVal.textContent = e.target.value;
    });
  }

  if (evalCasesRange && evalCasesVal) {
    evalCasesRange.addEventListener("input", (e) => {
      evalCasesVal.textContent = e.target.value;
    });
  }

  // Auto-disable hybrid toggle if BM25 is selected as primary store
  if (playStoreSelect && playHybridCheck) {
    playStoreSelect.addEventListener("change", () => {
      if (playStoreSelect.value === "bm25") {
        playHybridCheck.checked = false;
        playHybridCheck.disabled = true;
      } else {
        playHybridCheck.disabled = false;
      }
    });
  }

  if (evalStoreSelect && evalHybridCheck) {
    evalStoreSelect.addEventListener("change", () => {
      if (evalStoreSelect.value === "bm25") {
        evalHybridCheck.checked = false;
        evalHybridCheck.disabled = true;
      } else {
        evalHybridCheck.disabled = false;
      }
    });
  }

  // Modal close
  if (btnCloseModal && modal) {
    btnCloseModal.addEventListener("click", () => {
      modal.classList.add("hidden");
    });
  }
  if (modal) {
    window.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.add("hidden");
    });
  }

  // System Prompt Editor Toggle & Preset Sync
  if (btnTogglePrompts && promptsEditorContainer) {
    btnTogglePrompts.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const isHidden = promptsEditorContainer.classList.toggle("hidden");
      btnTogglePrompts.innerHTML = isHidden ? "👁️ View / Edit System Prompt" : "🔼 Collapse System Prompt";
    });
  }

  if (playPromptSelect) {
    playPromptSelect.addEventListener("change", (e) => {
      const selected = e.target.value;
      if (selected !== "custom" && promptPresets[selected]) {
        if (playSystemPrompt) playSystemPrompt.value = promptPresets[selected].system || "";
      }
    });
  }

  const markPlayPromptCustom = () => {
    if (playPromptSelect && playPromptSelect.value !== "custom") {
      playPromptSelect.value = "custom";
    }
  };

  if (playSystemPrompt) {
    playSystemPrompt.addEventListener("input", markPlayPromptCustom);
  }

  // Benchmark System Prompt Editor Toggle & Preset Sync
  if (btnToggleEvalPrompts && evalPromptsEditorContainer) {
    btnToggleEvalPrompts.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const isHidden = evalPromptsEditorContainer.classList.toggle("hidden");
      btnToggleEvalPrompts.innerHTML = isHidden ? "👁️ View / Edit System Prompt" : "🔼 Collapse System Prompt";
    });
  }

  if (evalPromptSelect) {
    evalPromptSelect.addEventListener("change", (e) => {
      const selected = e.target.value;
      if (selected !== "custom" && promptPresets[selected]) {
        if (evalSystemPrompt) evalSystemPrompt.value = promptPresets[selected].system || "";
      }
    });
  }

  const markEvalPromptCustom = () => {
    if (evalPromptSelect && evalPromptSelect.value !== "custom") {
      evalPromptSelect.value = "custom";
    }
  };

  if (evalSystemPrompt) {
    evalSystemPrompt.addEventListener("input", markEvalPromptCustom);
  }

  if (evalProviderSelect && evalModelInput) {
    evalProviderSelect.addEventListener("change", (e) => {
      const selectedProvider = e.target.value;
      if (defaultProviderModels[selectedProvider]) {
        evalModelInput.value = defaultProviderModels[selectedProvider];
      }
    });
  }

  // =========================================================================
  // 1. Initial Data Fetching
  // =========================================================================
  async function loadPromptTemplates() {
    try {
      const res = await fetch("/api/prompts/templates");
      if (!res.ok) return;
      const data = await res.json();
      promptPresets = data.presets || {};
      defaultSynthesisPrompt = data.synthesis_system_prompt || "";

      // Prefill playground prompts if default preset exists
      // Prefill playground and benchmark prompts if default preset exists
      if (promptPresets["default"]) {
        if (playSystemPrompt && !playSystemPrompt.value) {
          playSystemPrompt.value = promptPresets["default"].system || "";
        }
        if (evalSystemPrompt && !evalSystemPrompt.value) {
          evalSystemPrompt.value = promptPresets["default"].system || "";
        }
      }

      // Prefill golden generator system prompt
      if (genSystemPrompt && !genSystemPrompt.value) {
        genSystemPrompt.value = defaultSynthesisPrompt;
      }
    } catch (err) {
      console.error("Failed to load prompt templates:", err);
    }
  }

  async function loadDocuments() {
    try {
      const res = await fetch("/api/documents");
      const docs = await res.json();
      playDocSelect.innerHTML = "";
      genDocSelect.innerHTML = "";
      if (evalDocSelect) evalDocSelect.innerHTML = "";

      docs.forEach(doc => {
        const opt1 = document.createElement("option");
        opt1.value = doc.path;
        opt1.textContent = doc.name;
        playDocSelect.appendChild(opt1);

        const opt2 = document.createElement("option");
        opt2.value = doc.path;
        opt2.textContent = doc.name;
        genDocSelect.appendChild(opt2);

        if (evalDocSelect) {
          const opt3 = document.createElement("option");
          opt3.value = doc.path;
          opt3.textContent = doc.name;
          evalDocSelect.appendChild(opt3);
        }
      });
    } catch (err) {
      console.error("Failed to load documents:", err);
    }
  }

  async function loadRuns(selectedFilename = null) {
    try {
      const res = await fetch("/api/results");
      const runs = await res.json();

      selectRun.innerHTML = "";
      compareRunA.innerHTML = "";
      compareRunB.innerHTML = "";

      if (runs.length === 0) {
        selectRun.innerHTML = "<option value=''>No runs found</option>";
        return;
      }

      runs.forEach((run, idx) => {
        const optText = `${run.filename} (${run.type} - ${run.date})`;
        const opt1 = new Option(optText, run.filename);
        const optA = new Option(optText, run.filename);
        const optB = new Option(optText, run.filename);

        selectRun.appendChild(opt1);
        compareRunA.appendChild(optA);
        compareRunB.appendChild(optB);
      });

      if (runs.length > 1) {
        compareRunB.selectedIndex = 1;
      }

      // Determine target run: exact requested file if found, otherwise the first run
      const targetRun = (selectedFilename && runs.some(r => r.filename === selectedFilename))
        ? selectedFilename
        : runs[0].filename;

      selectRun.value = targetRun;
      loadRunDetail(targetRun);
    } catch (err) {
      console.error("Failed to load runs:", err);
    }
  }

  async function loadDatasets() {
    try {
      const res = await fetch("/api/datasets");
      const datasets = await res.json();
      selectDataset.innerHTML = "";
      if (evalDatasetSelect) evalDatasetSelect.innerHTML = "";

      if (datasets.length === 0) {
        selectDataset.innerHTML = "<option value=''>No datasets found</option>";
        if (evalDatasetSelect) evalDatasetSelect.innerHTML = "<option value=''>No datasets found</option>";
        return;
      }

      datasets.forEach(ds => {
        const opt = new Option(`${ds.filename} (${ds.count} pairs)`, ds.filename);
        selectDataset.appendChild(opt);

        if (evalDatasetSelect) {
          const optEval = new Option(`${ds.filename} (${ds.count} pairs)`, `evals/datasets/${ds.filename}`);
          evalDatasetSelect.appendChild(optEval);
        }
      });

      if (datasets.length > 0) {
        loadDatasetDetail(datasets[0].filename);
      }
    } catch (err) {
      console.error("Failed to load datasets:", err);
    }
  }

  // =========================================================================
  // 2. Playground: Live Query
  // =========================================================================
  const defaultProviderModels = {
    openai: "gpt-4o-mini",
    deepseek: "deepseek-chat",
    ollama: "llama3"
  };

  if (playProviderSelect && playModelInput) {
    playProviderSelect.addEventListener("change", (e) => {
      const selectedProvider = e.target.value;
      if (defaultProviderModels[selectedProvider]) {
        playModelInput.value = defaultProviderModels[selectedProvider];
      }
    });
  }

  btnRunQuery.addEventListener("click", async () => {
    const query = playQueryText.value.trim();
    if (!query) {
      alert("Please enter a question or query.");
      return;
    }

    // Set loading state
    btnRunQuery.disabled = true;
    btnQueryText.textContent = "Executing RAG...";
    btnQuerySpinner.classList.remove("hidden");
    playAnswerBox.className = "answer-content empty";
    playAnswerBox.textContent = "Retrieving context and generating response...";
    playChunksBox.innerHTML = "<div class='placeholder-text'>Fetching chunks...</div>";
    playTimingBox.classList.add("hidden");

    const playEmbeddings = document.getElementById("play-embeddings");
    const playChunkSize = document.getElementById("play-chunk-size");
    const playChunkOverlap = document.getElementById("play-chunk-overlap");

    const payload = {
      query: query,
      doc_path: playDocSelect.value,
      loader_type: playLoaderSelect ? playLoaderSelect.value : "auto",
      splitter_type: playSplitterSelect ? playSplitterSelect.value : "recursive",
      vector_store: playStoreSelect.value,
      embedding_provider: playEmbeddings ? playEmbeddings.value : "openai",
      chunk_size: playChunkSize ? (parseInt(playChunkSize.value, 10) || 500) : 500,
      chunk_overlap: playChunkOverlap ? (parseInt(playChunkOverlap.value, 10) || 100) : 100,
      top_k: parseInt(playTopkRange.value, 10),
      provider: playProviderSelect.value,
      model_name: playModelInput.value.trim() || "gpt-4o-mini",
      prompt_template: playPromptSelect.value,
      system_prompt: playSystemPrompt ? playSystemPrompt.value : undefined,
      use_reranker: playRerankCheck ? playRerankCheck.checked : false,
      use_hybrid: playHybridCheck ? playHybridCheck.checked : false
    };

    try {
      const res = await fetch("/api/playground/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Query execution failed");
      }

      const data = await res.json();

      // Update Answer
      playAnswerBox.className = "answer-content";
      playAnswerBox.textContent = data.answer || "No response generated.";

      // Update Timings
      timeTotal.textContent = `${data.total_latency_ms.toFixed(1)}ms`;
      timeRetrieval.textContent = `${data.retrieval_latency_ms.toFixed(1)}ms`;
      timeGeneration.textContent = `${data.generation_latency_ms.toFixed(1)}ms`;
      playTimingBox.classList.remove("hidden");

      // Update Chunks
      chunkCountSpan.textContent = data.chunks ? data.chunks.length : 0;
      if (data.chunks && data.chunks.length > 0) {
        playChunksBox.innerHTML = "";
        data.chunks.forEach((chunk, i) => {
          const card = document.createElement("div");
          card.className = "chunk-card";
          const pageStr = chunk.page !== null && chunk.page !== undefined ? `Page ${chunk.page + 1}` : "Page N/A";
          card.innerHTML = `
            <div class="chunk-header">
              <span>Chunk #${i + 1} (${chunk.chars} chars)</span>
              <span>${pageStr}</span>
            </div>
            <div class="chunk-body">${escapeHtml(chunk.content)}</div>
          `;
          playChunksBox.appendChild(card);
        });
      } else {
        playChunksBox.innerHTML = "<div class='placeholder-text'>No chunks returned.</div>";
      }

    } catch (err) {
      playAnswerBox.className = "answer-content empty";
      playAnswerBox.textContent = `❌ Error: ${err.message}`;
    } finally {
      btnRunQuery.disabled = false;
      btnQueryText.textContent = "⚡ Execute Pipeline";
      btnQuerySpinner.classList.add("hidden");
    }
  });

  // =========================================================================
  // 3. Benchmark Runs Detail Viewer
  // 3. Benchmark Evaluation Launcher & Metrics Registry
  // =========================================================================
  async function loadMetricsCatalog() {
    try {
      const res = await fetch("/api/evaluate/metrics");
      if (!res.ok) throw new Error("Failed to load metrics catalog");
      const metrics = await res.json();

      if (!metricsCheckboxContainer) return;
      metricsCheckboxContainer.innerHTML = "";

      // Group metrics by category
      const categories = {};
      metrics.forEach(m => {
        if (!categories[m.category]) {
          categories[m.category] = {
            label: m.category_label || m.category,
            items: []
          };
        }
        categories[m.category].items.push(m);
      });

      // Fixed order preference: retriever -> generator -> safety -> ops
      const catOrder = ["retriever", "generator", "safety", "ops"];
      const sortedCatKeys = Object.keys(categories).sort((a, b) => {
        const idxA = catOrder.indexOf(a);
        const idxB = catOrder.indexOf(b);
        return (idxA === -1 ? 99 : idxA) - (idxB === -1 ? 99 : idxB);
      });

      sortedCatKeys.forEach(catKey => {
        const cat = categories[catKey];
        const card = document.createElement("div");
        card.className = "metric-category-card";
        card.setAttribute("data-category", catKey);

        const badgeClass = catKey === "retriever" ? "retriever" : (catKey === "generator" ? "generator" : (catKey === "safety" ? "safety" : "ops"));

        let itemsHtml = "";
        cat.items.forEach(m => {
          itemsHtml += `
            <label class="metric-checkbox-item" data-category="${escapeHtml(m.category)}" data-id="${escapeHtml(m.id)}">
              <input type="checkbox" value="${escapeHtml(m.id)}" ${m.default_checked ? "checked" : ""} />
              <div class="metric-text">
                <strong>${escapeHtml(m.name)}</strong>
                <p>${escapeHtml(m.description)}</p>
              </div>
            </label>
          `;
        });

        card.innerHTML = `
          <span class="category-badge ${badgeClass}">${escapeHtml(cat.label)}</span>
          <div class="category-items-list">
            ${itemsHtml}
          </div>
        `;
        metricsCheckboxContainer.appendChild(card);
      });

      // Track manual unchecks so scope switches don't re-check metrics explicitly turned off by user
      // Track manual unchecks so scope switches don't re-check metrics explicitly turned off or opt-in by default
      metricsCheckboxContainer.querySelectorAll(".metric-checkbox-item").forEach(item => {
        const cb = item.querySelector('input[type="checkbox"]');
        if (cb) {
          if (!cb.checked) {
            item.dataset.manuallyUnchecked = "true";
          }
          cb.addEventListener("change", () => {
            if (!cb.checked) {
              item.dataset.manuallyUnchecked = "true";
            } else {
              delete item.dataset.manuallyUnchecked;
            }
          });
        }
      });

      // Apply initial scope filtering
      applyScopeFilter();
    } catch (err) {
      console.error("Failed to load metrics catalog:", err);
      if (metricsCheckboxContainer) {
        metricsCheckboxContainer.innerHTML = "<div class='placeholder-text'>Failed to load metrics catalog.</div>";
      }
    }
  }

  function applyScopeFilter() {
    const activeRadio = document.querySelector('input[name="eval-scope"]:checked');
    const scope = activeRadio ? activeRadio.value : "all";

    // Update active visual state on scope cards
    scopeCards.forEach(card => {
      const radio = card.querySelector('input[type="radio"]');
      if (radio && radio.checked) {
        card.classList.add("active");
      } else {
        card.classList.remove("active");
      }
    });

    // Update metric checkboxes based on scope:
    // - retriever_only: generator metrics disabled/unchecked, retriever & ops enabled
    // - generator_only: retriever metrics disabled/unchecked, generator & ops enabled
    // - all: all metrics enabled
    // - ops metrics: always enabled regardless of scope
    const items = document.querySelectorAll(".metric-checkbox-item");
    items.forEach(item => {
      const category = item.getAttribute("data-category");
      const checkbox = item.querySelector('input[type="checkbox"]');
      if (!checkbox) return;

      if (scope === "retriever_only") {
        if (category === "generator" || category === "safety") {
          item.classList.add("disabled");
          checkbox.disabled = true;
          checkbox.checked = false;
        } else {
          item.classList.remove("disabled");
          checkbox.disabled = false;
          if (!item.dataset?.manuallyUnchecked) {
            checkbox.checked = true;
          }
        }
      } else if (scope === "generator_only") {
        if (category === "retriever") {
          item.classList.add("disabled");
          checkbox.disabled = true;
          checkbox.checked = false;
        } else {
          item.classList.remove("disabled");
          checkbox.disabled = false;
          if (!item.dataset?.manuallyUnchecked) {
            checkbox.checked = true;
          }
        }
      } else {
        // scope === "all": All enabled
        item.classList.remove("disabled");
        checkbox.disabled = false;
        if (!item.dataset?.manuallyUnchecked) {
          checkbox.checked = true;
        }
      }
    });

    // Toggle benchmark generator & prompt config visibility based on scope
    const isRetrieverOnly = (scope === "retriever_only");
    if (evalGeneratorConfigRow) {
      evalGeneratorConfigRow.style.display = isRetrieverOnly ? "none" : "";
    }
    if (evalPromptsEditorContainer) {
      evalPromptsEditorContainer.style.display = isRetrieverOnly ? "none" : "";
    }
    if (evalProviderSelect && evalProviderSelect.parentElement) {
      evalProviderSelect.parentElement.style.display = isRetrieverOnly ? "none" : "";
    }
  }

  scopeRadios.forEach(radio => {
    radio.addEventListener("change", applyScopeFilter);
  });

  scopeCards.forEach(card => {
    card.addEventListener("click", () => {
      const radio = card.querySelector('input[type="radio"]');
      if (radio && !radio.checked) {
        radio.checked = true;
        applyScopeFilter();
      }
    });
  });

  let isBenchmarkRunning = false;
  if (btnStartEval) {
    btnStartEval.addEventListener("click", async () => {
      if (isBenchmarkRunning) return;

      const docPath = evalDocSelect.value;
      const datasetPath = evalDatasetSelect.value;
      const vectorStore = evalStoreSelect.value;
      const maxCases = parseInt(evalCasesRange.value, 10);

      const activeRadio = document.querySelector('input[name="eval-scope"]:checked');
      const scope = activeRadio ? activeRadio.value : "all";

      // Collect all active checked metric IDs
      const selectedMetrics = [];
      document.querySelectorAll('#metrics-checkbox-container input[type="checkbox"]:checked').forEach(cb => {
        if (!cb.disabled) {
          selectedMetrics.push(cb.value);
        }
      });

      if (selectedMetrics.length === 0) {
        alert("Please select at least one evaluation metric to benchmark.");
        return;
      }

      if (!docPath || !datasetPath) {
        alert("Please select both a document corpus and an evaluation dataset.");
        return;
      }

      isBenchmarkRunning = true;
      btnStartEval.disabled = true;
      btnEvalText.textContent = "Running Benchmark (this may take 20-60s)...";
      btnEvalSpinner.classList.remove("hidden");
      evalStatusMsg.className = "status-banner hidden";

      const evalEmbeddings = document.getElementById("eval-embeddings");
      const evalChunkSize = document.getElementById("eval-chunk-size");
      const evalChunkOverlap = document.getElementById("eval-chunk-overlap");

      const payload = {
        doc_path: docPath,
        dataset_path: datasetPath,
        vector_store: vectorStore,
        loader_type: evalLoaderSelect ? evalLoaderSelect.value : "auto",
        splitter_type: evalSplitterSelect ? evalSplitterSelect.value : "recursive",
        embedding_provider: evalEmbeddings ? evalEmbeddings.value : "openai",
        chunk_size: evalChunkSize ? (parseInt(evalChunkSize.value, 10) || 500) : 500,
        chunk_overlap: evalChunkOverlap ? (parseInt(evalChunkOverlap.value, 10) || 100) : 100,
        top_k: 3,
        use_reranker: evalRerankCheck ? evalRerankCheck.checked : false,
        use_hybrid: evalHybridCheck ? evalHybridCheck.checked : false,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider: evalProviderSelect ? evalProviderSelect.value : "openai",
        model_name: evalModelInput && evalModelInput.value.trim() ? evalModelInput.value.trim() : "gpt-4o-mini",
        temperature: 0.0,
        prompt_template: "default",
        prompt_template: evalPromptSelect ? evalPromptSelect.value : "default",
        system_prompt: evalSystemPrompt && evalSystemPrompt.value.trim() ? evalSystemPrompt.value.trim() : undefined,
        eval_model: "gpt-4o-mini",
        max_cases: maxCases,
        scope: scope,
        selected_metrics: selectedMetrics
      };

      try {
        const res = await fetch("/api/evaluate/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Benchmark execution failed");

        evalStatusMsg.className = "status-banner success";
        evalStatusMsg.textContent = `🎉 ${data.message || "Benchmark evaluation completed successfully!"}`;
        evalStatusMsg.classList.remove("hidden");

        // Reload runs list and select the newly generated benchmark run
        await loadRuns(data.filename);
      } catch (err) {
        evalStatusMsg.className = "status-banner error";
        evalStatusMsg.textContent = `❌ ${err.message}`;
        evalStatusMsg.classList.remove("hidden");
      } finally {
        isBenchmarkRunning = false;
        btnStartEval.disabled = false;
        btnEvalText.textContent = "▶️ Run Selected Benchmark";
        btnEvalSpinner.classList.add("hidden");
      }
    });
  }

  // =========================================================================
  // 4. Benchmark Runs Detail Viewer
  // =========================================================================
  selectRun.addEventListener("change", (e) => {
    if (e.target.value) loadRunDetail(e.target.value);
  });

  btnRefreshRuns.addEventListener("click", () => {
    loadRuns();
  });

  async function loadRunDetail(filename) {
    try {
      const res = await fetch(`/api/results/${encodeURIComponent(filename)}`);
      if (!res.ok) throw new Error("Failed to load run data");
      const data = await res.json();
      currentRunData = data;

      renderRunMetadata(data);
      renderMetricsSummary(data);
      renderTestCasesTable(data);
    } catch (err) {
      console.error("Error loading run detail:", err);
    }
  }

  function renderRunMetadata(data) {
    runMetaStrip.innerHTML = "";
    runMetaStrip.classList.remove("hidden");

    const hp = data.hyperparameters || {};
    const costVal = (data.evaluationCost !== undefined && data.evaluationCost !== null)
      ? data.evaluationCost
      : (data.cost_metrics?.total_cost_usd !== undefined ? data.cost_metrics.total_cost_usd : 0);

    const durVal = (data.runDuration !== undefined && data.runDuration !== null)
      ? data.runDuration.toFixed(2) + "s"
      : (data.latency_metrics?.total_latency_ms?.mean
          ? ((data.latency_metrics.total_latency_ms.mean * (data.testCases?.length || data.per_query_results?.length || 1)) / 1000).toFixed(2) + "s"
          : "N/A");

    const isHybrid = hp.use_hybrid === true || hp.use_hybrid === "true" || hp.use_hybrid === "True";
    const isRerank = hp.use_reranker === true || hp.use_reranker === "true" || hp.use_reranker === "True";

    const pills = [
      `Type: <strong>${hp.eval_type || (data.timestamp ? "ops_eval" : "N/A")}</strong>`,
      `Doc: <strong>${hp.document || "N/A"}</strong>`,
      `Dataset: <strong>${hp.dataset || "N/A"}</strong>`,
      `Store: <strong>${hp.vector_store || hp.vector_store_type || "N/A"}</strong>`,
      `Loader: <strong>${hp.loader_type || "auto"}</strong>`,
      `Splitter: <strong>${hp.splitter_type || "recursive"}</strong>`,
      `Hybrid: <strong>${isHybrid ? "Enabled (BM25+Dense)" : "Disabled"}</strong>`,
      `Model: <strong>${hp.model_name || "N/A"}</strong>`,
      `Reranker: <strong>${isRerank ? "Enabled" : "Disabled"}</strong>`,
      `Top-K: <strong>${hp.top_k || "N/A"}</strong>`,
      `Duration: <strong>${durVal}</strong>`,
      `Cost: <strong>$${typeof costVal === "number" ? costVal.toFixed(5) : "0.00"}</strong>`
    ];

    pills.forEach(p => {
      const span = document.createElement("div");
      span.className = "meta-pill";
      span.innerHTML = p;
      runMetaStrip.appendChild(span);
    });
  }

  function renderMetricsSummary(data) {
    metricsGrid.innerHTML = "";
    opsStatsSection.classList.add("hidden");

    // Standard DeepEval metricsScores
    if (data.metricsScores && data.metricsScores.length > 0) {
      data.metricsScores.forEach(m => {
        const avg = m.scores && m.scores.length > 0
          ? (m.scores.reduce((a, b) => a + b, 0) / m.scores.length).toFixed(2)
          : "N/A";

        const passRate = (m.passes + m.fails) > 0
          ? ((m.passes / (m.passes + m.fails)) * 100).toFixed(0) + "%"
          : "N/A";

        const card = document.createElement("div");
        card.className = "metric-card";
        const isPassed = m.fails === 0;
        card.innerHTML = `
          <div class="metric-title">${m.metric}</div>
          <div class="metric-value" style="color: ${avg >= 0.7 ? '#3fb950' : '#f85149'}">${avg}</div>
          <span class="metric-badge ${isPassed ? 'badge-pass' : 'badge-fail'}">
            ${isPassed ? 'PASS' : 'FAIL'} (${passRate} pass rate)
          </span>
        `;
        metricsGrid.appendChild(card);
      });
    } else if (data.latency_metrics) {
      // If no DeepEval quality metrics (pure Ops run), show Ops summary in the top cards grid
      const lat = data.latency_metrics;
      const tok = data.token_metrics || {};
      const cost = data.cost_metrics || {};
      const topOpsCards = [
        { title: "Average Latency", val: `${lat.total_latency_ms?.mean?.toFixed(1) || 0}ms`, badge: "Mean E2E", color: "#58a6ff" },
        { title: "P95 Tail Latency", val: `${lat.total_latency_ms?.p95?.toFixed(1) || 0}ms`, badge: "Tail Latency", color: "#d29922" },
        { title: "Generation Throughput", val: `${tok.avg_throughput_tokens_per_sec?.toFixed(1) || 0} t/s`, badge: "Tokens/sec", color: "#3fb950" },
        { title: "Total Cost", val: `$${cost.total_cost_usd ? cost.total_cost_usd.toFixed(5) : "0.00"}`, badge: "API Cost", color: "#a371f7" }
      ];
      topOpsCards.forEach(c => {
        const card = document.createElement("div");
        card.className = "metric-card";
        card.innerHTML = `
          <div class="metric-title">${c.title}</div>
          <div class="metric-value" style="color: ${c.color}">${c.val}</div>
          <span class="metric-badge badge-info">${c.badge}</span>
        `;
        metricsGrid.appendChild(card);
      });
    }

    // Operational Latency & Ops Run Data
    if (data.latency_metrics) {
      opsStatsSection.classList.remove("hidden");
      opsStatsGrid.innerHTML = "";

      const lat = data.latency_metrics;
      const tok = data.token_metrics || {};
      const cost = data.cost_metrics || {};

      const opsCards = [
        { title: "P50 Latency", val: `${lat.total_latency_ms?.p50?.toFixed(1) || 0}ms`, badge: "Median" },
        { title: "P95 Latency", val: `${lat.total_latency_ms?.p95?.toFixed(1) || 0}ms`, badge: "Tail Latency" },
        { title: "Throughput", val: `${tok.avg_throughput_tokens_per_sec?.toFixed(1) || 0} t/s`, badge: "Tokens/sec" },
        { title: "Cost per 1k", val: `$${cost.projected_cost_per_1k_queries_usd?.toFixed(4) || 0}`, badge: "Projected" }
      ];

      opsCards.forEach(c => {
        const card = document.createElement("div");
        card.className = "metric-card";
        card.innerHTML = `
          <div class="metric-title">${c.title}</div>
          <div class="metric-value">${c.val}</div>
          <span class="metric-badge badge-info">${c.badge}</span>
        `;
        opsStatsGrid.appendChild(card);
      });
    }
  }

  function renderTestCasesTable(data) {
    testCasesBody.innerHTML = "";
    const cases = data.testCases || data.per_query_results || data.per_query_ops || [];
    casesCountSpan.textContent = cases.length;

    if (cases.length === 0) {
      testCasesBody.innerHTML = "<tr><td colspan='5' class='placeholder-text'>No test cases recorded in this run.</td></tr>";
      return;
    }

    cases.forEach((tc, idx) => {
      const tr = document.createElement("tr");
      const inputQuery = tc.input || tc.query || "N/A";
      let isSuccess = tc.success;
      if (isSuccess === undefined) {
        isSuccess = tc.flaky !== undefined ? !tc.flaky : true;
      }

      let scoresHtml = "";
      if (tc.metricsData && tc.metricsData.length > 0) {
        scoresHtml = tc.metricsData.map(m => {
          const color = m.score >= (m.threshold || 0.7) ? "#3fb950" : "#f85149";
          return `<span style="font-size: 11px; margin-right: 6px; color: ${color}; font-weight: 600;">${m.name}: ${m.score.toFixed(2)}</span>`;
        }).join("");
      } else if (tc.total_ms) {
        scoresHtml = `<span style="font-size: 11px; color: #58a6ff;">Total: ${tc.total_ms.toFixed(1)}ms | Ret: ${tc.retrieval_ms?.toFixed(1)}ms</span>`;
      }

      // Check for attached per-query ops info
      const opsInfo = tc.total_ms ? tc : (data.per_query_ops && data.per_query_ops[idx]);
      if (opsInfo && tc.metricsData && tc.metricsData.length > 0) {
        const opsBadge = `<span style="font-size: 11px; color: #58a6ff; margin-left: 4px;">(${opsInfo.total_ms.toFixed(0)}ms | $${opsInfo.cost_usd ? opsInfo.cost_usd.toFixed(5) : "0.00"})</span>`;
        scoresHtml = `${scoresHtml} ${opsBadge}`;
      } else if (opsInfo && !scoresHtml) {
        scoresHtml = `<span style="font-size: 11px; color: #58a6ff;">Total: ${opsInfo.total_ms.toFixed(1)}ms | Ret: ${opsInfo.retrieval_ms?.toFixed(1)}ms | Gen: ${opsInfo.generation_ms?.toFixed(1)}ms</span>`;
      }

      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td style="max-width: 400px; font-weight: 500;">${escapeHtml(inputQuery.slice(0, 120))}${inputQuery.length > 120 ? '...' : ''}</td>
        <td>
          <span class="metric-badge ${isSuccess ? 'badge-pass' : 'badge-fail'}">
            ${isSuccess ? 'PASSED' : 'FAILED'}
          </span>
        </td>
        <td>${scoresHtml || "N/A"}</td>
        <td>
          <button class="btn btn-secondary btn-sm btn-view-case" data-index="${idx}" style="padding: 4px 8px; font-size: 11px;">View</button>
        </td>
      `;
      testCasesBody.appendChild(tr);
    });

    // Attach View click handler
    document.querySelectorAll(".btn-view-case").forEach(btn => {
      btn.addEventListener("click", () => {
        const index = parseInt(btn.getAttribute("data-index"), 10);
        openCaseModal(cases[index]);
      });
    });
  }

  function openCaseModal(tc) {
    modalTitle.textContent = `Test Case Detail: ${tc.name || "Query #" + (tc.query_index || 1)}`;
    const query = tc.input || tc.query || "N/A";
    const actual = tc.actualOutput || "N/A";
    const expected = tc.expectedOutput || "N/A";

    let contextHtml = "";
    const ctxList = tc.retrievalContext || tc.retrieval_context || tc.context || [];
    if (ctxList.length > 0) {
      contextHtml = ctxList.map((c, i) => `
        <div class="chunk-card" style="margin-bottom: 8px;">
          <div class="chunk-header"><span>Context #${i + 1}</span></div>
          <div class="chunk-body">${escapeHtml(typeof c === 'string' ? c : JSON.stringify(c))}</div>
        </div>
      `).join("");
    } else {
      contextHtml = "<div class='placeholder-text'>No context chunks available.</div>";
    }

    let metricsHtml = "";
    if (tc.metricsData) {
      metricsHtml = `
        <h4 style="margin: 16px 0 8px; font-size: 12px; color: var(--text-muted); text-transform: uppercase;">Metrics Breakdown</h4>
        <div class="cards-grid" style="grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); margin-bottom: 16px;">
          ${tc.metricsData.map(m => `
            <div class="metric-card" style="padding: 10px;">
              <div class="metric-title">${m.name}</div>
              <div class="metric-value" style="font-size: 20px; color: ${m.score >= m.threshold ? '#3fb950' : '#f85149'}">${m.score.toFixed(2)}</div>
              <span class="metric-badge ${m.success ? 'badge-pass' : 'badge-fail'}">${m.success ? 'PASS' : 'FAIL'}</span>
            </div>
          `).join("")}
        </div>
      `;
    }

    modalBody.innerHTML = `
      <div style="margin-bottom: 14px;">
        <strong style="color: var(--text-muted); font-size: 12px; text-transform: uppercase;">User Input:</strong>
        <div style="margin-top: 4px; font-size: 14px; font-weight: 500;">${escapeHtml(query)}</div>
      </div>

      <div style="margin-bottom: 14px;">
        <strong style="color: var(--text-muted); font-size: 12px; text-transform: uppercase;">Generated Output:</strong>
        <div class="result-box" style="margin-top: 4px; padding: 12px;">${escapeHtml(actual)}</div>
      </div>

      ${expected !== "N/A" ? `
      <div style="margin-bottom: 14px;">
        <strong style="color: var(--text-muted); font-size: 12px; text-transform: uppercase;">Expected Ground Truth Output:</strong>
        <div style="margin-top: 4px; font-size: 13px; color: var(--text-muted);">${escapeHtml(expected)}</div>
      </div>` : ''}

      ${metricsHtml}

      <div>
        <strong style="color: var(--text-muted); font-size: 12px; text-transform: uppercase;">Retrieved Chunks:</strong>
        <div style="margin-top: 8px;">${contextHtml}</div>
      </div>
    `;

    modal.classList.remove("hidden");
  }

  // =========================================================================
  // 4. Compare Runs
  // =========================================================================
  btnDoCompare.addEventListener("click", async () => {
    const fileA = compareRunA.value;
    const fileB = compareRunB.value;

    if (!fileA || !fileB) {
      alert("Please select both runs to compare.");
      return;
    }
    if (fileA === fileB) {
      alert("Please select two different runs for comparison.");
      return;
    }

    btnDoCompare.disabled = true;
    btnDoCompare.textContent = "Comparing...";

    try {
      const [resA, resB] = await Promise.all([
        fetch(`/api/results/${encodeURIComponent(fileA)}`).then(r => r.json()),
        fetch(`/api/results/${encodeURIComponent(fileB)}`).then(r => r.json())
      ]);

      renderComparison(resA, resB, fileA, fileB);
    } catch (err) {
      alert("Failed to compare runs: " + err.message);
    } finally {
      btnDoCompare.disabled = false;
      btnDoCompare.textContent = "⚖️ Compare";
    }
  });

  function renderComparison(runA, runB, nameA, nameB) {
    compareContainer.classList.remove("hidden");
    comparisonBody.innerHTML = "";

    const hpA = runA.hyperparameters || {};
    const hpB = runB.hyperparameters || {};

    // Hyperparameters rows
    const hpKeys = [
      "eval_type",
      "vector_store",
      "loader_type",
      "splitter_type",
      "use_hybrid",
      "model_name",
      "use_reranker",
      "top_k",
      "chunk_size",
      "chunk_overlap",
      "prompt_template"
    ];
    hpKeys.forEach(k => {
      const vA = hpA[k] !== undefined ? String(hpA[k]) : "N/A";
      const vB = hpB[k] !== undefined ? String(hpB[k]) : "N/A";
      const diff = vA === vB ? "<span style='color: var(--text-dim)'>Identical</span>" : "<strong style='color: #58a6ff'>Changed</strong>";

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>Config: ${k}</strong></td>
        <td>${vA}</td>
        <td>${vB}</td>
        <td>${diff}</td>
      `;
      comparisonBody.appendChild(tr);
    });

    // Metrics Comparison
    const getMetricMap = (run) => {
      const map = {};
      if (run.metricsScores) {
        run.metricsScores.forEach(m => {
          const avg = m.scores && m.scores.length > 0 ? (m.scores.reduce((a, b) => a + b, 0) / m.scores.length) : 0;
          map[m.metric] = avg;
        });
      }
      return map;
    };

    const mapA = getMetricMap(runA);
    const mapB = getMetricMap(runB);
    const allMetrics = Array.from(new Set([...Object.keys(mapA), ...Object.keys(mapB)]));

    allMetrics.forEach(metric => {
      const scoreA = mapA[metric] !== undefined ? mapA[metric] : null;
      const scoreB = mapB[metric] !== undefined ? mapB[metric] : null;

      let deltaStr = "N/A";
      if (scoreA !== null && scoreB !== null) {
        const delta = scoreB - scoreA;
        const color = delta > 0 ? "#3fb950" : (delta < 0 ? "#f85149" : "var(--text-dim)");
        const sign = delta > 0 ? "+" : "";
        deltaStr = `<span style="color: ${color}; font-weight: 700;">${sign}${(delta * 100).toFixed(1)}%</span>`;
      }

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>Metric: ${metric}</strong></td>
        <td>${scoreA !== null ? scoreA.toFixed(2) : "N/A"}</td>
        <td>${scoreB !== null ? scoreB.toFixed(2) : "N/A"}</td>
        <td>${deltaStr}</td>
      `;
      comparisonBody.appendChild(tr);
    });

    // Duration / Cost comparison
    const durA = runA.runDuration ? runA.runDuration.toFixed(2) + "s" : (runA.latency_metrics?.total_latency_ms?.mean ? (runA.latency_metrics.total_latency_ms.mean / 1000).toFixed(2) + "s (avg)" : "N/A");
    const durB = runB.runDuration ? runB.runDuration.toFixed(2) + "s" : (runB.latency_metrics?.total_latency_ms?.mean ? (runB.latency_metrics.total_latency_ms.mean / 1000).toFixed(2) + "s (avg)" : "N/A");
    const trDur = document.createElement("tr");
    trDur.innerHTML = `
      <td><strong>Total Run Duration</strong></td>
      <td>${durA}</td>
      <td>${durB}</td>
      <td>-</td>
    `;
    comparisonBody.appendChild(trDur);

    const costA = (runA.evaluationCost !== null && runA.evaluationCost !== undefined)
      ? runA.evaluationCost
      : runA.cost_metrics?.total_cost_usd;
    const costB = (runB.evaluationCost !== null && runB.evaluationCost !== undefined)
      ? runB.evaluationCost
      : runB.cost_metrics?.total_cost_usd;
    if ((costA !== null && costA !== undefined) || (costB !== null && costB !== undefined)) {
      const costAStr = typeof costA === "number" ? `$${costA.toFixed(5)}` : "N/A";
      const costBStr = typeof costB === "number" ? `$${costB.toFixed(5)}` : "N/A";
      let costDelta = "-";
      if (typeof costA === "number" && typeof costB === "number" && costA > 0) {
        const pct = ((costB - costA) / costA) * 100;
        const color = pct <= 0 ? "#3fb950" : "#f85149";
        costDelta = `<span style="color: ${color}; font-weight: 700;">${pct > 0 ? "+" : ""}${pct.toFixed(1)}%</span>`;
      }
      const trCost = document.createElement("tr");
      trCost.innerHTML = `
        <td><strong>Total Cost (USD)</strong></td>
        <td>${costAStr}</td>
        <td>${costBStr}</td>
        <td>${costDelta}</td>
      `;
      comparisonBody.appendChild(trCost);
    }
  }

  // =========================================================================
  // 5. Golden Datasets & Generator
  // =========================================================================
  selectDataset.addEventListener("change", (e) => {
    if (e.target.value) loadDatasetDetail(e.target.value);
  });

  btnRefreshDatasets.addEventListener("click", () => {
    loadDatasets();
  });

  async function loadDatasetDetail(filename) {
    try {
      const res = await fetch(`/api/datasets/${encodeURIComponent(filename)}`);
      const items = await res.json();
      datasetCountSpan.textContent = items.length;
      datasetPreview.innerHTML = "";

      if (items.length === 0) {
        datasetPreview.innerHTML = "<div class='placeholder-text'>Dataset is empty.</div>";
        return;
      }

      items.forEach((item, idx) => {
        const card = document.createElement("div");
        card.className = "qa-card";
        const metaStr = item.metadata?.user_instruction ? `Instruction: ${item.metadata.user_instruction}` : (item.source_file || "Golden QA");
        card.innerHTML = `
          <div class="qa-q">Q${idx + 1}: ${escapeHtml(item.input)}</div>
          <div class="qa-a"><strong>Answer:</strong> ${escapeHtml(item.expected_output || "N/A")}</div>
          <div class="qa-meta"><span>📁 ${escapeHtml(metaStr)}</span></div>
        `;
        datasetPreview.appendChild(card);
      });
    } catch (err) {
      console.error("Failed to load dataset detail:", err);
    }
  }

  // Generate Goldens Handler
  btnGenerateGoldens.addEventListener("click", async () => {
    const docPath = genDocSelect.value;
    const instruction = genInstructionText.value.trim();
    const count = parseInt(genCountRange.value, 10);
    const strategy = genStrategySelect.value;
    const outfile = genOutfileInput.value.trim() || "custom_goldens.json";

    btnGenerateGoldens.disabled = true;
    btnGenText.textContent = "Synthesizing Dataset...";
    btnGenSpinner.classList.remove("hidden");
    genStatusMsg.className = "status-banner hidden";

    const payload = {
      doc_path: docPath,
      system_prompt: genSystemPrompt ? genSystemPrompt.value : undefined,
      instruction: instruction || "Generate clear, representative questions and answers.",
      loader_type: genLoaderSelect ? genLoaderSelect.value : "auto",
      splitter_type: genSplitterSelect ? genSplitterSelect.value : "recursive",
      max_goldens: count,
      sample_strategy: strategy,
      output_file: outfile
    };

    try {
      const res = await fetch("/api/goldens/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Synthesis failed");

      genStatusMsg.className = "status-banner success";
      genStatusMsg.textContent = `🎉 Generated ${data.count} golden QA pairs successfully to ${data.file}!`;
      genStatusMsg.classList.remove("hidden");

      // Refresh datasets list
      await loadDatasets();
    } catch (err) {
      genStatusMsg.className = "status-banner error";
      genStatusMsg.textContent = `❌ ${err.message}`;
      genStatusMsg.classList.remove("hidden");
    } finally {
      btnGenerateGoldens.disabled = false;
      btnGenText.textContent = "🚀 Synthesize Dataset";
      btnGenSpinner.classList.add("hidden");
    }
  });

  // Utility to escape HTML
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initial Load
  loadDocuments();
  loadRuns();
  loadDatasets();
  loadMetricsCatalog();
  loadPromptTemplates();
  console.log("RAG Eval Suite frontend initialized successfully.");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}

