(() => {
  const screens = {
    start: document.getElementById("screen-start"),
    loading: document.getElementById("screen-loading"),
    team: document.getElementById("screen-team"),
    final: document.getElementById("screen-final"),
  };

  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  const N_GAMES = 82;
  // Perfect run: five JACKPOTs with unbroken streak multipliers
  // 1000 + 1250 + 1500 + 1750 + 2000
  const BOARD_CEILING = [1000, 1250, 1500, 1750, 2000];
  const MAX_SCORE = BOARD_CEILING.reduce((a, b) => a + b, 0);
  const DAILY_STORAGE_KEY = "guessthewins.daily.v1";

  const LOADING_MESSAGES = [
    "Opening the window…",
    "Running 300 simulated seasons…",
    "The house is calculating…",
  ];

  const VERDICT_EMOJI = {
    JACKPOT: "🟩",
    "SO CLOSE": "🟩",
    SHARP: "🟨",
    "NOT BAD": "🟨",
    COLD: "🟧",
    AIRBALL: "🟥",
  };

  const POS_CYCLE = ["G", "G", "F", "F", "C"]; // fallback only if API omits position

  // ---------- DOM refs ----------
  const hud = document.getElementById("hud");
  const hudScoreEl = document.getElementById("hud-score");
  const hudPipsEl = document.getElementById("hud-pips");
  const hudStreakEl = document.getElementById("hud-streak");

  const startErrorEl = document.getElementById("start-error");
  const btnDeal = document.getElementById("btn-deal");
  const startFooterEl = document.getElementById("start-footer");
  const dailyPanelEl = document.getElementById("daily-panel");
  const freePanelEl = document.getElementById("free-panel");
  const dailyDateLabelEl = document.getElementById("daily-date-label");
  const dailyDoneEl = document.getElementById("daily-done");
  const dailyDoneScoreEl = document.getElementById("daily-done-score");
  const dailyDoneNoteEl = document.getElementById("daily-done-note");
  const btnShareDaily = document.getElementById("btn-share-daily");

  const loadingTextEl = document.getElementById("loading-text");

  const teamCardEl = document.getElementById("team-card");
  const boardNumEl = document.getElementById("board-num");
  const teamHeadingEl = document.getElementById("team-heading");
  const teamDiffChipEl = document.getElementById("team-diff-chip");
  const playerListEl = document.getElementById("player-list");

  const betAreaEl = document.getElementById("bet-area");
  const betValueEl = document.getElementById("bet-value");
  const betSubEl = document.getElementById("bet-sub");
  const betSliderEl = document.getElementById("bet-slider");
  const btnLock = document.getElementById("btn-lock");
  const teamErrorEl = document.getElementById("team-error");

  const revealAreaEl = document.getElementById("reveal-area");
  const revealActualValueEl = document.getElementById("reveal-actual-value");
  const revealActualSubEl = document.getElementById("reveal-actual-sub");
  const barYouValueEl = document.getElementById("bar-you-value");
  const barYouMarkerEl = document.getElementById("bar-you-marker");
  const barActualValueEl = document.getElementById("bar-actual-value");
  const barActualMarkerEl = document.getElementById("bar-actual-marker");
  const barRangeEl = document.getElementById("bar-range");
  const barCaptionEl = document.getElementById("bar-caption");
  const oodPillEl = document.getElementById("ood-pill");
  const verdictStampEl = document.getElementById("verdict-stamp");
  const verdictOffbyEl = document.getElementById("verdict-offby");
  const pointsFlyEl = document.getElementById("points-fly");
  const btnNext = document.getElementById("btn-next");

  const finalTotalEl = document.getElementById("final-total");
  const finalOfMaxEl = document.getElementById("final-of-max");
  const finalMeterFillEl = document.getElementById("final-meter-fill");
  const finalPctEl = document.getElementById("final-pct");
  const finalVerdictEl = document.getElementById("final-verdict");
  const finalRecapEl = document.getElementById("final-recap");
  const btnShare = document.getElementById("btn-share");
  const btnAgain = document.getElementById("btn-again");
  const finalFooterEl = document.getElementById("final-footer");

  const confettiLayer = document.getElementById("confetti-layer");

  // ---------- game state ----------
  let currentRound = null;
  let teamIndex = 0;
  let score = 0;
  let streak = 0;
  let teamResults = [];
  let loadingInterval = null;
  let hudPips = [];
  let dailyDate = null;
  let dailyResult = null;

  // ---------- helpers ----------
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function clampWins(v) {
    return Math.max(0, Math.min(N_GAMES, Math.round(v)));
  }

  function clampPct(v) {
    return Math.max(0, Math.min(100, v));
  }

  function fameTierLabel(tier) {
    return tier.replace(/_/g, " ");
  }

  function formatCompact(n) {
    const v = Math.round(n);
    if (v >= 1000) {
      const k = v / 1000;
      const trimmed = k % 1 === 0 ? String(k) : k.toFixed(1).replace(/\.0$/, "");
      return `${trimmed}k`;
    }
    return String(v);
  }

  function recordLine(wins) {
    const w = clampWins(wins);
    const losses = N_GAMES - w;
    const pct = ((w / N_GAMES) * 100).toFixed(1);
    return `projected record ${w}\u2013${losses} \u00b7 ${pct}% win rate`;
  }

  function schedule(fn, ms) {
    if (prefersReducedMotion) {
      fn();
      return null;
    }
    return setTimeout(fn, ms);
  }

  function animateNumber(from, to, durationMs, onFrame) {
    if (prefersReducedMotion) {
      onFrame(to);
      return;
    }
    const start = performance.now();
    function frame(now) {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      onFrame(from + (to - from) * eased);
      if (t < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function selectedFormat() {
    const input = document.querySelector('input[name="format"]:checked');
    return input ? input.value : "daily";
  }

  function isDailyFormat() {
    return selectedFormat() === "daily";
  }

  function loadDailyResult() {
    try {
      const raw = localStorage.getItem(DAILY_STORAGE_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }

  function saveDailyResult(payload) {
    try {
      localStorage.setItem(DAILY_STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // ignore quota / private mode
    }
  }

  function buildShareText(result) {
    const grid = (result.labels || [])
      .map((label) => VERDICT_EMOJI[label] || "⬜")
      .join("");
    return [
      `GuessTheWins Daily ${result.date}`,
      grid,
      `Score: ${Number(result.score).toLocaleString("en-US")} / ${MAX_SCORE.toLocaleString("en-US")}`,
    ].join("\n");
  }

  async function copyShareText(result) {
    const text = buildShareText(result);
    try {
      if (navigator.share) {
        await navigator.share({ text });
        return "shared";
      }
    } catch (err) {
      if (err && err.name === "AbortError") return "cancelled";
    }
    try {
      await navigator.clipboard.writeText(text);
      return "copied";
    } catch {
      window.prompt("Copy your result:", text);
      return "prompted";
    }
  }

  function showScreen(name) {
    Object.entries(screens).forEach(([key, node]) => {
      node.classList.toggle("hidden", key !== name);
    });
    if (name !== "loading" && loadingInterval) {
      clearInterval(loadingInterval);
      loadingInterval = null;
    }
    if (name === "loading") startLoadingRotation();
    hud.classList.toggle("hidden", name !== "team" && name !== "final");
  }

  function startLoadingRotation() {
    let i = 0;
    loadingTextEl.textContent = LOADING_MESSAGES[0];
    if (loadingInterval) clearInterval(loadingInterval);
    if (prefersReducedMotion) return;
    loadingInterval = setInterval(() => {
      i = (i + 1) % LOADING_MESSAGES.length;
      loadingTextEl.textContent = LOADING_MESSAGES[i];
    }, 1400);
  }

  // ---------- HUD ----------
  function buildHudPips() {
    hudPipsEl.textContent = "";
    hudPips = [];
    for (let i = 0; i < 5; i++) {
      const pip = el("span", "pip");
      hudPipsEl.appendChild(pip);
      hudPips.push(pip);
    }
  }

  function markCurrentPip(index) {
    hudPips.forEach((pip, i) => {
      pip.classList.toggle("current", i === index);
    });
  }

  function fillPip(index) {
    if (hudPips[index]) {
      hudPips[index].classList.add("filled");
      hudPips[index].classList.remove("current");
    }
  }

  function bumpHudScore(newScore) {
    const from = Number(hudScoreEl.textContent.replace(/,/g, "")) || 0;
    animateNumber(from, newScore, 600, (v) => {
      hudScoreEl.textContent = Math.round(v).toLocaleString("en-US");
    });
  }

  function trimMultiplier(m) {
    return m.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  }

  function showStreakBadge(multiplier) {
    hudStreakEl.textContent = `\u00d7${trimMultiplier(multiplier)}`;
    hudStreakEl.classList.remove("hidden");
  }

  function hideStreakBadge() {
    hudStreakEl.classList.add("hidden");
  }

  // ---------- player rows ----------
  function buildPlayerRow(p, index) {
    const row = el("div", "player-row");
    const pos = p.position || POS_CYCLE[index] || "—";
    row.appendChild(el("span", "pos", pos));

    const nameWrap = el("div", "player-name-wrap");
    nameWrap.appendChild(el("span", "player-name", p.full_name));
    if (p.fame_tier) {
      nameWrap.appendChild(el("span", "tier-pill", fameTierLabel(p.fame_tier)));
    }
    row.appendChild(nameWrap);
    row.appendChild(
      el("span", "player-meta", `${p.season} \u00b7 ${p.team_abbr || "—"}`)
    );
    return row;
  }

  // ---------- bet slider (wins 0–82) ----------
  function setBetWins(v) {
    const wins = clampWins(v);
    betValueEl.textContent = String(wins);
    betSubEl.textContent = recordLine(wins);
    betSliderEl.value = String(wins);
    btnLock.textContent = `Lock Bet \u00b7 ${wins} Wins`;
  }

  // ---------- scoring ----------
  function scoreVerdict(errWins) {
    if (errWins <= 1) return { base: 1000, label: "JACKPOT", cls: "verdict-great" };
    if (errWins <= 3) return { base: 750, label: "SO CLOSE", cls: "verdict-great" };
    if (errWins <= 6) return { base: 500, label: "SHARP", cls: "verdict-great" };
    if (errWins <= 10) return { base: 250, label: "NOT BAD", cls: "verdict-neutral" };
    if (errWins <= 15) return { base: 100, label: "COLD", cls: "verdict-worst" };
    return { base: 0, label: "AIRBALL", cls: "verdict-worst" };
  }

  function applyScoring(errWins) {
    const v = scoreVerdict(errWins);
    let multiplier = 1;
    if (errWins <= 6) {
      streak += 1;
      multiplier = Math.min(2, 1 + 0.25 * (streak - 1));
    } else {
      streak = 0;
      multiplier = 1;
    }
    const points = Math.round((v.base * multiplier) / 10) * 10;
    return { ...v, multiplier, points };
  }

  // ---------- confetti ----------
  function fireConfetti(count) {
    if (prefersReducedMotion) return;
    const colors = ["var(--brass)", "var(--brass-deep)", "var(--ticket)", "#ffffff"];
    for (let i = 0; i < count; i++) {
      const piece = el("div", "confetti-piece");
      const duration = 1.3 + Math.random() * 1.0;
      const delay = Math.random() * 0.25;
      const drift = `${(Math.random() * 200 - 100).toFixed(0)}px`;
      const spin = `${(360 + Math.random() * 480).toFixed(0)}deg`;
      piece.style.left = `${Math.random() * 100}%`;
      piece.style.background = colors[i % colors.length];
      piece.style.animationDuration = `${duration}s`;
      piece.style.animationDelay = `${delay}s`;
      piece.style.setProperty("--drift", drift);
      piece.style.setProperty("--spin", spin);
      confettiLayer.appendChild(piece);
      setTimeout(() => piece.remove(), (duration + delay) * 1000 + 150);
    }
  }

  // ---------- start screen / daily ----------
  function syncFormatPanels() {
    const daily = isDailyFormat();
    dailyPanelEl.classList.toggle("hidden", !daily);
    freePanelEl.classList.toggle("hidden", daily);

    const alreadyPlayed =
      daily &&
      dailyResult &&
      dailyDate &&
      dailyResult.date === dailyDate;

    if (daily) {
      if (alreadyPlayed) {
        btnDeal.classList.add("hidden");
        dailyDoneEl.classList.remove("hidden");
        dailyDoneScoreEl.textContent = `${Number(dailyResult.score).toLocaleString("en-US")} pts`;
        dailyDoneNoteEl.textContent = "You already settled today\u2019s slip. Share it, or switch to Free Play.";
        startFooterEl.textContent = "Come back tomorrow for a new shared board";
      } else {
        btnDeal.classList.remove("hidden");
        dailyDoneEl.classList.add("hidden");
        btnDeal.textContent = "Play Today\u2019s Boards";
        startFooterEl.textContent = "House rules apply \u00b7 Five boards \u00b7 Shared worldwide";
      }
    } else {
      btnDeal.classList.remove("hidden");
      dailyDoneEl.classList.add("hidden");
      btnDeal.textContent = "Open a Slip";
      startFooterEl.textContent = "House rules apply \u00b7 Five boards per session";
    }
  }

  async function refreshDailyMeta() {
    try {
      const res = await fetch("/api/daily");
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to load daily");
      dailyDate = data.daily_date;
      dailyDateLabelEl.textContent = `Today \u00b7 ${dailyDate} UTC`;
      const stored = loadDailyResult();
      dailyResult =
        stored && stored.date === dailyDate ? stored : null;
    } catch {
      dailyDateLabelEl.textContent = "Today\u2019s boards";
    }
    syncFormatPanels();
  }

  // ---------- team stage ----------
  function replayDealIn() {
    teamCardEl.style.animation = "none";
    void teamCardEl.offsetWidth;
    teamCardEl.style.animation = "";
  }

  function showTeam(index) {
    const team = currentRound.teams[index];
    teamIndex = index;

    boardNumEl.textContent = String(index + 1).padStart(2, "0");
    teamHeadingEl.textContent = "Five-Man Roster";
    if (currentRound.daily) {
      teamDiffChipEl.textContent = `DAILY \u00b7 ${currentRound.daily_date || ""}`;
    } else {
      teamDiffChipEl.textContent = currentRound.difficulty.toUpperCase();
    }

    playerListEl.textContent = "";
    team.players.forEach((p, i) => {
      playerListEl.appendChild(buildPlayerRow(p, i));
    });

    setBetWins(41);
    teamErrorEl.textContent = "";
    btnLock.disabled = false;

    betAreaEl.classList.remove("hidden");
    revealAreaEl.classList.add("hidden");
    btnNext.classList.add("hidden");

    markCurrentPip(index);
    replayDealIn();
  }

  // ---------- reveal ----------
  function startReveal(result) {
    const errWins = result.error * N_GAMES;
    const scoring = applyScoring(errWins);
    const guessWins = result.guess_win_pct * N_GAMES;
    const actualWins = result.actual_wins;

    betAreaEl.classList.add("hidden");
    revealAreaEl.classList.remove("hidden");
    btnNext.classList.add("hidden");

    verdictStampEl.className = "verdict-stamp";
    verdictStampEl.textContent = "";
    verdictOffbyEl.textContent = "";
    pointsFlyEl.textContent = "";
    pointsFlyEl.classList.remove("fly");
    oodPillEl.classList.toggle("hidden", !result.extrapolation_warning);

    barYouMarkerEl.style.left = "0%";
    barActualMarkerEl.style.left = "0%";
    barRangeEl.style.width = "0%";
    barRangeEl.style.left = "0%";
    barYouValueEl.textContent = "0";
    barActualValueEl.textContent = "0";
    barCaptionEl.textContent = "Likely range";
    revealActualValueEl.textContent = "0";
    revealActualSubEl.textContent = "";

    animateNumber(0, actualWins, 650, (v) => {
      revealActualValueEl.textContent = v.toFixed(0);
    });

    schedule(() => {
      const g = Math.round(guessWins * 10) / 10;
      const a = Math.round(actualWins * 10) / 10;
      barYouValueEl.textContent = String(g);
      barActualValueEl.textContent = String(a);
      barYouMarkerEl.style.left = `${clampPct((guessWins / N_GAMES) * 100)}%`;
      barActualMarkerEl.style.left = `${clampPct((actualWins / N_GAMES) * 100)}%`;

      const p10Pct = (result.p10_wins / N_GAMES) * 100;
      const p90Pct = (result.p90_wins / N_GAMES) * 100;
      barRangeEl.style.left = `${clampPct(p10Pct)}%`;
      barRangeEl.style.width = `${clampPct(p90Pct - p10Pct)}%`;

      barCaptionEl.textContent =
        `Likely ${result.p10_wins.toFixed(0)}\u2013${result.p90_wins.toFixed(0)}`;
      revealActualSubEl.textContent =
        `${a.toFixed(1)} wins \u00b7 ${(result.actual_win_pct * 100).toFixed(1)}% win rate`;
    }, 650);

    schedule(() => {
      verdictStampEl.textContent = scoring.label;
      verdictStampEl.classList.add(scoring.cls, "show");
      verdictOffbyEl.textContent = `OFF BY ${errWins.toFixed(1)} WINS`;
    }, 1200);

    schedule(() => {
      pointsFlyEl.textContent = `+${scoring.points}`;
      pointsFlyEl.classList.add("fly");
      score += scoring.points;
      bumpHudScore(score);
      if (scoring.multiplier > 1) {
        showStreakBadge(scoring.multiplier);
      } else {
        hideStreakBadge();
      }
      if (scoring.points >= 750) fireConfetti(18);
    }, 1350);

    schedule(() => {
      btnNext.textContent = teamIndex === 4 ? "See Settlement" : "Next Board";
      btnNext.classList.remove("hidden");
      fillPip(teamIndex);
    }, 1650);

    teamResults.push({
      points: scoring.points,
      ceiling: BOARD_CEILING[teamIndex],
      errWins,
      label: scoring.label,
    });
  }

  // ---------- final ----------
  function finalVerdictLabel(total) {
    if (total >= 4000) return "HIGH ROLLER";
    if (total >= 3000) return "SHARP SHOOTER";
    if (total >= 2000) return "HOT HAND";
    if (total >= 1000) return "LUCKY ROOKIE";
    return "THE HOUSE WINS";
  }

  function buildRecapCard(r, index) {
    const card = el("div", "recap-card");
    const ceiling = r.ceiling || BOARD_CEILING[index];
    card.appendChild(el("span", "recap-team-num", `BOARD ${index + 1}`));
    card.appendChild(el("span", "recap-verdict", r.label));
    const pointsRow = el("span", "recap-points");
    pointsRow.appendChild(document.createTextNode(`+${r.points}`));
    pointsRow.appendChild(el("span", "recap-ceiling", ` / ${ceiling}`));
    card.appendChild(pointsRow);
    card.appendChild(
      el("span", "recap-offby", `OFF BY ${r.errWins.toFixed(1)}`)
    );
    return card;
  }

  function persistDailyIfNeeded() {
    if (!currentRound || !currentRound.daily || !currentRound.daily_date) return;
    const payload = {
      date: currentRound.daily_date,
      score,
      labels: teamResults.map((r) => r.label),
      errWins: teamResults.map((r) => r.errWins),
    };
    saveDailyResult(payload);
    dailyResult = payload;
    dailyDate = currentRound.daily_date;
  }

  function renderFinalScreen() {
    const total = score;
    const isDaily = Boolean(currentRound && currentRound.daily);

    if (isDaily) persistDailyIfNeeded();

    finalTotalEl.textContent = "0";
    finalOfMaxEl.textContent = `0 / ${formatCompact(MAX_SCORE)}`;
    finalMeterFillEl.style.width = "0%";
    finalPctEl.textContent = `0% of max`;

    animateNumber(0, total, 1000, (v) => {
      const n = Math.round(v);
      finalTotalEl.textContent = n.toLocaleString("en-US");
      finalOfMaxEl.textContent = `${formatCompact(n)} / ${formatCompact(MAX_SCORE)}`;
      finalMeterFillEl.style.width = `${clampPct((n / MAX_SCORE) * 100)}%`;
      finalPctEl.textContent = `${Math.round((n / MAX_SCORE) * 100)}% of max`;
    });

    finalVerdictEl.textContent = finalVerdictLabel(total);

    finalRecapEl.textContent = "";
    teamResults.forEach((r, i) => finalRecapEl.appendChild(buildRecapCard(r, i)));

    btnShare.classList.toggle("hidden", !isDaily);
    if (isDaily) {
      btnAgain.textContent = "Back to Lobby";
      finalFooterEl.textContent = "Come back tomorrow for a new shared board";
    } else {
      btnAgain.textContent = "Cash Out & Replay";
      finalFooterEl.textContent = "Thanks for playing \u00b7 The window is always open";
    }

    showScreen("final");

    if (total >= 3000) {
      schedule(() => fireConfetti(24), 200);
    }
  }

  // ---------- round lifecycle ----------
  function initRoundState(data) {
    currentRound = data;
    teamIndex = 0;
    score = 0;
    streak = 0;
    teamResults = [];
    hudScoreEl.textContent = "0";
    hideStreakBadge();
    buildHudPips();
  }

  async function dealRound() {
    if (isDailyFormat() && dailyResult && dailyDate && dailyResult.date === dailyDate) {
      startErrorEl.textContent = "You already played today\u2019s boards.";
      syncFormatPanels();
      return;
    }

    const difficultyInput = document.querySelector(
      'input[name="difficulty"]:checked'
    );
    const modeInput = document.querySelector('input[name="mode"]:checked');
    const difficulty = difficultyInput ? difficultyInput.value : "medium";
    const mode = modeInput ? modeInput.value : "competitive";
    const daily = isDailyFormat();

    startErrorEl.textContent = "";
    btnDeal.disabled = true;
    showScreen("loading");

    try {
      const res = await fetch("/api/round", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(daily ? { daily: true } : { difficulty, mode }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to open a slip");
      initRoundState(data);
      showTeam(0);
      showScreen("team");
    } catch (err) {
      startErrorEl.textContent = err.message;
      showScreen("start");
      syncFormatPanels();
    } finally {
      btnDeal.disabled = false;
    }
  }

  async function lockInTeam() {
    if (!currentRound) return;
    const wins = Number(betSliderEl.value);
    const guess = wins / N_GAMES;

    btnLock.disabled = true;
    btnLock.textContent = "Settling\u2026";
    teamErrorEl.textContent = "";

    try {
      const res = await fetch("/api/score_team", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          round_id: currentRound.round_id,
          ticket: currentRound.ticket,
          team_index: teamIndex,
          guess,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to settle bet");
      startReveal(data);
    } catch (err) {
      btnLock.disabled = false;
      setBetWins(wins);
      teamErrorEl.textContent = err.message;
    }
  }

  function goNext() {
    if (teamIndex >= 4) {
      renderFinalScreen();
    } else {
      showTeam(teamIndex + 1);
    }
  }

  function resetToStart() {
    currentRound = null;
    teamIndex = 0;
    score = 0;
    streak = 0;
    teamResults = [];
    hudScoreEl.textContent = "0";
    hideStreakBadge();
    hudPipsEl.textContent = "";
    hudPips = [];
    showScreen("start");
    syncFormatPanels();
  }

  async function shareCurrentDaily(fromBtn) {
    const result =
      dailyResult ||
      (currentRound &&
        currentRound.daily && {
          date: currentRound.daily_date,
          score,
          labels: teamResults.map((r) => r.label),
        });
    if (!result || !result.date) return;

    const status = await copyShareText(result);
    if (!fromBtn) return;
    const original = fromBtn.textContent;
    if (status === "copied") fromBtn.textContent = "Copied!";
    else if (status === "shared") fromBtn.textContent = "Shared!";
    else if (status === "cancelled") return;
    else fromBtn.textContent = "Ready to paste";
    setTimeout(() => {
      fromBtn.textContent = original;
    }, 1600);
  }

  // ---------- wiring ----------
  betSliderEl.addEventListener("input", () => {
    setBetWins(Number(betSliderEl.value));
  });

  document.querySelectorAll('input[name="format"]').forEach((input) => {
    input.addEventListener("change", syncFormatPanels);
  });

  btnDeal.addEventListener("click", dealRound);
  btnLock.addEventListener("click", lockInTeam);
  btnNext.addEventListener("click", goNext);
  btnAgain.addEventListener("click", resetToStart);
  btnShare.addEventListener("click", () => shareCurrentDaily(btnShare));
  btnShareDaily.addEventListener("click", () => shareCurrentDaily(btnShareDaily));

  setBetWins(41);
  refreshDailyMeta();
})();
