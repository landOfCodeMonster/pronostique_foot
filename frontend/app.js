"use strict";

const pct = (x) => `${Math.round(x * 100)}%`;

const STAGES = {
  GROUP_STAGE: "Phase de groupes",
  LAST_16: "Huitièmes",
  QUARTER_FINALS: "Quarts",
  SEMI_FINALS: "Demies",
  THIRD_PLACE: "Petite finale",
  FINAL: "Finale",
};

const RELIABILITY = {
  faible: "low",
  moyen: "mid",
  "élevé": "high",
};

function fmtKickoff(iso) {
  try {
    return new Date(iso)
      .toLocaleString("fr-FR", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })
      .toUpperCase();
  } catch {
    return iso;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function card(m, i) {
  const stage = STAGES[m.stage] || "Match";
  const level = RELIABILITY[m.reliability] || "low";
  const home = escapeHtml(m.home_team);
  const away = escapeHtml(m.away_team);
  return `
  <article class="match" style="--i:${i}">
    <div class="match__top">
      <span class="match__stage">${stage}</span>
      <span class="match__kick">${fmtKickoff(m.utc_date)}</span>
    </div>

    <div class="board">
      <div class="board__team board__team--home"><span class="board__name">${home}</span></div>
      <div class="board__score">
        <span>${m.pred_home}</span><span class="board__sep">–</span><span>${m.pred_away}</span>
      </div>
      <div class="board__team board__team--away"><span class="board__name">${away}</span></div>
    </div>

    <div class="odds">
      <div class="odds__bar" role="img"
           aria-label="Probabilités : ${home} ${pct(m.prob_home)}, nul ${pct(m.prob_draw)}, ${away} ${pct(m.prob_away)}">
        <span class="odds__seg odds__seg--home" data-w="${m.prob_home * 100}"></span>
        <span class="odds__seg odds__seg--draw" data-w="${m.prob_draw * 100}"></span>
        <span class="odds__seg odds__seg--away" data-w="${m.prob_away * 100}"></span>
      </div>
      <div class="odds__legend">
        <span class="odds__key"><i class="dot dot--home"></i>1 <b>${pct(m.prob_home)}</b></span>
        <span class="odds__key"><i class="dot dot--draw"></i>N <b>${pct(m.prob_draw)}</b></span>
        <span class="odds__key"><i class="dot dot--away"></i>2 <b>${pct(m.prob_away)}</b></span>
      </div>
    </div>

    <div class="match__bottom">
      <div class="chips">
        <span class="chip">+2.5 buts <b>${pct(m.prob_over25)}</b></span>
        <span class="chip">Les deux marquent <b>${pct(m.prob_btts)}</b></span>
      </div>
      <div class="signal signal--${level}" title="Fiabilité : ${escapeHtml(m.reliability)}">
        <span></span><span></span><span></span><em>${escapeHtml(m.reliability)}</em>
      </div>
    </div>
  </article>`;
}

function animateBars() {
  requestAnimationFrame(() => {
    document.querySelectorAll(".odds__seg").forEach((seg) => {
      seg.style.width = `${seg.dataset.w}%`;
    });
  });
}

async function load() {
  const feed = document.getElementById("matches");
  const count = document.getElementById("count");
  try {
    const res = await fetch("/api/matches/upcoming");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (!data.length) {
      feed.innerHTML =
        '<p class="state">Aucun match à venir pour l\'instant. Revenez à l\'approche d\'une journée de Coupe du Monde.</p>';
      count.textContent = "";
      return;
    }

    feed.innerHTML = data.map(card).join("");
    count.textContent = `${data.length} match${data.length > 1 ? "s" : ""} analysé${data.length > 1 ? "s" : ""}.`;
    animateBars();
  } catch (e) {
    feed.innerHTML =
      '<p class="state state--error">Données indisponibles. Vérifiez la clé API et que le serveur tourne.</p>';
    count.textContent = "";
  }
}

load();
