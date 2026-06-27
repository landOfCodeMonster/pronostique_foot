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

const RELIABILITY = { faible: "low", moyen: "mid", "élevé": "high" };
const LIVE = new Set(["IN_PLAY", "PAUSED"]);

let allMatches = [];

const isLive = (m) => LIVE.has(m.status);

function fmtKickoff(iso) {
  try {
    return new Date(iso)
      .toLocaleString("fr-FR", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })
      .toUpperCase();
  } catch {
    return iso;
  }
}

// Lowercase + strip accents so "bresil" matches "Brésil".
function normalize(s) {
  return String(s).toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function card(m, i) {
  const stage = STAGES[m.stage] || "Match";
  const level = RELIABILITY[m.reliability] || "low";
  const home = escapeHtml(m.home_team);
  const away = escapeHtml(m.away_team);
  const live = isLive(m);
  const status = live
    ? '<span class="live">En cours</span>'
    : `<span class="match__kick">${fmtKickoff(m.utc_date)}</span>`;
  const lh = m.live_home == null ? "·" : m.live_home;
  const la = m.live_away == null ? "·" : m.live_away;
  const scoreBlock = live
    ? `<div class="board__score board__score--live"><span>${lh}</span><span class="board__sep">–</span><span>${la}</span></div>`
    : `<div class="board__score"><span>${m.pred_home}</span><span class="board__sep">–</span><span>${m.pred_away}</span></div>`;
  const forecast = live
    ? `<p class="forecast">Pronostic <b>${m.pred_home}–${m.pred_away}</b></p>`
    : "";
  return `
  <article class="match${live ? " match--live" : ""}" style="--i:${i}">
    <div class="match__top">
      <span class="match__stage">${stage}</span>
      ${status}
    </div>

    <div class="board">
      <div class="board__team board__team--home"><span class="board__name">${home}</span></div>
      ${scoreBlock}
      <div class="board__team board__team--away"><span class="board__name">${away}</span></div>
    </div>
    ${forecast}

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

function render(query = "") {
  const feed = document.getElementById("matches");
  const count = document.getElementById("count");
  const q = normalize(query.trim());
  const list = q
    ? allMatches.filter((m) => normalize(m.home_team).includes(q) || normalize(m.away_team).includes(q))
    : allMatches;

  if (!list.length) {
    feed.innerHTML = q
      ? `<p class="state">Aucun match pour « ${escapeHtml(query.trim())} ».</p>`
      : '<p class="state">Aucun match à venir pour l\'instant. Revenez à l\'approche d\'une journée de Coupe du Monde.</p>';
    count.textContent = q ? "" : "";
    return;
  }

  feed.innerHTML = list.map(card).join("");
  const n = list.length;
  const total = allMatches.length;
  const label = q ? `${n} / ${total} match${total > 1 ? "s" : ""}` : `${n} match${n > 1 ? "s" : ""} analysé${n > 1 ? "s" : ""}.`;
  count.textContent = label;
  animateBars();
}

async function load() {
  const feed = document.getElementById("matches");
  try {
    const res = await fetch("/api/matches/upcoming");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    allMatches = await res.json();
    render(document.getElementById("search").value);
  } catch (e) {
    feed.innerHTML =
      '<p class="state state--error">Données indisponibles. Vérifiez la clé API et que le serveur tourne.</p>';
    document.getElementById("count").textContent = "";
  }
}

document.getElementById("search").addEventListener("input", (e) => render(e.target.value));

load();
