"use strict";
const $ = (s) => document.querySelector(s);
const PALETTE = ["#ff5050", "#ffaa28", "#f0e63c", "#5adc6e", "#46c8e6", "#6e82ff", "#c86eff"];

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.text()).slice(0, 200));
  return r.json();
}
const fmtTime = (t) => (t == null ? "—" : `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, "0")}`);

/* ------------------------------------------------------------------ pista + carros em canvas */
class View {
  constructor(canvas) {
    this.c = canvas; this.ctx = canvas.getContext("2d"); this.track = null; this.follow = false;
  }
  setTrack(t) {
    this.track = t;
    const xs = t.left.concat(t.right).map((p) => p[0]), ys = t.left.concat(t.right).map((p) => p[1]);
    this.b = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
    const mk = (pts) => { const p = new Path2D(); pts.forEach((q, i) => (i ? p.lineTo(q[0], q[1]) : p.moveTo(q[0], q[1]))); p.closePath(); return p; };
    this.pCenter = mk(t.center); this.pL = mk(t.left); this.pR = mk(t.right);
  }
  transform(focus) {
    const { width: W, height: H } = this.c, b = this.b, pad = 30;
    let s = Math.min((W - 2 * pad) / (b.x1 - b.x0), (H - 2 * pad) / (b.y1 - b.y0)), cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
    if (this.follow && focus) { s = Math.max(s * 3.2, 1.6); cx = focus[0]; cy = focus[1]; }
    this.ctx.setTransform(s, 0, 0, -s, W / 2 - cx * s, H / 2 + cy * s);
    return s;
  }
  draw(cars, focus) {
    const ctx = this.ctx; ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#16241a"; ctx.fillRect(0, 0, this.c.width, this.c.height);
    if (!this.track) return;
    const s = this.transform(focus), t = this.track;
    ctx.lineJoin = ctx.lineCap = "round";
    ctx.strokeStyle = "#3e4046"; ctx.lineWidth = t.width; ctx.stroke(this.pCenter);
    ctx.strokeStyle = "#ebebeb"; ctx.lineWidth = Math.max(0.35, 1.2 / s); ctx.stroke(this.pL); ctx.stroke(this.pR);
    const st = t.start, nx = -Math.sin(st.heading), ny = Math.cos(st.heading);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2.4; ctx.beginPath();
    ctx.moveTo(st.x + nx * t.width / 2, st.y + ny * t.width / 2); ctx.lineTo(st.x - nx * t.width / 2, st.y - ny * t.width / 2); ctx.stroke();
    const k = Math.max(1, 9 / (4.6 * s));
    for (const car of cars) {
      ctx.save(); ctx.translate(car.x, car.y); ctx.rotate(car.h); ctx.scale(k, k);
      ctx.fillStyle = car.color; ctx.strokeStyle = "#000"; ctx.lineWidth = 0.15;
      ctx.fillRect(-2.3, -0.95, 4.6, 1.9); ctx.strokeRect(-2.3, -0.95, 4.6, 1.9);
      ctx.fillStyle = "#fff"; ctx.fillRect(1.2, -0.5, 0.8, 1.0); ctx.restore();
    }
  }
}

function poseAt(car, t) {
  const f = car.frames, x = t / car.dt, i = Math.min(Math.floor(x), f.length - 1), j = Math.min(i + 1, f.length - 1), a = x - Math.floor(x);
  const p = f[i], q = f[j];
  let dh = q[2] - p[2]; dh = Math.atan2(Math.sin(dh), Math.cos(dh));
  const w = i === j ? 0 : Math.min(a, 1);
  return { x: p[0] + (q[0] - p[0]) * w, y: p[1] + (q[1] - p[1]) * w, h: p[2] + dh * w };
}

/* ------------------------------------------------------------------ player */
class Player {
  constructor(view, ui) {
    this.view = view; this.ui = ui; this.cars = []; this.t = 0; this.playing = true; this.last = performance.now();
    ui.play.onclick = () => { this.playing = !this.playing; ui.play.textContent = this.playing ? "⏸" : "▶"; };
    ui.scrub.oninput = () => { this.t = (ui.scrub.value / 1000) * this.duration; };
    requestAnimationFrame((n) => this.loop(n));
  }
  load(cars) { this.cars = cars; this.t = 0; this.duration = Math.max(...cars.map((c) => (c.frames.length - 1) * c.dt)); }
  loop(now) {
    const dt = (now - this.last) / 1000; this.last = now;
    if (this.cars.length) {
      if (this.playing) { this.t += dt * parseFloat(this.ui.speed.value); if (this.t > this.duration + 1.5) this.t = 0; }
      const tt = Math.min(this.t, this.duration);
      this.ui.scrub.value = (tt / this.duration) * 1000;
      const poses = this.cars.map((c) => ({ ...poseAt(c, Math.min(tt, (c.frames.length - 1) * c.dt)), color: c.color }));
      this.view.follow = this.ui.follow ? this.ui.follow.checked : false;
      this.view.draw(poses.slice().reverse(), [poses[0].x, poses[0].y]);
      const el = this.ui.clock; if (el) el.textContent = fmtTime(tt);
    }
    requestAnimationFrame((n) => this.loop(n));
  }
}

/* ------------------------------------------------------------------ gráficos de linha */
function chart(canvas, series, { xs, pct = false, yMin = null } = {}) {
  const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height, m = { l: 46, r: 12, t: 12, b: 26 };
  ctx.clearRect(0, 0, W, H);
  const vals = series.flatMap((s) => s.data.filter((v) => v != null && isFinite(v)));
  if (vals.length < 2) { ctx.fillStyle = "#8d96a8"; ctx.fillText("sem dados", W / 2 - 24, H / 2); return; }
  let lo = yMin ?? Math.min(...vals), hi = Math.max(...vals); if (hi === lo) hi = lo + 1;
  const n = xs.length, X = (i) => m.l + ((W - m.l - m.r) * i) / Math.max(n - 1, 1), Y = (v) => H - m.b - ((H - m.t - m.b) * (v - lo)) / (hi - lo);
  ctx.strokeStyle = "#2a3140"; ctx.fillStyle = "#8d96a8"; ctx.font = "12px system-ui"; ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k++) { const v = lo + ((hi - lo) * k) / 4, y = Y(v); ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(W - m.r, y); ctx.stroke(); ctx.fillText(Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2).replace(/\.?0+$/, ""), 4, y + 4); }
  [0, Math.floor((n - 1) / 2), n - 1].forEach((i) => ctx.fillText(xs[i], X(i) - 8, H - 8));
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.w || 2; ctx.beginPath(); let pen = false;
    s.data.forEach((v, i) => { if (v == null || !isFinite(v)) { pen = false; return; } pen ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)); pen = true; });
    ctx.stroke();
    if (s.dots) s.data.forEach((v, i) => { if (v != null && isFinite(v)) { ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X(i), Y(v), 3, 0, 7); ctx.fill(); } });
  }
  let lx = m.l + 6; ctx.font = "12px system-ui";
  for (const s of series) { ctx.fillStyle = s.color; ctx.fillRect(lx, m.t + 2, 10, 10); ctx.fillStyle = "#c9d1e0"; ctx.fillText(s.name, lx + 14, m.t + 11); lx += 22 + ctx.measureText(s.name).width; }
}

/* ------------------------------------------------------------------ abas */
document.querySelectorAll("#tabs button").forEach((b) => (b.onclick = () => {
  document.querySelectorAll("#tabs button, .tab").forEach((e) => e.classList.remove("active"));
  b.classList.add("active"); $("#tab-" + b.dataset.tab).classList.add("active");
}));

/* ------------------------------------------------------------------ corrida */
const raceView = new View($("#canvas-race"));
const racePlayer = new Player(raceView, { play: $("#play"), scrub: $("#scrub"), speed: $("#speed"), follow: $("#follow") });
const trackSpec = () => ($("#trackKind").value === "seed" ? `seed:${parseInt($("#seed").value || "0", 10)}` : "interlagos");

$("#trackKind").onchange = () => { $("#seedBox").hidden = $("#trackKind").value !== "seed"; };
$("#randSeed").onclick = () => { $("#seed").value = Math.floor(Math.random() * 100000); };

async function runRace(spec, model) {
  const btn = $("#run"), st = $("#status"); btn.disabled = true; st.textContent = "simulando no servidor…";
  try {
    const d = await getJSON(`/api/simulate?track=${encodeURIComponent(spec)}&model=${encodeURIComponent(model)}`);
    raceView.setTrack(d.track); d.car.color = "#ff5050"; racePlayer.load([d.car]); st.textContent = "";
    const r = d.result, res = $("#result"); res.hidden = false;
    res.innerHTML = `<div><b>${d.track.name}</b><span>${(d.track.length / 1000).toFixed(2)} km · largura ${d.track.width} m</span></div>
      <div><b class="${r.finished ? "ok" : "bad"}">${r.finished ? "Completou" : r.reason}</b><span>${d.car.label}</span></div>
      <div><b>${(r.progress * 100).toFixed(1)}%</b><span>da volta</span></div>
      <div><b>${r.finished ? fmtTime(r.time) : "—"}</b><span>tempo de volta</span></div>
      <div><b>${(r.avg_speed * 3.6).toFixed(0)} km/h</b><span>velocidade média</span></div>`;
  } catch (e) { st.textContent = "erro: " + e.message; }
  btn.disabled = false;
}
$("#run").onclick = () => runRace(trackSpec(), $("#model").value);

/* ------------------------------------------------------------------ comparar gerações */
const cmpView = new View($("#canvas-cmp"));
const cmpPlayer = new Player(cmpView, { play: $("#cplay"), scrub: $("#cscrub"), speed: $("#cspeed") });

async function loadStatic() {
  try {
    const [rep, hist] = await Promise.all([getJSON("/data/replays.json"), getJSON("/data/history.json")]);
    cmpView.setTrack(rep.track);
    rep.cars.forEach((c, i) => (c.color = PALETTE[i % PALETTE.length]));
    cmpPlayer.load(rep.cars);
    $("#legend").innerHTML = rep.cars.map((c) => `<span><i style="background:${c.color}"></i>${c.label} — ${c.finished ? "completou em " + fmtTime(c.time) : c.reason + " a " + (c.progress * 100).toFixed(0) + "%"}</span>`).join("");
    if (!raceView.track) raceView.setTrack(rep.track);
    const xs = hist.map((r) => r.gen), col = (k) => hist.map((r) => r[k]);
    chart($("#ch-fit"), [{ name: "melhor", data: col("best_fit"), color: "#ffaa28" }, { name: "média", data: col("mean_fit"), color: "#46c8e6" }, { name: "mediana", data: col("median_fit"), color: "#6e82ff", w: 1 }], { xs });
    chart($("#ch-prog"), [{ name: "melhor", data: col("best_progress").map((v) => v * 100), color: "#5adc6e" }, { name: "média", data: col("mean_progress").map((v) => v * 100), color: "#46c8e6" }], { xs });
    chart($("#ch-time"), [{ name: "melhor volta", data: col("best_time"), color: "#c86eff", dots: true }], { xs });
    chart($("#ch-val"), [{ name: "progresso médio", data: col("val_progress").map((v) => (v == null ? null : v * 100)), color: "#ff5050", dots: true }, { name: "pistas completas", data: col("val_finish_rate").map((v) => (v == null ? null : v * 100)), color: "#5adc6e", dots: true }], { xs, yMin: 0 });
    const last = hist[hist.length - 1];
    $("#evo-note").textContent = `${hist.length} gerações · ${last.n_finishers} dos carros completaram a volta na última · tempo de treino ${(last.elapsed_s / 60).toFixed(1)} min.`;
  } catch (e) { $("#legend").textContent = "Dados de treino ainda não exportados (python -m f1ia export-web)."; }
  try {
    const g = await getJSON("/data/generalization.json");
    $("#gen-sum").innerHTML = `Melhor carro da geração <b>${g.generation}</b>, treinado <u>só</u> em Interlagos, em ${g.summary.n} pistas novas geradas por semente: completou <b>${(g.summary.finish_rate * 100).toFixed(0)}%</b> (progresso médio ${(g.summary.mean_progress * 100).toFixed(1)}%).`;
    $("#gen-table tbody").innerHTML = g.rows.map((r) => `<tr><td>${r.track}</td><td>${(r.length / 1000).toFixed(2)} km</td><td>${(r.progress * 100).toFixed(1)}%</td><td>${r.time == null ? "—" : fmtTime(r.time)}</td><td class="${r.finished ? "ok" : "bad"}">${r.reason_name}</td><td><button data-seed="${r.track.split(":")[1]}">ver</button></td></tr>`).join("");
    document.querySelectorAll("#gen-table button").forEach((b) => (b.onclick = () => {
      $("#trackKind").value = "seed"; $("#seedBox").hidden = false; $("#seed").value = b.dataset.seed;
      if ($("#model option[value='interlagos_best']")) $("#model").value = "interlagos_best";
      document.querySelector('[data-tab="race"]').click(); $("#run").click();
    }));
  } catch (e) { $("#gen-sum").textContent = "Relatório de generalização indisponível."; }
}

async function loadModels() {
  try {
    const d = await getJSON("/api/models");
    for (const m of d.models) { const o = document.createElement("option"); o.value = m.name; o.textContent = `IA treinada: ${m.name} (ger ${m.generation})`; $("#model").appendChild(o); }
    if (d.models.length) $("#model").value = d.models[0].name;
  } catch (e) { /* API indisponível: segue só com o piloto automático */ }
}

loadModels().then(loadStatic);
