/* AI Engineer Drill — quiz engine, scoring, study notes, stats. No dependencies. */
(() => {
  'use strict';

  // ---------------------------------------------------------------- data
  const BANK = (window.QUIZ_BANK || []).slice().sort((a, b) => (a.order || 99) - (b.order || 99));
  const TOPIC = {};
  const ALLQ = [];
  const QMAP = {};
  BANK.forEach(t => {
    TOPIC[t.id] = t;
    t.questions.forEach(q => { q.topic = t.id; ALLQ.push(q); QMAP[q.id] = q; });
  });

  const LEVEL_W = { 1: 1, 2: 2, 3: 3 };
  const LEVEL_NAME = { 1: 'Cơ bản', 2: 'Trung cấp', 3: 'Nâng cao' };
  const TYPE_NAME = { single: 'Một đáp án', multi: 'Nhiều đáp án', numeric: 'Tính toán' };
  const LETTERS = 'ABCDEF';
  const NEG = 0.25;

  // ---------------------------------------------------------------- storage (per-browser convenience)
  const store = {
    get(k, d) { try { const v = localStorage.getItem('aiq.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('aiq.' + k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } },
    del(k) { try { localStorage.removeItem('aiq.' + k); } catch (e) { /* ignore */ } },
  };
  let progress = store.get('progress', {});           // qid -> {n, c, last, t}
  let flags = new Set(store.get('flags', []));
  let history = store.get('history', []);
  const DEFAULTS = {
    topics: BANK.map(t => t.id), count: 20, levels: [1, 2, 3], types: ['single', 'multi', 'numeric'],
    source: 'all', mode: 'practice', timePer: 90, strictMulti: false, negative: false, trapOnly: false,
  };
  let settings = Object.assign({}, DEFAULTS, store.get('settings', {}));
  settings.topics = settings.topics.filter(id => TOPIC[id]);
  if (!settings.topics.length) settings.topics = DEFAULTS.topics.slice();
  let quiz = store.get('quiz', null);
  if (quiz && !(quiz.items || []).every(it => QMAP[it.id])) quiz = null;
  let lastResult = store.get('last', null);
  if (lastResult && !(lastResult.items || []).every(it => QMAP[it.id])) lastResult = null;

  const saveQuiz = () => (quiz ? store.set('quiz', quiz) : store.del('quiz'));
  const saveProgress = () => store.set('progress', progress);
  const saveFlags = () => store.set('flags', [...flags]);

  // ---------------------------------------------------------------- utils
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const app = $('#app');
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const range = n => Array.from({ length: n }, (_, i) => i);
  function shuffle(a) {
    a = a.slice();
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }
  const pct = (a, b) => (b ? Math.round((a / b) * 1000) / 10 : 0);
  const fmtNum = x => (Math.round(x * 100) / 100).toLocaleString('vi-VN');
  function fmtTime(sec) {
    sec = Math.max(0, Math.round(sec));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    const mm = String(m).padStart(2, '0'), ss = String(s).padStart(2, '0');
    return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
  }
  const fmtDate = t => new Date(t).toLocaleString('vi-VN', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
  function colorFor(p) { return p >= 80 ? 'var(--good)' : p >= 55 ? 'var(--part)' : 'var(--bad)'; }

  // ---------------------------------------------------------------- tiny markdown renderer
  function inline(s) {
    // code spans become placeholders so bold/italic can wrap them and never reach inside them
    const codes = [];
    const text = String(s).replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return '\u0000' + (codes.length - 1) + '\u0000'; });
    return esc(text)
      .replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[\s(])\*([^*\s][^*\n]*?[^*\s]|[^*\s])\*(?=[\s).,:;!?]|$)/g, '$1<em>$2</em>')
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
      .replace(/\u0000(\d+)\u0000/g, (_, i) => '<code>' + esc(codes[Number(i)]) + '</code>');
  }
  const LIST_RE = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
  const isTableSep = l => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);
  const splitRow = l => l.trim().replace(/^\|/, '').replace(/\|$/, '').split(/(?<!\\)\|/).map(c => c.trim().replace(/\\\|/g, '|'));
  function md(src) {
    if (src == null) return '';
    const lines = String(src).replace(/\r\n?/g, '\n').split('\n');
    let html = '', i = 0;
    const startsBlock = l => /^```/.test(l) || /^#{1,6}\s/.test(l) || LIST_RE.test(l) || /^>\s?/.test(l) || /^\s*\|/.test(l) || /^-{3,}\s*$/.test(l);
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }
      let m;
      if ((m = line.match(/^```\s*([\w+-]*)/))) {
        const buf = []; i++;
        while (i < lines.length && !/^```\s*$/.test(lines[i])) buf.push(lines[i++]);
        i++;
        html += `<pre><code${m[1] ? ` class="lang-${esc(m[1])}"` : ''}>${esc(buf.join('\n'))}</code></pre>`;
        continue;
      }
      if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
        const lv = Math.min(5, m[1].length + 2);
        html += `<h${lv}>${inline(m[2])}</h${lv}>`; i++; continue;
      }
      if (/^-{3,}\s*$/.test(line)) { html += '<hr>'; i++; continue; }
      if (/^\s*\|/.test(line) && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        const head = splitRow(line); i += 2;
        let t = '<div class="tbl"><table><thead><tr>' + head.map(c => `<th>${inline(c)}</th>`).join('') + '</tr></thead><tbody>';
        while (i < lines.length && /^\s*\|/.test(lines[i])) {
          t += '<tr>' + splitRow(lines[i]).map(c => `<td>${inline(c)}</td>`).join('') + '</tr>'; i++;
        }
        html += t + '</tbody></table></div>'; continue;
      }
      if (/^>\s?/.test(line)) {
        const buf = [];
        while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, ''));
        html += '<blockquote>' + md(buf.join('\n')) + '</blockquote>'; continue;
      }
      if (LIST_RE.test(line)) {
        const items = [];
        while (i < lines.length) {
          const l = lines[i], lm = l.match(LIST_RE);
          if (lm) { items.push({ ind: lm[1].length, ord: /\d/.test(lm[2]), text: lm[3] }); i++; }
          else if (l.trim() && /^\s{2,}/.test(l) && items.length) { items[items.length - 1].text += ' ' + l.trim(); i++; }
          else break;
        }
        const stack = [];
        for (const it of items) {
          while (stack.length > 1 && it.ind < stack[stack.length - 1].ind) html += '</li></' + stack.pop().tag + '>';
          const top = stack[stack.length - 1];
          if (!top || it.ind > top.ind) { const tag = it.ord ? 'ol' : 'ul'; stack.push({ ind: it.ind, tag }); html += `<${tag}><li>`; }
          else html += '</li><li>';
          html += inline(it.text);
        }
        while (stack.length) html += '</li></' + stack.pop().tag + '>';
        continue;
      }
      const buf = [];
      while (i < lines.length && lines[i].trim() && !(buf.length && startsBlock(lines[i]))) buf.push(lines[i++]);
      html += '<p>' + buf.map(inline).join('<br>') + '</p>';
    }
    return html;
  }

  // ---------------------------------------------------------------- scoring
  function parseNum(s) {
    if (s == null) return NaN;
    s = String(s).trim().replace(/\s+/g, '').replace(/%$/, '');
    if (!s) return NaN;
    const commas = (s.match(/,/g) || []).length;
    if (commas === 1 && !s.includes('.')) s = s.replace(',', '.');
    else s = s.replace(/,/g, '');
    const fr = s.match(/^(-?[\d.]+(?:e[+-]?\d+)?)\/(-?[\d.]+(?:e[+-]?\d+)?)$/i);
    if (fr) return Number(fr[1]) / Number(fr[2]);
    if (/^-?[\d.]+(e[+-]?\d+)?$/i.test(s) || /^-?\d*\.?\d+(e[+-]?\d+)?$/i.test(s)) return Number(s);
    return NaN;
  }
  const isAnswered = (q, r) => {
    if (r == null) return false;
    if (q.type === 'multi') return Array.isArray(r) && r.length > 0;
    if (q.type === 'numeric') return String(r).trim() !== '';
    return true;
  };
  // Returns fraction of credit 0..1
  function gradeFrac(q, r, strict) {
    if (!isAnswered(q, r)) return 0;
    if (q.type === 'single') return r === q.answer ? 1 : 0;
    if (q.type === 'multi') {
      const ans = new Set(q.answer);
      const tp = r.filter(x => ans.has(x)).length, fp = r.length - tp;
      if (strict) return tp === ans.size && fp === 0 ? 1 : 0;
      return Math.max(0, (tp - fp) / ans.size);
    }
    const v = parseNum(r), a = Number(q.answer);
    if (Number.isNaN(v)) return 0;
    const tol = Number(q.tolerance || 0) + 1e-9 * Math.max(1, Math.abs(a));
    return Math.abs(v - a) <= tol ? 1 : 0;
  }
  function gradeItem(q, r, opts) {
    const w = LEVEL_W[q.level] || 1;
    const answered = isAnswered(q, r);
    const frac = gradeFrac(q, r, opts.strictMulti);
    let pts = frac * w;
    if (opts.negative && answered && frac === 0) pts = -NEG * w;
    const status = !answered ? 'na' : frac >= 0.999 ? 'ok' : frac > 0 ? 'pa' : 'ko';
    return { frac, pts, max: w, status, answered };
  }
  function recordProgress(qid, frac) {
    const p = progress[qid] || { n: 0, c: 0, last: 0, t: 0 };
    p.n += 1; p.c += frac; p.last = frac; p.t = Date.now();
    progress[qid] = p;
  }
  function correctText(q) {
    if (q.type === 'numeric') return `${fmtNum(q.answer)}${q.unit ? ' ' + q.unit : ''}${q.tolerance ? ` (± ${fmtNum(q.tolerance)})` : ''}`;
    const idx = q.type === 'single' ? [q.answer] : q.answer;
    return idx.map(i => q.options[i]).join(' · ');
  }
  function respText(q, r) {
    if (!isAnswered(q, r)) return '— bỏ trống —';
    if (q.type === 'numeric') return String(r) + (q.unit ? ' ' + q.unit : '');
    const idx = q.type === 'single' ? [r] : r.slice().sort((a, b) => a - b);
    return idx.map(i => q.options[i]).join(' · ');
  }
  function band(p) {
    if (p >= 90) return ['Xuất sắc', 'Sẵn sàng cho vòng technical khó. Tiếp tục luyện câu Nâng cao và câu bẫy.'];
    if (p >= 75) return ['Vững', 'Nền tảng tốt. Ôn lại các chủ đề có thanh màu vàng/đỏ bên dưới.'];
    if (p >= 60) return ['Khá', 'Còn lỗ hổng rõ ràng. Đọc mục Kiến thức của chủ đề yếu rồi làm lại câu sai.'];
    if (p >= 40) return ['Trung bình', 'Cần ôn lại nền tảng trước khi luyện đề tổng hợp.'];
    return ['Cần học lại', 'Bắt đầu từ tab Kiến thức, sau đó luyện từng chủ đề ở chế độ Luyện tập.'];
  }

  // ---------------------------------------------------------------- pool & quiz lifecycle
  function sourceOk(q, src) {
    const p = progress[q.id];
    if (src === 'new') return !p;
    if (src === 'wrong') return !!p && p.last < 1;
    if (src === 'flagged') return flags.has(q.id);
    if (src === 'weak') return !p || p.c / p.n < 0.7;
    return true;
  }
  function buildPool(s) {
    return ALLQ.filter(q => s.topics.includes(q.topic) && s.levels.includes(q.level) && s.types.includes(q.type)
      && (!s.trapOnly || q.trap) && sourceOk(q, s.source));
  }
  function makeItem(q) {
    const perm = q.options ? (q.noShuffle ? range(q.options.length) : shuffle(range(q.options.length))) : null;
    return { id: q.id, perm };
  }
  function startQuiz(questions, s, title) {
    const items = questions.map(makeItem);
    quiz = {
      title: title || 'Đề tổng hợp',
      items, idx: 0, resp: {}, checked: {}, mode: s.mode,
      strictMulti: !!s.strictMulti, negative: !!s.negative,
      startedAt: Date.now(),
      limit: s.mode === 'exam' && s.timePer ? s.timePer * items.length : 0,
    };
    saveQuiz();
    go('quiz');
  }
  function finishQuiz(reason) {
    if (!quiz) return;
    const opts = { strictMulti: quiz.strictMulti, negative: quiz.negative };
    const items = quiz.items.map(it => {
      const q = QMAP[it.id], r = quiz.resp[it.id];
      const g = gradeItem(q, r, opts);
      // practice mode already recorded checked questions
      if (g.answered && !(quiz.mode === 'practice' && quiz.checked[it.id])) recordProgress(it.id, g.frac);
      return Object.assign({ id: it.id, resp: r === undefined ? null : r }, g);
    });
    saveProgress();
    const max = items.reduce((a, b) => a + b.max, 0);
    const raw = items.reduce((a, b) => a + b.pts, 0);
    const score = Math.max(0, raw);
    lastResult = {
      at: Date.now(), title: quiz.title, mode: quiz.mode, reason: reason || 'submit',
      dur: (Date.now() - quiz.startedAt) / 1000, strictMulti: quiz.strictMulti, negative: quiz.negative,
      items, score, max, pct: pct(score, max),
    };
    store.set('last', lastResult);
    history.unshift({
      at: lastResult.at, title: lastResult.title, mode: lastResult.mode, n: items.length,
      ok: items.filter(x => x.status === 'ok').length, score, max, pct: lastResult.pct, dur: lastResult.dur,
      topics: [...new Set(items.map(x => QMAP[x.id].topic))],
    });
    history = history.slice(0, 60);
    store.set('history', history);
    quiz = null; saveQuiz();
    go('result');
  }

  // ---------------------------------------------------------------- routing
  let view = 'home';
  let notesTopic = BANK[0] ? BANK[0].id : null;
  let notesQuery = '';
  let bankFilter = { topic: 'all', level: 0, type: 'all', trap: false, q: '', open: false };
  let reviewFilter = 'all';
  let timerHandle = null;

  function go(v) {
    if (location.hash !== '#' + v) { location.hash = v; return; }
    render();
  }
  window.addEventListener('hashchange', render);

  function render() {
    let v = (location.hash || '#home').slice(1) || 'home';
    if (!['home', 'quiz', 'result', 'notes', 'bank', 'stats'].includes(v)) v = 'home';
    if (v === 'quiz' && !quiz) v = 'home';
    if (v === 'result' && !lastResult) v = 'home';
    view = v;
    clearInterval(timerHandle);
    $$('.tabs a').forEach(a => a.classList.toggle('active', a.dataset.nav === (v === 'quiz' || v === 'result' ? 'home' : v)));
    if (!ALLQ.length) { app.innerHTML = `<div class="panel empty"><h2>Chưa có dữ liệu câu hỏi</h2><p>Chạy <code>python tools/build.py</code> để tạo <code>data/bank.js</code>.</p></div>`; return; }
    ({ home: renderHome, quiz: renderQuiz, result: renderResult, notes: renderNotes, bank: renderBank, stats: renderStats })[v]();
  }

  // ---------------------------------------------------------------- HOME
  function topicMastery(tid) {
    const qs = TOPIC[tid].questions;
    let seen = 0, c = 0, n = 0;
    qs.forEach(q => { const p = progress[q.id]; if (p) { seen++; c += p.c; n += p.n; } });
    return { seen, total: qs.length, acc: n ? (c / n) * 100 : 0 };
  }
  function seg(name, options, current, multi) {
    return `<div class="seg" role="group">${options.map(([val, label]) => {
      const checked = multi ? current.includes(val) : String(current) === String(val);
      return `<label><input type="${multi ? 'checkbox' : 'radio'}" name="${name}" value="${val}" ${checked ? 'checked' : ''}><span>${label}</span></label>`;
    }).join('')}</div>`;
  }
  function renderHome() {
    const seenAll = Object.keys(progress).filter(id => QMAP[id]).length;
    const totC = Object.entries(progress).filter(([id]) => QMAP[id]).reduce((a, [, p]) => a + p.c, 0);
    const totN = Object.entries(progress).filter(([id]) => QMAP[id]).reduce((a, [, p]) => a + p.n, 0);
    const best = history.length ? Math.max(...history.map(h => h.pct)) : null;
    const resume = quiz ? `<div class="panel resume" style="margin-bottom:18px">
        <div><b>Bạn đang làm dở: ${esc(quiz.title)}</b><div class="small muted">${Object.keys(quiz.resp).length}/${quiz.items.length} câu đã trả lời · ${quiz.mode === 'exam' ? 'Thi thử' : 'Luyện tập'}</div></div>
        <span class="spacer"></span>
        <button class="btn primary" data-act="resume">Tiếp tục</button>
        <button class="btn ghost" data-act="discard">Bỏ bài này</button></div>` : '';
    app.innerHTML = `
      ${resume}
      <section class="hero">
        <div>
          <div class="eyebrow">Ôn phỏng vấn AI Engineer</div>
          <h1>Trắc nghiệm lý thuyết & tính toán, có bẫy, có giải thích từng câu</h1>
          <p>${ALLQ.length} câu trong ${BANK.length} chủ đề, trọng tâm NLP, LLM, RAG và Agent. Câu Cơ bản 1 điểm, Trung cấp 2 điểm, Nâng cao 3 điểm.</p>
        </div>
        <div class="hero-stats">
          <div><b>${seenAll}<small class="muted" style="font-size:13px">/${ALLQ.length}</small></b><span>câu đã làm</span></div>
          <div><b>${totN ? Math.round((totC / totN) * 100) + '%' : '—'}</b><span>tỉ lệ đúng</span></div>
          <div><b>${best == null ? '—' : best + '%'}</b><span>điểm cao nhất</span></div>
        </div>
      </section>
      <section class="panel">
        <div class="row" style="margin-bottom:12px">
          <h2 style="font-size:18px">1. Chọn chủ đề</h2><span class="spacer"></span>
          <button class="btn sm ghost" data-act="topics-focus" title="NLP, LLM, Inference, RAG, Agent, Eval">Trọng tâm GenAI</button>
          <button class="btn sm ghost" data-act="topics-all">Chọn hết</button>
          <button class="btn sm ghost" data-act="topics-none">Bỏ chọn</button>
        </div>
        <div class="topics">${BANK.map(t => {
          const m = topicMastery(t.id);
          return `<label class="topic"><input type="checkbox" name="topic" value="${t.id}" ${settings.topics.includes(t.id) ? 'checked' : ''}>
            <span class="ic" aria-hidden="true">${esc(t.icon || '•')}</span>
            <span style="min-width:0;flex:1"><span class="tt">${esc(t.name)}</span>
              <span class="meta num" style="display:block">${t.questions.length} câu · đã làm ${m.seen}${m.seen ? ` · đúng ${Math.round(m.acc)}%` : ''}</span>
              <span class="mastery"><i style="width:${(m.seen / m.total) * 100}%;background:${m.seen ? colorFor(m.acc) : 'var(--accent)'}"></i></span></span>
            <span class="tick" aria-hidden="true">✓</span></label>`;
        }).join('')}</div>

        <h2 style="font-size:18px;margin:26px 0 14px">2. Cấu hình đề</h2>
        <div class="settings">
          <div class="field"><span class="lbl">Chế độ</span>
            ${seg('mode', [['practice', 'Luyện tập'], ['exam', 'Thi thử']], settings.mode)}
            <span class="hint">${settings.mode === 'exam' ? 'Không hiện đáp án cho đến khi nộp bài. Có giới hạn thời gian.' : 'Kiểm tra từng câu, xem giải thích ngay.'}</span></div>
          <div class="field"><span class="lbl">Số câu</span>
            ${seg('count', [[10, '10'], [20, '20'], [30, '30'], [50, '50'], [0, 'Tất cả']], settings.count)}</div>
          <div class="field"><span class="lbl">Độ khó</span>
            ${seg('levels', [[1, 'Cơ bản'], [2, 'Trung cấp'], [3, 'Nâng cao']], settings.levels, true)}</div>
          <div class="field"><span class="lbl">Dạng câu</span>
            ${seg('types', [['single', 'Một đáp án'], ['multi', 'Nhiều đáp án'], ['numeric', 'Tính toán']], settings.types, true)}</div>
          <div class="field"><span class="lbl">Nguồn câu hỏi</span>
            ${seg('source', [['all', 'Tất cả'], ['new', 'Chưa làm'], ['weak', 'Còn yếu'], ['wrong', 'Sai lần trước'], ['flagged', 'Đã đánh dấu']], settings.source)}</div>
          <div class="field"><span class="lbl">Thời gian (thi thử)</span>
            ${seg('timePer', [[60, '60 s/câu'], [90, '90 s/câu'], [150, '150 s/câu'], [0, 'Không giới hạn']], settings.timePer)}</div>
          <div class="field"><span class="lbl">Luật chấm</span>
            <label class="checkrow"><input type="checkbox" id="strictMulti" ${settings.strictMulti ? 'checked' : ''}> Nhiều đáp án: đúng hết mới có điểm</label>
            <label class="checkrow"><input type="checkbox" id="negative" ${settings.negative ? 'checked' : ''}> Trừ ${NEG * 100}% điểm câu khi trả lời sai</label>
            <label class="checkrow"><input type="checkbox" id="trapOnly" ${settings.trapOnly ? 'checked' : ''}> Chỉ lấy câu có bẫy</label>
            <span class="hint">Mặc định câu nhiều đáp án chấm từng phần: (số chọn đúng − số chọn sai) / số đáp án đúng.</span></div>
        </div>
        <div class="startbar">
          <button class="btn primary" data-act="start" id="startBtn">Bắt đầu làm bài</button>
          <span class="pool" id="poolInfo"></span>
        </div>
      </section>`;
    updatePoolInfo();
  }
  function readSettingsFromForm() {
    const val = n => { const el = $(`input[name="${n}"]:checked`, app); return el ? el.value : null; };
    const vals = n => $$(`input[name="${n}"]:checked`, app).map(e => e.value);
    settings.topics = vals('topic');
    settings.mode = val('mode') || 'practice';
    settings.count = Number(val('count') ?? 20);
    settings.levels = vals('levels').map(Number);
    settings.types = vals('types');
    settings.source = val('source') || 'all';
    settings.timePer = Number(val('timePer') ?? 90);
    settings.strictMulti = $('#strictMulti', app).checked;
    settings.negative = $('#negative', app).checked;
    settings.trapOnly = $('#trapOnly', app).checked;
    store.set('settings', settings);
  }
  function updatePoolInfo() {
    const pool = buildPool(settings);
    const n = settings.count ? Math.min(settings.count, pool.length) : pool.length;
    const info = $('#poolInfo'), btn = $('#startBtn');
    if (!info) return;
    info.textContent = pool.length
      ? `Kho phù hợp: ${pool.length} câu → đề gồm ${n} câu${settings.mode === 'exam' && settings.timePer ? ` · ${fmtTime(n * settings.timePer)}` : ''}`
      : 'Không có câu nào khớp bộ lọc. Hãy nới điều kiện.';
    btn.disabled = !pool.length;
  }

  // ---------------------------------------------------------------- QUIZ
  function itemStatusClass(it) {
    const q = QMAP[it.id], r = quiz.resp[it.id];
    if (quiz.mode === 'practice' && quiz.checked[it.id]) return gradeItem(q, r, quiz).status;
    return isAnswered(q, r) ? 'ans' : '';
  }
  function renderOptions(q, it, r, revealed) {
    const multi = q.type === 'multi';
    const ans = new Set(multi ? q.answer : [q.answer]);
    const sel = new Set(multi ? (r || []) : (r == null ? [] : [r]));
    return `<div class="opts" role="${multi ? 'group' : 'radiogroup'}">${it.perm.map((oi, pos) => {
      const isSel = sel.has(oi), isAns = ans.has(oi);
      let cls = 'opt' + (multi ? ' multi' : ''), mark = '';
      if (revealed) {
        if (isSel && isAns) { cls += ' right'; mark = '✓ đúng'; }
        else if (isSel && !isAns) { cls += ' wrong'; mark = '✗ sai'; }
        else if (isAns) { cls += ' missed'; mark = 'đáp án'; }
      } else if (isSel) cls += ' sel';
      return `<button type="button" class="${cls}" data-act="pick" data-oi="${oi}" ${revealed ? 'disabled' : ''}
        role="${multi ? 'checkbox' : 'radio'}" aria-checked="${isSel}">
        <span class="key">${LETTERS[pos]}</span><span class="otext md">${md(q.options[oi])}</span><span class="mark">${mark}</span></button>`;
    }).join('')}</div>`;
  }
  function renderFeedback(q, r) {
    const g = gradeItem(q, r, quiz || { strictMulti: false, negative: false });
    const map = { ok: ['ok', 'Chính xác'], pa: ['pa', 'Đúng một phần'], ko: ['ko', 'Chưa đúng'], na: ['ko', 'Bỏ trống'] };
    const [cls, label] = map[g.status];
    return `<div class="feedback ${cls}" role="status">
      <div class="verdict">${label} <span class="chip num">${g.pts >= 0 ? '+' : ''}${fmtNum(g.pts)} / ${g.max} điểm</span>${q.trap ? '<span class="chip trap">câu có bẫy</span>' : ''}</div>
      ${q.type === 'numeric' ? `<div style="margin-top:6px">Đáp án: <b class="num">${esc(correctText(q))}</b></div>` : ''}
      <div class="explain md">${md(q.explain)}</div></div>`;
  }
  function renderQuiz() {
    const it = quiz.items[quiz.idx], q = QMAP[it.id], r = quiz.resp[it.id];
    const revealed = quiz.mode === 'practice' && !!quiz.checked[it.id];
    const answered = Object.keys(quiz.resp).filter(id => isAnswered(QMAP[id], quiz.resp[id])).length;
    const isLast = quiz.idx === quiz.items.length - 1;
    const t = TOPIC[q.topic];
    let body;
    if (q.type === 'numeric') {
      body = `<div class="numeric"><label class="sr" for="numIn" hidden>Đáp án số</label>
        <input id="numIn" inputmode="decimal" autocomplete="off" placeholder="Nhập số, vd 12.5" value="${esc(r ?? '')}" ${revealed ? 'disabled' : ''}>
        ${q.unit ? `<span class="unit">${esc(q.unit)}</span>` : ''}</div>
        <div class="qtype-hint">Chấp nhận dấu phẩy thập phân (12,5), dạng 1e9 hoặc phân số 1/3.${q.tolerance ? ` Sai số cho phép ± ${fmtNum(q.tolerance)}.` : ''}</div>`;
    } else {
      body = renderOptions(q, it, r, revealed) +
        `<div class="qtype-hint">${q.type === 'multi' ? 'Chọn TẤT CẢ phương án đúng (có thể 2, 3 hoặc nhiều hơn).' : 'Chọn một phương án.'}</div>`;
    }
    const actions = [];
    actions.push(`<button class="btn" data-act="prev" ${quiz.idx === 0 ? 'disabled' : ''}>← Câu trước</button>`);
    if (quiz.mode === 'practice' && !revealed) actions.push(`<button class="btn primary" data-act="check" ${isAnswered(q, r) ? '' : 'disabled'} id="checkBtn">Kiểm tra</button>`);
    if (!isLast) actions.push(`<button class="btn ${quiz.mode === 'practice' && !revealed ? '' : 'primary'}" data-act="next">${quiz.mode === 'practice' && !revealed ? 'Bỏ qua →' : 'Câu tiếp →'}</button>`);
    else actions.push(`<button class="btn primary" data-act="submit">Nộp bài</button>`);
    actions.push(`<span class="spacer"></span><button class="btn ghost sm" data-act="flag">${flags.has(q.id) ? '★ Đã đánh dấu' : '☆ Đánh dấu ôn lại'}</button>`);

    app.innerHTML = `<div class="quiz">
      <section class="panel">
        <div class="qhead">
          <span class="qnum">Câu ${quiz.idx + 1}/${quiz.items.length}</span>
          <span class="chip accent">${esc(t.icon || '')} ${esc(t.name)}</span>
          <span class="chip">${esc(q.sub)}</span>
          <span class="chip l${q.level}">${LEVEL_NAME[q.level]} · ${LEVEL_W[q.level]}đ</span>
          <span class="chip">${TYPE_NAME[q.type]}</span>
        </div>
        <div class="qtext md">${md(q.q)}</div>
        ${body}
        ${revealed ? renderFeedback(q, r) : ''}
        <div class="qactions">${actions.join('')}</div>
      </section>
      <aside class="rail">
        <div class="panel stack" style="gap:12px">
          <div class="row"><div>
            <div class="eyebrow">${quiz.limit ? 'Thời gian còn lại' : 'Thời gian'}</div>
            <div class="timer" id="timer">--:--</div></div>
            <span class="spacer"></span>
            <div style="text-align:right"><div class="eyebrow">Đã làm</div><div class="num" style="font-weight:700">${answered}/${quiz.items.length}</div></div>
          </div>
          <div class="progress"><i style="width:${pct(answered, quiz.items.length)}%"></i></div>
          <div class="grid-nav">${quiz.items.map((x, i) => {
            const st = itemStatusClass(x);
            return `<button type="button" data-act="goto" data-i="${i}" class="${st} ${i === quiz.idx ? 'cur' : ''} ${flags.has(x.id) ? 'flag' : ''}" aria-label="Câu ${i + 1}">${i + 1}</button>`;
          }).join('')}</div>
          <div class="legend">${quiz.mode === 'practice'
            ? '<span><i style="background:var(--good-soft);border-color:var(--good)"></i>Đúng</span><span><i style="background:var(--part-soft);border-color:var(--part)"></i>Một phần</span><span><i style="background:var(--bad-soft);border-color:var(--bad)"></i>Sai</span>'
            : '<span><i style="background:var(--accent-soft);border-color:var(--accent)"></i>Đã trả lời</span>'}<span><i style="background:var(--warn);border-color:var(--warn);border-radius:50%"></i>Đánh dấu</span></div>
          <button class="btn primary" data-act="submit">Nộp bài & chấm điểm</button>
          <button class="btn ghost sm" data-act="quit">Thoát (giữ bài làm dở)</button>
          <div class="kbd-help"><kbd>1</kbd>–<kbd>6</kbd> hoặc <kbd>A</kbd>–<kbd>F</kbd> chọn · <kbd>Enter</kbd> kiểm tra/tiếp · <kbd>←</kbd><kbd>→</kbd> chuyển câu · <kbd>S</kbd> đánh dấu</div>
        </div>
      </aside></div>`;
    tickTimer();
    timerHandle = setInterval(tickTimer, 1000);
    const ni = $('#numIn');
    if (ni && !revealed) {
      ni.addEventListener('input', () => {
        quiz.resp[it.id] = ni.value; saveQuiz();
        const cb = $('#checkBtn'); if (cb) cb.disabled = !ni.value.trim();
      });
      ni.focus();
    }
  }
  function tickTimer() {
    if (!quiz) return;
    const el = $('#timer'); if (!el) return;
    const el2 = (Date.now() - quiz.startedAt) / 1000;
    if (quiz.limit) {
      const left = quiz.limit - el2;
      el.textContent = fmtTime(left);
      el.classList.toggle('low', left < 60);
      if (left <= 0) { clearInterval(timerHandle); finishQuiz('timeout'); }
    } else el.textContent = fmtTime(el2);
  }
  function pick(oi) {
    const it = quiz.items[quiz.idx], q = QMAP[it.id];
    if (quiz.mode === 'practice' && quiz.checked[it.id]) return;
    if (q.type === 'multi') {
      const s = new Set(quiz.resp[it.id] || []);
      s.has(oi) ? s.delete(oi) : s.add(oi);
      quiz.resp[it.id] = [...s];
      if (!s.size) delete quiz.resp[it.id];
    } else quiz.resp[it.id] = oi;
    saveQuiz(); renderQuiz();
  }
  function checkCurrent() {
    const it = quiz.items[quiz.idx], q = QMAP[it.id], r = quiz.resp[it.id];
    if (quiz.mode !== 'practice' || quiz.checked[it.id] || !isAnswered(q, r)) return;
    quiz.checked[it.id] = true;
    recordProgress(it.id, gradeFrac(q, r, quiz.strictMulti)); saveProgress();
    saveQuiz(); renderQuiz();
  }
  function move(d) {
    const n = quiz.idx + d;
    if (n < 0 || n >= quiz.items.length) return;
    quiz.idx = n; saveQuiz(); renderQuiz(); window.scrollTo({ top: 0 });
  }
  function confirmSubmit() {
    const unanswered = quiz.items.filter(it => !isAnswered(QMAP[it.id], quiz.resp[it.id])).length;
    const unchecked = quiz.mode === 'practice' ? quiz.items.filter(it => isAnswered(QMAP[it.id], quiz.resp[it.id]) && !quiz.checked[it.id]).length : 0;
    let msg = unanswered ? `Còn <b>${unanswered}</b> câu bỏ trống, sẽ được tính 0 điểm.` : 'Bạn đã trả lời hết các câu.';
    if (unchecked) msg += ` ${unchecked} câu đã chọn nhưng chưa bấm Kiểm tra sẽ được chấm luôn.`;
    showModal('Nộp bài?', msg, [['Làm tiếp', null], ['Nộp bài', () => finishQuiz('submit'), 'primary']]);
  }

  // ---------------------------------------------------------------- RESULT
  function groupStats(items, keyFn) {
    const g = {};
    items.forEach(x => {
      const k = keyFn(x);
      g[k] = g[k] || { pts: 0, max: 0, n: 0, ok: 0 };
      g[k].pts += Math.max(0, x.pts); g[k].max += x.max; g[k].n++; if (x.status === 'ok') g[k].ok++;
    });
    return g;
  }
  function barsHtml(entries) {
    return `<div class="bars">${entries.map(([name, s]) => {
      const p = pct(s.pts, s.max);
      return `<div class="bar"><span class="name" title="${esc(name)}">${esc(name)}</span>
        <span class="track"><i style="width:${p}%;background:${colorFor(p)}"></i></span>
        <span class="v">${p}% · ${s.ok}/${s.n}</span></div>`;
    }).join('')}</div>`;
  }
  function optionList(q, r) {
    if (q.type === 'numeric') return '';
    const ans = new Set(q.type === 'multi' ? q.answer : [q.answer]);
    const sel = new Set(q.type === 'multi' ? (r || []) : (r == null ? [] : [r]));
    return `<ul class="olist">${q.options.map((o, i) => {
      const c = ans.has(i), s = sel.has(i);
      const mark = c && s ? '✓' : c ? '○' : s ? '✗' : '';
      return `<li class="${c ? 'c' : s ? 'x' : ''}"><span class="num">${mark}</span><span class="md">${md(o)}</span></li>`;
    }).join('')}</ul><div class="small muted" style="margin-top:4px">✓ bạn chọn đúng · ○ đáp án đúng bạn bỏ sót · ✗ bạn chọn sai</div>`;
  }
  function reviewCard(x, i) {
    const q = QMAP[x.id], t = TOPIC[q.topic];
    const label = { ok: 'Đúng', pa: 'Một phần', ko: 'Sai', na: 'Bỏ trống' }[x.status];
    return `<details class="rcard ${x.status}" ${x.status !== 'ok' ? 'open' : ''}>
      <summary>
        <div class="row" style="gap:6px"><span class="qnum">#${i + 1}</span>
          <span class="chip">${esc(t.name)}</span><span class="chip">${esc(q.sub)}</span>
          <span class="chip l${q.level}">${LEVEL_NAME[q.level]}</span>${q.trap ? '<span class="chip trap">bẫy</span>' : ''}
          <span class="spacer"></span><b class="num small">${label} · ${x.pts >= 0 ? '+' : ''}${fmtNum(x.pts)}/${x.max}</b></div>
        <div class="rq md">${md(q.q)}</div>
      </summary>
      ${optionList(q, x.resp)}
      ${q.type === 'numeric' ? `<div class="answer-line"><span>Bạn trả lời: <b class="num">${esc(respText(q, x.resp))}</b></span><span>Đáp án: <b class="num">${esc(correctText(q))}</b></span></div>` : ''}
      <div class="feedback" style="margin-top:12px"><div class="eyebrow">Giải thích</div><div class="explain md">${md(q.explain)}</div></div>
      <div class="row" style="margin-top:10px"><button class="btn sm ghost" data-act="flag-id" data-id="${q.id}">${flags.has(q.id) ? '★ Đã đánh dấu' : '☆ Đánh dấu ôn lại'}</button></div>
    </details>`;
  }
  function renderResult() {
    const R = lastResult;
    const [bname, bdesc] = band(R.pct);
    const items = R.items;
    const cnt = s => items.filter(x => x.status === s).length;
    const traps = items.filter(x => QMAP[x.id].trap);
    const trapOk = traps.filter(x => x.status === 'ok').length;
    const byTopic = groupStats(items, x => TOPIC[QMAP[x.id].topic].name);
    const byLevel = groupStats(items, x => LEVEL_NAME[QMAP[x.id].level]);
    const byType = groupStats(items, x => TYPE_NAME[QMAP[x.id].type]);
    const subs = groupStats(items.filter(x => x.status !== 'ok'), x => QMAP[x.id].sub);
    const weakSubs = Object.entries(subs).sort((a, b) => b[1].n - a[1].n).slice(0, 8);
    const C = 2 * Math.PI * 64;
    const filters = [['all', 'Tất cả'], ['bad', 'Sai & bỏ trống'], ['pa', 'Một phần'], ['ok', 'Đúng'], ['trap', 'Câu bẫy']];
    const shown = items.map((x, i) => [x, i]).filter(([x]) => reviewFilter === 'all' || (reviewFilter === 'bad' && (x.status === 'ko' || x.status === 'na'))
      || (reviewFilter === 'trap' && QMAP[x.id].trap) || x.status === reviewFilter);
    app.innerHTML = `<div class="stack" style="gap:20px">
      <section class="panel score-hero">
        <div class="ring"><svg viewBox="0 0 150 150" aria-hidden="true">
          <circle cx="75" cy="75" r="64" fill="none" stroke="var(--surface-2)" stroke-width="12"/>
          <circle cx="75" cy="75" r="64" fill="none" stroke="${colorFor(R.pct)}" stroke-width="12" stroke-linecap="round"
            stroke-dasharray="${C}" stroke-dashoffset="${C * (1 - R.pct / 100)}"/></svg>
          <div class="val"><div><b>${R.pct}%</b><span class="num">${fmtNum(R.score)}/${R.max} điểm</span></div></div></div>
        <div style="min-width:0">
          <div class="eyebrow">${esc(R.title)} · ${R.mode === 'exam' ? 'Thi thử' : 'Luyện tập'}${R.reason === 'timeout' ? ' · hết giờ, tự nộp' : ''}</div>
          <div class="band" style="color:${colorFor(R.pct)}">${bname}</div>
          <p class="muted" style="margin:4px 0 0">${bdesc}</p>
          <div class="kpis">
            <div><b>${cnt('ok')}/${items.length}</b><span>câu đúng hoàn toàn</span></div>
            <div><b>${cnt('pa')}</b><span>đúng một phần</span></div>
            <div><b>${cnt('ko')} · ${cnt('na')}</b><span>sai · bỏ trống</span></div>
            <div><b>${traps.length ? trapOk + '/' + traps.length : '—'}</b><span>né được bẫy</span></div>
            <div><b>${fmtTime(R.dur)}</b><span>thời gian làm</span></div>
          </div>
          <div class="small muted" style="margin-top:10px">Luật chấm: Cơ bản 1đ · Trung cấp 2đ · Nâng cao 3đ; nhiều đáp án ${R.strictMulti ? 'đúng hết mới có điểm' : 'chấm từng phần'}${R.negative ? `; trả lời sai bị trừ ${NEG * 100}% điểm câu` : ''}.</div>
          <div class="row" style="margin-top:14px">
            <button class="btn primary" data-act="retry-wrong" ${cnt('ok') === items.length ? 'disabled' : ''}>Làm lại các câu chưa đúng</button>
            <button class="btn" data-act="retry-all">Làm lại cả đề</button>
            <button class="btn ghost" data-act="home">Tạo đề mới</button>
          </div>
        </div>
      </section>
      <div class="two">
        <section class="panel"><h3 style="font-size:16px;margin-bottom:14px">Theo chủ đề</h3>${barsHtml(Object.entries(byTopic))}</section>
        <section class="panel stack">
          <div><h3 style="font-size:16px;margin-bottom:14px">Theo độ khó</h3>${barsHtml(Object.entries(byLevel))}</div>
          <div><h3 style="font-size:16px;margin-bottom:14px">Theo dạng câu</h3>${barsHtml(Object.entries(byType))}</div>
        </section>
      </div>
      ${weakSubs.length ? `<section class="panel"><h3 style="font-size:16px;margin-bottom:10px">Mảng kiến thức cần ôn</h3>
        <div class="row">${weakSubs.map(([s, v]) => `<span class="chip trap">${esc(s)} · ${v.n} câu</span>`).join('')}</div></section>` : ''}
      <section class="stack">
        <div class="row"><h3 style="font-size:18px">Xem lại bài làm</h3><span class="spacer"></span>
          ${seg('rfilter', filters, reviewFilter)}</div>
        <div class="review">${shown.length ? shown.map(([x, i]) => reviewCard(x, i)).join('') : '<div class="panel empty">Không có câu nào trong mục này.</div>'}</div>
      </section></div>`;
  }

  // ---------------------------------------------------------------- NOTES
  function highlight(html, qstr) {
    if (!qstr) return html;
    const re = new RegExp('(' + qstr.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
    return html.split(/(<[^>]+>)/g).map(p => (p.startsWith('<') ? p : p.replace(re, '<mark>$1</mark>'))).join('');
  }
  function renderNotes() {
    const t = TOPIC[notesTopic] || BANK[0];
    const qstr = notesQuery.trim().toLowerCase();
    let content;
    if (qstr) {
      const hits = [];
      BANK.forEach(tp => tp.notes.forEach((n, i) => {
        if ((n.title + '\n' + n.md).toLowerCase().includes(qstr)) hits.push([tp, n, i]);
      }));
      content = `<div class="eyebrow">${hits.length} mục khớp “${esc(notesQuery)}”</div>` + (hits.length ? hits.map(([tp, n]) =>
        `<article class="note"><div class="eyebrow">${esc(tp.name)}</div><h2>${highlight(inline(n.title), notesQuery.trim())}</h2><div class="md">${highlight(md(n.md), notesQuery.trim())}</div></article>`).join('')
        : '<div class="empty">Không tìm thấy. Thử từ khoá khác, ví dụ “KV cache”, “BM25”, “LoRA”.</div>');
    } else {
      content = `<div class="row"><div><div class="eyebrow">Tổng hợp kiến thức</div><h1 style="font-size:26px">${esc(t.icon || '')} ${esc(t.name)}</h1></div>
          <span class="spacer"></span><button class="btn primary sm" data-act="practice-topic" data-id="${t.id}">Luyện ${t.questions.length} câu chủ đề này</button></div>
        <nav class="toc">${t.notes.map((n, i) => `<a href="#notes" data-act="toc" data-i="${i}">${esc(n.title)}</a>`).join('')}</nav>
        ${t.notes.map((n, i) => `<article class="note" id="note-${i}"><h2>${inline(n.title)}</h2><div class="md">${md(n.md)}</div></article>`).join('')}`;
    }
    app.innerHTML = `<div class="split">
      <aside class="side">
        <input class="search" id="notesSearch" type="search" placeholder="Tìm trong kiến thức…" value="${esc(notesQuery)}" style="margin-bottom:8px">
        ${BANK.map(tp => `<button type="button" data-act="notes-topic" data-id="${tp.id}" class="${tp.id === t.id && !qstr ? 'active' : ''}">
          <span aria-hidden="true">${esc(tp.icon || '•')}</span><span>${esc(tp.name)}</span><span class="cnt">${tp.notes.length}</span></button>`).join('')}
      </aside>
      <section class="panel" style="min-width:0">${content}</section></div>`;
    const s = $('#notesSearch');
    s.addEventListener('input', debounce(() => { notesQuery = s.value; const pos = s.selectionStart; renderNotes(); const s2 = $('#notesSearch'); s2.focus(); s2.setSelectionRange(pos, pos); }, 200));
  }
  function debounce(fn, ms) { let h; return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); }; }

  // ---------------------------------------------------------------- BANK (browse)
  function renderBank() {
    const f = bankFilter;
    const list = bankList();
    const LIMIT = 80;
    app.innerHTML = `<div class="stack">
      <div><div class="eyebrow">Ngân hàng câu hỏi</div><h1 style="font-size:26px">Duyệt & học theo câu</h1>
        <p class="muted" style="margin:4px 0 0">Mở từng câu để xem đáp án và giải thích. Dùng để học, không tính điểm.</p></div>
      <section class="panel stack" style="gap:12px">
        <input class="search" id="bankSearch" type="search" placeholder="Tìm theo nội dung, ví dụ: RRF, LoRA, MCP, IoU…" value="${esc(f.q)}">
        <div class="row">
          ${seg('btopic', [['all', 'Tất cả'], ...BANK.map(t => [t.id, esc(t.name.split(/[:,&]/)[0].trim())])], f.topic)}
        </div>
        <div class="row">
          ${seg('blevel', [[0, 'Mọi độ khó'], [1, 'Cơ bản'], [2, 'Trung cấp'], [3, 'Nâng cao']], f.level)}
          ${seg('btype', [['all', 'Mọi dạng'], ['single', 'Một đáp án'], ['multi', 'Nhiều đáp án'], ['numeric', 'Tính toán']], f.type)}
          <label class="checkrow"><input type="checkbox" id="btrap" ${f.trap ? 'checked' : ''}> Chỉ câu bẫy</label>
          <label class="checkrow"><input type="checkbox" id="bopen" ${f.open ? 'checked' : ''}> Mở sẵn đáp án</label>
          <span class="spacer"></span>
          <button class="btn sm primary" data-act="practice-list" ${list.length ? '' : 'disabled'}>Luyện ${Math.min(list.length, 50)} câu này</button>
        </div>
        <div class="pool">${list.length} câu khớp${list.length > LIMIT ? ` · hiển thị ${LIMIT} câu đầu, hãy lọc thêm` : ''}</div>
      </section>
      <div class="review">${list.slice(0, LIMIT).map(q => {
        const t = TOPIC[q.topic], p = progress[q.id];
        return `<details class="rcard ${p ? (p.last >= 1 ? 'ok' : p.last > 0 ? 'pa' : 'ko') : 'na'}" ${f.open ? 'open' : ''}>
          <summary><div class="row" style="gap:6px"><span class="qnum">${esc(q.id)}</span><span class="chip">${esc(t.name)}</span><span class="chip">${esc(q.sub)}</span>
            <span class="chip l${q.level}">${LEVEL_NAME[q.level]}</span><span class="chip">${TYPE_NAME[q.type]}</span>${q.trap ? '<span class="chip trap">bẫy</span>' : ''}
            <span class="spacer"></span>${p ? `<span class="small muted num">làm ${p.n} lần · ${Math.round((p.c / p.n) * 100)}%</span>` : ''}</div>
            <div class="rq md">${highlight(md(q.q), f.q.trim())}</div></summary>
          ${optionList(q, null).replace(/<div class="small muted"[\s\S]*$/, '')}
          ${q.type === 'numeric' ? `<div class="answer-line"><span>Đáp án: <b class="num">${esc(correctText(q))}</b></span></div>` : ''}
          <div class="feedback" style="margin-top:12px"><div class="eyebrow">Giải thích</div><div class="explain md">${md(q.explain)}</div></div>
          <div class="row" style="margin-top:10px"><button class="btn sm ghost" data-act="flag-id" data-id="${q.id}">${flags.has(q.id) ? '★ Đã đánh dấu' : '☆ Đánh dấu ôn lại'}</button></div>
        </details>`;
      }).join('') || '<div class="panel empty">Không có câu nào khớp bộ lọc.</div>'}</div></div>`;
    const s = $('#bankSearch');
    s.addEventListener('input', debounce(() => { bankFilter.q = s.value; const pos = s.selectionStart; renderBank(); const s2 = $('#bankSearch'); s2.focus(); s2.setSelectionRange(pos, pos); }, 250));
  }
  function bankList() {
    const f = bankFilter, qstr = f.q.trim().toLowerCase();
    return ALLQ.filter(q => (f.topic === 'all' || q.topic === f.topic) && (!f.level || q.level === f.level)
      && (f.type === 'all' || q.type === f.type) && (!f.trap || q.trap)
      && (!qstr || (q.q + ' ' + q.sub + ' ' + (q.options || []).join(' ') + ' ' + q.explain).toLowerCase().includes(qstr)));
  }

  // ---------------------------------------------------------------- STATS
  function renderStats() {
    const entries = Object.entries(progress).filter(([id]) => QMAP[id]);
    const subAgg = {};
    entries.forEach(([id, p]) => {
      const q = QMAP[id], k = TOPIC[q.topic].name + ' › ' + q.sub;
      subAgg[k] = subAgg[k] || { c: 0, n: 0, qs: 0 };
      subAgg[k].c += p.c; subAgg[k].n += p.n; subAgg[k].qs++;
    });
    const weak = Object.entries(subAgg).filter(([, v]) => v.n >= 2).map(([k, v]) => [k, (v.c / v.n) * 100, v])
      .sort((a, b) => a[1] - b[1]).slice(0, 12);
    const trend = history.slice(0, 20).reverse();
    const W = 640, H = 160, P = 28;
    const pts = trend.map((h, i) => [P + (trend.length === 1 ? (W - 2 * P) / 2 : (i * (W - 2 * P)) / (trend.length - 1)), H - P - (h.pct / 100) * (H - 2 * P)]);
    const chart = trend.length ? `<div class="scroll-x"><svg viewBox="0 0 ${W} ${H}" style="width:100%;min-width:420px;height:auto" role="img" aria-label="Điểm ${trend.length} lần làm gần nhất">
        ${[0, 50, 100].map(v => { const y = H - P - (v / 100) * (H - 2 * P); return `<line x1="${P}" x2="${W - P}" y1="${y}" y2="${y}" stroke="var(--line)" stroke-dasharray="3 4"/><text x="${P - 6}" y="${y + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${v}</text>`; }).join('')}
        ${pts.length > 1 ? `<path d="M${pts.map(p => p.join(',')).join(' L')} L${pts[pts.length - 1][0]},${H - P} L${pts[0][0]},${H - P} Z" fill="var(--accent-soft)"/>
        <polyline points="${pts.map(p => p.join(',')).join(' ')}" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linejoin="round"/>` : ''}
        ${pts.map((p, i) => `<circle cx="${p[0]}" cy="${p[1]}" r="${i === pts.length - 1 ? 5 : 3}" fill="${i === pts.length - 1 ? 'var(--accent)' : 'var(--surface)'}" stroke="var(--accent)" stroke-width="2"><title>${trend[i].pct}% · ${fmtDate(trend[i].at)}</title></circle>`).join('')}
        ${pts.length ? `<text x="${pts[pts.length - 1][0]}" y="${pts[pts.length - 1][1] - 10}" text-anchor="middle" font-size="12" font-weight="700" fill="var(--ink)">${trend[trend.length - 1].pct}%</text>` : ''}
      </svg></div>` : '<div class="empty">Chưa có lần làm bài nào.</div>';
    app.innerHTML = `<div class="stack" style="gap:20px">
      <div class="row"><div><div class="eyebrow">Thống kê cá nhân (lưu trên trình duyệt này)</div><h1 style="font-size:26px">Tiến độ ôn tập</h1></div>
        <span class="spacer"></span><button class="btn danger sm" data-act="reset">Xoá toàn bộ tiến độ</button></div>
      <section class="panel"><h3 style="font-size:16px;margin-bottom:10px">Điểm các lần làm gần nhất</h3>${chart}</section>
      <div class="two">
        <section class="panel"><h3 style="font-size:16px;margin-bottom:14px">Mức thành thạo theo chủ đề</h3>
          <div class="bars">${BANK.map(t => {
            const m = topicMastery(t.id);
            return `<div class="bar"><span class="name">${esc(t.icon || '')} ${esc(t.name)}</span>
              <span class="track"><i style="width:${m.seen ? m.acc : 0}%;background:${colorFor(m.acc)}"></i></span>
              <span class="v">${m.seen ? Math.round(m.acc) + '%' : '—'} · ${m.seen}/${m.total}</span></div>`;
          }).join('')}</div>
          <p class="small muted">Tỉ lệ đúng trung bình trên các câu đã làm · số câu đã làm/tổng.</p></section>
        <section class="panel"><h3 style="font-size:16px;margin-bottom:14px">Mảng yếu nhất (≥ 2 lượt làm)</h3>
          ${weak.length ? `<div class="bars">${weak.map(([k, p, v]) => `<div class="bar"><span class="name" title="${esc(k)}">${esc(k)}</span>
            <span class="track"><i style="width:${p}%;background:${colorFor(p)}"></i></span><span class="v">${Math.round(p)}% · ${v.qs} câu</span></div>`).join('')}</div>`
            : '<div class="empty">Làm thêm vài đề để thấy mảng yếu.</div>'}
          <div class="row" style="margin-top:14px"><button class="btn sm" data-act="practice-weak">Luyện 20 câu còn yếu</button>
          <button class="btn sm" data-act="practice-flagged" ${flags.size ? '' : 'disabled'}>Luyện ${flags.size} câu đã đánh dấu</button></div></section>
      </div>
      <section class="panel"><h3 style="font-size:16px;margin-bottom:10px">Lịch sử</h3>
        ${history.length ? `<div class="scroll-x"><table class="hist"><thead><tr><th>Thời điểm</th><th>Đề</th><th>Chế độ</th><th class="num">Số câu</th><th class="num">Đúng</th><th class="num">Điểm</th><th class="num">%</th><th class="num">Thời gian</th></tr></thead>
          <tbody>${history.map(h => `<tr><td>${fmtDate(h.at)}</td><td>${esc(h.title)}</td><td>${h.mode === 'exam' ? 'Thi thử' : 'Luyện tập'}</td><td class="num">${h.n}</td><td class="num">${h.ok}</td>
            <td class="num">${fmtNum(h.score)}/${h.max}</td><td class="num" style="color:${colorFor(h.pct)};font-weight:700">${h.pct}%</td><td class="num">${fmtTime(h.dur)}</td></tr>`).join('')}</tbody></table></div>`
          : '<div class="empty">Chưa có lịch sử.</div>'}</section></div>`;
  }

  // ---------------------------------------------------------------- modal
  function showModal(title, html, buttons) {
    const m = $('#modal');
    m.innerHTML = `<div class="box" role="dialog" aria-modal="true" aria-labelledby="mTitle"><h3 id="mTitle">${esc(title)}</h3><div>${html}</div>
      <div class="row">${buttons.map((b, i) => `<button class="btn ${b[2] || ''}" data-mi="${i}">${esc(b[0])}</button>`).join('')}</div></div>`;
    m.hidden = false;
    const close = () => { m.hidden = true; m.innerHTML = ''; };
    m.onclick = e => {
      if (e.target === m) return close();
      const b = e.target.closest('[data-mi]'); if (!b) return;
      const fn = buttons[Number(b.dataset.mi)][1]; close(); if (fn) fn();
    };
    const last = m.querySelector('[data-mi]:last-child'); if (last) last.focus();
  }

  // ---------------------------------------------------------------- events
  const FOCUS = ['nlp', 'llm', 'infer', 'rag', 'agent', 'eval'];
  function practiceSet(qs, title, n) {
    const s = Object.assign({}, settings, { mode: 'practice' });
    startQuiz(shuffle(qs).slice(0, n || qs.length), s, title);
  }
  app.addEventListener('change', e => {
    const el = e.target;
    if (view === 'home') {
      readSettingsFromForm();
      if (el.name === 'mode') renderHome(); else updatePoolInfo();
    } else if (view === 'result' && el.name === 'rfilter') { reviewFilter = el.value; renderResult(); }
    else if (view === 'bank') {
      if (el.name === 'btopic') bankFilter.topic = el.value;
      if (el.name === 'blevel') bankFilter.level = Number(el.value);
      if (el.name === 'btype') bankFilter.type = el.value;
      if (el.id === 'btrap') bankFilter.trap = el.checked;
      if (el.id === 'bopen') bankFilter.open = el.checked;
      renderBank();
    }
  });
  app.addEventListener('click', e => {
    const b = e.target.closest('[data-act]'); if (!b) return;
    const act = b.dataset.act;
    switch (act) {
      case 'topics-all': case 'topics-none': case 'topics-focus':
        $$('input[name="topic"]', app).forEach(c => { c.checked = act === 'topics-all' || (act === 'topics-focus' && FOCUS.includes(c.value)); });
        readSettingsFromForm(); updatePoolInfo(); break;
      case 'start': {
        readSettingsFromForm();
        const pool = buildPool(settings); if (!pool.length) return;
        const n = settings.count ? Math.min(settings.count, pool.length) : pool.length;
        const names = settings.topics.length === BANK.length ? 'Tất cả chủ đề' : settings.topics.map(id => TOPIC[id].name.split(/[:,&]/)[0].trim()).join(', ');
        startQuiz(shuffle(pool).slice(0, n), settings, names);
        break;
      }
      case 'resume': go('quiz'); break;
      case 'discard': showModal('Bỏ bài đang làm?', 'Các câu trả lời chưa nộp sẽ mất.', [['Giữ lại', null], ['Bỏ bài', () => { quiz = null; saveQuiz(); renderHome(); }, 'danger']]); break;
      case 'pick': pick(Number(b.dataset.oi)); break;
      case 'check': checkCurrent(); break;
      case 'next': move(1); break;
      case 'prev': move(-1); break;
      case 'goto': quiz.idx = Number(b.dataset.i); saveQuiz(); renderQuiz(); break;
      case 'flag': { const id = quiz.items[quiz.idx].id; flags.has(id) ? flags.delete(id) : flags.add(id); saveFlags(); renderQuiz(); break; }
      case 'flag-id': { const id = b.dataset.id; flags.has(id) ? flags.delete(id) : flags.add(id); saveFlags(); b.textContent = flags.has(id) ? '★ Đã đánh dấu' : '☆ Đánh dấu ôn lại'; break; }
      case 'submit': confirmSubmit(); break;
      case 'quit': go('home'); break;
      case 'home': go('home'); break;
      case 'retry-wrong': {
        const qs = lastResult.items.filter(x => x.status !== 'ok').map(x => QMAP[x.id]);
        startQuiz(shuffle(qs), Object.assign({}, settings, { mode: lastResult.mode, strictMulti: lastResult.strictMulti, negative: lastResult.negative }), 'Làm lại câu chưa đúng');
        break;
      }
      case 'retry-all': {
        const qs = lastResult.items.map(x => QMAP[x.id]);
        startQuiz(shuffle(qs), Object.assign({}, settings, { mode: lastResult.mode, strictMulti: lastResult.strictMulti, negative: lastResult.negative }), lastResult.title);
        break;
      }
      case 'notes-topic': notesTopic = b.dataset.id; notesQuery = ''; renderNotes(); window.scrollTo({ top: 0 }); break;
      case 'toc': { e.preventDefault(); const el = $('#note-' + b.dataset.i); if (el) el.scrollIntoView({ behavior: 'smooth' }); break; }
      case 'practice-topic': practiceSet(TOPIC[b.dataset.id].questions, TOPIC[b.dataset.id].name, 20); break;
      case 'practice-list': practiceSet(bankList(), 'Luyện từ ngân hàng câu', 50); break;
      case 'practice-weak': {
        const qs = ALLQ.filter(q => { const p = progress[q.id]; return p && p.c / p.n < 0.7; });
        const pool = qs.length >= 5 ? qs : qs.concat(shuffle(ALLQ.filter(q => !progress[q.id])));
        practiceSet(pool.slice(0, Math.max(qs.length, 20)), 'Ôn câu còn yếu', 20); break;
      }
      case 'practice-flagged': practiceSet([...flags].filter(id => QMAP[id]).map(id => QMAP[id]), 'Câu đã đánh dấu'); break;
      case 'reset': showModal('Xoá toàn bộ tiến độ?', 'Lịch sử, tỉ lệ đúng từng câu và câu đánh dấu trên trình duyệt này sẽ bị xoá. Không thể hoàn tác.',
        [['Huỷ', null], ['Xoá hết', () => {
          progress = {}; history = []; flags = new Set(); lastResult = null;
          ['progress', 'history', 'flags', 'last'].forEach(store.del); renderStats();
        }, 'danger']]); break;
    }
  });
  document.addEventListener('keydown', e => {
    if (!$('#modal').hidden) { if (e.key === 'Escape') { $('#modal').hidden = true; } return; }
    if (view !== 'quiz' || !quiz || e.ctrlKey || e.metaKey || e.altKey) return;
    const inInput = e.target.tagName === 'INPUT';
    const it = quiz.items[quiz.idx], q = QMAP[it.id];
    if (e.key === 'Enter') {
      e.preventDefault();
      const revealed = quiz.checked[it.id];
      if (quiz.mode === 'practice' && !revealed && isAnswered(q, quiz.resp[it.id])) checkCurrent();
      else if (quiz.idx < quiz.items.length - 1) move(1);
      else confirmSubmit();
      return;
    }
    if (inInput) return;
    if (e.key === 'ArrowRight') move(1);
    else if (e.key === 'ArrowLeft') move(-1);
    else if (e.key.toLowerCase() === 's') { flags.has(it.id) ? flags.delete(it.id) : flags.add(it.id); saveFlags(); renderQuiz(); }
    else if (it.perm) {
      let pos = -1;
      if (/^[1-6]$/.test(e.key)) pos = Number(e.key) - 1;
      else if (/^[a-f]$/i.test(e.key)) pos = LETTERS.indexOf(e.key.toUpperCase());
      if (pos >= 0 && pos < it.perm.length) pick(it.perm[pos]);
    }
  });
  $$('.topbar [data-nav]').forEach(a => a.addEventListener('click', () => { if (view === 'quiz') saveQuiz(); }));

  // theme toggle (explicit choice overrides system)
  const themeBtn = $('#themeBtn');
  const applyTheme = t => { if (t) document.documentElement.setAttribute('data-theme', t); else document.documentElement.removeAttribute('data-theme'); };
  applyTheme(store.get('theme', null));
  themeBtn.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme')
      || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    applyTheme(next); store.set('theme', next);
  });

  // exposed for tests
  window.__quiz = { md, parseNum, gradeFrac, gradeItem };
  render();
})();
