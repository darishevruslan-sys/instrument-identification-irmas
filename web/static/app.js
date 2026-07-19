// ====== State ======
let currentFile = null;
let analyzeData = null; // last /api/analyze response (contains probabilities)
let probChart = null;
let currentFeature = 'mel';
let currentMode = 'quick'; // 'quick' or 'full'

// ====== DOM ======
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const uploadError = document.getElementById('upload-error');
const audioCard = document.getElementById('audio-card');
const featureCard = document.getElementById('feature-card');
const resultsCard = document.getElementById('results-card');
const pipelineCard = document.getElementById('pipeline-card');
const modelStatusPill = document.getElementById('model-status-pill');
const progressWrap = document.getElementById('progress-wrap');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const fullOptions = document.getElementById('full-options');

// ====== Model info ======
async function loadModelInfo() {
    try {
        const res = await fetch('/api/model-info');
        const info = await res.json();
        const badge = document.querySelector('.hero-badge');
        if (info.ready) {
            modelStatusPill.textContent =
                `Модель готова · ${info.model_size} · ${info.feature_type.toUpperCase()} · ${info.class_codes.length} классов`;
            badge.classList.add('ready');
        } else {
            modelStatusPill.textContent = 'Модель не загружена';
            badge.classList.add('error');
            // show a non-blocking hint in the upload card
            const warn = document.createElement('div');
            warn.className = 'alert alert-warning';
            warn.style.marginTop = '16px';
            warn.innerHTML = '<strong>Checkpoint не найден.</strong> ' + info.message;
            document.getElementById('upload-card').appendChild(warn);
        }
    } catch (e) {
        modelStatusPill.textContent = 'Сервер недоступен';
        document.querySelector('.hero-badge').classList.add('error');
    }
}
loadModelInfo();

// ====== Upload handling ======
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function showError(msg) {
    uploadError.textContent = msg;
    uploadError.classList.remove('hidden');
}
function clearError() {
    uploadError.classList.add('hidden');
}

function handleFile(file) {
    clearError();
    // basic client-side check
    const allowed = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.aif'];
    const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
    if (!allowed.includes(ext)) {
        showError(`Неподдерживаемый формат: ${ext}. Поддерживаются: ${allowed.join(', ')}`);
        return;
    }
    currentFile = file;
    // reset previous results
    analyzeData = null;
    [audioCard, featureCard, resultsCard, pipelineCard].forEach(c => c.classList.add('hidden'));
    progressWrap.classList.add('hidden');

    // kick off the analysis based on current mode
    runAnalyze(file, currentMode);
}

// ====== Mode tabs ======
document.querySelectorAll('#mode-tabs .tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('#mode-tabs .tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentMode = tab.dataset.mode;
        fullOptions.classList.toggle('hidden', currentMode !== 'full');
        // if we have a file already, re-analyze in the new mode
        if (currentFile) {
            analyzeData = null;
            [audioCard, featureCard, resultsCard, pipelineCard].forEach(c => c.classList.add('hidden'));
            runAnalyze(currentFile, currentMode);
        }
    });
});

// ====== /api/analyze or /api/analyze-full ======
function showProgress(show, text, determinate, pct) {
    if (show) {
        progressWrap.classList.remove('hidden');
        progressText.textContent = text || 'Анализ…';
        if (determinate) {
            progressFill.classList.remove('indeterminate');
            progressFill.style.width = (pct || 0) + '%';
        } else {
            progressFill.classList.add('indeterminate');
            progressFill.style.width = '';
        }
    } else {
        progressWrap.classList.add('hidden');
    }
}

async function runAnalyze(file, mode) {
    const fd = new FormData();
    fd.append('file', file);

    // show skeleton state on cards
    audioCard.classList.remove('hidden');
    featureCard.classList.remove('hidden');
    resultsCard.classList.remove('hidden');
    pipelineCard.classList.remove('hidden');
    setAllLoading(true);
    showProgress(true, mode === 'full' ? 'Анализ всей песни…' : 'Анализ 3 секунд…', false);

    try {
        let url, options;
        if (mode === 'full') {
            const hop = document.getElementById('hop-duration').value;
            fd.append('hop_duration', hop);
            url = '/api/analyze-full';
            options = { method: 'POST', body: fd };
        } else {
            url = '/api/analyze';
            options = { method: 'POST', body: fd };
        }

        const res = await fetch(url, options);
        if (!res.ok) {
            let detail = 'Ошибка при анализе файла.';
            try { const j = await res.json(); detail = j.detail || detail; } catch (_) {}
            throw new Error(detail);
        }
        analyzeData = await res.json();
        showProgress(false);
        renderAudio(analyzeData, mode);
        renderResults(analyzeData, parseFloat(document.getElementById('threshold').value), mode);
        renderPipeline(analyzeData.pipeline_steps);
        // build feature image for the default tab
        requestFeatureImage(currentFeature);
    } catch (err) {
        showProgress(false);
        setAllLoading(false);
        [audioCard, featureCard, resultsCard, pipelineCard].forEach(c => c.classList.add('hidden'));
        showError(err.message);
    }
}

function setAllLoading(loading) {
    if (loading) {
        document.getElementById('meta-grid').innerHTML = '<div class="skeleton" style="padding:20px">Анализ…</div>';
        document.getElementById('results-top').innerHTML = '';
        document.getElementById('instrument-table').innerHTML = '<div class="skeleton" style="padding:20px">Прогон модели…</div>';
        document.getElementById('pipeline-list').innerHTML = '';
        document.getElementById('feature-skeleton').textContent = 'Строим визуализацию…';
        document.getElementById('feature-skeleton').classList.remove('hidden');
        document.getElementById('feature-image').classList.add('hidden');
    }
}

// ====== Render: audio metadata ======
function renderAudio(data, mode) {
    document.getElementById('audio-player').src = URL.createObjectURL(currentFile);
    document.getElementById('audio-name').textContent = data.filename;

    const raw = data.raw;
    const m = data.model;
    const grid = document.getElementById('meta-grid');

    // In "full" mode, show sliding-window info instead of crop/pad
    const sw = data.sliding_window;
    const isFull = mode === 'full' || (data.mode === 'full');

    if (isFull && sw) {
        grid.innerHTML = `
            <div class="meta-item"><div class="label">Длительность</div><div class="value">${sw.total_duration} с</div></div>
            <div class="meta-item"><div class="label">Sample rate</div><div class="value">${raw.sample_rate} Гц</div></div>
            <div class="meta-item"><div class="label">Окон обработано</div><div class="value">${sw.num_windows}</div></div>
            <div class="meta-item"><div class="label">Окно / шаг</div><div class="value">${sw.window_duration} с / ${sw.hop_duration} с</div></div>
        `;
        const hint = document.getElementById('audio-crop-hint');
        hint.innerHTML = `<span class="mode-badge full">🎵 Вся песня · sliding window</span>`;
    } else {
        grid.innerHTML = `
            <div class="meta-item"><div class="label">Длительность (исх.)</div><div class="value">${raw.duration} с</div></div>
            <div class="meta-item"><div class="label">Sample rate</div><div class="value">${raw.sample_rate} Гц</div></div>
            <div class="meta-item"><div class="label">Каналы</div><div class="value">${raw.channels} (${raw.mono ? 'моно' : 'стерео'})</div></div>
            <div class="meta-item"><div class="label">Модель видит</div><div class="value">${m.duration} с · ${m.target_samples} сэмпл.</div></div>
        `;

        // crop/pad hint
        const hint = document.getElementById('audio-crop-hint');
        if (m.was_cropped) {
            hint.innerHTML = `<span class="crop-badge">⌤ crop до ${m.duration} с</span>`;
        } else if (m.was_padded) {
            hint.innerHTML = `<span class="crop-badge">⌥ pad до ${m.duration} с</span>`;
        } else {
            hint.innerHTML = `<span class="crop-badge neutral">✓ ровно ${m.duration} с</span>`;
        }
    }
}

// ====== Render: results ======
function renderResults(data, threshold, mode) {
    const classes = data.classes; // sorted desc by prob
    const isFull = mode === 'full' || (data.mode === 'full');

    // top predicted
    const top = document.getElementById('results-top');
    const predicted = classes.filter(c => c.probability >= threshold);
    let topHtml = '';
    if (isFull && data.sliding_window) {
        topHtml += `<span class="mode-badge full">Вся песня · ${data.sliding_window.num_windows} окон</span>`;
    }
    if (predicted.length) {
        topHtml += predicted.map(c =>
            `<span class="result-pill">🎵 ${c.name} <span class="pct">${(c.probability * 100).toFixed(1)}%</span></span>`
        ).join('');
    } else {
        topHtml += `<div class="no-predict">Ни один инструмент не превзошёл threshold ${threshold.toFixed(2)}. Снизьте порог, чтобы увидеть предсказания.</div>`;
    }
    top.innerHTML = topHtml;

    // bar chart
    renderChart(classes, threshold);

    // detailed table
    const table = document.getElementById('instrument-table');
    table.innerHTML = classes.map((c, i) => {
        const on = c.probability >= threshold;
        const widthPct = Math.max(1, Math.min(100, c.probability * 100)).toFixed(1);
        return `
        <div class="inst-row ${on ? 'predicted' : ''}">
            <div class="inst-rank">${i + 1}</div>
            <div class="inst-body">
                <div class="inst-name">${c.name} <span class="inst-code">${c.code}</span></div>
                <div class="inst-bar-wrap"><div class="inst-bar" style="width:${widthPct}%"></div></div>
            </div>
            <div class="inst-numbers">
                <div class="inst-prob">${(c.probability * 100).toFixed(1)}%</div>
                <div class="inst-logit">logit ${c.logit >= 0 ? '+' : ''}${c.logit.toFixed(3)}</div>
                <span class="inst-status ${on ? 'on' : 'off'}">${on ? '✓ predicted' : 'not predicted'}</span>
            </div>
        </div>`;
    }).join('');
}

function renderChart(classes, threshold) {
    const ctx = document.getElementById('prob-chart').getContext('2d');
    const labels = classes.map(c => c.name);
    const probs = classes.map(c => c.probability * 100);
    const colors = classes.map(c => c.probability >= threshold ? '#16a34a' : '#6366f1');
    if (probChart) probChart.destroy();
    probChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Вероятность, %',
                data: probs,
                backgroundColor: colors,
                borderRadius: 6,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `p = ${(ctx.parsed.y / 100).toFixed(4)}`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true, max: 100,
                    ticks: { callback: (v) => v + '%', color: '#94a3b8', font: { size: 11 } },
                    grid: { color: '#e2e8f0' }
                },
                x: {
                    ticks: { color: '#475569', font: { size: 11, weight: '600' } },
                    grid: { display: false }
                }
            },
            animation: { duration: 400 }
        }
    });
}

// ====== Threshold slider (re-render without re-querying the model) ======
const thresholdInput = document.getElementById('threshold');
const thresholdVal = document.getElementById('threshold-val');
thresholdInput.addEventListener('input', (e) => {
    const t = parseFloat(e.target.value);
    thresholdVal.textContent = t.toFixed(2);
    if (analyzeData) renderResults(analyzeData, t, analyzeData.mode || currentMode);
});

// ====== Render: pipeline (accordion with expandable explanations) ======
const PIPELINE_CHEVRON = '<svg class="pipe-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>';

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    }[ch]));
}

function pipelineLineClass(line) {
    const text = String(line ?? '').trim();
    if (/^[•-]\s/.test(text)) return 'pipe-detail-line bullet';
    if (/^\d+\./.test(text)) return 'pipe-detail-line numbered';
    if (text.endsWith(':')) return 'pipe-detail-line lead';
    return 'pipe-detail-line';
}

function formatPipelineLine(line) {
    return escapeHtml(String(line ?? '').trim()).replace(/`([^`]+)`/g, '<code>$1</code>');
}

function renderPipeline(steps) {
    const list = document.getElementById('pipeline-list');
    list.innerHTML = steps.map(s => {
        const hasExplanation = s.explanation && s.explanation.length > 0;
        const explanationHtml = hasExplanation ? s.explanation.map(line => {
            return `<p class="${pipelineLineClass(line)}">${formatPipelineLine(line)}</p>`;
        }).join('') : '';
        const headContent = `
                <span class="pipe-num" aria-hidden="true">${s.step}</span>
                <span class="pipe-content">
                    <span class="pipe-title-row">
                        <span class="pipe-title">${escapeHtml(s.title)}</span>
                        ${hasExplanation ? `<span class="pipe-action"><span class="pipe-state">Подробнее</span>${PIPELINE_CHEVRON}</span>` : ''}
                    </span>
                    <span class="pipe-detail">${escapeHtml(s.detail)}</span>
                </span>
        `;

        return `
        <div class="pipe-step" data-step="${s.step}">
            ${hasExplanation
                ? `<button type="button" class="pipe-head" aria-expanded="false">${headContent}</button>`
                : `<div class="pipe-head pipe-head-static">${headContent}</div>`}
            ${hasExplanation ? `<div class="pipe-body"><div class="pipe-body-inner">${explanationHtml}</div></div>` : ''}
        </div>`;
    }).join('');

    // Attach click handlers for accordion
    list.querySelectorAll('.pipe-step').forEach(stepEl => {
        const head = stepEl.querySelector('.pipe-head');
        if (!stepEl.querySelector('.pipe-body') || !head) return;
        head.addEventListener('click', () => {
            const isOpen = stepEl.classList.toggle('open');
            head.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
            const state = head.querySelector('.pipe-state');
            if (state) state.textContent = isOpen ? 'Свернуть' : 'Подробнее';
        });
    });
}

// ====== Feature image tabs ======
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentFeature = tab.dataset.feature;
        requestFeatureImage(currentFeature);
    });
});

async function requestFeatureImage(featureType) {
    if (!currentFile) return;
    const sk = document.getElementById('feature-skeleton');
    const img = document.getElementById('feature-image');
    sk.textContent = `Строим ${featureType === 'mel' ? 'Mel-spectrogram' : featureType === 'mfcc' ? 'MFCC' : 'Waveform'}…`;
    sk.classList.remove('hidden');
    img.classList.add('hidden');

    const fd = new FormData();
    fd.append('file', currentFile);
    fd.append('feature_type', featureType);

    try {
        const res = await fetch('/api/feature-image', { method: 'POST', body: fd });
        if (!res.ok) {
            let detail = 'Ошибка построения визуализации.';
            try { const j = await res.json(); detail = j.detail || detail; } catch (_) {}
            throw new Error(detail);
        }
        const data = await res.json();
        img.src = data.image;
        img.classList.remove('hidden');
        sk.classList.add('hidden');
        const captions = {
            mel: 'Mel-спектрограмма (логарифмическая шкала дБ) — основное представление, которое «видит» CNN.',
            mfcc: 'MFCC — мел-кепстральные коэффициенты, компактное описание тембра звука.',
            waveform: 'Осциллограмма — изменение амплитуды сигнала во времени.'
        };
        document.getElementById('feature-caption').textContent = captions[featureType] || '';
    } catch (err) {
        sk.textContent = 'Ошибка: ' + err.message;
    }
}
