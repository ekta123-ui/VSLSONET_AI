const page = document.body.dataset.page;
const charts = new Map();
const palette = ['#fa4f58', '#55d8c0', '#eab36b', '#75aaff', '#c9a5ff', '#f18b50'];

async function getJson(url) {
  const response = await fetch(url, { headers: { Accept: 'application/json' } });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function notify(message) {
  const notice = document.getElementById('page-notice');
  if (!notice) return;
  notice.textContent = message;
  notice.hidden = false;
}

function formatNumber(value) {
  return value === null || value === undefined || Number.isNaN(Number(value))
    ? 'N/A'
    : new Intl.NumberFormat().format(Number(value));
}

function formatPercent(value) {
  return value === null || value === undefined || Number.isNaN(Number(value))
    ? 'N/A'
    : `${(Number(value) * 100).toFixed(1)}%`;
}

function formatMetric(value, digits = 1) {
  return value === null || value === undefined || Number.isNaN(Number(value))
    ? 'N/A'
    : Number(value).toFixed(digits);
}

function setOptions(select, values, { selected = '', all = false } = {}) {
  if (!select) return;
  select.replaceChildren();
  if (all) select.add(new Option('All', 'All'));
  for (const value of values) {
    select.add(new Option(value, value));
  }
  if (selected && [...select.options].some((option) => option.value === selected)) {
    select.value = selected;
  }
}

function showEmpty(element, message) {
  if (!element) return;
  element.replaceChildren();
  const row = document.createElement('tr');
  const cell = document.createElement('td');
  cell.colSpan = 9;
  cell.className = 'empty-state';
  cell.textContent = message;
  row.append(cell);
  element.append(row);
}

function renderTable(body, rows, columns, emptyMessage = 'Insufficient data for this selection.') {
  if (!body) return;
  body.replaceChildren();
  if (!rows.length) {
    showEmpty(body, emptyMessage);
    return;
  }
  for (const item of rows) {
    const row = document.createElement('tr');
    for (const column of columns) {
      const cell = document.createElement('td');
      const value = column.value(item);
      if (column.className) cell.className = column.className;
      if (column.rolePill && value && value !== 'N/A') {
        const pill = document.createElement('span');
        pill.className = 'role-pill';
        pill.textContent = value;
        cell.append(pill);
      } else if (column.link) {
        const link = document.createElement('a');
        link.href = column.link(item);
        link.textContent = value ?? 'N/A';
        cell.append(link);
      } else {
        cell.textContent = value ?? 'N/A';
      }
      row.append(cell);
    }
    body.append(row);
  }
}

function renderChart(canvasId, config) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return;
  charts.get(canvasId)?.destroy();
  charts.set(canvasId, new Chart(canvas, config));
}

function barOptions(percentAxis = false) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#c4cfcd', usePointStyle: true, boxWidth: 8, font: { size: 10 } } },
      tooltip: { backgroundColor: '#101718', borderColor: '#2b3a3b', borderWidth: 1, titleColor: '#f0f4f2', bodyColor: '#c4cfcd' },
    },
    scales: {
      x: { ticks: { color: '#9baaa7', maxRotation: 0, autoSkip: true }, grid: { display: false } },
      y: {
        beginAtZero: true,
        max: percentAxis ? 1 : undefined,
        ticks: { color: '#9baaa7', callback: percentAxis ? (value) => `${(value * 100).toFixed(0)}%` : undefined },
        grid: { color: '#2b3a3b' },
      },
    },
  };
}

function doughnutOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    cutout: '68%',
    plugins: { legend: { display: false }, tooltip: { callbacks: { label: (item) => `${item.label}: ${formatNumber(item.raw)}` } } },
  };
}

async function initializeDashboard() {
  const [summary, maps, sentiment, comparison] = await Promise.all([
    getJson('/api/dashboard-summary'),
    getJson('/api/maps'),
    getJson('/api/sentiment-distribution'),
    getJson('/api/model-comparison'),
  ]);

  document.getElementById('metric-map-groups').textContent = formatNumber(summary.map_groups);
  document.getElementById('metric-maps').textContent = formatNumber(summary.maps);
  document.getElementById('metric-agents').textContent = formatNumber(summary.agents);
  document.getElementById('metric-players').textContent = formatNumber(summary.player_labels);
  document.getElementById('snapshot-range').textContent = `${summary.date_start} - ${summary.date_end}`;

  const mapSelect = document.getElementById('dashboard-map');
  setOptions(mapSelect, maps.maps);
  const modelNames = ['SVM', 'LSTM'];
  const modelStats = modelNames.map((name) => {
    const result = comparison.models[name];
    const report = result.classification_report;
    return { name, accuracy: result.accuracy, macroF1: report['macro avg']['f1-score'] };
  });
  renderChart('dashboard-model-chart', {
    type: 'bar',
    data: {
      labels: modelStats.map((item) => item.name),
      datasets: [
        { label: 'Accuracy', data: modelStats.map((item) => item.accuracy), backgroundColor: '#fa4f58', borderRadius: 3 },
        { label: 'Macro F1', data: modelStats.map((item) => item.macroF1), backgroundColor: '#55d8c0', borderRadius: 3 },
      ],
    },
    options: barOptions(true),
  });
  renderSentimentChart('dashboard-sentiment-chart', 'sentiment-legend', sentiment);

  const updateMap = async () => {
    const result = await getJson(`/api/agent-map-stats?map=${encodeURIComponent(mapSelect.value)}`);
    const rows = result.stats.slice(0, 10);
    renderChart('dashboard-agent-chart', {
      type: 'bar',
      data: {
        labels: rows.map((item) => item.agent_name),
        datasets: [
          { label: 'Pick rate', data: rows.map((item) => item.pick_rate), backgroundColor: '#fa4f58', borderRadius: 2 },
          { label: 'Map win rate', data: rows.map((item) => item.map_win_rate), backgroundColor: '#55d8c0', borderRadius: 2 },
        ],
      },
      options: barOptions(true),
    });
    renderTable(document.getElementById('dashboard-agent-table'), rows, [
      { value: (item) => item.agent_name, link: (item) => `/agents/${encodeURIComponent(item.agent_name)}` },
      { value: (item) => item.role || 'N/A', rolePill: true },
      { value: (item) => formatNumber(item.appearances) },
      { value: (item) => formatPercent(item.pick_rate) },
      { value: (item) => item.resolved_maps ? formatPercent(item.map_win_rate) : 'N/A' },
    ]);
  };
  mapSelect.addEventListener('change', () => updateMap().catch((error) => notify(error.message)));
  await updateMap();
}

function renderSentimentChart(canvasId, legendId, sentiment) {
  renderChart(canvasId, {
    type: 'doughnut',
    data: {
      labels: sentiment.labels,
      datasets: [{ data: sentiment.labels.map((label) => sentiment.counts[label]), backgroundColor: ['#fa4f58', '#eab36b', '#55d8c0'], borderColor: '#131d1e', borderWidth: 3 }],
    },
    options: doughnutOptions(),
  });
  const legend = document.getElementById(legendId);
  if (legend) {
    legend.replaceChildren();
    sentiment.labels.forEach((label, index) => {
      const item = document.createElement('div');
      item.className = 'legend-item';
      const swatch = document.createElement('span');
      swatch.className = 'legend-swatch';
      swatch.style.backgroundColor = ['#fa4f58', '#eab36b', '#55d8c0'][index];
      const text = document.createElement('span');
      text.innerHTML = `<strong>${formatPercent(sentiment.percentages[label] / 100)}</strong>${label} · ${formatNumber(sentiment.counts[label])}`;
      item.append(swatch, text);
      legend.append(item);
    });
  }
}

async function initializeMaps() {
  const [maps, roles] = await Promise.all([getJson('/api/maps'), getJson('/api/roles')]);
  const mapSelect = document.getElementById('map-filter');
  const roleSelect = document.getElementById('map-role-filter');
  const sideControls = document.getElementById('map-side-filter');
  setOptions(mapSelect, maps.maps, { selected: document.body.dataset.selectedMap });
  setOptions(roleSelect, roles.roles, { all: true });
  let side = 'both';
  sideControls.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-side]');
    if (!button) return;
    side = button.dataset.side;
    sideControls.querySelectorAll('button').forEach((item) => item.classList.toggle('selected', item === button));
    update().catch((error) => notify(error.message));
  });
  const update = async () => {
    const mapName = mapSelect.value;
    const [map, stats] = await Promise.all([
      getJson(`/api/map/${encodeURIComponent(mapName)}`),
      getJson(`/api/agent-map-performance?map=${encodeURIComponent(mapName)}&role=${encodeURIComponent(roleSelect.value)}&side=${encodeURIComponent(side)}`),
    ]);
    const sorted = stats.stats;
    document.getElementById('map-group-count').textContent = formatNumber(map.map_groups);
    document.getElementById('map-pick-count').textContent = formatNumber(map.player_agent_picks);
    document.getElementById('map-agent-count').textContent = formatNumber(sorted.length);
    document.getElementById('map-table-title').textContent = `${map.map} · ${roleSelect.value} · ${side === 'both' ? 'overall' : `${side} ACS`}`;
    renderTable(document.getElementById('map-agent-table'), sorted, [
      { value: (item) => item.agent_name, link: (item) => `/agents/${encodeURIComponent(item.agent_name)}` },
      { value: (item) => item.role || 'N/A', rolePill: true },
      { value: (item) => formatNumber(item.appearances) },
      { value: (item) => formatPercent(item.pick_rate) },
      { value: (item) => item.resolved_maps ? formatPercent(item.map_win_rate) : 'N/A' },
      { value: (item) => item.attack_acs_samples ? `${formatMetric(item.attack_acs_median)} · n=${formatNumber(item.attack_acs_samples)}` : 'N/A' },
      { value: (item) => item.defense_acs_samples ? `${formatMetric(item.defense_acs_median)} · n=${formatNumber(item.defense_acs_samples)}` : 'N/A' },
    ]);
    const chartRows = sorted.slice(0, 10);
    renderChart('map-agent-chart', {
      type: 'bar',
      data: {
        labels: chartRows.map((item) => item.agent_name),
        datasets: [{
          label: side === 'attack' ? 'Median T-side ACS' : side === 'defense' ? 'Median CT-side ACS' : 'Map win rate',
          data: chartRows.map((item) => side === 'attack' ? item.attack_acs_median : side === 'defense' ? item.defense_acs_median : item.map_win_rate),
          backgroundColor: chartRows.map((_, index) => palette[index % palette.length]),
          borderRadius: 2,
        }],
      },
      options: barOptions(side === 'both'),
    });
  };
  mapSelect.addEventListener('change', () => update().catch((error) => notify(error.message)));
  roleSelect.addEventListener('change', () => update().catch((error) => notify(error.message)));
  await update();
}

async function initializeAgents() {
  const { agents } = await getJson('/api/agent-catalog');
  const select = document.getElementById('agent-filter');
  const names = agents.map((item) => item.agent);
  setOptions(select, names, { selected: document.body.dataset.selectedAgent || names[0] });
  if (!select.value) return;
  const update = async () => {
    const result = await getJson(`/api/agents/${encodeURIComponent(select.value)}`);
    const historical = result.historical;
    const maps = historical?.maps || [];
    const overall = historical?.overall;
    const review = result.review_sentiment;
    document.getElementById('agent-name').textContent = result.agent;
    document.getElementById('agent-role').textContent = result.role || 'Role unavailable';
    document.getElementById('agent-role-mark').textContent = (result.role || 'NA').slice(0, 2).toUpperCase();
    document.getElementById('agent-appearances').textContent = overall ? formatNumber(overall.appearances) : 'N/A';
    document.getElementById('agent-pick-rate').textContent = overall ? formatPercent(overall.pick_rate) : 'N/A';
    document.getElementById('agent-win-rate').textContent = overall?.map_win_rate === null || overall?.map_win_rate === undefined ? 'N/A' : formatPercent(overall.map_win_rate);
    document.getElementById('agent-attack-acs').textContent = overall?.attack_acs_samples ? `${formatMetric(overall.attack_acs_median)} · n=${formatNumber(overall.attack_acs_samples)}` : 'N/A';
    document.getElementById('agent-defense-acs').textContent = overall?.defense_acs_samples ? `${formatMetric(overall.defense_acs_median)} · n=${formatNumber(overall.defense_acs_samples)}` : 'N/A';
    document.getElementById('agent-best-map').textContent = result.best_maps?.[0]?.map || 'Insufficient data';
    document.getElementById('agent-worst-map').textContent = result.worst_maps?.[0]?.map || 'Insufficient data';
    document.getElementById('agent-positive').textContent = review?.total_mentions ? formatPercent(review.positive_percentage / 100) : 'N/A';
    document.getElementById('agent-neutral').textContent = review?.total_mentions ? formatPercent(review.neutral_percentage / 100) : 'N/A';
    document.getElementById('agent-negative').textContent = review?.total_mentions ? formatPercent(review.negative_percentage / 100) : 'N/A';
    document.getElementById('agent-positive-count').textContent = `${formatNumber(review?.positive_reviews || 0)} reviews`;
    document.getElementById('agent-neutral-count').textContent = `${formatNumber(review?.neutral_reviews || 0)} reviews`;
    document.getElementById('agent-negative-count').textContent = `${formatNumber(review?.negative_reviews || 0)} reviews`;
    document.getElementById('agent-mentions').textContent = formatNumber(review?.total_mentions || 0);
    document.getElementById('agent-sentiment-method').textContent = review?.method || 'No review-mention analysis is available.';
    const trend = review?.trend || [];
    document.getElementById('agent-sentiment-trend-wrap').hidden = trend.length === 0;
    document.getElementById('agent-sentiment-trend-empty').hidden = trend.length > 0;
    if (trend.length) {
      renderChart('agent-sentiment-trend', {
        type: 'line',
        data: {
          labels: trend.map((item) => item.date),
          datasets: [
            { label: 'Positive', data: trend.map((item) => item.Positive), borderColor: '#55d8c0', backgroundColor: '#55d8c0', tension: 0.2 },
            { label: 'Neutral', data: trend.map((item) => item.Neutral), borderColor: '#eab36b', backgroundColor: '#eab36b', tension: 0.2 },
            { label: 'Negative', data: trend.map((item) => item.Negative), borderColor: '#fa4f58', backgroundColor: '#fa4f58', tension: 0.2 },
          ],
        },
        options: barOptions(),
      });
    } else {
      document.getElementById('agent-sentiment-trend-empty').textContent = review?.total_mentions
        ? 'No valid review dates for these mentions.'
        : 'No detected review mentions for this agent.';
    }
    renderTable(document.getElementById('agent-map-table'), maps, [
      { value: (item) => item.map, link: (item) => `/maps/${encodeURIComponent(item.map)}` },
      { value: (item) => formatNumber(item.appearances) },
      { value: (item) => formatPercent(item.pick_rate) },
      { value: (item) => item.resolved_maps ? formatPercent(item.map_win_rate) : 'N/A' },
      { value: (item) => item.attack_acs_samples ? `${formatMetric(item.attack_acs_median)} · n=${formatNumber(item.attack_acs_samples)}` : 'N/A' },
      { value: (item) => item.defense_acs_samples ? `${formatMetric(item.defense_acs_median)} · n=${formatNumber(item.defense_acs_samples)}` : 'N/A' },
    ]);
    renderChart('agent-map-chart', {
      type: 'bar',
      data: {
        labels: maps.map((item) => item.map),
        datasets: [
          { label: 'Pick rate', data: maps.map((item) => item.pick_rate), backgroundColor: '#fa4f58', borderRadius: 2 },
          { label: 'Map win rate', data: maps.map((item) => item.map_win_rate), backgroundColor: '#55d8c0', borderRadius: 2 },
        ],
      },
      options: barOptions(true),
    });
    renderChart('agent-side-acs-chart', {
      type: 'bar',
      data: {
        labels: maps.map((item) => item.map),
        datasets: [
          { label: 'Median T-side ACS', data: maps.map((item) => item.attack_acs_median), backgroundColor: '#fa4f58', borderRadius: 2 },
          { label: 'Median CT-side ACS', data: maps.map((item) => item.defense_acs_median), backgroundColor: '#55d8c0', borderRadius: 2 },
        ],
      },
      options: barOptions(),
    });
  };
  select.addEventListener('change', () => update().catch((error) => notify(error.message)));
  await update();
}

function attachSegmented(containerId, initialValue, onChange) {
  const container = document.getElementById(containerId);
  let value = initialValue;
  container.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-side]');
    if (!button) return;
    value = button.dataset.side;
    container.querySelectorAll('button').forEach((item) => item.classList.toggle('selected', item === button));
    onChange(value);
  });
  return () => value;
}

async function initializeRecommendations() {
  const [maps, roles, agents] = await Promise.all([getJson('/api/maps'), getJson('/api/roles'), getJson('/api/agents')]);
  const mapSelect = document.getElementById('recommend-map');
  const roleSelect = document.getElementById('recommend-role');
  const synergyAgent = document.getElementById('synergy-agent');
  setOptions(mapSelect, maps.maps, { selected: document.body.dataset.selectedMap });
  setOptions(roleSelect, roles.roles, { all: true });
  setOptions(synergyAgent, agents.agents.map((item) => item.agent));
  let side = 'both';
  let update;
  attachSegmented('recommend-side', side, (value) => {
    side = value;
    update?.().catch((error) => notify(error.message));
  });

  update = async () => {
    const mapName = mapSelect.value;
    const role = roleSelect.value;
    const result = await getJson(`/api/recommend?map=${encodeURIComponent(mapName)}&side=${side}&role=${encodeURIComponent(role)}`);
    document.getElementById('recommend-title').textContent = `${mapName} · ${role} · ${side}`;
    document.getElementById('recommendation-method').textContent = side === 'both'
      ? 'Sorted by Wilson 95% lower bound for historical map win rate; minimum 20 agent-map picks. This is not a causal agent effect.'
      : `Sorted by percentile of median ${side === 'attack' ? 'T-side' : 'CT-side'} ACS; minimum map-pick and side-sample thresholds apply. This is not a side win rate.`;
    const list = document.getElementById('recommendation-list');
    list.replaceChildren();
    if (!result.recommendations.length) {
      list.textContent = 'Insufficient data for this map, role, and side combination.';
    }
    for (const item of result.recommendations) {
      const card = document.createElement('article');
      card.className = 'recommendation-card';
      const top = document.createElement('div');
      top.className = 'recommendation-top';
      const title = document.createElement('strong');
      title.textContent = `#${item.rank} ${item.agent} · ${item.role}`;
      const score = document.createElement('span');
      score.className = 'recommendation-score';
      score.textContent = `${formatMetric(item.score)} pts`;
      top.append(title, score);
      const why = document.createElement('ul');
      item.why.forEach((reason) => {
        const entry = document.createElement('li');
        entry.textContent = reason;
        why.append(entry);
      });
      card.append(top, why);
      list.append(card);
    }
    await updateCompositions(mapName);
    await updateSynergy(mapName);
  };
  const updateCompositions = async (mapName) => {
    const result = await getJson(`/api/team-composition?map=${encodeURIComponent(mapName)}`);
    const container = document.getElementById('composition-list');
    container.replaceChildren();
    if (!result.compositions.length) {
      container.textContent = 'Insufficient repeated lineup data for this map.';
      return;
    }
    result.compositions.forEach((item) => {
      const card = document.createElement('article');
      card.className = 'composition-card';
      const summary = document.createElement('p');
      summary.textContent = `${formatPercent(item.win_rate)} win rate · ${formatNumber(item.appearances)} map appearances`;
      const lineup = document.createElement('div');
      lineup.className = 'composition-agents';
      item.composition.split(' | ').forEach((agent) => {
        const chip = document.createElement('span');
        chip.textContent = agent;
        lineup.append(chip);
      });
      card.append(summary, lineup);
      container.append(card);
    });
  };
  const updateSynergy = async (mapName) => {
    if (!synergyAgent.value) return;
    const result = await getJson(`/api/agent-synergy?agent=${encodeURIComponent(synergyAgent.value)}&map=${encodeURIComponent(mapName)}`);
    renderTable(document.getElementById('synergy-table'), result.pairings, [
      { value: (item) => item.partner },
      { value: (item) => item.map },
      { value: (item) => formatNumber(item.appearances) },
      { value: (item) => formatPercent(item.win_rate) },
    ]);
  };
  mapSelect.addEventListener('change', () => update().catch((error) => notify(error.message)));
  roleSelect.addEventListener('change', () => update().catch((error) => notify(error.message)));
  document.getElementById('run-recommendation').addEventListener('click', () => update().catch((error) => notify(error.message)));
  synergyAgent.addEventListener('change', () => updateSynergy(mapSelect.value).catch((error) => notify(error.message)));
  await update();
}

async function initializeCompare() {
  const [maps, agents] = await Promise.all([getJson('/api/maps'), getJson('/api/agents')]);
  const mapSelect = document.getElementById('compare-map');
  const agentSelect = document.getElementById('compare-agents');
  setOptions(mapSelect, maps.maps, { all: true });
  setOptions(agentSelect, agents.agents.map((item) => item.agent));
  if (agentSelect.options.length >= 2) {
    agentSelect.options[0].selected = true;
    agentSelect.options[1].selected = true;
  }
  const update = async () => {
    const selected = [...agentSelect.selectedOptions].map((option) => option.value);
    if (selected.length < 2 || selected.length > 5) {
      notify('Choose between two and five agents to compare.');
      return;
    }
    const query = new URLSearchParams();
    selected.forEach((agent) => query.append('agent', agent));
    if (mapSelect.value !== 'All') query.set('map', mapSelect.value);
    const result = await getJson(`/api/agent-comparison?${query}`);
    const records = result.stats;
    renderTable(document.getElementById('compare-table'), records, [
      { value: (item) => item.agent_name },
      { value: (item) => item.role || 'N/A', rolePill: true },
      { value: (item) => item.map },
      { value: (item) => formatNumber(item.appearances) },
      { value: (item) => formatPercent(item.pick_rate) },
      { value: (item) => item.resolved_maps ? formatPercent(item.map_win_rate) : 'N/A' },
      { value: (item) => item.attack_acs_samples ? formatMetric(item.attack_acs_median) : 'N/A' },
      { value: (item) => item.defense_acs_samples ? formatMetric(item.defense_acs_median) : 'N/A' },
    ]);
    const chartRows = records.filter((item) => item.resolved_maps).sort((left, right) => right.map_win_rate - left.map_win_rate);
    renderChart('compare-chart', {
      type: 'bar',
      data: {
        labels: chartRows.map((item) => `${item.agent_name} · ${item.map}`),
        datasets: [{ label: 'Map win rate', data: chartRows.map((item) => item.map_win_rate), backgroundColor: chartRows.map((_, index) => palette[index % palette.length]), borderRadius: 2 }],
      },
      options: barOptions(true),
    });
  };
  document.getElementById('run-comparison').addEventListener('click', () => update().catch((error) => notify(error.message)));
  await update();
}

async function initializeSentiment() {
  const [distribution, scoreData, agentSentiment] = await Promise.all([
    getJson('/api/sentiment/distribution'),
    getJson('/api/sentiment/scores'),
    getJson('/api/sentiment/by-agent'),
  ]);
  renderSentimentChart('sentiment-chart', 'sentiment-counts', distribution);
  document.getElementById('sentiment-source-note').textContent = distribution.label_source;
  renderChart('score-distribution-chart', {
    type: 'bar',
    data: {
      labels: scoreData.score_distribution.map((item) => String(item.score)),
      datasets: [{ label: 'Reviews', data: scoreData.score_distribution.map((item) => item.count), backgroundColor: '#75aaff', borderRadius: 2 }],
    },
    options: barOptions(),
  });
  const sentimentOrder = ['Negative', 'Neutral', 'Positive'];
  renderChart('score-sentiment-chart', {
    type: 'bar',
    data: {
      labels: sentimentOrder,
      datasets: [{ label: 'Reviews', data: sentimentOrder.map((label) => scoreData.sentiment_distribution[label].count), backgroundColor: ['#fa4f58', '#eab36b', '#55d8c0'], borderRadius: 2 }],
    },
    options: barOptions(),
  });
  renderTable(document.getElementById('score-sentiment-table'), sentimentOrder.map((label) => ({
    label,
    range: scoreData.label_mapping[label.toLowerCase()],
    ...scoreData.sentiment_distribution[label],
  })), [
    { value: (item) => item.label },
    { value: (item) => item.range },
    { value: (item) => formatNumber(item.count) },
    { value: (item) => formatPercent(item.percentage / 100) },
    { value: (item) => item.average_score === null ? 'N/A' : formatMetric(item.average_score, 2) },
  ]);
  renderTable(document.getElementById('sentiment-agent-table'), agentSentiment.agents, [
    { value: (item) => item.agent, link: (item) => `/agents/${encodeURIComponent(item.agent)}` },
    { value: (item) => item.role || 'N/A', rolePill: true },
    { value: (item) => formatNumber(item.positive_reviews) },
    { value: (item) => formatPercent(item.positive_percentage / 100) },
    { value: (item) => formatNumber(item.neutral_reviews) },
    { value: (item) => formatPercent(item.neutral_percentage / 100) },
    { value: (item) => formatNumber(item.negative_reviews) },
    { value: (item) => formatPercent(item.negative_percentage / 100) },
    { value: (item) => formatNumber(item.total_mentions) },
  ], 'No agent names detected in the review dataset.');
  document.getElementById('agent-sentiment-note').textContent = agentSentiment.method;
  const form = document.getElementById('dashboard-predict-form');
  const reviewInput = document.getElementById('dashboard-review');
  const output = document.getElementById('prediction-results');
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const review = reviewInput.value.trim();
    if (!review) {
      notify('Enter a review before analyzing.');
      return;
    }
    try {
      const response = await fetch('/api/sentiment/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ review }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Prediction failed.');
      await renderPredictionAnalysis(output, result);
      output.hidden = false;
      document.getElementById('page-notice').hidden = true;
    } catch (error) {
      notify(error.message);
    }
  });
  document.getElementById('clear-review').addEventListener('click', () => {
    reviewInput.value = '';
    output.replaceChildren();
    output.hidden = true;
  });
}

function createAnalysisSection(title) {
  const section = document.createElement('section');
  section.className = 'analysis-section';
  const heading = document.createElement('h3');
  heading.textContent = title;
  section.append(heading);
  return section;
}

function addAnalysisMetric(container, label, value) {
  const metric = document.createElement('div');
  metric.className = 'analysis-metric';
  const name = document.createElement('span');
  name.textContent = label;
  const result = document.createElement('strong');
  result.textContent = value;
  metric.append(name, result);
  container.append(metric);
}

function normalizeLookup(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]/g, '');
}

async function renderPredictionAnalysis(output, result) {
  output.replaceChildren();
  output.classList.add('has-analysis');

  const overall = document.createElement('article');
  overall.className = 'prediction-overall';
  const overallLabel = document.createElement('strong');
  overallLabel.textContent = `Overall sentiment · ${result.overall.label}`;
  const overallNote = document.createElement('small');
  overallNote.textContent = `Selected model: ${result.selected_model}. SVM confidence is unavailable because it is not calibrated.`;
  overall.append(overallLabel, overallNote);
  output.append(overall);

  const models = createAnalysisSection('MODEL PREDICTIONS');
  const modelGrid = document.createElement('div');
  modelGrid.className = 'analysis-grid';
  for (const [name, prediction] of Object.entries({ SVM: result.svm, LSTM: result.lstm })) {
    const card = document.createElement('article');
    card.className = 'prediction-result';
    const label = document.createElement('span');
    label.textContent = name;
    const sentiment = document.createElement('strong');
    sentiment.textContent = prediction.label;
    const detail = document.createElement('small');
    detail.textContent = prediction.confidence === null ? 'Uncalibrated model; confidence unavailable' : `Model score: ${(prediction.confidence * 100).toFixed(1)}%`;
    card.append(label, sentiment, detail);
    modelGrid.append(card);
  }
  models.append(modelGrid);
  output.append(models);

  const analysis = result.analysis;
  const aspectRows = [
    ...analysis.agents.map((item) => ({ label: `Agent: ${item.agent}`, sentiment: item.sentiment })),
    ...analysis.maps.map((item) => ({ label: `Map: ${item.map}`, sentiment: item.sentiment })),
    ...Object.entries(analysis.topics)
      .filter(([, topic]) => topic.detected)
      .map(([name, topic]) => ({ label: `Topic: ${name}`, sentiment: topic.sentiment || 'No polarity' })),
  ];
  const graph = createAnalysisSection('NLP SENTIMENT GRAPH');
  const graphNote = document.createElement('p');
  graphNote.className = 'panel-note';
  graphNote.textContent = 'Polarity detected near named agents, maps, and topic keywords. This rule based aspect analysis is separate from the SVM/LSTM review prediction.';
  graph.append(graphNote);
  if (aspectRows.length) {
    const chartWrap = document.createElement('div');
    chartWrap.className = 'chart-wrap nlp-chart-wrap';
    const canvas = document.createElement('canvas');
    canvas.id = 'prediction-nlp-chart';
    canvas.setAttribute('aria-label', 'NLP aspect sentiment by detected agent, map, and topic');
    chartWrap.append(canvas);
    graph.append(chartWrap);
    const labels = ['Positive', 'Neutral', 'Negative', 'Mixed', 'No polarity'];
    renderChart(canvas.id, {
      type: 'bar',
      data: {
        labels: aspectRows.map((item) => item.label),
        datasets: labels.map((label, index) => ({
          label,
          data: aspectRows.map((item) => item.sentiment === label ? 1 : 0),
          backgroundColor: ['#55d8c0', '#eab36b', '#fa4f58', '#c9a5ff', '#71817f'][index],
          borderRadius: 2,
        })),
      },
      options: {
        ...barOptions(),
        indexAxis: 'y',
        scales: {
          x: { stacked: true, beginAtZero: true, max: 1, ticks: { display: false }, grid: { display: false } },
          y: { stacked: true, ticks: { color: '#c4cfcd' }, grid: { display: false } },
        },
        plugins: { ...barOptions().plugins, legend: { ...barOptions().plugins.legend, position: 'bottom' } },
      },
    });
  } else {
    graph.append(document.createTextNode('No agent, map, or topic aspects were detected in this review.'));
  }
  output.append(graph);

  const agents = createAnalysisSection('DETECTED AGENTS');
  if (!analysis.agents.length) {
    agents.append(document.createTextNode(analysis.agent_detection_message));
  }
  for (const item of analysis.agents) {
    const card = document.createElement('article');
    card.className = 'analysis-card';
    const title = document.createElement('strong');
    title.textContent = `${item.agent} · ${item.sentiment}`;
    const role = item.historical_performance?.role || 'Role unavailable';
    const details = document.createElement('p');
    details.textContent = `${role}${item.evidence.length ? ` · Evidence: ${item.evidence.join(', ')}` : ' · No polarity words found near the mention.'}`;
    card.append(title, details);
    const overallStats = item.historical_performance?.overall;
    if (overallStats) {
      const metrics = document.createElement('div');
      metrics.className = 'analysis-metrics';
      addAnalysisMetric(metrics, 'Pick rate', formatPercent(overallStats.pick_rate));
      addAnalysisMetric(metrics, 'Map win rate', overallStats.map_win_rate === null ? 'N/A' : formatPercent(overallStats.map_win_rate));
      addAnalysisMetric(metrics, 'T-side median ACS', overallStats.attack_acs_samples ? `${formatMetric(overallStats.attack_acs_median)} · n=${formatNumber(overallStats.attack_acs_samples)}` : 'N/A');
      addAnalysisMetric(metrics, 'CT-side median ACS', overallStats.defense_acs_samples ? `${formatMetric(overallStats.defense_acs_median)} · n=${formatNumber(overallStats.defense_acs_samples)}` : 'N/A');
      card.append(metrics);
    } else {
      const unavailable = document.createElement('p');
      unavailable.textContent = 'Insufficient historical performance data for this agent.';
      card.append(unavailable);
    }
    agents.append(card);
  }
  output.append(agents);

  const maps = createAnalysisSection('DETECTED MAPS');
  if (!analysis.maps.length) {
    maps.append(document.createTextNode(analysis.map_detection_message));
  }
  for (const item of analysis.maps) {
    const card = document.createElement('article');
    card.className = 'analysis-card';
    const title = document.createElement('strong');
    title.textContent = `${item.map} · ${item.sentiment}`;
    const evidence = document.createElement('p');
    evidence.textContent = item.evidence.length ? `Context evidence: ${item.evidence.join(', ')}` : 'No polarity words found near the map mention.';
    card.append(title, evidence);
    try {
      const map = await getJson(`/api/maps/${encodeURIComponent(item.map)}`);
      const details = document.createElement('p');
      details.textContent = `${formatNumber(map.map_groups)} historical map groups · ${map.data_period.start} to ${map.data_period.end}`;
      card.append(details);
      const agentRows = analysis.agents.map((agent) => map.agents.find((entry) => normalizeLookup(entry.agent_name) === normalizeLookup(agent.agent))).filter(Boolean);
      if (agentRows.length) {
        const metrics = document.createElement('div');
        metrics.className = 'analysis-metrics';
        agentRows.forEach((entry) => {
          addAnalysisMetric(metrics, `${entry.agent_name} pick rate`, formatPercent(entry.pick_rate));
          addAnalysisMetric(metrics, `${entry.agent_name} map win rate`, entry.resolved_maps ? formatPercent(entry.map_win_rate) : 'N/A');
          addAnalysisMetric(metrics, `${entry.agent_name} T-side ACS`, entry.attack_acs_samples ? `${formatMetric(entry.attack_acs_median)} · n=${formatNumber(entry.attack_acs_samples)}` : 'N/A');
          addAnalysisMetric(metrics, `${entry.agent_name} CT-side ACS`, entry.defense_acs_samples ? `${formatMetric(entry.defense_acs_median)} · n=${formatNumber(entry.defense_acs_samples)}` : 'N/A');
        });
        card.append(metrics);
      }
    } catch (error) {
      const unavailable = document.createElement('p');
      unavailable.textContent = 'Insufficient historical map data.';
      card.append(unavailable);
    }
    maps.append(card);
  }
  output.append(maps);

  const topics = createAnalysisSection('TOPIC ANALYSIS');
  const detectedTopics = Object.entries(analysis.topics).filter(([, topic]) => topic.detected);
  if (!detectedTopics.length) {
    topics.append(document.createTextNode('No supported topic keywords detected.'));
  }
  detectedTopics.forEach(([name, topic]) => {
    const card = document.createElement('article');
    card.className = 'analysis-card';
    const title = document.createElement('strong');
    title.textContent = `${name} · ${topic.sentiment || 'No polarity detected'}`;
    const evidence = document.createElement('p');
    evidence.textContent = topic.evidence.length ? `Context evidence: ${topic.evidence.join(', ')}` : topic.basis;
    card.append(title, evidence);
    topics.append(card);
  });
  const method = document.createElement('p');
  method.className = 'panel-note';
  method.textContent = analysis.aspect_method;
  topics.append(method);
  output.append(topics);
}

async function initializeModels() {
  const [evaluation, matchEvaluation] = await Promise.all([
    getJson('/api/model-comparison'),
    getJson('/api/match-model-results'),
  ]);
  const models = Object.entries(evaluation.models).map(([name, result]) => ({ name, ...result }));
  const labels = evaluation.labels;
  renderTable(document.getElementById('models-table'), models, [
    { value: (item) => item.name },
    { value: (item) => formatPercent(item.accuracy) },
    { value: (item) => formatPercent(item.classification_report['macro avg']['f1-score']) },
    { value: (item) => formatPercent(item.classification_report['weighted avg']['f1-score']) },
    { value: (item) => formatMetric(item.training_time_seconds, 2) },
    { value: (item) => formatMetric(item.inference_latency_ms_per_review, 2) },
    { value: (item) => formatNumber(item.test_samples) },
  ]);
  renderChart('models-chart', {
    type: 'bar',
    data: {
      labels: models.map((item) => item.name),
      datasets: [
        { label: 'Accuracy', data: models.map((item) => item.accuracy), backgroundColor: '#fa4f58', borderRadius: 2 },
        { label: 'Macro F1', data: models.map((item) => item.classification_report['macro avg']['f1-score']), backgroundColor: '#55d8c0', borderRadius: 2 },
        { label: 'Weighted F1', data: models.map((item) => item.classification_report['weighted avg']['f1-score']), backgroundColor: '#eab36b', borderRadius: 2 },
      ],
    },
    options: barOptions(true),
  });
  const container = document.getElementById('confusion-matrices');
  container.replaceChildren();
  models.forEach((model) => {
    const section = document.createElement('section');
    section.className = 'panel matrix-card';
    const heading = document.createElement('h3');
    heading.textContent = `${model.name} · Confusion matrix`;
    const grid = document.createElement('div');
    grid.className = 'matrix-grid';
    model.confusion_matrix.forEach((row, rowIndex) => row.forEach((value, columnIndex) => {
      const cell = document.createElement('div');
      cell.className = 'matrix-cell';
      const count = document.createElement('strong');
      count.textContent = formatNumber(value);
      const label = document.createElement('small');
      label.textContent = `${labels[rowIndex]} → ${labels[columnIndex]}`;
      cell.append(count, label);
      grid.append(cell);
    }));
    section.append(heading, grid);
    container.append(section);
  });

  const matchModel = matchEvaluation;
  const baseline = matchModel.baseline;
  const matchRows = [
    {
      name: matchModel.model,
      accuracy: matchModel.accuracy,
      macroF1: matchModel.f1_macro,
      precision: matchModel.precision_macro,
      recall: matchModel.recall_macro,
      trainingMaps: matchModel.train_map_outcomes,
      testMaps: matchModel.test_map_outcomes,
    },
    {
      name: baseline.model,
      accuracy: baseline.accuracy,
      macroF1: baseline.f1_macro,
      precision: baseline.classification_report['macro avg'].precision,
      recall: baseline.classification_report['macro avg'].recall,
      trainingMaps: matchModel.train_map_outcomes,
      testMaps: matchModel.test_map_outcomes,
    },
  ];
  renderTable(document.getElementById('match-model-table'), matchRows, [
    { value: (item) => item.name },
    { value: (item) => formatPercent(item.accuracy) },
    { value: (item) => formatMetric(item.macroF1, 3) },
    { value: (item) => formatMetric(item.precision, 3) },
    { value: (item) => formatMetric(item.recall, 3) },
    { value: (item) => formatNumber(item.trainingMaps) },
    { value: (item) => formatNumber(item.testMaps) },
  ]);
  renderChart('match-model-chart', {
    type: 'bar',
    data: {
      labels: matchRows.map((item) => item.name),
      datasets: [
        { label: 'Accuracy', data: matchRows.map((item) => item.accuracy), backgroundColor: '#fa4f58', borderRadius: 2 },
        { label: 'Macro F1', data: matchRows.map((item) => item.macroF1), backgroundColor: '#55d8c0', borderRadius: 2 },
      ],
    },
    options: barOptions(true),
  });
  document.getElementById('match-model-warning').textContent =
    `Experimental: accuracy is ${(matchModel.accuracy * 100).toFixed(1)}% versus ${(baseline.accuracy * 100).toFixed(1)}% for the majority baseline; macro F1 is ${(matchModel.f1_macro * 100).toFixed(1)}% versus ${(baseline.f1_macro * 100).toFixed(1)}%. Split is grouped by reconstructed date/team pairing; event-level IDs are unavailable.`;

  const matchMatrices = document.getElementById('match-model-matrices');
  matchMatrices.replaceChildren();
  [matchModel, baseline].forEach((model) => {
    const matrix = model.confusion_matrix;
    const card = document.createElement('article');
    card.className = 'panel matrix-card';
    const heading = document.createElement('h3');
    heading.textContent = `${model.model} · Confusion matrix`;
    const grid = document.createElement('div');
    grid.className = 'matrix-grid';
    matrix.forEach((row, rowIndex) => row.forEach((value, columnIndex) => {
      const cell = document.createElement('div');
      cell.className = 'matrix-cell';
      const count = document.createElement('strong');
      count.textContent = formatNumber(value);
      const label = document.createElement('small');
      label.textContent = `${['Team 2 wins', 'Team 1 wins'][rowIndex]} → ${['Team 2 wins', 'Team 1 wins'][columnIndex]}`;
      cell.append(count, label);
      grid.append(cell);
    }));
    card.append(heading, grid);
    matchMatrices.append(card);
  });
}

async function initializePage() {
  try {
    if (page === 'dashboard') await initializeDashboard();
    if (page === 'maps') await initializeMaps();
    if (page === 'agents') await initializeAgents();
    if (page === 'recommend') await initializeRecommendations();
    if (page === 'compare') await initializeCompare();
    if (page === 'sentiment') await initializeSentiment();
    if (page === 'models') await initializeModels();
  } catch (error) {
    notify(error.message || 'Unable to load this view.');
    console.error(error);
  }
}

document.addEventListener('DOMContentLoaded', initializePage);
