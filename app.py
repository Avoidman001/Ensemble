import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

from engine import (
    log_returns, ewma_vol, simulate_gbm, simulate_iid_bootstrap,
    simulate_stationary_bootstrap, first_passage_probabilities,
    walk_forward_calibration, sanity_check_gbm, sanity_check_bootstrap_variance,
)

st.set_page_config(page_title="M15 Quant Academy", layout="wide", page_icon="◆")

# ---------------------------------------------------------------------------
# Дизайн: ink-navy фон, тёплый янтарный акцент (тикерная лента), serif+mono
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink: #0B1220;
    --ink-2: #111A2C;
    --paper: #ECE7DC;
    --amber: #D4A24C;
    --teal: #4C8C82;
    --line: rgba(236, 231, 220, 0.14);
}

.stApp { background-color: var(--ink); color: var(--paper); }
h1, h2, h3 { font-family: 'Source Serif 4', serif !important; color: var(--paper) !important; font-weight: 600 !important; }
p, li, div, span, label { font-family: 'Inter', sans-serif; }
.stMarkdown code, .katex { font-family: 'IBM Plex Mono', monospace !important; }

.formula-block {
    background: var(--ink-2);
    border-left: 3px solid var(--amber);
    padding: 1.1rem 1.4rem;
    margin: 0.8rem 0 1.2rem 0;
    border-radius: 4px;
}
.explain-block {
    border-top: 1px solid var(--line);
    padding-top: 0.9rem;
    margin-top: 0.4rem;
    color: #C9C2B3;
    line-height: 1.55;
}
.metric-row {
    display: flex; gap: 1.4rem; flex-wrap: wrap; margin: 0.6rem 0 1.4rem 0;
}
.metric-card {
    background: var(--ink-2);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    min-width: 160px;
}
.metric-card .label { font-size: 0.78rem; color: #9C9484; text-transform: none; }
.metric-card .value { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; color: var(--amber); }

hr { border-color: var(--line) !important; }
.stTabs [data-baseweb="tab"] { font-family: 'Source Serif 4', serif; font-size: 1.02rem; }
.stButton button {
    background-color: var(--amber) !important;
    color: var(--ink) !important;
    border: none !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


def metric_card(label: str, value: str):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def plotly_dark_layout(fig, title=""):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        font=dict(family="Inter, sans-serif", color="#ECE7DC"),
        margin=dict(t=50, l=10, r=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Заголовок
# ---------------------------------------------------------------------------
st.markdown("### M15 QUANT ACADEMY")
st.markdown(
    "<p style='color:#9C9484; margin-top:-0.6rem;'>"
    "Разбор математики, которая стоит за Monte Carlo-движками для трейдинга — "
    "с формулами, живыми демо и калькулятором на реальных данных."
    "</p>", unsafe_allow_html=True)

tab_learn, tab_calc = st.tabs(["Обучение", "Калькулятор"])

# ===========================================================================
# ВКЛАДКА 1 — ОБУЧЕНИЕ
# ===========================================================================
with tab_learn:

    # --- 1. GBM ---------------------------------------------------------
    st.markdown("## 1. Геометрическое броуновское движение (GBM)")
    st.markdown("""
    <div class="explain-block">
    Цена актива не может стать отрицательной и движется мультипликативно —
    в процентах от текущего значения, а не в фиксированных пунктах.
    GBM — стандартная модель именно такого поведения, та же, что лежит
    в основе формулы Блэка-Шоулза для опционов.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-block">', unsafe_allow_html=True)
    st.latex(r"S_{t+1} = S_t \cdot \exp\Big( (\mu - \tfrac{1}{2}\sigma^2) + \sigma Z_t \Big), \quad Z_t \sim \mathcal{N}(0,1)")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain-block">
    <b>Когда использовать:</b> как базовый, аналитически прозрачный сценарий —
    быстрый ориентир, если доходности приблизительно нормальны.
    <b>Ограничение:</b> реальные доходности имеют "толстые хвосты" — GBM их недооценивает.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        demo_sigma = st.slider("σ (волатильность за бар)", 0.001, 0.05, 0.015, 0.001, key="gbm_sigma")
        demo_mu = st.slider("μ (дрифт за бар)", -0.005, 0.005, 0.0, 0.0005, key="gbm_mu")
        demo_n = st.slider("Число путей", 10, 300, 60, 10, key="gbm_n")
    with c2:
        demo_paths = simulate_gbm(100.0, demo_mu, demo_sigma, 60, demo_n, seed=1)
        fig = go.Figure()
        for i in range(demo_n):
            fig.add_trace(go.Scatter(y=demo_paths[:, i], mode="lines",
                                      line=dict(width=1, color="rgba(212,162,76,0.25)"),
                                      showlegend=False))
        fig.add_trace(go.Scatter(y=np.median(demo_paths, axis=1), mode="lines",
                                  line=dict(width=2.5, color="#4C8C82"), name="Медиана"))
        st.plotly_chart(plotly_dark_layout(fig, "Симулированные GBM-пути"), use_container_width=True)

    st.divider()

    # --- 2. Volatility drag ---------------------------------------------
    st.markdown("## 2. Снос волатильности: почему стоит «−0.5σ²»")
    st.markdown("""
    <div class="explain-block">
    Если просто складывать случайные проценты доходности, среднее геометрическое
    (то, что реально накапливается) оказывается меньше среднего арифметического.
    Классический пример: +50% в один день и −50% на следующий дают среднее
    арифметическое 0%, но капитал реально падает на 25%.
    Член <code>−0.5σ²</code> в формуле GBM — поправка именно на этот эффект.
    </div>
    """, unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        up = st.slider("Доходность день 1, %", -60, 60, 50, key="drag_up")
    with colB:
        down = st.slider("Доходность день 2, %", -60, 60, -50, key="drag_down")
    arithmetic_mean = (up + down) / 2
    real_result = (1 + up / 100) * (1 + down / 100) - 1
    m1, m2 = st.columns(2)
    with m1:
        metric_card("Среднее арифметическое", f"{arithmetic_mean:.1f}%")
    with m2:
        metric_card("Реальный итог капитала", f"{real_result*100:.1f}%")

    st.divider()

    # --- 3. EWMA vol ------------------------------------------------------
    st.markdown("## 3. EWMA-волатильность и кластеризация")
    st.markdown("""
    <div class="explain-block">
    Волатильность рынка непостоянна — за сильным движением чаще следует ещё одно
    сильное движение, а не тишина ("volatility clustering"). Плоское std по всему
    окну размазывает вчерашний шторм и полугодовую тишину в одно число.
    EWMA вместо этого экспоненциально забывает старые бары.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-block">', unsafe_allow_html=True)
    st.latex(r"\sigma_t^2 = \lambda \cdot \sigma_{t-1}^2 + (1-\lambda) \cdot r_{t-1}^2")
    st.markdown('</div>', unsafe_allow_html=True)
    lam_demo = st.slider("λ (память модели)", 0.80, 0.99, 0.94, 0.01, key="ewma_lam")
    rng_demo = np.random.default_rng(7)
    calm = rng_demo.standard_normal(150) * 0.005
    storm = rng_demo.standard_normal(80) * 0.028
    calm2 = rng_demo.standard_normal(120) * 0.006
    synthetic_rets = np.concatenate([calm, storm, calm2])
    vol_est = ewma_vol(synthetic_rets, lam=lam_demo)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=np.abs(synthetic_rets), mode="lines",
                               line=dict(width=1, color="rgba(236,231,220,0.35)"), name="|доходность|"))
    fig2.add_trace(go.Scatter(y=vol_est, mode="lines", line=dict(width=2.5, color="#D4A24C"),
                               name="EWMA-волатильность"))
    st.plotly_chart(plotly_dark_layout(fig2, "Синтетический пример: спокойный → штормовой → снова спокойный режим"),
                     use_container_width=True)
    st.markdown("""
    <div class="explain-block">
    <b>Когда использовать:</b> почти всегда для форекаста волатильности вперёд —
    промышленный стандарт (RiskMetrics, J.P. Morgan). λ≈0.94 — типичное значение
    для дневных данных, ниже — быстрее реагирует, но шумнее.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- 4. IID sampling ---------------------------------------------------
    st.markdown("## 4. IID-семплинг из нормального распределения")
    st.markdown("""
    <div class="explain-block">
    "Независимые одинаково распределённые" — каждое случайное число не зависит
    от предыдущих и берётся из одного и того же распределения. Именно так GBM
    генерирует шоки Z. Это самый простой и быстрый способ, но он предполагает,
    что рынок "без памяти" и его доходности — колокол Гаусса.
    </div>
    """, unsafe_allow_html=True)
    fig3 = go.Figure()
    z_demo = np.random.default_rng(3).standard_normal(20000)
    fig3.add_trace(go.Histogram(x=z_demo, nbinsx=60, marker_color="#4C8C82", name="IID N(0,1)"))
    st.plotly_chart(plotly_dark_layout(fig3, "20 000 сэмплов из standard_normal"), use_container_width=True)

    st.divider()

    # --- 5. Stationary bootstrap ---------------------------------------
    st.markdown("## 5. Stationary bootstrap (Politis & Romano, 1994)")
    st.markdown("""
    <div class="explain-block">
    Вместо генерации чисел из формулы — пересборка настоящих исторических
    доходностей в новые последовательности. Это сохраняет реальную форму
    распределения (толстые хвосты, асимметрию). Но если тянуть отдельные точки
    независимо (IID bootstrap), теряется автокорреляция и кластеризация
    волатильности. Stationary bootstrap решает это блоками СЛУЧАЙНОЙ длины
    (геометрическое распределение) вместо точек или блоков фиксированной длины —
    это математически корректнее сохраняет стационарность ряда.
    </div>
    """, unsafe_allow_html=True)
    block_len_demo = st.slider("Средняя длина блока", 1.0, 40.0, 10.0, 1.0, key="block_len_demo")
    geom_p = 1.0 / block_len_demo
    block_lengths_demo = np.random.default_rng(4).geometric(geom_p, size=5000)
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=block_lengths_demo, marker_color="#D4A24C"))
    st.plotly_chart(plotly_dark_layout(fig4, "Распределение длины блока (геометрическое)"), use_container_width=True)
    st.markdown("""
    <div class="explain-block">
    <b>Когда использовать:</b> когда важно сохранить временную структуру ряда
    (автокорреляцию, кластеризацию волатильности) — то есть почти всегда для
    финансовых доходностей, в отличие от IID bootstrap.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- 6. Ensemble --------------------------------------------------
    st.markdown("## 6. Ансамбль трёх методов")
    st.markdown("""
    <div class="explain-block">
    GBM даёт аналитически прозрачный, но "слишком гладкий" сценарий.
    IID bootstrap даёт реальную форму распределения, но без памяти рынка.
    Stationary bootstrap добавляет память (автокорреляцию, кластеры волатильности).
    Объединяя пути всех трёх в один пул перед расчётом вероятностей, ансамбль не
    зависит целиком от одного набора допущений.
    </div>
    """, unsafe_allow_html=True)
    ens_sigma = 0.015
    ens_hist = np.random.default_rng(9).standard_normal(1000) * 0.014 + \
               np.where(np.random.default_rng(10).random(1000) < 0.05,
                        np.random.default_rng(11).standard_normal(1000) * 0.04, 0)
    p_gbm = simulate_gbm(100, 0, ens_sigma, 40, 150, seed=1)
    p_iid = simulate_iid_bootstrap(100, ens_hist, 40, 150, seed=2)
    p_stat = simulate_stationary_bootstrap(100, ens_hist, 40, 150, mean_block_length=8, seed=3)
    fig5 = go.Figure()
    for arr, color, name in [(p_gbm, "rgba(76,140,130,0.5)", "GBM"),
                              (p_iid, "rgba(212,162,76,0.5)", "IID bootstrap"),
                              (p_stat, "rgba(200,90,90,0.5)", "Stationary bootstrap")]:
        for i in range(30):
            fig5.add_trace(go.Scatter(y=arr[:, i], mode="lines", line=dict(width=1, color=color),
                                       showlegend=(i == 0), name=name))
    st.plotly_chart(plotly_dark_layout(fig5, "Три метода, наложенные друг на друга (по 30 путей каждый)"),
                     use_container_width=True)

    st.divider()

    # --- 7. Calibration --------------------------------------------------
    st.markdown("## 7. Калибровочный тест (в духе Kupiec / Hosmer-Lemeshow)")
    st.markdown("""
    <div class="explain-block">
    Красивая математика ничего не значит, пока не доказано, что она предсказывает
    реальность. Идея: если модель говорит "70% вероятность", то в реальной истории
    такие случаи должны сбываться примерно в 70% случаев — не в 40% и не в 95%.
    Разбиваем предсказания на группы (децили) и сравниваем среднее предсказание
    с фактической частотой в каждой группе. Статистика хи-квадрат показывает,
    насколько это расхождение случайно или систематично.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-block">', unsafe_allow_html=True)
    st.latex(r"HL = \sum_{g=1}^{G} \frac{(O_g - E_g)^2}{E_g (1 - E_g/n_g)} \; \sim \; \chi^2_{G-2}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explain-block">
    Живой прогон этого теста на реальных данных — во вкладке «Калькулятор».
    </div>
    """, unsafe_allow_html=True)

# ===========================================================================
# ВКЛАДКА 2 — КАЛЬКУЛЯТОР
# ===========================================================================
with tab_calc:
    st.markdown("## Данные")
    c1, c2, c3 = st.columns(3)
    with c1:
        ticker = st.text_input("Тикер (Yahoo Finance)", value="AAPL")
    with c2:
        interval = st.selectbox("Интервал", ["15m", "30m", "1h", "1d"], index=0)
    with c3:
        period_options = {"15m": "60d", "30m": "60d", "1h": "730d", "1d": "5y"}
        period = period_options[interval]
        st.text_input("Период (лимит yfinance)", value=period, disabled=True)

    if interval in ("15m", "30m"):
        st.caption("⚠ yfinance отдаёт внутридневные бары (15m/30m) только за последние ~60 дней — ограничение API, не этого инструмента.")

    fetch = st.button("Загрузить данные")

    if fetch or "prices" in st.session_state:
        if fetch:
            with st.spinner("Загружаю котировки..."):
                try:
                    data = yf.download(ticker, period=period, interval=interval, progress=False)
                except Exception as e:
                    st.error(f"Ошибка загрузки: {e}")
                    data = pd.DataFrame()
            if data.empty:
                st.error("Данные не получены — проверь тикер или попробуй другой интервал.")
            else:
                close = data["Close"].dropna()
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                st.session_state["prices"] = close.values
                st.session_state["price_index"] = close.index
                st.session_state["ticker"] = ticker

        if "prices" in st.session_state:
            prices = st.session_state["prices"]
            st.success(f"{st.session_state['ticker']}: загружено {len(prices)} баров")

            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(y=prices, mode="lines", line=dict(color="#D4A24C", width=1.5)))
            st.plotly_chart(plotly_dark_layout(fig_price, "Цена закрытия"), use_container_width=True)

            st.divider()
            st.markdown("## Параметры модели")

            colv1, colv2 = st.columns(2)
            with colv1:
                lam = st.slider("λ EWMA", 0.80, 0.99, 0.94, 0.01)
            with colv2:
                block_len = st.slider("Средняя длина блока (stationary bootstrap)", 2.0, 40.0, 10.0, 1.0)

            colh1, colh2, colh3 = st.columns(3)
            with colh1:
                horizon = st.slider("Горизонт (баров вперёд)", 5, 100, 20, 1)
            with colh2:
                tp_pct = st.slider("Take-profit, %", 0.2, 10.0, 2.0, 0.1) / 100
            with colh3:
                sl_pct = st.slider("Stop-loss, %", 0.2, 10.0, 2.0, 0.1) / 100

            drift_mode = st.radio("Дрифт (μ)", ["Нулевой (рекомендуется)", "Историческое среднее (шумно на коротких данных)"],
                                   horizontal=True)
            n_sims = st.slider("Число путей на метод", 200, 5000, 1500, 100)

            run = st.button("Запустить симуляцию")

            if run:
                rets = log_returns(prices)
                sigma_series = ewma_vol(rets, lam=lam)
                sigma_now = sigma_series[-1]
                S0 = prices[-1]
                mu = 0.0 if drift_mode.startswith("Нулевой") else rets.mean()

                m1, m2, m3 = st.columns(3)
                with m1:
                    metric_card("Текущая цена", f"{S0:.2f}")
                with m2:
                    metric_card("σ за бар (EWMA)", f"{sigma_now*100:.3f}%")
                with m3:
                    metric_card("σ аннуализ. (√252·бары/день)", f"{sigma_now*np.sqrt(252*26)*100:.1f}%")

                gbm_paths = simulate_gbm(S0, mu, sigma_now, horizon, n_sims, seed=1)
                iid_paths = simulate_iid_bootstrap(S0, rets, horizon, n_sims, seed=2)
                stat_paths = simulate_stationary_bootstrap(S0, rets, horizon, n_sims,
                                                            mean_block_length=block_len, seed=3)
                ensemble_paths = np.concatenate([gbm_paths, iid_paths, stat_paths], axis=1)

                st.divider()
                st.markdown("## Веерная диаграмма путей (ансамбль)")
                pct = np.percentile(ensemble_paths, [5, 25, 50, 75, 95], axis=1)
                fig_fan = go.Figure()
                fig_fan.add_trace(go.Scatter(y=pct[4], line=dict(width=0), showlegend=False))
                fig_fan.add_trace(go.Scatter(y=pct[0], fill="tonexty", line=dict(width=0),
                                              fillcolor="rgba(212,162,76,0.12)", name="5–95%"))
                fig_fan.add_trace(go.Scatter(y=pct[3], line=dict(width=0), showlegend=False))
                fig_fan.add_trace(go.Scatter(y=pct[1], fill="tonexty", line=dict(width=0),
                                              fillcolor="rgba(212,162,76,0.28)", name="25–75%"))
                fig_fan.add_trace(go.Scatter(y=pct[2], line=dict(width=2.5, color="#D4A24C"), name="Медиана"))
                fig_fan.add_hline(y=S0*(1+tp_pct), line_dash="dash", line_color="#4C8C82", annotation_text="TP")
                fig_fan.add_hline(y=S0*(1-sl_pct), line_dash="dash", line_color="#C85A5A", annotation_text="SL")
                st.plotly_chart(plotly_dark_layout(fig_fan, "Ансамбль: перцентили путей"), use_container_width=True)

                st.divider()
                st.markdown("## Вероятности по методам")
                rows = []
                for name, arr in [("GBM", gbm_paths), ("IID bootstrap", iid_paths),
                                   ("Stationary bootstrap", stat_paths), ("Ансамбль (все три)", ensemble_paths)]:
                    p = first_passage_probabilities(arr, S0, tp_pct, sl_pct)
                    rows.append({"Метод": name, "P(TP первым)": f"{p['p_tp_first']*100:.1f}%",
                                 "P(SL первым)": f"{p['p_sl_first']*100:.1f}%",
                                 "P(ни то ни другое)": f"{p['p_neither']*100:.1f}%"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.session_state["calib_params"] = dict(lam=lam, horizon=horizon, tp_pct=tp_pct,
                                                          sl_pct=sl_pct, mu=mu)

            if "calib_params" in st.session_state:
                st.divider()
                st.markdown("## Калибровка на истории (честный walk-forward)")
                st.caption("Без заглядывания в будущее: на каждом шаге волатильность считается только по данным ДО этой точки.")
                run_calib = st.button("Запустить калибровочный тест")
                if run_calib:
                    p = st.session_state["calib_params"]
                    with st.spinner("Прогоняю walk-forward тест по истории..."):
                        try:
                            result = walk_forward_calibration(
                                prices, lam=p["lam"], horizon=p["horizon"],
                                tp_pct=p["tp_pct"], sl_pct=p["sl_pct"], mu=p["mu"],
                                n_origins=150, n_sims_per_origin=300)
                        except ValueError as e:
                            st.error(str(e))
                            result = None
                    if result:
                        table = result["table"]
                        fig_cal = go.Figure()
                        fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                                      line=dict(dash="dash", color="#9C9484"), name="Идеальная калибровка"))
                        fig_cal.add_trace(go.Scatter(x=table["mean_predicted"], y=table["mean_realized"],
                                                      mode="markers+lines", marker=dict(size=10, color="#D4A24C"),
                                                      name="Модель"))
                        fig_cal.update_layout(xaxis_title="Предсказанная вероятность",
                                               yaxis_title="Реальная частота")
                        st.plotly_chart(plotly_dark_layout(fig_cal, "Reliability diagram"), use_container_width=True)

                        cc1, cc2, cc3 = st.columns(3)
                        with cc1:
                            metric_card("HL-статистика", f"{result['hl_stat']:.2f}")
                        with cc2:
                            metric_card("Степени свободы", f"{result['dof']}")
                        with cc3:
                            metric_card("p-value", f"{result['p_value']:.3f}")

                        if result["p_value"] < 0.05:
                            st.warning("p-value < 0.05: модель статистически значимо разъезжается с реальностью на этой истории. "
                                       "Это не значит 'модель бесполезна' — значит, её нужно калибровать дальше или сузить горизонт/условия.")
                        else:
                            st.success("p-value ≥ 0.05: нет статистических оснований отвергать калибровку модели на этой истории.")

            st.divider()
            with st.expander("Проверка математики (sanity checks)"):
                if st.button("Прогнать проверки"):
                    with st.spinner("Считаю..."):
                        gbm_check = sanity_check_gbm()
                        boot_check = sanity_check_bootstrap_variance(log_returns(prices))
                    st.markdown("**GBM: эмпирика против теории (закон логнормального распределения)**")
                    st.json(gbm_check)
                    st.markdown("**Stationary bootstrap (длина блока≈1) против IID bootstrap — должны совпадать**")
                    st.json(boot_check)
