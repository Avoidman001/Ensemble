"""
engine.py — математическое ядро M15 Quant Academy.

Всё в этом файле работает в "per-bar" единицах (на одну M15-свечу),
если явно не указано иное. Аннуализация делается только для отображения.

Компоненты:
1. log_returns          — логарифмические доходности
2. ewma_vol             — EWMA-волатильность (RiskMetrics-стиль)
3. simulate_gbm         — геометрическое броуновское движение
4. simulate_iid_bootstrap        — IID bootstrap по историческим доходностям
5. simulate_stationary_bootstrap — stationary bootstrap (Politis & Romano, 1994)
6. first_passage_probabilities   — вероятности TP-first / SL-first / neither
7. walk_forward_calibration      — калибровочный тест в стиле Hosmer-Lemeshow
8. sanity_checks                 — проверка математики против теории
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Базовые доходности
# ---------------------------------------------------------------------------

def log_returns(prices: np.ndarray) -> np.ndarray:
    """Логарифмические доходности: r_t = ln(P_t / P_{t-1})."""
    prices = np.asarray(prices, dtype=float)
    return np.diff(np.log(prices))


# ---------------------------------------------------------------------------
# 2. EWMA-волатильность
# ---------------------------------------------------------------------------

def ewma_vol(returns: np.ndarray, lam: float = 0.94, init_window: int = 30) -> np.ndarray:
    """
    Рекурсивная EWMA-оценка дисперсии (RiskMetrics):
        sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}^2

    Возвращает массив той же длины, что и returns — sigma[t] это оценка
    волатильности НА МОМЕНТ t, построенная только на данных до t
    (каузально, без заглядывания в будущее — важно для честного бэктеста).
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    sigma2 = np.empty(n)

    init_window = min(init_window, n)
    sigma2[0] = np.var(returns[:init_window]) if init_window > 1 else returns[0] ** 2

    for t in range(1, n):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * returns[t - 1] ** 2

    return np.sqrt(sigma2)


# ---------------------------------------------------------------------------
# 3. GBM симуляция
# ---------------------------------------------------------------------------

def simulate_gbm(S0: float, mu: float, sigma: float, horizon: int, n_sims: int,
                  seed: int | None = None) -> np.ndarray:
    """
    Пути GBM: S(t) = S0 * exp( cumsum( (mu - 0.5*sigma^2) + sigma * Z ) )
    Z ~ IID N(0,1). mu и sigma — per-bar величины (dt = 1 бар).

    Возвращает массив shape (horizon, n_sims).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(horizon, n_sims))
    increments = (mu - 0.5 * sigma ** 2) + sigma * z
    log_paths = np.cumsum(increments, axis=0)
    return S0 * np.exp(log_paths)


# ---------------------------------------------------------------------------
# 4. IID bootstrap
# ---------------------------------------------------------------------------

def simulate_iid_bootstrap(S0: float, hist_returns: np.ndarray, horizon: int,
                            n_sims: int, seed: int | None = None) -> np.ndarray:
    """
    Классический IID bootstrap: каждый шаг пути — случайная точка из
    исторических лог-доходностей, независимо. Разрушает автокорреляцию
    и кластеризацию волатильности, но не требует нормальности.
    """
    rng = np.random.default_rng(seed)
    n = len(hist_returns)
    idx = rng.integers(0, n, size=(horizon, n_sims))
    sampled_returns = hist_returns[idx]
    log_paths = np.cumsum(sampled_returns, axis=0)
    return S0 * np.exp(log_paths)


# ---------------------------------------------------------------------------
# 5. Stationary bootstrap (Politis & Romano, 1994)
# ---------------------------------------------------------------------------

def simulate_stationary_bootstrap(S0: float, hist_returns: np.ndarray, horizon: int,
                                   n_sims: int, mean_block_length: float = 10.0,
                                   seed: int | None = None) -> np.ndarray:
    """
    Stationary bootstrap: блоки СЛУЧАЙНОЙ длины (геометрическое распределение,
    параметр p = 1/mean_block_length). С вероятностью p стартуем новый блок
    со случайной точки, иначе продолжаем текущий блок (индекс + 1, циклически).

    Это академический стандарт для сохранения автокорреляции и кластеризации
    волатильности в bootstrap-выборке (в отличие от IID bootstrap выше).

    Векторизовано по n_sims: цикл идёт только по horizon шагам, не по путям.
    """
    rng = np.random.default_rng(seed)
    n = len(hist_returns)
    p = 1.0 / mean_block_length

    idx = np.empty((horizon, n_sims), dtype=np.int64)
    idx[0, :] = rng.integers(0, n, size=n_sims)

    restart_draws = rng.random(size=(horizon, n_sims))
    for t in range(1, horizon):
        restart_mask = restart_draws[t] < p
        continued = (idx[t - 1, :] + 1) % n
        fresh = rng.integers(0, n, size=n_sims)
        idx[t, :] = np.where(restart_mask, fresh, continued)

    sampled_returns = hist_returns[idx]
    log_paths = np.cumsum(sampled_returns, axis=0)
    return S0 * np.exp(log_paths)


# ---------------------------------------------------------------------------
# 6. Вероятности достижения TP/SL (first-passage по симулированным путям)
# ---------------------------------------------------------------------------

def first_passage_probabilities(paths: np.ndarray, S0: float, tp_pct: float,
                                 sl_pct: float) -> dict:
    """
    paths: shape (horizon, n_sims), уже включает S0 как базу (paths[0] — первый шаг ОТ S0).
    tp_pct, sl_pct — положительные числа в долях (0.01 = 1%).

    Возвращает P(TP раньше SL), P(SL раньше TP), P(ни то ни другое за горизонт).
    """
    tp_level = S0 * (1 + tp_pct)
    sl_level = S0 * (1 - sl_pct)

    hit_tp = paths >= tp_level
    hit_sl = paths <= sl_level

    horizon, n_sims = paths.shape
    tp_first = np.zeros(n_sims, dtype=bool)
    sl_first = np.zeros(n_sims, dtype=bool)
    resolved = np.zeros(n_sims, dtype=bool)

    for t in range(horizon):
        newly_tp = hit_tp[t] & ~resolved
        newly_sl = hit_sl[t] & ~resolved
        # если оба уровня пробиты в одном баре — считаем как SL (консервативно)
        tp_first |= newly_tp & ~newly_sl
        sl_first |= newly_sl
        resolved |= newly_tp | newly_sl

    neither = ~resolved

    return {
        "p_tp_first": tp_first.mean(),
        "p_sl_first": sl_first.mean(),
        "p_neither": neither.mean(),
    }


# ---------------------------------------------------------------------------
# 7. Walk-forward калибровочный тест (Hosmer-Lemeshow style, идея из Kupiec POF)
# ---------------------------------------------------------------------------

def walk_forward_calibration(prices: np.ndarray, lam: float, horizon: int,
                              tp_pct: float, sl_pct: float, n_origins: int = 150,
                              n_sims_per_origin: int = 300, mu: float = 0.0,
                              n_buckets: int = 10, seed: int = 42) -> dict:
    """
    Честный walk-forward тест без заглядывания в будущее:
    1. Для каждой "точки отсчёта" origin_t считаем EWMA-vol, используя
       только данные ДО этой точки.
    2. Прогоняем GBM-симуляцию (быстрый прокси всего ансамбля) и получаем
       p_hat = P(TP раньше SL за горизонт), предсказанную моделью.
    3. Смотрим РЕАЛЬНОЕ будущее (следующие `horizon` баров) и проверяем,
       что случилось на самом деле: TP раньше SL -> outcome=1, иначе 0.
    4. Группируем origins по децилям p_hat и сравниваем среднее предсказание
       с реальной частотой в группе — это тест калибровки в духе
       Hosmer-Lemeshow (та же семья идей, что Kupiec POF test для VaR,
       но корректно работает при ПЕРЕМЕННОЙ p, а не только при константной).

    Возвращает DataFrame по бакетам + статистику хи-квадрат + p-value.
    """
    prices = np.asarray(prices, dtype=float)
    rets = log_returns(prices)
    sigma_series = ewma_vol(rets, lam=lam)

    n = len(prices)
    max_origin = n - horizon - 1
    min_origin = 30  # нужна история для начальной EWMA-оценки

    if max_origin <= min_origin:
        raise ValueError("Недостаточно исторических данных для калибровочного теста.")

    origins = np.linspace(min_origin, max_origin, num=min(n_origins, max_origin - min_origin),
                           dtype=int)
    origins = np.unique(origins)

    rng_master = np.random.default_rng(seed)
    predicted = np.empty(len(origins))
    realized = np.empty(len(origins))

    for i, t in enumerate(origins):
        sigma_t = sigma_series[t - 1]  # оценка волатильности, известная на момент t
        S0 = prices[t]

        sim_paths = simulate_gbm(S0, mu, sigma_t, horizon, n_sims_per_origin,
                                  seed=int(rng_master.integers(0, 2**31 - 1)))
        probs = first_passage_probabilities(sim_paths, S0, tp_pct, sl_pct)
        predicted[i] = probs["p_tp_first"]

        # реальное будущее
        future = prices[t: t + horizon + 1]
        tp_level = S0 * (1 + tp_pct)
        sl_level = S0 * (1 - sl_pct)
        outcome = 0
        for future_price in future[1:]:
            if future_price >= tp_level:
                outcome = 1
                break
            if future_price <= sl_level:
                outcome = 0
                break
        realized[i] = outcome

    df = pd.DataFrame({"predicted": predicted, "realized": realized})
    df["bucket"] = pd.qcut(df["predicted"], q=min(n_buckets, df["predicted"].nunique()),
                            duplicates="drop")

    grouped = df.groupby("bucket", observed=True).agg(
        n=("realized", "size"),
        mean_predicted=("predicted", "mean"),
        mean_realized=("realized", "mean"),
        sum_predicted=("predicted", "sum"),
        sum_realized=("realized", "sum"),
    ).reset_index()

    # Hosmer-Lemeshow статистика
    O = grouped["sum_realized"].values
    E = grouped["sum_predicted"].values
    ng = grouped["n"].values
    denom = E * (1 - E / ng)
    denom = np.where(denom <= 0, np.nan, denom)
    hl_terms = (O - E) ** 2 / denom
    hl_stat = np.nansum(hl_terms)
    dof = max(len(grouped) - 2, 1)
    p_value = 1 - stats.chi2.cdf(hl_stat, dof)

    return {
        "table": grouped,
        "hl_stat": hl_stat,
        "dof": dof,
        "p_value": p_value,
        "n_origins_used": len(origins),
    }


# ---------------------------------------------------------------------------
# 8. Sanity checks — проверка математики против теории
# ---------------------------------------------------------------------------

def sanity_check_gbm(S0: float = 100.0, mu: float = 0.0, sigma: float = 0.01,
                      horizon: int = 20, n_sims: int = 200_000, seed: int = 1) -> dict:
    """
    Терминальное распределение GBM теоретически логнормально:
    ln(S_T/S0) ~ N( (mu - 0.5*sigma^2)*T , sigma^2*T )
    Сверяем эмпирические среднее/дисперсию с теорией.
    """
    paths = simulate_gbm(S0, mu, sigma, horizon, n_sims, seed=seed)
    terminal_log_ret = np.log(paths[-1] / S0)

    theory_mean = (mu - 0.5 * sigma ** 2) * horizon
    theory_var = sigma ** 2 * horizon

    return {
        "empirical_mean": terminal_log_ret.mean(),
        "theory_mean": theory_mean,
        "empirical_var": terminal_log_ret.var(),
        "theory_var": theory_var,
        "mean_abs_error": abs(terminal_log_ret.mean() - theory_mean),
        "var_abs_error": abs(terminal_log_ret.var() - theory_var),
    }


def sanity_check_bootstrap_variance(hist_returns: np.ndarray, horizon: int = 20,
                                     n_sims: int = 50_000, seed: int = 1) -> dict:
    """
    При mean_block_length -> 1, stationary bootstrap должен статистически
    совпадать по дисперсии терминальной доходности с IID bootstrap
    (блоки длины ~1 = почти независимые точки).
    """
    S0 = 100.0
    iid_paths = simulate_iid_bootstrap(S0, hist_returns, horizon, n_sims, seed=seed)
    stat_paths = simulate_stationary_bootstrap(S0, hist_returns, horizon, n_sims,
                                                mean_block_length=1.01, seed=seed)

    iid_term = np.log(iid_paths[-1] / S0)
    stat_term = np.log(stat_paths[-1] / S0)

    return {
        "iid_var": iid_term.var(),
        "stationary_block1_var": stat_term.var(),
        "relative_diff_pct": abs(iid_term.var() - stat_term.var()) / iid_term.var() * 100,
    }
