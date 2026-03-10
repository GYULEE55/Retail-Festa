# -*- coding: utf-8 -*-
"""
🎯 AI 기반 수요예측 및 매입 최적화 통합 플랫폼 v2.0
제4회 유통데이터 활용 경진대회 - 사고팔조팀

✨ 핵심 기능:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 🤖 8-Model 지능형 추천 시스템
   - SARIMAX, ARIMA, Prophet, LSTM, XGBoost, Random Forest,
     Linear Regression, Exponential Smoothing
   - 데이터 특성 기반 자동 모델 선정
   - Top 3 모델 추천 + 신뢰도 점수

2. 💰 EOQ/ROP 기반 최적 매입 추천
   - 경제적 주문량 자동 계산
   - 재주문점 기반 긴급도 판정
   - 재고회전율(ITR) 과잉재고 경보

3. 📊 Publication-Quality 시각화
   - 8개 모델 비교 차트
   - 매입 최적화 대시보드
   - 비용 절감 효과 분석

4. 📁 종합 리포트 자동 생성
   - Excel: 상세 분석 결과
   - CSV: 데이터 추출용
   - PNG: 발표/포트폴리오용
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 실행 방법:
   python3 통합_실행_v2.py

Author: 사고팔조팀 (Think & Decide)
Date: 2025
Version: 2.0
"""

import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore")

LINE = "=" * 80
SUBLINE = "-" * 80


def print_section(title: str) -> None:
    print("\n" + LINE)
    print(title)
    print(LINE)


def as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if not isinstance(value, (int, float, np.number, str)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# 프로젝트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "scripts"))


def _load_local_module(module_filename: str, module_name: str):
    module_path = project_root / module_filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈 로드 실패: {module_filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_module = _load_local_module("모델_추천_엔진_v2.py", "model_engine_v2")
purchase_module = _load_local_module(
    "추천알고리즘_매입최적화.py", "purchase_recommendation"
)

AdvancedModelRecommendationEngine = model_module.AdvancedModelRecommendationEngine
PurchaseRecommendationSystem = purchase_module.PurchaseRecommendationSystem
calculate_cost_savings = purchase_module.calculate_cost_savings

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 시각화 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "#F6F8FC"
plt.rcParams["axes.facecolor"] = "#FFFFFF"
plt.rcParams["axes.edgecolor"] = "#D6DDE8"
plt.rcParams["grid.color"] = "#DDE3EE"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlepad"] = 12

# 색상 팔레트
COLOR_PALETTE = {
    "primary": "#2563EB",
    "secondary": "#14B8A6",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "ink": "#1F2937",
    "muted": "#94A3B8",
    "panel": "#FFFFFF",
    "panel_soft": "#EEF2FF",
}

# 우선순위 색상
PRIORITY_COLORS = {
    "긴급": COLOR_PALETTE["danger"],
    "보통": COLOR_PALETTE["warning"],
    "불필요": COLOR_PALETTE["success"],
}

FONT_SCALE = {
    "title": 17,
    "axis": 13,
    "tick": 11,
    "annotation": 11,
    "kpi": 14,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 생성 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def generate_realistic_sales_data():
    """
    실전 데이터 시뮬레이션 (2023-2024년 판매 데이터)

    각 상품별 특성:
    - 생수/음료/건강: 강한 계절성 + 상승 추세 + 날씨 영향
    - 봉지라면: 불규칙 + 주말 집중 + 높은 변동성
    - 탄산음료: 주간 계절성 + 여름 피크 + 노이즈
    - 신선식품: 강한 추세 + 계절성 + 낮은 노이즈
    """
    np.random.seed(42)

    # 시간 인덱스 (1년치 데이터)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")

    # 1. 생수/음료/건강 - 계절성(주간+월간) + 추세 + 날씨 효과
    t = np.arange(365)
    weekly_season = 50 * np.sin(2 * np.pi * t / 7)  # 주간 패턴
    monthly_season = 100 * np.sin(2 * np.pi * t / 30)  # 월간 패턴
    trend = t * 0.5  # 상승 추세
    noise = np.random.normal(0, 20, 365)  # 노이즈
    sales_water = weekly_season + monthly_season + trend + noise + 500
    sales_water = pd.Series(
        np.maximum(sales_water, 0), index=dates, name="생수/음료/건강"
    )

    # 2. 봉지라면 - 불규칙 패턴 + 주말 피크
    base_sales = np.random.normal(300, 80, 365)
    weekend_boost = np.array([50 if date.dayofweek >= 5 else 0 for date in dates])
    sales_ramen = base_sales + weekend_boost
    sales_ramen = pd.Series(np.maximum(sales_ramen, 0), index=dates, name="봉지라면")

    # 3. 탄산음료 - 강한 주간 계절성 + 여름 피크
    weekly_strong = 150 * np.sin(2 * np.pi * t / 7)
    summer_boost = 80 * np.exp(-((t - 180) ** 2) / 5000)  # 여름 피크 (7월)
    noise2 = np.random.normal(0, 30, 365)
    sales_soda = weekly_strong + summer_boost + noise2 + 200
    sales_soda = pd.Series(np.maximum(sales_soda, 0), index=dates, name="탄산음료")

    # 4. 신선식품 - 강한 추세 + 약한 계절성
    strong_trend = t * 1.2
    seasonal_weak = 30 * np.sin(2 * np.pi * t / 365)
    noise3 = np.random.normal(0, 15, 200)  # 200일치만
    sales_fresh = strong_trend[:200] + seasonal_weak[:200] + noise3 + 100
    sales_fresh = pd.Series(
        np.maximum(sales_fresh, 0),
        index=pd.date_range("2023-01-01", periods=200, freq="D"),
        name="신선식품",
    )

    return {
        "생수/음료/건강": sales_water,
        "봉지라면": sales_ramen,
        "탄산음료": sales_soda,
        "신선식품": sales_fresh,
    }


def prepare_purchase_data():
    """매입 최적화용 데이터 준비"""
    return pd.DataFrame(
        {
            "중분류": ["생수/음료/건강", "신선식품", "봉지라면", "탄산음료"],
            "순매출수량_예측": [4500, 320, 2800, 1900],
            "순매출금액": [4500000, 640000, 2800000, 1900000],
            "EOQ": [3532, 531, 2400, 1600],
            "ROP": [2077, 82, 1200, 800],
            "현재재고": [1500, 400, 1000, 1200],
            "L": [4.4, 1.8, 3.0, 2.5],
            "d": [134, 16, 95, 65],
            "ITR": [3.2, 5.1, 4.8, 2.9],
        }
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 시각화 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _style_panel(ax):
    ax.set_facecolor(COLOR_PALETTE["panel"])
    for spine in ax.spines.values():
        spine.set_color("#CDD6E4")
        spine.set_linewidth(1.0)


def _plot_model_scores(ax, model_results):
    product_name = "생수/음료/건강"
    all_scores = model_results[product_name]["all_scores"]
    models = list(all_scores.keys())
    scores = list(all_scores.values())
    score_colors = [
        COLOR_PALETTE["primary"],
        COLOR_PALETTE["secondary"],
        COLOR_PALETTE["warning"],
    ] + [COLOR_PALETTE["muted"]] * max(0, len(models) - 3)

    bars = ax.barh(
        models,
        scores,
        color=score_colors[: len(models)],
        edgecolor="#334155",
        linewidth=0.8,
    )
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(
            score + 0.12,
            i,
            f"{score:.1f}",
            va="center",
            fontsize=FONT_SCALE["annotation"],
            color=COLOR_PALETTE["ink"],
        )

    ax.set_xlabel("적합도 점수 (0-10)", fontsize=FONT_SCALE["axis"])
    ax.set_title(
        f"{product_name} - 8개 모델 적합도 (대표 품목)",
        fontsize=FONT_SCALE["title"],
    )
    ax.set_xlim(0, max(10.5, max(scores) + 0.8))
    ax.tick_params(labelsize=FONT_SCALE["tick"])
    ax.grid(axis="x", alpha=0.35)
    ax.axvline(
        7.5,
        color=COLOR_PALETTE["danger"],
        linestyle="--",
        alpha=0.65,
        label="우수 기준",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=FONT_SCALE["tick"])


def _plot_confidence(ax, model_results):
    products = list(model_results.keys())
    top_models = [
        model_results[p]["recommendations"][0]["model_name"] for p in products
    ]
    confidences = [
        model_results[p]["recommendations"][0]["confidence"] for p in products
    ]
    x_pos = np.arange(len(products))
    bars = ax.bar(
        x_pos,
        confidences,
        color=COLOR_PALETTE["secondary"],
        edgecolor="#0F172A",
        linewidth=0.8,
        alpha=0.9,
    )

    for bar, conf, model in zip(bars, confidences, top_models):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{conf:.1f}%\n{model}",
            ha="center",
            va="bottom",
            fontsize=FONT_SCALE["annotation"],
            color=COLOR_PALETTE["ink"],
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(products, fontsize=FONT_SCALE["tick"], rotation=8, ha="right")
    ax.set_ylabel("신뢰도 (%)", fontsize=FONT_SCALE["axis"])
    ax.set_title("상품별 추천 모델 신뢰도", fontsize=FONT_SCALE["title"])
    ax.set_ylim(0, 116)
    ax.grid(axis="y", alpha=0.35)
    ax.axhline(
        30, color=COLOR_PALETTE["success"], linestyle="--", alpha=0.8, label="신뢰 기준"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=FONT_SCALE["tick"])


def _plot_purchase(ax, purchase_results):
    products_pur = purchase_results["중분류"].values
    recommended_qty = purchase_results["추천수량"].values
    current_stock = purchase_results["현재재고"].values
    priorities = purchase_results["우선순위"].values
    x = np.arange(len(products_pur))
    width = 0.34

    bars_stock = ax.bar(
        x - width / 2,
        current_stock,
        width,
        label="현재 재고",
        color=COLOR_PALETTE["panel_soft"],
        edgecolor="#64748B",
        linewidth=0.8,
    )
    bars_reco = ax.bar(
        x + width / 2,
        recommended_qty,
        width,
        label="추천 수량",
        color=[PRIORITY_COLORS[p] for p in priorities],
        edgecolor="#374151",
        linewidth=0.8,
    )

    max_qty = max(float(np.max(current_stock)), float(np.max(recommended_qty)))
    ax.set_ylim(0, max_qty * 1.35)

    for bars in [bars_stock, bars_reco]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max_qty * 0.018,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=FONT_SCALE["annotation"],
                color=COLOR_PALETTE["ink"],
            )

    for i, priority in enumerate(priorities):
        ax.text(
            i,
            max(current_stock[i], recommended_qty[i]) + max_qty * 0.09,
            priority,
            ha="center",
            fontsize=FONT_SCALE["annotation"],
            color=COLOR_PALETTE["ink"],
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="#F8FAFC",
                edgecolor="#CBD5E1",
                linewidth=0.8,
            ),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        products_pur, fontsize=FONT_SCALE["tick"], rotation=8, ha="right"
    )
    ax.set_ylabel("수량", fontsize=FONT_SCALE["axis"])
    ax.set_title("매입 최적화 추천 (EOQ/ROP)", fontsize=FONT_SCALE["title"])
    ax.legend(loc="upper left", fontsize=FONT_SCALE["tick"], ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.35)


def _plot_savings(ax, savings_analysis):
    categories = list(savings_analysis.keys())
    amounts = [savings_analysis[cat]["총_절감액"] for cat in categories]
    rates = [savings_analysis[cat]["절감률(%)"] for cat in categories]
    max_abs_amount = max(abs(float(v)) for v in amounts) if amounts else 1.0
    saving_colors = [
        COLOR_PALETTE["success"] if float(v) >= 0 else COLOR_PALETTE["danger"]
        for v in amounts
    ]

    bars = ax.barh(
        categories,
        amounts,
        color=saving_colors,
        edgecolor="#334155",
        linewidth=0.8,
        alpha=0.85,
    )
    for i, (bar, amount, rate) in enumerate(zip(bars, amounts, rates)):
        amount_f = float(amount)
        x_text = amount_f + (
            0.03 * max_abs_amount if amount_f >= 0 else -0.03 * max_abs_amount
        )
        align = "left" if amount_f >= 0 else "right"
        trend_word = "절감" if amount_f >= 0 else "증가"
        ax.text(
            x_text,
            i,
            f"{amount_f:,.0f}원\n({rate:.1f}% {trend_word})",
            va="center",
            ha=align,
            fontsize=FONT_SCALE["annotation"],
            color=COLOR_PALETTE["ink"],
        )

    ax.axvline(0, color="#6B7280", linewidth=1.0)
    ax.set_xlim(-max_abs_amount * 1.15, max_abs_amount * 1.15)
    ax.set_xlabel("비용 변화 (원)", fontsize=FONT_SCALE["axis"])
    ax.set_title("비용 변화 분석", fontsize=FONT_SCALE["title"])
    ax.tick_params(labelsize=FONT_SCALE["tick"])
    ax.grid(axis="x", alpha=0.35)


def _plot_kpi(ax, model_results, purchase_results, savings_analysis):
    ax.set_facecolor(COLOR_PALETTE["panel"])
    ax.axis("off")

    amounts = [float(v["총_절감액"]) for v in savings_analysis.values()]
    total_savings = sum(amounts)
    avg_confidence = np.mean(
        [
            model_results[p]["recommendations"][0]["confidence"]
            for p in model_results.keys()
        ]
    )
    urgent_count = len(purchase_results[purchase_results["우선순위"] == "긴급"])
    total_products = len(purchase_results)

    kpi_text = (
        "핵심 성과 지표\n\n"
        f"총 비용 절감액: {total_savings:,.0f}원\n"
        f"평균 모델 신뢰도: {avg_confidence:.1f}%\n"
        f"긴급 발주 상품: {urgent_count}/{total_products}개\n"
        f"분석 완료 상품: {total_products}개"
    )

    ax.text(
        0.5,
        0.5,
        kpi_text,
        ha="center",
        va="center",
        fontsize=FONT_SCALE["kpi"],
        color=COLOR_PALETTE["ink"],
        linespacing=1.6,
        bbox=dict(
            boxstyle="round,pad=0.95",
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            linewidth=1.6,
        ),
    )


def create_comprehensive_visualization(
    model_results, purchase_results, savings_analysis
):
    """
    종합 대시보드 생성

    Layout:
    ┌─────────────────┬─────────────────┐
    │  모델 추천 차트   │  신뢰도 비교     │
    ├─────────────────┼─────────────────┤
    │  매입 최적화     │  비용 절감 효과  │
    ├─────────────────┴─────────────────┤
    │         핵심 지표 요약             │
    └───────────────────────────────────┘
    """
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(
        3, 2, height_ratios=[1.1, 1.1, 0.75], hspace=0.44, wspace=0.30
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    for ax in [ax1, ax2, ax3, ax4]:
        _style_panel(ax)

    _plot_model_scores(ax1, model_results)
    _plot_confidence(ax2, model_results)
    _plot_purchase(ax3, purchase_results)
    _plot_savings(ax4, savings_analysis)
    _plot_kpi(ax5, model_results, purchase_results, savings_analysis)

    plt.suptitle(
        "AI 기반 수요예측 및 매입 최적화 통합 대시보드",
        fontsize=19,
        fontweight="bold",
        color=COLOR_PALETTE["ink"],
        y=0.975,
    )
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.06, right=0.98)
    return fig


def create_split_visualizations(model_results, purchase_results, savings_analysis):
    fig_model = plt.figure(figsize=(21, 8))
    gs_model = gridspec.GridSpec(1, 2, wspace=0.28)
    ax_m1 = fig_model.add_subplot(gs_model[0, 0])
    ax_m2 = fig_model.add_subplot(gs_model[0, 1])
    for ax in [ax_m1, ax_m2]:
        _style_panel(ax)
    _plot_model_scores(ax_m1, model_results)
    _plot_confidence(ax_m2, model_results)
    fig_model.suptitle(
        "모델 추천 분석", fontsize=22, fontweight="bold", color=COLOR_PALETTE["ink"]
    )
    fig_model.text(
        0.5,
        0.90,
        "상품별 예측모델 적합도와 신뢰도 비교",
        ha="center",
        va="center",
        fontsize=13,
        color="#475569",
    )
    fig_model.subplots_adjust(top=0.84, bottom=0.14, left=0.06, right=0.98)

    fig_ops = plt.figure(figsize=(21, 8))
    gs_ops = gridspec.GridSpec(1, 2, wspace=0.28)
    ax_o1 = fig_ops.add_subplot(gs_ops[0, 0])
    ax_o2 = fig_ops.add_subplot(gs_ops[0, 1])
    for ax in [ax_o1, ax_o2]:
        _style_panel(ax)
    _plot_purchase(ax_o1, purchase_results)
    _plot_savings(ax_o2, savings_analysis)
    fig_ops.suptitle(
        "매입/비용 분석", fontsize=22, fontweight="bold", color=COLOR_PALETTE["ink"]
    )
    fig_ops.text(
        0.5,
        0.90,
        "추천 발주 수량과 비용 변화 결과",
        ha="center",
        va="center",
        fontsize=13,
        color="#475569",
    )
    fig_ops.subplots_adjust(top=0.84, bottom=0.14, left=0.06, right=0.98)

    fig_kpi = plt.figure(figsize=(12, 4.8))
    ax_kpi = fig_kpi.add_subplot(111)
    _plot_kpi(ax_kpi, model_results, purchase_results, savings_analysis)
    fig_kpi.suptitle(
        "KPI 요약", fontsize=18, fontweight="bold", color=COLOR_PALETTE["ink"]
    )
    fig_kpi.subplots_adjust(top=0.93, bottom=0.08, left=0.04, right=0.96)

    return {
        "model": fig_model,
        "ops": fig_ops,
        "kpi": fig_kpi,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 실행 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    """통합 실행 메인 함수"""

    print(LINE)
    print("AI 기반 수요예측 및 매입 최적화 통합 플랫폼 v2.0")
    print("   제4회 유통데이터 활용 경진대회 - 사고팔조팀")
    print(LINE)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART 1: 모델 추천 시스템
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print_section("PART 1: 8-Model 지능형 추천 시스템")

    # 데이터 생성
    sales_data_dict = generate_realistic_sales_data()

    # 모델 추천 엔진 초기화
    engine = AdvancedModelRecommendationEngine()

    # 모델 추천 결과 저장
    model_results = {}

    print("\n상품별 최적 모델 분석 중...\n")

    for product_name, sales_data in sales_data_dict.items():
        result = engine.recommend_models(sales_data, product_name, top_k=3)
        model_results[product_name] = result

        print(SUBLINE)
        print(f"상품: {product_name}")
        print(
            f"   데이터: {result['data_characteristics']['sample_size']}일 | "
            f"계절성: {'✓' if result['data_characteristics']['has_seasonality'] else '✗'} | "
            f"추세: {result['data_characteristics']['trend_strength']:.2f} | "
            f"노이즈: {result['data_characteristics']['noise_level']:.2f}"
        )

        print("\n   Top 3 추천 모델:")
        for rec in result["recommendations"]:
            print(f"      {rec['rank']}위. {rec['model_name']} ({rec['full_name']})")
            print(
                f"          점수: {rec['score']:.1f}/10 | 신뢰도: {rec['confidence']:.1f}% | "
                f"복잡도: {rec['complexity']}"
            )

        print(f"\n   요약: {result['summary']}")

    print("\n" + SUBLINE)
    print("모델 추천 완료\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART 2: 매입 최적화 추천
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print_section("PART 2: EOQ/ROP 기반 매입 최적화")

    # 추천 시스템 초기화
    recommender = PurchaseRecommendationSystem(
        safety_stock_multiplier=1.2, min_itr_threshold=4.0, max_order_qty_multiplier=3.0
    )

    # 매입 데이터 준비
    purchase_data = prepare_purchase_data()

    # 추천 실행
    recommendations = recommender.batch_recommend(purchase_data)

    # 중복 컬럼 제거 (batch_recommend가 원본 DF와 결과를 concat하므로 중복 발생 가능)
    recommendations = recommendations.loc[:, ~recommendations.columns.duplicated()]

    # 우선순위 정렬
    prioritized = recommender.prioritize(recommendations)

    print("\n매입 최적화 추천 결과\n")

    # 요약 리포트 먼저 출력
    summary = recommender.generate_summary_report(recommendations)

    print(f"- 총 분석 상품: {summary['총_상품수']}개")
    print(
        f"- 긴급 발주 필요: {summary['긴급_상품수']}개 ({summary['긴급_비율(%)']:.1f}%)"
    )
    print(f"- 과잉재고 경고: {summary['과잉재고_경고_상품수']}개")
    print(f"- 총 추천 매입량: {summary['총_추천수량']:,.0f}개")
    if summary["예상_총비용"]:
        print(f"- 예상 총비용: {summary['예상_총비용']:,.1f}원")
    print(
        f"- 평균 재고일 개선: {summary['평균_현재재고일']:.1f}일 -> {summary['평균_재입고후재고일']:.1f}일"
    )
    print()

    # 상품별 상세 결과 테이블
    print(
        f"{'센터':<8} {'상품':<16} {'우선순위':<8} {'추천수량':>10} {'재고일':>10} {'ITR':>6} {'경고':<4}"
    )
    print(SUBLINE)

    for _, row in prioritized.iterrows():
        # 우선순위 아이콘
        if row["우선순위"] == "긴급":
            priority_icon = "긴급"
        elif row["우선순위"] == "보통":
            priority_icon = "보통"
        else:
            priority_icon = "불필요"

        # ITR 경고 표시
        itr_warning = "Y" if bool(row["과잉재고_경고"]) else "N"

        # 센터는 임시로 상품명 기반 분류 (실제로는 데이터에서 가져와야 함)
        center = "A센터" if row["중분류"] in ["생수/음료/건강", "신선식품"] else "B센터"

        print(
            f"{center:<8s} {row['중분류']:<16s} {priority_icon:<8s} {row['추천수량']:>10,.0f} {row['예상재고일']:>8.1f}일 {row['ITR']:>6.1f} {itr_warning:<4s}"
        )

    print()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART 3: 비용 절감 분석
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print_section("PART 3: 비용 절감 효과 분석")

    savings_analysis = {}

    # 상품별 절감액 계산
    original_purchases = [5000, 600, 3000, 2200]  # 기존 발주량 (예시)

    for i in range(len(prioritized)):
        row = prioritized.iloc[i]
        product = str(row["중분류"])
        recommended = float(row["추천수량"])
        original = float(original_purchases[i])
        predicted_qty = float(row["순매출수량_예측"])
        unit_cost = (
            float(row["순매출금액"]) / predicted_qty if predicted_qty > 0 else 1000.0
        )

        savings = calculate_cost_savings(
            original_purchase=original,
            recommended_purchase=recommended,
            unit_cost=unit_cost,
            holding_cost_rate=0.2,
        )

        savings_analysis[product] = savings

        total_delta = as_float(savings.get("총_절감액"), 0.0)
        trend_word = "절감" if total_delta >= 0 else "증가"
        print(f"\n{product}")
        print(f"   기존 발주: {original:,.0f}개 -> 추천 발주: {recommended:,.0f}개")
        print(
            f"   총 비용 변화: {savings['총_절감액']:,.0f}원 ({savings['절감률(%)']:.1f}% {trend_word})"
        )
        print(
            f"   └─ 매입비용: {savings['매입비용_절감']:,.0f}원 | 보관비용: {savings['보관비용_절감']:,.0f}원"
        )

    total_savings = sum(s["총_절감액"] for s in savings_analysis.values())
    print("\n" + SUBLINE)
    print(f"전체 절감액 합계: {total_savings:,.0f}원")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART 4: 시각화 및 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print_section("PART 4: 결과 시각화 및 저장")

    # 출력 디렉토리 생성
    output_dir = Path("images")
    results_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # 통합 대시보드 생성
    print("\n통합 대시보드 생성 중...")
    fig = create_comprehensive_visualization(
        model_results, prioritized, savings_analysis
    )
    split_figs = create_split_visualizations(
        model_results, prioritized, savings_analysis
    )

    # PNG 저장
    png_path = output_dir / "통합_실행_결과_v2.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"- PNG 저장: {png_path}")

    model_png_path = output_dir / "포트폴리오_모델분석_v2.png"
    split_figs["model"].savefig(
        model_png_path, dpi=300, bbox_inches="tight", facecolor="white"
    )
    print(f"- 포트폴리오(모델) 저장: {model_png_path}")

    ops_png_path = output_dir / "포트폴리오_매입비용분석_v2.png"
    split_figs["ops"].savefig(
        ops_png_path, dpi=300, bbox_inches="tight", facecolor="white"
    )
    print(f"- 포트폴리오(매입/비용) 저장: {ops_png_path}")

    kpi_png_path = output_dir / "포트폴리오_KPI_v2.png"
    split_figs["kpi"].savefig(
        kpi_png_path, dpi=300, bbox_inches="tight", facecolor="white"
    )
    print(f"- 포트폴리오(KPI) 저장: {kpi_png_path}")

    plt.close(fig)
    plt.close(split_figs["model"])
    plt.close(split_figs["ops"])
    plt.close(split_figs["kpi"])

    # Excel 저장
    excel_path = results_dir / "통합_실행_결과_v2.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # 모델 추천 결과
        model_summary = []
        for product, result in model_results.items():
            for rec in result["recommendations"]:
                model_summary.append(
                    {
                        "상품명": product,
                        "순위": rec["rank"],
                        "추천모델": rec["model_name"],
                        "모델설명": rec["full_name"],
                        "점수": rec["score"],
                        "신뢰도(%)": rec["confidence"],
                        "복잡도": rec["complexity"],
                    }
                )
        pd.DataFrame(model_summary).to_excel(writer, sheet_name="모델추천", index=False)

        # 매입 최적화 결과
        prioritized.to_excel(writer, sheet_name="매입최적화", index=False)

        # 비용 절감 분석
        savings_df = pd.DataFrame(savings_analysis).T
        savings_df.index.name = "상품명"
        savings_df.to_excel(writer, sheet_name="비용절감")

    print(f"- Excel 저장: {excel_path}")

    # CSV 저장
    csv_path = results_dir / "통합_실행_결과_v2.csv"
    prioritized.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"- CSV 저장: {csv_path}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 최종 요약
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + LINE)
    print("통합 실행 완료")
    print(LINE)
    print("\n생성된 파일:")
    print(f"   - {png_path}")
    print(f"   - {model_png_path}")
    print(f"   - {ops_png_path}")
    print(f"   - {kpi_png_path}")
    print(f"   - {excel_path}")
    print(f"   - {csv_path}")

    print("\n핵심 결과:")
    print(f"   - 분석 상품: {len(model_results)}개")
    print("   - 추천 모델 수: 8개")
    print(f"   - 긴급 발주: {summary['긴급_상품수']}개")
    print(f"   - 총 절감액: {total_savings:,.0f}원")

    print("\n" + LINE)
    print("사고팔조팀 - AI 기반 수요예측 및 매입 최적화 시스템")
    print('   "센터별로 사고, 데이터로 판다"')
    print(LINE + "\n")

    # plt.show()  # 자동화 환경에서는 주석 처리


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
